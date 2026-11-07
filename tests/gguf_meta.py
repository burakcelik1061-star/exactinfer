"""Bağımsız, küçük GGUF okuyucu (Python standart kütüphanesi + isteğe bağlı numpy).

Testlerde motorun C++ okuyucusundan bağımsız bir başvuru olsun diye ayrı yazıldı.
"""

import struct

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _parse(path):
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(n))[0]

        def rstr():
            n = rd("<Q")
            return f.read(n).decode("utf-8", errors="surrogateescape")

        def rval(t):
            if t in _SCALAR:
                return rd(_SCALAR[t])
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("<I"), rd("<Q")
                return [rval(et) for _ in range(n)]
            raise ValueError(f"bilinmeyen tur {t}")

        if f.read(4) != b"GGUF":
            raise ValueError("GGUF degil")
        rd("<I")
        n_t = rd("<Q")
        n_kv = rd("<Q")
        meta = {}
        for _ in range(n_kv):
            k = rstr()
            meta[k] = rval(rd("<I"))
        tensors = {}
        for _ in range(n_t):
            name = rstr()
            nd = rd("<I")
            ne = [rd("<Q") for _ in range(nd)]
            ty = rd("<I")
            off = rd("<Q")
            tensors[name] = (ne, ty, off)
        align = meta.get("general.alignment", 32)
        data_start = (f.tell() + align - 1) // align * align
        return meta, tensors, data_start


def read_metadata(path):
    return _parse(path)[0]


# ggml türü → (blok boyu, blok baytı)
TYPE_SIZE = {0: (1, 4), 1: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2)}


class GgufTensors:
    """Tensörleri numpy dizisi olarak (bellek eşlemeli) verir; fp32 döner.

    Nicemli türler llama.cpp'nin resmi `gguf` paketindeki numpy çözücüleriyle açılır (motorun C++/CUDA
    çözücülerinden bağımsız başvuru).
    """

    def __init__(self, path):
        import numpy as np

        self.meta, self._t, start = _parse(path)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        self._start = start
        self._np = np

    def has(self, name):
        return name in self._t

    def type_of(self, name):
        return self._t[name][1]

    def shape(self, name):
        """numpy şekli ggml'in tersidir: ne=[E, R] → (R, E)."""
        return tuple(reversed(self._t[name][0]))

    def raw_rows(self, name, r0=0, n=None):
        """Satırların ham baytları: (n, satır_baytı) uint8."""
        ne, ty, off = self._t[name]
        blk, size = TYPE_SIZE[ty]
        row_bytes = ne[0] // blk * size
        rows = 1
        for d in ne[1:]:
            rows *= d
        n = rows - r0 if n is None else n
        a = self._mm[self._start + off + r0 * row_bytes: self._start + off + (r0 + n) * row_bytes]
        return a.reshape(n, row_bytes)

    def rows(self, name, r0=0, n=None):
        """Satırları fp32 olarak çözer: (n, ne[0])."""
        np = self._np
        ne, ty, off = self._t[name]
        raw = self.raw_rows(name, r0, n)
        if ty == 0:
            return raw.view(np.float32).copy()
        if ty == 1:
            return raw.view(np.float16).astype(np.float32)
        from gguf import GGMLQuantizationType, quants

        return quants.dequantize(np.ascontiguousarray(raw), GGMLQuantizationType(ty)).astype(np.float32)

    def get(self, name):
        return self.rows(name).reshape(self.shape(name))
