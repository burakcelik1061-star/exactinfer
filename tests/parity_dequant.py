"""Nicem çözücü testi: motorun çözdüğü ağırlıklar, llama.cpp'nin resmi `gguf` paketindeki numpy çözücüleriyle
karşılaştırılır. Her nicemli tensörden rastgele bir satır öbeği alınır (tüm katmanlar, tüm türler).

    python tests/parity_dequant.py [--model qwen3:4b-instruct] [--rows 16] [--all-rows]
"""

import argparse
import collections
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EL, model_path  # noqa: E402
from gguf_meta import GgufTensors  # noqa: E402

NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    path = model_path(a.model)
    g = GgufTensors(path)
    rnd = random.Random(a.seed)
    stats = collections.defaultdict(lambda: [0, 0, 0, 0.0])  # tensör, değer, birebir aynı, en büyük göreli fark
    bad = []
    for name, (ne, ty, _) in g._t.items():
        if ty in (0, 1):
            continue
        rows = 1
        for d in ne[1:]:
            rows *= d
        n = min(a.rows, rows)
        r0 = rnd.randrange(0, rows - n + 1)
        r = subprocess.run([str(EL), "dequant", str(path), name, str(r0), str(n)], capture_output=True, check=False)
        if r.returncode != 0:
            sys.exit(f"motor hatasi ({name}): {r.stderr.decode(errors='replace')}")
        got = np.frombuffer(r.stdout, dtype=np.float32).reshape(n, ne[0])
        want = g.rows(name, r0, n)
        st = stats[NAMES.get(ty, ty)]
        st[0] += 1
        st[1] += want.size
        st[2] += int((got == want).sum())
        rel = float((np.abs(got - want) / np.maximum(np.abs(want), 1e-30)).max())
        st[3] = max(st[3], rel)
        if not np.allclose(got, want, rtol=1e-6, atol=0):
            bad.append((name, r0, rel))

    print(f"model: {a.model}")
    for t, (nt, nv, eq, rel) in sorted(stats.items()):
        print(f"  {t:5s} {nt:4d} tensor, {nv:10d} deger: birebir ayni {eq / nv * 100:7.3f}%  en buyuk goreli fark {rel:.2e}")
    for name, r0, rel in bad[:10]:
        print(f"  FARK: {name} satir {r0}: goreli {rel:.2e}")
    ok = not bad and stats
    print("SONUC:", "TAMAM" if ok else "BASARISIZ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
