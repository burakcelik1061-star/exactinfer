"""src/unicode_tables.h dosyasını üretir: harf (\\p{L}) ve rakam (\\p{N}) kod noktası aralıkları.

    python tools/gen_unicode.py

Ön-tokenleştirici düzenli ifadesindeki sınıflar için gerekir. Kaynak: Python'un unicodedata veritabanı.
"""

import sys
import unicodedata
from pathlib import Path


def ranges(pred):
    out, start = [], None
    for cp in range(sys.maxunicode + 2):
        ok = cp <= sys.maxunicode and pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    return out


def emit(name, rs):
    lines = [f"inline constexpr CpRange {name}[] = {{"]
    row = []
    for a, b in rs:
        row.append(f"{{0x{a:X}, 0x{b:X}}}")
        if len(row) == 6:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("    " + ", ".join(row) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    letters = ranges(lambda c: unicodedata.category(c).startswith("L"))
    numbers = ranges(lambda c: unicodedata.category(c).startswith("N"))
    body = "\n\n".join([emit("kLetter", letters), emit("kNumber", numbers)])
    text = (
        "// OTOMATIK URETILDI: tools/gen_unicode.py (Unicode " + unicodedata.unidata_version + "). Elle duzenlemeyin.\n"
        "#pragma once\n#include <cstdint>\n\nnamespace el {\n\n"
        "struct CpRange { uint32_t lo, hi; };\n\n" + body + "\n\n}  // namespace el\n"
    )
    out = Path(__file__).resolve().parent.parent / "src" / "unicode_tables.h"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{out}: {len(letters)} harf, {len(numbers)} rakam araligi (Unicode {unicodedata.unidata_version})")


if __name__ == "__main__":
    main()
