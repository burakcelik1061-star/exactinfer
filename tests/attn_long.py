"""Uzun bağlamda istem dikkati sınaması: aynı uzun istem, motorun üç dikkat çekirdeğiyle (varsayılan yazmaç döşemeli,
EL_ATTN=tile32 eski karolu, EL_ATTN=naive tek blok) çalıştırılır; üretilen tokenler aynı ve logprob farkları fp32
gürültüsü düzeyinde olmalı. parity_reference.py kısa istemlerle (<= 133 token) sınar; bu test 512'lik toplu istem
parçalarının pos0 > 0 olduğu, binlerce anahtarlı durumu kapsar.

    python tests\\attn_long.py [--model qwen3:4b-instruct] [--tokens 6000] [--backend cuda]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import EL, ollama_blob

for _s in (sys.stdout, sys.stderr):
    _s.reconfigure(encoding="utf-8", errors="replace")

# Doğal, tekrarsız uzun metin: kaynak dosyalarımızın kendisi (Türkçe yorumlar + kod)
ROOT = Path(__file__).resolve().parent.parent


def long_prompt(el, model, want):
    text = ""
    for f in sorted((ROOT / "src").rglob("*.cpp")) + sorted((ROOT / "src").rglob("*.cu")):
        text += f.read_text(encoding="utf-8")
    d = Path(tempfile.mkdtemp())
    lo, hi = 0, len(text)
    while lo < hi:  # istenen token sayısına en yakın önek (ikili arama)
        mid = (lo + hi + 1) // 2
        (d / "p.txt").write_text(text[:mid], encoding="utf-8", newline="")
        n = len(json.loads(subprocess.run([str(el), "tokenize", str(model), "-f", str(d / "p.txt")],
                                          capture_output=True, check=True).stdout))
        lo, hi = (mid, hi) if n <= want else (lo, mid - 1)
    body = text[:lo]
    p = d / "prompt.txt"
    p.write_text("<|im_start|>user\nAşağıdaki kodu özetle.\n\n" + body + "<|im_end|>\n<|im_start|>assistant\n"
                 "<think>\n\n</think>\n\n", encoding="utf-8", newline="")
    return p


def run(model, prompt, backend, n, attn):
    env = dict(os.environ)
    env.pop("EL_ATTN", None)
    if attn:
        env["EL_ATTN"] = attn
    r = subprocess.run([str(EL), "generate", str(model), "-f", str(prompt), "-n", str(n), "--top", "2",
                        "--backend", backend], capture_output=True, env=env)
    if r.returncode:
        sys.exit(r.stderr.decode("utf-8", "replace"))
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--backend", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("-n", type=int, default=24)
    a = ap.parse_args()
    model = ollama_blob(a.model)
    prompt = long_prompt(EL, model, a.tokens)
    runs = {name: run(model, prompt, a.backend, a.n, attn) for name, attn in
            (("yazmaç döşemeli", None), ("tile32", "tile32"), ("naive", "naive"))}
    base = runs["yazmaç döşemeli"]
    ids = [t["id"] for t in base["tokens"]]
    print(f"istem {len(base['prompt_ids'])} token, {len(ids)} üretilen")
    bad = 0
    for name, r in runs.items():
        if r is base:
            continue
        same = [t["id"] for t in r["tokens"]] == ids
        dlp = max(abs(x["logprob"] - y["logprob"]) for x, y in zip(base["tokens"], r["tokens"]))
        ok = same and dlp < 1e-3
        bad += not ok
        print(f"{'TAMAM' if ok else 'HATA '}  {name:8s} tokenler {'aynı' if same else 'FARKLI'}  max|Δlp|={dlp:.2e}")
    print("SONUC:", "TAMAM" if not bad else f"{bad} hata")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
