"""Tokenleştirici eşlik testi: motorun çıktısı ile HuggingFace `tokenizers` (Rust) aynı id'leri vermeli.

Başvuru tokenleştirici aynı GGUF'taki sözlük ve birleştirmelerden, Qwen2'nin tokenizer.json tanımıyla kurulur
(bölme düzenli ifadesi + ByteLevel, normalizasyon yok — llama.cpp/Ollama da normalize etmez).

    python tests/parity_tokenizer.py [--n 3000] [--seed 1]
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EL, ROOT, model_path  # noqa: E402
from gguf_meta import read_metadata  # noqa: E402

from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers  # noqa: E402

QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def reference(meta):
    tokens = meta["tokenizer.ggml.tokens"]
    types = meta["tokenizer.ggml.token_type"]
    vocab = {t: i for i, t in enumerate(tokens)}
    merges = [tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    specials = [tokens[i] for i, t in enumerate(types) if t in (3, 4)]
    tok.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in specials])
    return tok, specials


FIXED = [
    "The capital of France is",
    "Merhaba dünya! Bugün hava çok güzel, değil mi?",
    "İSTANBUL ışık ĞÜŞİÖÇ ğüşıöç",
    "I'm sure you'll see we've done it; they'd say 'hello' — DON'T, it's FINE'S",
    "  leading spaces and trailing   ",
    "a  b   c    d\te\t\tf",
    "line1\nline2\r\nline3\n\n\nline4\r\r\n",
    "   \n   \n\t \n x",
    "1234567890 3.14159 1,000,000 0x1F 2026-11-02 12:30",
    "def f(x):\n    return x**2 + 1  # yorum\n",
    "if (a != b && c <= d) { return -1; } else { x += 2; }",
    "<|im_start|>user\nSaat kaç?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "<|im_start|><|im_end|><|endoftext|>",
    "<tool_call>\n{\"name\": \"system_status\", \"arguments\": {}}\n</tool_call>",
    "<|im_start  not special |>",
    "你好，世界！こんにちは 안녕하세요",
    "مرحبا بالعالم שלום עולם",
    "Emoji: 😀👍🏽🇹🇷 👨‍👩‍👧‍👦 ✨",
    "nbsp: x  y thin space ideographic　space",
    "combining: é ä ñ",
    "roman Ⅻ ½ ² ٣ ໓",
    "!!!???...,,,;;;:::",
    " !!!\n\n",
    "  \u0085",
    "",
    " ",
    "\n",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "https://example.com/path?q=1&r=two#frag user@mail.com",
]

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "çğıöşüÇĞİÖŞÜâîû",
    "0123456789",
    " " * 6 + "\t\n\r",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "'s't're've'm'll'd'S'T'RE",
    "  　 \u0085​﻿",
    "你好世界中文日本語한국어",
    "αβγδΩЖжЯя",
    "😀👍🏽🇹🇷✨‍",
    "́̈̃",
    "½²٣Ⅻ໓",
]


def fuzz(n, seed):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = rnd.randint(1, 60)
        pools = rnd.sample(POOLS, rnd.randint(1, 4))
        out.append("".join(rnd.choice(rnd.choice(pools)) for _ in range(k)))
    return out


def own_sources():
    texts = []
    for p in sorted((ROOT / "src").glob("*.[ch]*")) + sorted((ROOT / "tests").glob("*.py")):
        if p.name != "unicode_tables.h":
            texts.append(p.read_text(encoding="utf-8"))
    return texts


def engine_encode(model, texts, special=True):
    data = "".join(t.encode("utf-8").hex() + "\n" for t in texts).encode()
    args = [str(EL), "tokenize", str(model), "--lines"] + ([] if special else ["--no-special"])
    r = subprocess.run(args, input=data, capture_output=True, check=False)
    if r.returncode != 0:
        sys.exit(f"motor hatasi: {r.stderr.decode(errors='replace')}")
    out = [json.loads(line) for line in r.stdout.decode().splitlines()]
    if len(out) != len(texts):
        sys.exit(f"motor {len(texts)} metin icin {len(out)} satir dondurdu")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    model = model_path()
    meta = read_metadata(model)
    ref, specials = reference(meta)

    groups = {"sabit": FIXED, "kaynak-kod": own_sources(), "rastgele": fuzz(a.n, a.seed)}
    total = bad = 0
    for name, texts in groups.items():
        got = engine_encode(model, texts)
        want = [e.ids for e in ref.encode_batch(texts, add_special_tokens=False)]
        fails = [(t, g, w) for t, g, w in zip(texts, got, want) if g != w]
        n_tok = sum(len(w) for w in want)
        print(f"{name:11s} {len(texts) - len(fails):5d}/{len(texts)} metin ayni  ({n_tok} token)")
        for t, g, w in fails[:5]:
            print(f"   FARK: {t!r}\n     motor: {g}\n     hf   : {w}")
        total += len(texts)
        bad += len(fails)

    # Özel token ayrıştırma kapalıyken yazımlar düz metin gibi parçalanmalı
    plain = ["<|im_start|>user\nmerhaba<|im_end|>"]
    got = engine_encode(model, plain, special=False)
    ref_plain = reference(meta)[0]
    ref_plain_tok = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(meta["tokenizer.ggml.tokens"])},
                                         merges=[tuple(m.split(" ", 1)) for m in meta["tokenizer.ggml.merges"]]))
    ref_plain_tok.pre_tokenizer = ref_plain.pre_tokenizer
    want = [ref_plain_tok.encode(plain[0]).ids]
    ok = got == want
    print(f"ozel-kapali {'TAMAM' if ok else 'FARK'}")
    total += 1
    bad += 0 if ok else 1

    print(f"\nSONUC: {total - bad}/{total} ayni" + ("  -> TAMAM" if bad == 0 else "  -> BASARISIZ"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
