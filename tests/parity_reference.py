"""Sıkı doğruluk testi: motor, numpy ile yazılmış bağımsız bir fp32 Qwen3 uygulamasına karşı.

Motor her istem için açgözlü N token üretir. Sonra istem + üretilen tokenler başvuruya tek seferde verilir
("teacher forcing") ve her adımda denetlenir:
  * motorun seçtiği token başvuruda da en olası token olmalı (gerçek bir eşitlik dışında: fark < --tie),
  * motorun bildirdiği logprob ile başvurununki arasındaki fark < --tol.

İki taraf da fp32 hesapladığı için farklar yalnızca toplama sırasından gelir (~1e-4). Ollama (GPU fp16) karşılaştırması
için tests/parity_model.py'ye bakın.

    python tests/parity_reference.py [--model qwen3:4b-instruct] [-n 48] [--tol 2e-3] [--tie 1e-3]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EL, model_path  # noqa: E402
from gguf_meta import GgufTensors  # noqa: E402
from parity_model import PROMPTS  # noqa: E402


def rmsnorm(x, w, eps):
    x64 = x.astype(np.float64)
    inv = 1.0 / np.sqrt((x64 * x64).mean(axis=-1, keepdims=True) + eps)
    return (x * inv.astype(np.float32)) * w


def reference_logprobs(g, seqs):
    """seqs: [(ids, start)] → her dizi için start.. konumlarının log-softmax'ı (T-start, V), float64.

    Katman katman ilerler: bir katmanın ağırlıkları bir kez fp32'ye çözülür ve tüm dizilere uygulanır, sonra atılır.
    Böylece bellekte aynı anda yalnızca bir katman bulunur (4B modelin tamamı fp32'de ~16 GB tutardı).
    """
    m = g.meta
    p = m["general.architecture"] + "."
    L, H = m[p + "block_count"], m[p + "attention.head_count"]
    HKV = m.get(p + "attention.head_count_kv", H)
    E = m[p + "embedding_length"]
    D = m.get(p + "attention.key_length", E // H)
    eps = m.get(p + "attention.layer_norm_rms_epsilon", 1e-6)
    base = m.get(p + "rope.freq_base", 10000.0)
    inv_freq = base ** (-np.arange(0, D, 2, dtype=np.float64) / D)

    xs, ropes, masks = [], [], []
    for ids, _ in seqs:
        T = len(ids)
        xs.append(np.concatenate([g.rows("token_embd.weight", int(t), 1) for t in ids]))
        ang = np.arange(T, dtype=np.float64)[:, None] * inv_freq[None, :]
        ropes.append((np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]))
        masks.append(np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1))

    def rope(v, cs):  # (T, n, D), NEOX: yarımlar döner
        cos, sin = cs
        a, b = v[..., : D // 2], v[..., D // 2:]
        return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)

    for l in range(L):
        b = f"blk.{l}."
        w = {k: g.get(b + k + ".weight") for k in ("attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm",
                                                  "attn_k_norm", "attn_output", "ffn_norm", "ffn_gate", "ffn_up",
                                                  "ffn_down")}
        for si, x in enumerate(xs):
            T = x.shape[0]
            h = rmsnorm(x, w["attn_norm"], eps)
            q = (h @ w["attn_q"].T).reshape(T, H, D)
            k = (h @ w["attn_k"].T).reshape(T, HKV, D)
            v = (h @ w["attn_v"].T).reshape(T, HKV, D)
            q = rope(rmsnorm(q, w["attn_q_norm"], eps), ropes[si])
            k = rope(rmsnorm(k, w["attn_k_norm"], eps), ropes[si])
            k = np.repeat(k, H // HKV, axis=1)
            v = np.repeat(v, H // HKV, axis=1)
            s = np.einsum("thd,shd->hts", q, k) / np.float32(np.sqrt(D)) + masks[si]
            s = np.exp(s - s.max(axis=-1, keepdims=True))
            s /= s.sum(axis=-1, keepdims=True)
            att = np.einsum("hts,shd->thd", s, v).reshape(T, H * D)
            x = x + att @ w["attn_output"].T
            h = rmsnorm(x, w["ffn_norm"], eps)
            gt = h @ w["ffn_gate"].T
            up = h @ w["ffn_up"].T
            with np.errstate(over="ignore"):  # exp(-gt) taşarsa silu doğru biçimde 0'a gider
                act = gt / (1.0 + np.exp(-gt))
            xs[si] = x + (act * up) @ w["ffn_down"].T
        del w

    # Çıkış katmanı (çoğunlukla gömmeye bağlı, 151936 satır) satır parçalarıyla çözülür
    out_name = "output.weight" if g.has("output.weight") else "token_embd.weight"
    V = g.shape(out_name)[0]
    hs = [rmsnorm(x[start:], g.get("output_norm.weight"), eps) for x, (_, start) in zip(xs, seqs)]
    logits = [np.empty((hh.shape[0], V), dtype=np.float32) for hh in hs]
    for r0 in range(0, V, 16384):
        n = min(16384, V - r0)
        wo = g.rows(out_name, r0, n)
        for hh, lg in zip(hs, logits):
            lg[:, r0:r0 + n] = hh @ wo.T
    res = []
    for lg in logits:
        lg = lg.astype(np.float64)
        lg -= lg.max(axis=-1, keepdims=True)
        res.append(lg - np.log(np.exp(lg).sum(axis=-1, keepdims=True)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--tie", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--only", help="yalnız bu adlı istem")
    ap.add_argument("--model", default="qwen3:0.6b-fp16", help="Ollama model adı (EL_MODEL yolu önceliklidir)")
    a = ap.parse_args()
    print(f"arka uc: {a.backend}  model: {a.model}")

    model = model_path(a.model)
    g = GgufTensors(model)
    prompts = [(n, p) for n, p in PROMPTS if not a.only or n == a.only]
    runs = []
    for name, prompt in prompts:
        r = subprocess.run([str(EL), "generate", str(model), "-n", str(a.n), "--top", "2", "--threads", str(a.threads),
                            "--backend", a.backend],
                           input=prompt.encode("utf-8"), capture_output=True, check=False)
        if r.returncode != 0:
            sys.exit(f"motor hatasi: {r.stderr.decode(errors='replace')}")
        runs.append(json.loads(r.stdout))
    t0 = time.time()
    lps = reference_logprobs(g, [(d["prompt_ids"] + [t["id"] for t in d["tokens"]], len(d["prompt_ids"]) - 1)
                                 for d in runs])
    print(f"baslangic hesabi {time.time() - t0:.1f} sn")

    fails = 0
    worst = 0.0
    n_tok = 0
    for (name, _), d, lp in zip(prompts, runs, lps):
        P = len(d["prompt_ids"])
        status, note, dmax = "TAMAM", "", 0.0
        for i, t in enumerate(d["tokens"]):
            row = lp[i]
            dmax = max(dmax, abs(row[t["id"]] - t["logprob"]))
            best = int(row.argmax())
            if best != t["id"] and row[best] - row[t["id"]] >= a.tie:
                status = "BASARISIZ"
                note = f"adim {i}: motor {t['id']} secti, baslangic {best} (fark {row[best] - row[t['id']]:.5f})"
                break
        if dmax > a.tol:
            status = "BASARISIZ"
            note += f" |Δlogprob| {dmax:.2e} > {a.tol}"
        fails += status != "TAMAM"
        worst = max(worst, dmax)
        n_tok += len(d["tokens"])
        print(f"{name:13s} {status:9s} istem {P:4d} + {len(d['tokens']):2d} token  max|Δlp|={dmax:.2e}  {note}")

    print(f"\nSONUC: {len(prompts) - fails}/{len(prompts)} istem, {n_tok} uretilen token; "
          f"en buyuk |Δlogprob| {worst:.2e}" + ("  -> TAMAM" if fails == 0 else "  -> BASARISIZ"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
