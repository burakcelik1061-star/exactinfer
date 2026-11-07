"""Model eşlik testi: motorun açgözlü (greedy) çıktısı Ollama'yla (llama.cpp tabanlı, GPU fp16) karşılaştırılır.

Her istem için iki taraf da aynı ham istemden N token üretir. Karşılaştırma:
  * istem token sayısı aynı olmalı (Ollama'nın prompt_eval_count'u),
  * üretilen token dizisi aynı olmalı; ilk ayrılık yalnızca bir "yakın eşitlik"te kabul edilir: iki tarafın seçtiği
    tokenlerin olasılıkları birbirine --tie (logprob farkı) kadar yakınsa sayısal gürültü sayılır ve o istemin
    karşılaştırması orada biter (sonraki tokenler farklı bir geçmişe bağlıdır),
  * eşleşen her tokenin logprob farkı raporlanır; --tol üstü BAŞARISIZ.

Motor fp32 hesaplar, Ollama GPU'da fp16 (KV önbelleği de fp16); bu yüzden logprob'larda ~1e-2 mertebesinde fark ve
uzun bağlamda ~0.1'e varan yakın eşitlik ayrılıkları beklenir. Nicemli modellerde (Q8_0, Q4_K, ...) llama.cpp ayrıca
etkinleştirmeleri de matris çarpımından önce 8 bite (q8_1) yuvarlar; fark ~0.2'ye çıkar, varsayılan eşikler buna göre
genişler (--tol/--tie ile değiştirilebilir). Ayrılık noktasında hangi tarafın doğru olduğunu
tests/parity_reference.py (bağımsız fp32 başvuru) belirler.

    python tests/parity_model.py [--model qwen3:4b-instruct] [-n 48] [--tol 0.08] [--tie 0.1]
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EL, OLLAMA, model_path, ollama_unload  # noqa: E402
from gguf_meta import GgufTensors  # noqa: E402

MODEL = "qwen3:0.6b-fp16"  # --model ile değişir

SYSTEM_TR = ("Sen Aria adında Türkçe konuşan bir masaüstü asistanısın. Kısa ve net cevap ver. "
             "Yalnızca sana verilen araçları kullan.")


def chat(user, system=SYSTEM_TR, think=False):
    s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    s += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    if not think:
        s += "<think>\n\n</think>\n\n"
    return s


LONG = ("Bir zamanlar küçük bir kasabada yaşayan meraklı bir mühendis vardı. Her sabah atölyesine gider, "
        "eski radyoları söküp yeniden birleştirir, devrelerin nasıl çalıştığını anlamaya çalışırdı. ") * 6

PROMPTS = [
    ("en-dogal", "The capital of France is"),
    ("en-liste", "Here are five tips for writing clean code:\n1."),
    ("tr-dogal", "Türkiye'nin en kalabalık şehri"),
    ("tr-hikaye", LONG + "Bir gün"),
    ("uzun-baglam", chat(LONG * 3 + "\n\nBu metni tek cümleyle özetle.")),  # >1000 token: 512'lik parça sınırı
    ("kod", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("sayi", "2, 3, 5, 7, 11, 13,"),
    ("sohbet-tr", chat("Bilgisayarımın durumu nasıl? Kısaca söyle.")),
    ("sohbet-en", chat("What is a GPU? Answer in one sentence.", system="You are a helpful assistant.")),
    ("sohbet-dusun", chat("15 ile 27'nin toplamı kaç?", think=True)),
    ("arac-cagri", chat("Not defterini aç.", system=SYSTEM_TR + "\n\n# Tools\n<tools>\n"
                         '{"type":"function","function":{"name":"open_app","description":"Uygulama acar",'
                         '"parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}'
                         "\n</tools>\nAraç çağırmak için <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> yaz.")),
]


def ollama(prompt, n, top):
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": False, "logprobs": True, "top_logprobs": top,
            "options": {"temperature": 0, "top_k": 1, "num_predict": n, "seed": 1}}
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    def raw(t):  # özel tokenlerde "bytes" alanı gelmeyebiliyor
        return bytes(t["bytes"]) if "bytes" in t else t["token"].encode("utf-8")

    toks = [{"bytes": raw(t), "logprob": t["logprob"],
             "top": {raw(x): x["logprob"] for x in t["top_logprobs"]}} for t in d.get("logprobs", [])]
    return toks, d.get("prompt_eval_count", 0)  # önbellekten gelenler dahil toplam istem tokeni


def engine(model, prompt, n, top, threads, backend):
    args = [str(EL), "generate", str(model), "-n", str(n), "--top", str(top), "--backend", backend]
    if threads:
        args += ["--threads", str(threads)]
    r = subprocess.run(args, input=prompt.encode("utf-8"), capture_output=True, check=False)
    if r.returncode != 0:
        sys.exit(f"motor hatasi: {r.stderr.decode(errors='replace')}")
    d = json.loads(r.stdout)
    toks = [{"bytes": bytes.fromhex(t["hex"]), "logprob": t["logprob"],
             "top": {bytes.fromhex(x["hex"]): x["logprob"] for x in t["top"]}} for t in d["tokens"]]
    return toks, len(d["prompt_ids"]), d


def show(b):
    return repr(b.decode("utf-8", errors="replace"))


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=48)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tol", type=float, help="eşleşen tokenlerde izin verilen en büyük |Δlogprob| "
                    "(varsayılan F16: 0.08, nicemli: 0.25)")
    ap.add_argument("--tie", type=float, help="ayrılığın yakın eşitlik sayılacağı logprob farkı "
                    "(varsayılan F16: 0.1, nicemli: 0.25)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--only", help="yalnız bu adlı istem")
    ap.add_argument("--backend", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--model", default=MODEL, help="Ollama model adı (motor aynı GGUF dosyasını okur)")
    a = ap.parse_args()
    MODEL = a.model
    print(f"arka uc: {a.backend}  model: {MODEL}")

    model = model_path(MODEL)
    quant = GgufTensors(model).type_of("blk.0.ffn_down.weight") not in (0, 1, 30)  # F32/F16/BF16 dışı
    a.tol = a.tol if a.tol is not None else (0.25 if quant else 0.08)
    a.tie = a.tie if a.tie is not None else (0.25 if quant else 0.1)
    print(f"esikler: tol {a.tol}, tie {a.tie}" + ("  (nicemli model: Ollama etkinlestirmeleri 8 bite yuvarlar)"
                                                  if quant else ""))
    prompts = [(n, p) for n, p in PROMPTS if not a.only or n == a.only]
    refs = [ollama(p, a.n, a.top) for _, p in prompts]  # önce Ollama, sonra GPU'yu motora bırak
    ollama_unload(MODEL)
    total_match = total_cmp = 0
    worst = 0.0
    fails = ties = 0
    ms_tok = []
    for (name, prompt), (ref, ref_np) in zip(prompts, refs):
        got, got_np, raw = engine(model, prompt, a.n, a.top, a.threads, a.backend)
        if got:
            ms_tok.append(raw["ms_gen"] / max(1, len(got) - 1) if len(got) > 1 else None)

        status, note = "TAMAM", ""
        if ref_np and ref_np != got_np:
            status, note = "BASARISIZ", f"istem token sayisi farkli: ollama {ref_np}, motor {got_np}"
        match = 0
        dmax = 0.0
        for i in range(min(len(ref), len(got))):
            r, g = ref[i], got[i]
            if r["bytes"] == g["bytes"]:
                match += 1
                dmax = max(dmax, abs(r["logprob"] - g["logprob"]))
                continue
            # Ayrılık: iki seçimin olasılıkları yakın mı? (her iki tarafın kendi top listesine bak)
            gap_ref = r["logprob"] - r["top"].get(g["bytes"], float("-inf"))
            gap_got = g["logprob"] - g["top"].get(r["bytes"], float("-inf"))
            if max(gap_ref, gap_got) <= a.tie:
                if status == "TAMAM":
                    status = "YAKIN-ESITLIK"
                note = (f"adim {i}: ollama {show(r['bytes'])} / motor {show(g['bytes'])}, "
                        f"fark {max(gap_ref, gap_got):.3f}")
            else:
                status = "BASARISIZ"
                note = (f"adim {i}: ollama {show(r['bytes'])} ({r['logprob']:.3f}) / motor {show(g['bytes'])} "
                        f"({g['logprob']:.3f}), fark {max(gap_ref, gap_got):.3f}")
            break
        else:
            if len(ref) != len(got) and status == "TAMAM":
                # biri durdu (EOS) diğeri devam etti: kısa olanın sonunda EOS seçilmiş olmalı
                status, note = "BASARISIZ", f"uzunluk farkli: ollama {len(ref)}, motor {len(got)}"
        if dmax > a.tol:
            status = "BASARISIZ"
            note = (note + "; " if note else "") + f"logprob farki {dmax:.4f} > {a.tol}"

        worst = max(worst, dmax)
        total_match += match
        total_cmp += max(len(ref), len(got)) if status == "TAMAM" else match
        fails += status == "BASARISIZ"
        ties += status == "YAKIN-ESITLIK"
        text = b"".join(t["bytes"] for t in got[:match]).decode("utf-8", errors="replace")
        print(f"{name:13s} {status:13s} {match:3d}/{len(ref):3d} token ayni  max|Δlp|={dmax:.4f}  {note}")
        print(f"{'':13s} > {text[:110]!r}")

    speed = [m for m in ms_tok if m]
    print(f"\nSONUC: {fails} basarisiz, {ties} yakin-esitlik; eslesen token {total_match}; "
          f"en buyuk |Δlogprob| {worst:.4f}")
    if speed:
        print(f"motor ({a.backend}) uretim dongusu: ~{sum(speed) / len(speed):.1f} ms/token "
              "(ornekleme dahil; saf hiz icin: el bench)")
    print("-> TAMAM" if fails == 0 else "-> BASARISIZ")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
