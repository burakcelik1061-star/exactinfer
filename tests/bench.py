"""Hız kıyası: motor (`el bench`) ve Ollama, aynı model dosyası ve aynı GPU üzerinde.

  pp = istem işleme (prompt processing) hızı, tg = token üretme (text generation) hızı — token/sn.
Ollama'nın ölçümü kendi yanıtındaki prompt_eval_duration / eval_duration alanlarından alınır; önek önbelleğine
düşmemek için her tekrarda farklı bir istem kullanılır.

    python tests/bench.py [--model qwen3:4b-instruct] [--pp 512] [--tg 128] [--reps 3]
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EL, OLLAMA, model_path, ollama_unload  # noqa: E402

MODEL = "qwen3:0.6b-fp16"  # --model ile değişir
WORDS = ("river stone light cloud engine paper garden window silver music bridge forest market winter "
         "copper signal planet harbor lantern meadow").split()


def ollama_call(body):
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def ollama_bench(pp, tg, reps):
    ollama_call({"model": MODEL, "prompt": "hi", "raw": True, "stream": False, "options": {"num_predict": 1}})
    pps, tgs = [], []
    for r in range(reps):
        rnd = random.Random(1000 + r)
        prompt = " ".join(rnd.choice(WORDS) for _ in range(pp))  # her kelime ~1 token
        d = ollama_call({"model": MODEL, "prompt": prompt, "raw": True, "stream": False,
                         "options": {"num_predict": 1, "temperature": 0, "num_ctx": pp + tg + 64}})
        pps.append(d["prompt_eval_count"] / (d["prompt_eval_duration"] / 1e9))
        # üretim: EOS'a erken düşmeyen bir istem
        d = ollama_call({"model": MODEL, "prompt": f"Count from {r + 1} to 5000, separated by commas: {r + 1},",
                         "raw": True, "stream": False,
                         "options": {"num_predict": tg, "temperature": 0, "num_ctx": pp + tg + 64}})
        tgs.append(d["eval_count"] / (d["eval_duration"] / 1e9))
    return statistics.median(pps), statistics.median(tgs)


def engine_bench(model, backend, pp, tg, reps):
    r = subprocess.run([str(EL), "bench", str(model), "--backend", backend, "--pp", str(pp), "--tg", str(tg),
                        "--reps", str(reps)], capture_output=True, check=False)
    if r.returncode != 0:
        sys.exit(f"motor hatasi: {r.stderr.decode(errors='replace')}")
    d = json.loads(r.stdout)
    return d["pp_tps"], d["tg_tps"]


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", type=int, default=512)
    ap.add_argument("--tg", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="CPU arka ucunu da ölç (yavaş)")
    ap.add_argument("--model", default=MODEL, help="Ollama model adı (motor aynı GGUF dosyasını okur)")
    a = ap.parse_args()
    MODEL = a.model
    model = model_path(MODEL)
    ollama_unload(MODEL)  # ölçüm sırasında GPU'yu paylaşmasınlar

    rows = [("motor (cuda, fp32 hesap)", *engine_bench(model, "cuda", a.pp, a.tg, a.reps))]
    if a.cpu:
        rows.append(("motor (cpu, fp32 hesap)", *engine_bench(model, "cpu", min(a.pp, 64), min(a.tg, 32), 1)))
    try:
        rows.append(("ollama (cuda)", *ollama_bench(a.pp, a.tg, a.reps)))
    except OSError as e:
        print(f"ollama olculemedi: {e}")
    ollama_unload(MODEL)

    print(f"\n{'':28s} {'pp' + str(a.pp):>12s} {'tg' + str(a.tg):>12s}   (token/sn, ortanca)  {MODEL}")
    for name, pp, tg in rows:
        print(f"{name:28s} {pp:12.1f} {tg:12.1f}")


if __name__ == "__main__":
    main()
