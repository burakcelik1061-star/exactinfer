"""`el serve` sınaması: kalıcı sunucunun çıktısı, her istem için baştan çalıştırılan `el generate` ile birebir aynı
olmalı (önek önbelleği yalnız hızı değiştirir). Ayrıca bozuk isteklerin süreci düşürmediği ve kısıtlı modun
sunucuda da çalıştığı denetlenir.

    python tests\\serve_test.py [--model qwen3:0.6b-fp16] [--backend cpu|cuda]
"""

import argparse
import json
import subprocess
import sys

from common import EL, ollama_blob

SYSTEM = "<|im_start|>system\nSen yardımsever bir asistansın. Kısa yanıt ver.<|im_end|>\n"


def chat(*turns):
    s = SYSTEM
    for role, text in turns:
        s += f"<|im_start|>{role}\n{text}<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


TOOLS = [{"type": "function", "function": {"name": "set_volume", "description": "Ses seviyesini ayarlar.",
          "parameters": {"type": "object", "properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
                         "required": ["level"], "additionalProperties": False}}}]
TOOL_SYS = ("<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
            + json.dumps(TOOLS[0], ensure_ascii=False) +
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
            "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
            "<args-json-object>}\n</tool_call><|im_end|>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:0.6b-fp16")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    model = ollama_blob(a.model)
    n = 32

    p1 = chat(("user", "Türkiye'nin başkenti neresi?"))
    p2 = chat(("user", "Türkiye'nin başkenti neresi?"), ("assistant", "Ankara."), ("user", "Peki en kalabalık şehri?"))
    p3 = chat(("user", "Bir renk söyle."))
    p4 = TOOL_SYS + "<|im_start|>user\nSesi yüzde 150 yap.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def fresh(prompt, tools=None):
        cmd = [str(EL), "generate", str(model), "-n", str(n), "--top", "1", "--backend", a.backend, "--ctx", "4096"]
        if tools:
            from pathlib import Path
            import tempfile
            f = Path(tempfile.mkdtemp()) / "tools.json"
            f.write_text(json.dumps(tools), encoding="utf-8")
            cmd += ["--tools", str(f)]
        r = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True)
        if r.returncode:
            sys.exit(r.stderr.decode("utf-8", "replace"))
        j = json.loads(r.stdout)
        return "".join(t["hex"] for t in j["tokens"]), j

    srv = subprocess.Popen([str(EL), "serve", str(model), "--backend", a.backend, "--ctx", "4096"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ready = json.loads(srv.stdout.readline())
    assert ready.get("ready"), ready

    def req(obj=None, raw=None):
        srv.stdin.write((raw if raw is not None else json.dumps(obj, ensure_ascii=False)).encode("utf-8") + b"\n")
        srv.stdin.flush()
        return json.loads(srv.stdout.readline())

    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{'TAMAM' if ok else 'HATA ':5s}  {name}  {detail}")

    order = [("ilk istem", p1, None), ("aynı istem (tam önek)", p1, None), ("devam turu (ortak önek)", p2, None),
             ("başka istem (yalnız sistem öneki)", p3, None), ("kısıtlı mod", p4, TOOLS)]
    for name, prompt, tools in order:
        r = req({"id": name, "prompt": prompt, "n": n, **({"tools": tools} if tools else {})})
        want, j = fresh(prompt, tools)
        same = r.get("ok") and r["hex"] == want
        text = bytes.fromhex(r.get("hex", "")).decode("utf-8", "replace")
        extra = f" altered={r.get('altered_calls')} (generate: {j.get('altered_calls')})" if tools else ""
        if tools:
            same = same and r.get("altered_calls") == j.get("altered_calls")
        check(name, same and r.get("id") == name,
              f"önbellekten {r.get('reused')}/{r.get('prompt_tokens')} token, istem {r.get('ms_prompt')} ms | "
              f"{' '.join(text.split())[:70]!r}{extra}")

    # Örnekleme (serbest metinde): aynı tohum aynı çıktı; çağrının içi yine açgözlü (kısıtlı modda aynı çağrı)
    p5 = chat(("user", "Bana kısa bir hikâye anlat."))
    s1 = req({"prompt": p5, "n": n, "temperature": 0.8, "seed": 7})
    s2 = req({"prompt": p5, "n": n, "temperature": 0.8, "seed": 7})
    s3 = req({"prompt": p5, "n": n, "temperature": 0.8, "seed": 8})
    check("örnekleme: aynı tohum aynı çıktı", s1["ok"] and s1["hex"] == s2["hex"],
          f"tohum 8 farklı mı: {s3['hex'] != s1['hex']}")
    g = req({"prompt": p5, "n": n, "temperature": 0})
    check("temperature 0 = açgözlü", g["ok"] and g["hex"] == fresh(p5)[0])
    calls = set()
    for seed in range(6):
        r = req({"prompt": p4, "n": n, "tools": TOOLS, "temperature": 1.0, "seed": seed})
        t = bytes.fromhex(r["hex"]).decode("utf-8")
        calls.add(t[t.find("<tool_call>"):] if r["calls"] else "çağrı yok")
    want = bytes.fromhex(fresh(p4, TOOLS)[0]).decode("utf-8")
    want = want[want.find("<tool_call>"):] if "<tool_call>" in want else None
    check("örneklemede çağrı içi açgözlü", calls <= {want, "çağrı yok"}, f"{len(calls)} farklı sonuç")
    for bad in ({"temperature": -1}, {"temperature": 3}, {"top_k": 0}, {"top_p": 0}, {"top_p": 1.5}, {"seed": 1.5}):
        check(f"geçersiz örnekleme {bad}", req({"prompt": "x", **bad})["ok"] is False)

    # Tekrar cezası (DRY): kapalıyken açgözlüyle aynı; açıkken tekrar eden kalıp kırılır; çağrının içine dokunmaz
    rep = chat(("user", "Şu cümleyi alt alta 12 kez yaz, başka hiçbir şey yazma: kedi bahçede uyuyor ve rüya görüyor"))
    g = req({"prompt": rep, "n": 96})
    d0 = req({"prompt": rep, "n": 96, "dry_multiplier": 0})
    check("DRY kapalı = açgözlü", d0["ok"] and d0["hex"] == g["hex"] and "dry_hits" not in d0)
    d1 = req({"prompt": rep, "n": 96, "dry_multiplier": 0.8})
    phrase = "bahçede uyuyor ve rüya"
    cg, cd = (bytes.fromhex(r["hex"]).decode("utf-8", "replace").count(phrase) for r in (g, d1))
    check("DRY tekrarı kırar", d1["ok"] and d1.get("dry_hits", 0) > 0 and cd < cg,
          f"kalıp açgözlüde {cg}, DRY ile {cd} kez; ceza uygulanan adım {d1.get('dry_hits')}")
    r = req({"prompt": p4, "n": n, "tools": TOOLS, "dry_multiplier": 2.0})
    t = bytes.fromhex(r["hex"]).decode("utf-8")
    check("DRY çağrı içine dokunmaz", r["ok"] and want is not None and t[t.find("<tool_call>"):] == want)
    for bad in ({"dry_multiplier": -1}, {"dry_multiplier": 6}, {"dry_base": 0.5}, {"dry_allowed_length": 0},
                {"dry_allowed_length": 2.5}):
        check(f"geçersiz DRY {bad}", req({"prompt": "x", **bad})["ok"] is False)

    check("bozuk JSON süreci düşürmez", req(raw="{bozuk")["ok"] is False)
    check("eksik prompt", req({"n": 3})["ok"] is False)
    check("geçersiz n", req({"prompt": "x", "n": 0})["ok"] is False)
    check("geçersiz şema", req({"prompt": "x", "tools": [{"function": {}}]})["ok"] is False)
    r = req({"prompt": p1, "n": n})
    check("hatalardan sonra yine aynı çıktı", r["ok"] and r["hex"] == fresh(p1)[0], f"önbellekten {r.get('reused')}")
    srv.stdin.write(b'{"op": "quit"}\n')  # yanıt yok: süreç çıkar
    srv.stdin.flush()
    srv.wait(timeout=30)
    check("quit ile çıkış", srv.returncode == 0)
    print("SONUC:", "TAMAM" if not fails else f"{fails} hata")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
