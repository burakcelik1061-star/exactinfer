"""Kısıtlı / kısıtsız çözümleme karşılaştırması: aynı istemler, aynı model, `--tools` var ve yok.

Her yanıt bekçiye `parse` ile verilir. "Şemaya uygun" = bekçi çağrıyı kabul etti ya da yalnızca yol izinli kök
dışında diye reddetti (dosya sistemine bağlı ret; şemayla ifade edilemez). Kısıtın modelin seçimini değiştirdiği
çağrılar (`altered_calls`) bekçiye bildirilir; bekçi bunları onaya yükseltir. Hiçbir araç çalıştırılmaz.

    python tests\\constrain_compare.py [--model qwen3:0.6b-fp16] [--backend cpu|cuda] [-n 120]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from common import EL, ROOT, ollama_blob
from e2e_bekci import BEKCI, Bekci, prompt

JAIL = ("izinli klasörlerin dışında", "korumalı klasör", "ağ paylaşımına")

# (istem, beklenen araç; None = araç çağrısı beklenmiyor ya da serbest)
CASES = [
    ("Hesap makinesini aç.", "open_app"),
    ("Not defterini aç.", "open_app"),
    ("Dosya gezginini başlat.", "open_app"),
    ("Paint'i açar mısın?", "open_app"),
    ("Sesi yüzde 150 yap.", "set_volume"),
    ("Sesi tamamen kapat.", "set_volume"),
    ("Adında rapor geçen dosyaları ara, en fazla 10 sonuç göster.", "file_search"),
    ("Masaüstümdeki alışveriş.txt dosyasını oku.", "file_read"),
    ("Yarın saat 9'da toplantı var diye not ekle.", "note_add"),
    ("https://www.python.org adresini aç.", "open_url"),
    ("Bilgisayarın durumu nasıl?", "system_status"),
    ("Komut isteminde dir komutunu çalıştır.", None),
    ("Masaüstündeki eski klasörünü tamamen sil.", "file_delete"),
    ("Spotify'ı aç.", None),
    # JSON'u bozmaya yatkın istemler: iç tırnak, tek ters bölü, çok satır
    ('Şu notu ekle: Ali dedi ki "yarın erken gel" ve gitti.', "note_add"),
    (r"C:\Users\User\Documents\rapor.txt dosyasını oku.", "file_read"),
    ("Alışveriş listesi notu ekle: süt, ekmek, yumurta; her biri ayrı satırda olsun.", "note_add"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:0.6b-fp16")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("-n", type=int, default=120)
    ap.add_argument("--only", help="yalnız bu metni içeren istemler")
    a = ap.parse_args()
    model = ollama_blob(a.model)

    tmp = Path(tempfile.mkdtemp(prefix="constrain-cmp-"))
    text = (ROOT / "bekci" / "policy.example.toml").read_text(encoding="utf-8")
    text = text.replace('# log = "%LOCALAPPDATA%\\\\engine-lab\\\\actions.log"   # varsayılan',
                        f"log = '{tmp / 'actions.log'}'")
    text = text.replace('confirm_ui = "dialog"', 'confirm_ui = "client"').replace('hotkey = "ctrl+alt+end"', "")
    policy = tmp / "bekci.toml"
    policy.write_text(text, encoding="utf-8")
    b = Bekci(policy)
    tools = b.req(op="tools")["tools"]
    tools_file = tmp / "tools.json"
    tools_file.write_text(json.dumps(tools, ensure_ascii=False), encoding="utf-8")

    def run(user, constrained):
        cmd = [str(EL), "generate", str(model), "-n", str(a.n), "--top", "1", "--backend", a.backend]
        if constrained:
            cmd += ["--tools", str(tools_file)]
        r = subprocess.run(cmd, input=prompt(tools, user).encode("utf-8"), capture_output=True)
        if r.returncode:
            sys.exit(r.stderr.decode("utf-8", "replace"))
        j = json.loads(r.stdout)
        out = bytes.fromhex("".join(t["hex"] for t in j["tokens"])).decode("utf-8", "replace")
        b.req(op="turn")
        res = b.req(op="parse", text=out, **({"altered": j["altered_calls"]} if j.get("altered_calls") else {}))
        calls = [] if not res["ok"] else res["calls"]
        err = None if res["ok"] else res["error"]
        return out, calls, err, j

    def judge(calls, err, want):
        if err:
            return False, f"ayrıştırılamadı: {err[:70]}"
        if not calls:
            return want is None, "çağrı yok"
        parts, ok = [], True
        for c in calls:
            why = c.get("reason", "")
            fine = c["decision"] != "deny" or any(j in why for j in JAIL)
            ok &= fine
            shown = why and (not fine or "kısıtlı" in why)
            parts.append(f"{c.get('tool') or '-'}→{c['decision']}" + (f" ({why[:70]})" if shown else ""))
        if want is not None:
            ok &= any(c.get("tool") == want for c in calls)
        return ok, ", ".join(parts)

    score = {False: 0, True: 0}
    stats = {False: [0, 0.0], True: [0, 0.0]}  # token, ms
    forced = 0
    cases = [c for c in CASES if not a.only or a.only in c[0]]
    for user, want in cases:
        print(f"\n{user}   (beklenen: {want or 'serbest'})")
        for constrained in (False, True):
            out, calls, err, j = run(user, constrained)
            ok, desc = judge(calls, err, want)
            score[constrained] += ok
            stats[constrained][0] += len(j["tokens"])
            stats[constrained][1] += j["ms_gen"]
            extra = f"  [zorlanan token {j['forced']}/{j['constrained']}]" if constrained else ""
            forced += j.get("forced", 0)
            print(f"  {'kısıtlı ' if constrained else 'kısıtsız'} {'✓' if ok else '✗'} {desc}{extra}")
            print(f"           {' '.join(out.split())[:160]}")
    b.close()
    n = len(cases)
    print(f"\nmodel {a.model}, {a.backend}")
    for c in (False, True):
        tok, ms = stats[c]
        print(f"  {'kısıtlı ' if c else 'kısıtsız'}: {score[c]}/{n} doğru araç ve şemaya uygun, "
              f"{tok / (ms / 1000):.1f} token/sn üretim")
    print(f"  kısıtın değiştirdiği token: {forced}")


if __name__ == "__main__":
    main()
