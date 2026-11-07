"""Kısıtlı çözümleme eşleştiricisinin sınaması (model gerekmez).

1. Saldırı tablosu: elle yazılmış çağrılar `el constrain-test`e verilir; beklenen sonuç (complete / prefix / invalid)
   denetlenir. Politika şemasındaki geçersiz çağrıların bekçi tarafından da reddedildiği ayrıca doğrulanır.
2. Çapraz denetim: `el constrain-test --fuzz` rastgele yürüyüşlerle (her adımda kabul edilen baytlardan biri)
   binlerce tamamlanmış çağrı üretir; hiçbir yürüyüş çıkmaza girmemeli. Her çağrı bekçiye `parse` ile verilir:
   eşleştiricinin "tamam" dediği her çağrıyı bekçi de kabul etmeli. Tek izinli ret, yolun izinli kök klasörlerin
   dışında olmasıdır (dosya sistemine bağlı; şemayla ifade edilemez).

    python tests\\constrain_bekci.py [--walks 3000] [--seeds 3]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from common import EL, ROOT
from e2e_bekci import BEKCI, Bekci

# Bekçinin dosya sistemine bağlı (şemayla ifade edilemeyen) ret nedenleri
JAIL = ("izinli klasörlerin dışında", "korumalı klasör", "ağ paylaşımına")


def call(name, args):
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


# (ham metin, eşleştirici beklentisi, bekçi de reddetmeli mi)
POLICY_CASES = [
    (call("open_app", {"name": "Hesap Makinesi"}), "complete", False),
    (call("system_status", {}), "complete", False),
    ('{"name": "system_status", "arguments": {}}\n', "complete", False),
    (call("set_volume", {"level": 100}), "complete", False),
    (call("note_add", {"text": "süt al\nekmek al"}), "complete", False),
    (call("file_search", {"query": "rapor 2024", "limit": 50}), "complete", False),
    (call("open_url", {"url": "https://tr.wikipedia.org/wiki/Ankara"}), "complete", False),
    (call("file_read", {"path": "C:\\Users\\x\\Desktop\\not.txt"}), "complete", None),
    ('{"name": "open_', "prefix", None),
    ('{"name": "file_read", "arguments": {"path": "C:\\\\con', "prefix", None),  # "conx" olabilir
    # beyaz liste ve şema
    (call("shutdown", {}), "invalid", True),
    (call("run_shell", {"cmd": "del /s C:\\"}), "invalid", True),
    (call("open_app", {"name": "cmd.exe"}), "invalid", True),
    (call("open_app", {"name": "Paint", "args": "/c whoami"}), "invalid", True),
    (call("open_app", {}), "invalid", True),
    ('{"name": "system_status", "arguments": {}, "sudo": true}', "invalid", True),
    ('{"name": "set_volume", "arguments": {"level": 1, "level": 2}}', "invalid", True),
    ('["open_app"]', "invalid", True),
    ('{"name": "open_app", "arguments": "Paint"}', "invalid", True),
    # tür karışıklığı
    *[(call("set_volume", {"level": v}), "invalid", True) for v in (True, 50.0, "50", 101, -1, None)],
    ('{"name": "set_volume", "arguments": {"level": 05}}', "invalid", True),
    ('{"name": "set_volume", "arguments": {"level": -0}}', "invalid", True),
    ('{"name": "set_volume", "arguments": {"level": 1e1}}', "invalid", True),
    # metin içeriği
    (call("file_search", {"query": "rapor; del *"}), "invalid", True),
    (call("file_search", {"query": " rapor"}), "invalid", True),
    (call("note_add", {"text": "gizli\u202egpj.exe"}), "invalid", True),
    ('{"name": "note_add", "arguments": {"text": "gizli\\u202egpj.exe"}}', "invalid", True),
    ('{"name": "note_add", "arguments": {"text": "a\\u0000b"}}', "invalid", True),
    ('{"name": "note_add", "arguments": {"text": "a\\rb"}}', "invalid", True),
    # yollar
    (call("file_read", {"path": "C:\\x\\..\\Windows\\win.ini"}), "invalid", True),
    (call("file_read", {"path": "\\\\sunucu\\pay\\x.txt"}), "invalid", True),
    (call("file_read", {"path": "\\\\?\\C:\\Windows\\win.ini"}), "invalid", True),
    (call("file_read", {"path": "C:\\x.txt:gizli"}), "invalid", True),
    (call("file_read", {"path": "C:\\Users\\NUL"}), "invalid", True),
    (call("file_read", {"path": "C:\\Users\\con.txt"}), "invalid", True),
    (call("file_read", {"path": "C:\\Users\\COM1 .log"}), "invalid", True),
    (call("file_read", {"path": "C:\\x.txt."}), "invalid", True),
    (call("file_read", {"path": "C:\\a\\*.txt"}), "invalid", True),
    (call("file_read", {"path": "x.txt"}), "invalid", True),
    # adresler
    (call("open_url", {"url": "http://ornek.com"}), "invalid", True),
    (call("open_url", {"url": "https://kullanici@kotu.com/"}), "invalid", True),
    (call("open_url", {"url": "https:// ornek.com"}), "invalid", True),
    (call("open_url", {"url": "https:///yol"}), "invalid", True),
    (call("open_url", {"url": "javascript:alert(1)"}), "invalid", True),
    # yapı
    ('{"arguments": {}, "name": "system_status"}', "invalid", None),  # eşleştirici sırayı sabitler
    ('{"name": "system_status", "arguments": {}} sonra', "invalid", None),
    ('{' + " " * 10, "invalid", None),
    ('{"name": "system_status", "arguments": {},}', "invalid", True),
]

# Politikada olmayan türler için küçük bir şema
EXTRA_TOOLS = [{"type": "function", "function": {"name": "t", "parameters": {
    "type": "object", "additionalProperties": False, "required": ["x"],
    "properties": {"x": {"type": "number", "minimum": -1.5, "maximum": 2.25},
                   "b": {"type": "boolean"},
                   "n": {"type": "integer", "minimum": -20, "maximum": -3},
                   "e": {"type": "string", "enum": ["ab", "abc"]},
                   "k": {"type": "string", "maxLength": 3, "format": "name"}}}}}]


def num(x, rest=""):
    return '{"name": "t", "arguments": {"x": ' + x + rest + '}}'


EXTRA_CASES = [
    (num("2.25"), "complete"), (num("-1.5"), "complete"), (num("0"), "complete"), (num("2.2500"), "complete"),
    (num("2.26"), "invalid"), (num("-1.51"), "invalid"), (num("3"), "invalid"), (num("1e0"), "invalid"),
    (num("00"), "invalid"), (num(".5"), "invalid"), (num("1."), "invalid"),
    (num("1", ', "b": true'), "complete"), (num("1", ', "b": 1'), "invalid"), (num("1", ', "b": tru'), "invalid"),
    (num("1", ', "n": -3'), "complete"), (num("1", ', "n": -20'), "complete"), (num("1", ', "n": -21'), "invalid"),
    (num("1", ', "n": -2'), "invalid"), (num("1", ', "n": 5'), "invalid"), ('{"name": "t", "arguments": {"x": 1, "n": -', "prefix"),
    (num("1", ', "e": "ab"'), "complete"), (num("1", ', "e": "abc"'), "complete"), (num("1", ', "e": "a"'), "invalid"),
    (num("1", ', "k": "a b"'), "complete"), (num("1", ', "k": "ab "'), "invalid"), (num("1", ', "k": "abcd"'), "invalid"),
    (num("1", ', "k": "ş1"'), "complete"), (num("1", ', "k": "a;"'), "invalid"),
    ('{"name": "t", "arguments": {}}', "invalid"),  # zorunlu x yok
]


def constrain_test(tools_file, texts):
    lines = "\n".join(t.encode("utf-8").hex() for t in texts) + "\n"
    r = subprocess.run([str(EL), "constrain-test", str(tools_file)], input=lines.encode(), capture_output=True)
    if r.returncode:
        sys.exit(r.stderr.decode("utf-8", "replace"))
    return [l.split()[0] for l in r.stdout.decode().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if not BEKCI.exists():
        sys.exit(f"{BEKCI} yok; önce: cargo build --release --manifest-path bekci\\Cargo.toml")

    tmp = Path(tempfile.mkdtemp(prefix="constrain-"))
    root = tmp / "izinli"
    root.mkdir()
    text = (ROOT / "bekci" / "policy.example.toml").read_text(encoding="utf-8")
    text = text.replace('# log = "%LOCALAPPDATA%\\\\engine-lab\\\\actions.log"   # varsayılan',
                        f"log = '{tmp / 'actions.log'}'")
    text = text.replace('confirm_ui = "dialog"', 'confirm_ui = "client"').replace('hotkey = "ctrl+alt+end"', "")
    text = text.replace('roots = ["%USERPROFILE%\\\\Documents", "%USERPROFILE%\\\\Desktop", "%USERPROFILE%\\\\Downloads"]',
                        f"roots = ['{root}']")
    policy = tmp / "bekci.toml"
    policy.write_text(text, encoding="utf-8")
    b = Bekci(policy)
    tools_file = tmp / "tools.json"
    tools_file.write_text(json.dumps(b.req(op="tools")["tools"], ensure_ascii=False), encoding="utf-8")
    extra_file = tmp / "extra.json"
    extra_file.write_text(json.dumps(EXTRA_TOOLS), encoding="utf-8")

    def bekci_decisions(raw):
        b.req(op="turn")
        r = b.req(op="parse", text=f"<tool_call>\n{raw}\n</tool_call>")
        if not r["ok"]:
            return [("deny", r["error"])]
        return [(c["decision"], c.get("reason", "")) for c in r["calls"]]

    fails = 0
    got = constrain_test(tools_file, [t for t, _, _ in POLICY_CASES])
    for (raw, want, bekci_denies), g in zip(POLICY_CASES, got):
        ok = g == want
        if bekci_denies:
            d = bekci_decisions(raw)
            ok &= all(x == "deny" for x, _ in d)
        fails += not ok
        if not ok:
            print(f"HATA  eşleştirici={g} beklenen={want}  {raw[:100]}")
    print(f"saldırı tablosu (politika): {len(POLICY_CASES)} durum, {fails} hata")
    f2 = 0
    for (raw, want), g in zip(EXTRA_CASES, constrain_test(extra_file, [t for t, _ in EXTRA_CASES])):
        if g != want:
            f2 += 1
            print(f"HATA  eşleştirici={g} beklenen={want}  {raw}")
    print(f"saldırı tablosu (sayı/mantıksal/enum/ad): {len(EXTRA_CASES)} durum, {f2} hata")
    fails += f2

    decisions, mism, walks = Counter(), 0, 0
    for seed in range(1, a.seeds + 1):
        for tf in (tools_file, extra_file):
            r = subprocess.run([str(EL), "constrain-test", str(tf), "--fuzz", str(a.walks), "--seed", str(seed)],
                               capture_output=True)
            out = r.stdout.decode().splitlines()
            if r.returncode:
                fails += 1
                print(f"ÇIKMAZ (tohum {seed}, {tf.name}): {out[-1][:200] if out else r.stderr.decode()}")
                continue
            if tf is extra_file:
                walks += len(out)
                continue
            for h in out:
                raw = bytes.fromhex(h).decode("utf-8")
                walks += 1
                for dec, why in bekci_decisions(raw):
                    if dec == "deny" and any(j in why for j in JAIL):
                        dec = "deny (kök dışı yol)"
                    decisions[dec] += 1
                    if dec == "deny":
                        mism += 1
                        if mism <= 10:
                            print(f"UYUŞMAZLIK  bekçi: {why}\n            çağrı: {raw[:300]!r}")
    b.close()
    print(f"bulanık sınama: {walks} yürüyüş, çıkmaz yok; bekçi kararları: {dict(decisions)}")
    fails += mism
    print("SONUC:", "TAMAM" if not fails else f"{fails} hata")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
