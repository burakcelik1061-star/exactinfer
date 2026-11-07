"""Testlerin ortak ayarları: model yolu ve motor ikilisi."""

import json
import os
import sys
import urllib.request
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # Windows konsolu cp1254 olabilir; Türkçe/Unicode çıktı bozulmasın
    _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EL = Path(os.environ.get("EL_BIN", ROOT / "build" / "el.exe"))


def ollama_blob(model):
    """Ollama'nın yerel deposunda `model`in GGUF dosyasını bulur (ör. "qwen3:0.6b-fp16")."""
    name, _, tag = model.partition(":")
    base = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
    manifest = base / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
    layers = json.loads(manifest.read_text(encoding="utf-8"))["layers"]
    digest = next(l["digest"] for l in layers if l["mediaType"] == "application/vnd.ollama.image.model")
    return base / "blobs" / digest.replace(":", "-")


def model_path(default="qwen3:0.6b-fp16"):
    p = os.environ.get("EL_MODEL")
    if p:
        return Path(p)
    try:
        return ollama_blob(default)
    except (OSError, KeyError, StopIteration) as e:
        sys.exit(f"model bulunamadi ({e}); EL_MODEL ile GGUF yolunu verin ya da: ollama pull {default}")


OLLAMA = "http://127.0.0.1:11434"


def ollama_unload(model):
    """Ollama modeli GPU'dan boşaltır (keep_alive 0): 6 GB'lık kartta motorla ikisi birden sığmayabilir."""
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
    except OSError:
        pass  # Ollama çalışmıyorsa boşaltılacak bir şey yok
