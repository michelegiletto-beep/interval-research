"""Download public model weights from their official repository; no credentials."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
root = Path(__file__).resolve().parents[1]
destination = root / "models" / "base"
snapshot_download(BASE_MODEL, revision=REVISION, local_dir=destination, token=False,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md", "*.jinja"])
(root / "models" / "base-model.json").write_text(json.dumps({"repository": BASE_MODEL,
    "revision": REVISION, "license": "apache-2.0", "local_directory": "models/base"}, indent=2))
print(json.dumps({"downloaded": BASE_MODEL, "revision": REVISION, "destination": str(destination)}))
