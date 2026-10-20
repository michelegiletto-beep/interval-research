"""Gradio-SDK transport for the same FastAPI research application."""
from pathlib import Path
from hashlib import sha256
import json
import os
import subprocess
import sys
import tarfile

os.environ["INTERVAL_PUBLIC_DEMO"] = "1"
os.environ["INTERVAL_RUNTIME"] = "/tmp/interval-runtime"
os.environ.setdefault("OMP_NUM_THREADS", "2")
archive = Path(__file__).with_name("interval-source.tar.gz")
source = Path("/tmp") / ("interval-source-" + sha256(archive.read_bytes()).hexdigest()[:16])
source.mkdir(exist_ok=True)
with tarfile.open(archive) as package:
    package.extractall(source, filter="data")
manifest = json.loads((source / "release-manifest.json").read_text())
for item in manifest["files"]:
    assert sha256((source / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
sys.path.insert(0, str(source))
if not (source / "models" / "base" / "config.json").exists():
    subprocess.run([sys.executable, str(source / "scripts" / "download_model.py")], check=True)

import gradio as gr
from fastapi import FastAPI
from interval.server import app as research_app, status

with gr.Blocks() as diagnostics:
    gr.Markdown("# Interval runtime\nEmbedded model and numerical-service status. The research workspace is available at the root URL.")
    output = gr.JSON(value=status())
    gr.Button("Refresh runtime status").click(status, outputs=output, api_name="runtime_status")

app = gr.mount_gradio_app(FastAPI(), diagnostics, path="/runtime-status")
app.mount("/", research_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
