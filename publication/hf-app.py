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
import spaces
from fastapi import FastAPI
from interval.server import app as research_app, status
from interval.planner import load_model, score_candidates

# ZeroGPU intercepts this placement, then packs weights before serving requests.
# The CPU research planner retains its independent cache and numerical path.
gpu_tokenizer, gpu_model = load_model()
gpu_model.to("cuda")


@spaces.GPU(duration=30)
def check_gpu_planner(request):
    """Optional GPU inference check; the research workspace executes on CPU."""
    if not isinstance(request, str) or not 3 <= len(request) <= 1500:
        raise gr.Error("Enter a research request of 3–1500 characters.")
    from time import perf_counter
    started = perf_counter()
    scores = score_candidates(gpu_tokenizer, gpu_model, request)
    return {"recipe": max(scores, key=scores.get), "candidate_scores": scores,
            "device": str(next(gpu_model.parameters()).device), "inference_seconds": perf_counter() - started,
            "scope": "Diagnostic inference only; numerical execution is available in the main workspace."}


with gr.Blocks() as diagnostics:
    gr.Markdown("# Interval runtime\nEmbedded model and numerical-service status. The research workspace is available at the root URL.")
    output = gr.JSON(value=status())
    gr.Button("Refresh runtime status").click(status, outputs=output, api_name="runtime_status")
    gr.Markdown("## Optional GPU planner check\nRun the same adapted model on an allocated GPU. This checks procedure selection; it does not launch numerical experiments.")
    request = gr.Textbox(value="Compare the background estimators on this study", label="Research request")
    gpu_output = gr.JSON(label="GPU procedure proposal")
    gr.Button("Check adapted planner on GPU").click(check_gpu_planner, inputs=request, outputs=gpu_output, api_name="gpu_planner_check")

diagnostics.queue()
app = gr.mount_gradio_app(FastAPI(), diagnostics, path="/runtime-status")
app.mount("/", research_app)

if __name__ == "__main__":
    import uvicorn
    # mount_gradio_app uses Uvicorn instead of Blocks.launch, so explicitly run
    # the SDK's normal launch hook after registration of the real GPU function.
    from spaces import zero
    if hasattr(zero, "startup"):
        zero.startup()
    uvicorn.run(app, host="0.0.0.0", port=7860)
