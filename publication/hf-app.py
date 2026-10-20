"""Gradio-SDK transport for the same FastAPI research application."""
from pathlib import Path
from hashlib import sha256
import json
import os
import subprocess
import sys
import tarfile
import atexit
import time
import urllib.request

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

# Keep CPU optimization outside ZeroGPU's process-wide CUDA emulation.
# This is the unmodified local server, bound to a private loopback port.
child_env = dict(os.environ)
child_env["PYTHONPATH"] = str(source) + os.pathsep + child_env.get("PYTHONPATH", "")
cpu_backend = subprocess.Popen([sys.executable, "-m", "interval.cli", "serve", "--port", "8765"],
                               cwd=source, env=child_env)
atexit.register(cpu_backend.terminate)


def status():
    with urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=5) as response:
        return json.load(response)


for attempt in range(120):
    if cpu_backend.poll() is not None:
        raise RuntimeError("The isolated CPU research backend exited during startup")
    try:
        status()
        break
    except (OSError, ValueError):
        time.sleep(.25)
else:
    raise RuntimeError("The isolated CPU research backend did not become ready")

import gradio as gr
import spaces
import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response, RedirectResponse
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


@app.get("/runtime-status", include_in_schema=False)
def diagnostics_redirect():
    return RedirectResponse("/runtime-status/")


class ResearchProxy:
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 16384:
                await JSONResponse({"detail": "Request body exceeds its budget"}, status_code=413)(scope, receive, send)
                return
        excluded = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "content-length", "content-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded}
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                result = await client.request(request.method, "http://127.0.0.1:8765" + request.url.path,
                                              params=request.query_params, headers=headers, content=bytes(body))
            response = Response(result.content, status_code=result.status_code,
                                headers={k: v for k, v in result.headers.items() if k.lower() not in excluded})
        except httpx.RequestError:
            response = JSONResponse({"detail": "The research backend is unavailable"}, status_code=503)
        await response(scope, receive, send)


app.mount("/", ResearchProxy())

if __name__ == "__main__":
    import uvicorn
    # mount_gradio_app uses Uvicorn instead of Blocks.launch, so explicitly run
    # the SDK's normal launch hook after registration of the real GPU function.
    from spaces import zero
    if hasattr(zero, "startup"):
        zero.startup()
    uvicorn.run(app, host="0.0.0.0", port=7860)
