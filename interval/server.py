"""Single-user local service; public demo disables datasets and local training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import io
import json
import os
import re
import threading
import uuid
import zipfile
import time
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from . import __version__
from .experiment import ExperimentConfig, run_experiment, save_result
from .data import validate_uploaded_splits
from .planner import ROOT, RECIPES, plan

PUBLIC = os.getenv("INTERVAL_PUBLIC_DEMO", "0") == "1"
RUNTIME = Path(os.getenv("INTERVAL_RUNTIME", str(ROOT / "runtime")))
RUNTIME.mkdir(parents=True, exist_ok=True)
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="interval")
LOCK = threading.RLock()
JOBS = {}
DATASETS = {}
CAPACITY = threading.BoundedSemaphore(2)
LAST_ADMISSION = {}


def prune_public_records():
    """Remove only this public demo's identified generated records beyond its cap."""
    if not PUBLIC:
        return
    import shutil
    root = RUNTIME.resolve()
    identified = []
    for record in root.glob("*/job.json"):
        if record.parent.is_symlink() or not re.fullmatch(r"[a-f0-9]{32}", record.parent.name):
            continue
        try:
            saved = json.loads(record.read_text())
        except (OSError, ValueError):
            continue
        if saved.get("id") == record.parent.name and saved.get("status") in {"complete", "failed"}:
            identified.append(record)
    for record in sorted(identified, key=lambda p: p.stat().st_mtime, reverse=True)[20:]:
        target = record.parent.resolve()
        if target.parent == root and target.is_relative_to(root):
            shutil.rmtree(target)

app = FastAPI(title="Interval", version=__version__, docs_url="/api/docs", redoc_url=None)


class BodyBudgetExceeded(HTTPException):
    def __init__(self):
        super().__init__(status_code=413, detail="Request body exceeds its budget")


class BodyBudgetMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        budget = 13 * 1024 * 1024 if scope.get("path") == "/api/datasets" and not PUBLIC else 16384
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > budget:
                    raise BodyBudgetExceeded()
            return message

        try:
            await self.app(scope, bounded_receive, send)
        except BodyBudgetExceeded:
            await JSONResponse({"detail": "Request body exceeds its budget"}, status_code=413)(scope, receive, send)


app.add_middleware(BodyBudgetMiddleware)


@app.middleware("http")
async def boundaries(request: Request, call_next):
    host = request.url.hostname
    allowed = {"localhost", "127.0.0.1", "::1"} | set(filter(None, os.getenv("INTERVAL_ALLOWED_HOSTS", "").split(",")))
    if host not in allowed and not (PUBLIC and host and host.endswith(".hf.space")):
        return JSONResponse({"detail": "Host is not admitted"}, status_code=400)
    length = request.headers.get("content-length")
    limit = 13 * 1024 * 1024 if request.url.path == "/api/datasets" and not PUBLIC else 16384
    if length and (not length.isdigit() or int(length) > limit):
        return JSONResponse({"detail": "Request body exceeds its budget"}, status_code=413)
    origin = request.headers.get("origin")
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Cross-origin writes are not accepted"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'"
    return response


def checkpoint(job):
    destination = RUNTIME / job["id"]
    destination.mkdir(exist_ok=True)
    temporary = destination / "job.tmp"
    temporary.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(destination / "job.json")


def event(job, stage, message, detail=None):
    with LOCK:
        job["events"].append({"sequence": len(job["events"]) + 1, "utc": datetime.now(timezone.utc).isoformat(),
                              "stage": stage, "message": message, "detail": detail or {}})
        checkpoint(job)


def execute(job, body):
    try:
        job["status"] = "planning"
        event(job, "plan", "Interpreting the requested research procedure")
        proposal = plan(body.request, body.manual_recipe)
        if proposal["recipe"] not in RECIPES or proposal["stages"] != RECIPES[proposal["recipe"]]["stages"]:
            raise ValueError("Planner output does not match an admitted procedure")
        if PUBLIC and proposal["recipe"] in {"tails", "measurements"}:
            raise ValueError("Multi-run sweeps are available in the local application; select Compare estimators for the public demonstration")
        job["plan"] = proposal
        job["status"] = "running"
        event(job, "admit", "Admitted a bounded research procedure", proposal)
        configs = [body.config]
        if proposal["recipe"] == "tails":
            if body.dataset_id:
                raise ValueError("Tail sweeps require synthetic observations; uploaded data are never altered implicitly")
            configs = [body.config.model_copy(update={"tail": tail}) for tail in (.45, 1.1, 1.8)]
        splits = DATASETS.get(body.dataset_id) if body.dataset_id else None
        if body.dataset_id and splits is None:
            raise ValueError("Dataset is unavailable; import it again")
        if proposal["recipe"] == "measurements":
            configs = [body.config.model_copy(update={"shots": n}) for n in (1000, 5000, 20000)]
        elif proposal["recipe"] == "neural":
            configs = [body.config.model_copy(update={"neural_steps": 140 if PUBLIC else 300})]
        fitted = None
        for i, config in enumerate(configs):
            reuse = fitted if proposal["recipe"] == "measurements" else None
            result, fitted = run_experiment(config, lambda s, m, d: event(job, s, m, d), splits, reuse)
            save_result(result, RUNTIME / job["id"] / f"run-{i}", fitted)
            job["result_files"].append(f"run-{i}")
            job["latest_result"] = result
            event(job, "archive", "Saved result, neural weights and distribution arrays", {"content_sha256": result["content_sha256"], "run": i})
        job["status"] = "complete"
        event(job, "complete", "Procedure completed; evidence is available for inspection and export")
    except Exception as error:
        job["status"] = "failed"
        event(job, "failed", str(error)[:500])
    finally:
        try:
            checkpoint(job)
            if PUBLIC:
                with LOCK:
                    completed = [key for key, value in JOBS.items() if value["status"] in {"complete", "failed"}]
                    for key in completed[:-8]:
                        JOBS.pop(key, None)
                prune_public_records()
        finally:
            CAPACITY.release()


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: str = Field("Compare the classical, neural and quantum estimators.", min_length=3, max_length=1500)
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)
    manual_recipe: str | None = None
    project: str = Field("Research", max_length=100)
    dataset_id: str | None = None


def find_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "Unknown run")
    with LOCK:
        if job_id in JOBS:
            return JOBS[job_id]
    path = RUNTIME / job_id / "job.json"
    if not path.exists():
        raise HTTPException(404, "Unknown run")
    job = json.loads(path.read_text())
    if job["status"] in {"queued", "planning", "running"}:
        job["status"] = "interrupted"
    return job


@app.get("/api/status")
def status():
    receipt = ROOT / "models" / "training-receipt.json"
    training = json.loads(receipt.read_text()) if receipt.exists() else None
    return {"version": __version__, "public_demo": PUBLIC, "recipes": RECIPES,
            "base_model_installed": (ROOT / "models" / "base" / "config.json").exists(),
            "adapted_model_installed": (ROOT / "models" / "planner" / "adapter_config.json").exists(),
            "training": training, "runtime": "local open-weight inference", "single_user": True}


@app.post("/api/runs", status_code=202)
def launch(body: RunRequest, request: Request):
    if PUBLIC:
        if body.config.bits > 4 or body.config.quantum_steps > 100 or body.config.neural_steps > 140:
            raise HTTPException(422, "Public compute budget: at most 4 bits, 100 quantum steps and 140 neural steps")
        if body.manual_recipe in {"tails", "measurements"}:
            raise HTTPException(422, "Use the local application for multi-run sweeps")
        now = time.monotonic()
        with LOCK:
            previous = LAST_ADMISSION.get("global", 0.)
            if now - previous < 20:
                raise HTTPException(429, "Public compute is limited to one admission every 20 seconds")
            LAST_ADMISSION["global"] = now
    if body.manual_recipe is not None and body.manual_recipe not in RECIPES:
        raise HTTPException(422, "Choose an available procedure")
    if body.dataset_id and PUBLIC:
        raise HTTPException(403, "The public demonstration accepts synthetic observations only")
    if body.dataset_id and body.dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset is unavailable")
    if not CAPACITY.acquire(blocking=False):
        raise HTTPException(429, "The local worker has two admitted jobs. Wait for completion.")
    job_id = uuid.uuid4().hex
    job = {"id": job_id, "project": body.project, "status": "queued", "events": [], "result_files": [],
           "created_utc": datetime.now(timezone.utc).isoformat(), "request": body.request,
           "config": body.config.model_dump(), "plan": None, "latest_result": None}
    with LOCK:
        JOBS[job_id] = job
        checkpoint(job)
    POOL.submit(execute, job, body)
    return {"id": job_id, "status": "queued"}


@app.get("/api/runs")
def runs():
    if PUBLIC:
        return []  # A public visitor must not browse another visitor's prompts.
    files = sorted(RUNTIME.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    return [{key: item.get(key) for key in ("id", "project", "status", "created_utc")}
            for item in (json.loads(path.read_text()) for path in files)]


@app.get("/api/runs/{job_id}")
def get_run(job_id: str):
    return find_job(job_id)


@app.get("/api/runs/{job_id}/download")
def download(job_id: str):
    job = find_job(job_id)
    if job["status"] != "complete":
        raise HTTPException(409, "The run has not completed")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        directory = RUNTIME / job_id
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".npz", ".safetensors"}:
                archive.write(path, path.relative_to(directory).as_posix())
    return StreamingResponse(iter([data.getvalue()]), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="interval-{job_id[:8]}.zip"'})


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    if PUBLIC:
        raise HTTPException(403, "Use the local application for your own observations")
    payload = await file.read(12 * 1024 * 1024 + 1)
    try:
        splits = validate_uploaded_splits(payload)
    except Exception as error:
        raise HTTPException(422, str(error)) from error
    key = sha256(payload).hexdigest()[:24]
    if len(DATASETS) >= 4 and key not in DATASETS:
        raise HTTPException(429, "Restart the local service before importing more than four datasets")
    DATASETS[key] = splits
    return {"id": key, "sha256": sha256(payload).hexdigest(), "split_pixels": [int(s[0].before.size) for s in splits],
            "status": "Validated independent array identities; provenance requires owner verification"}


@app.get("/api/reference")
def reference():
    path = ROOT / "evidence" / "showcase" / "result.json"
    if not path.exists():
        path = ROOT / "evidence" / "smoke" / "result.json"
    if not path.exists():
        raise HTTPException(404, "No measured reference result is installed")
    return json.loads(path.read_text())


@app.get("/api/benchmark")
def benchmark():
    path = ROOT / "evidence" / "final-dense" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not installed"}


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
app.mount("/docs", StaticFiles(directory=ROOT / "docs", html=True), name="documents")
