"""Exercise a running synthetic demonstration, including model routing and export."""
from pathlib import Path
from time import monotonic, sleep
import argparse
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/hosted-smoke"))
    args = parser.parse_args()
    base = args.url.rstrip("/")
    origin = base
    args.output.mkdir(parents=True, exist_ok=True)

    def request(path, data=None, headers=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(base + path, data=body,
            headers={"Content-Type": "application/json", "Origin": origin, **(headers or {})})
        with urllib.request.urlopen(req, timeout=60) as response:
            return response.read()

    status = json.loads(request("/api/status"))
    assert status["public_demo"] and status["base_model_installed"] and status["adapted_model_installed"]
    try:
        request("/api/runs", {}, {"Origin": "https://example.invalid"})
        raise AssertionError("A cross-origin write was admitted")
    except urllib.error.HTTPError as error:
        assert error.code == 403, error.code
    try:
        request("/api/runs", {"request": "x" * 17000})
        raise AssertionError("An oversized body was admitted")
    except urllib.error.HTTPError as error:
        assert error.code == 413, error.code

    started = monotonic()
    launch = json.loads(request("/api/runs", {
        "project": "Public release verification", "request": "Compare the classical, neural and quantum background estimators.",
        "config": {"seed": 101, "bits": 3, "quantum_steps": 20, "neural_steps": 40,
                   "training_pixels": 256, "score_mode": "log"}}))
    path = "/api/runs/" + launch["id"]
    while True:
        job = json.loads(request(path))
        if job["status"] in {"complete", "failed", "interrupted"}:
            break
        if monotonic() - started > 180:
            raise TimeoutError("Public procedure exceeded three minutes")
        sleep(2)
    (args.output / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    assert job["status"] == "complete", job.get("events", [])[-1]
    assert job["plan"]["origin"] == "embedded-language-model-likelihood"
    assert job["plan"]["recipe"] == "compare"
    assert len(job["latest_result"]["metrics"]) == 9
    exported = request(path + "/download")
    with zipfile.ZipFile(io.BytesIO(exported)) as archive:
        names = archive.namelist()
        assert any(n.endswith("inference.json") for n in names)
        assert any(n.endswith("conditional-network.safetensors") for n in names)
        assert any(n.endswith("inference-arrays.npz") for n in names)
    (args.output / "evidence.zip").write_bytes(exported)
    receipt = {"url": base, "job_id": job["id"], "status": job["status"],
        "planner_origin": job["plan"]["origin"], "recipe": job["plan"]["recipe"],
        "adapter_sha256": job["plan"]["adapter_sha256"], "methods": len(job["latest_result"]["metrics"]),
        "result_sha256": job["latest_result"]["content_sha256"], "seconds": monotonic() - started,
        "cross_origin_write": "403", "oversized_body": "413", "export_files": len(names),
        "export_bytes": len(exported)}
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
