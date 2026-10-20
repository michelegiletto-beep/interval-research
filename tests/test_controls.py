import io
import json
import time
import zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from interval.data import split_scenes, Scene, validate_uploaded_splits
from interval.experiment import ExperimentConfig, run_experiment
from interval.experiment import save_result
from interval.estimators import restore_fitted, predict_background, METHOD_NAMES
from interval.evaluation import calibrate, spatial_mask
from interval.server import app


def upload_bytes(duplicate=False):
    groups = split_scenes(519, 1.1, 16)
    arrays = {}
    for part, scenes in zip(("train", "calibration", "test"), groups):
        scene = groups[0][0] if duplicate else scenes[0]
        arrays[f"{part}_before"] = scene.before
        arrays[f"{part}_after"] = scene.after
        if part != "train":
            arrays[f"{part}_mask"] = scene.changed
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    return data.getvalue()


def test_calibration_and_fit_are_invariant_to_test_annotations():
    splits = split_scenes(929, .7, 16)
    train, calibration, test = splits
    flipped = [Scene(s.before, s.after, ~s.changed, s.expected_background) for s in test]
    config = ExperimentConfig(seed=929, bits=2, side=16, quantum_steps=10, neural_steps=10)
    original, fitted = run_experiment(config, supplied_splits=splits)
    altered, alternate = run_experiment(config, supplied_splits=(train, calibration, flipped))
    np.testing.assert_allclose(fitted["quantum_joint"], alternate["quantum_joint"], atol=1e-7)
    assert [r["calibration"] for r in original["metrics"]] == [r["calibration"] for r in altered["metrics"]]
    assert original["metrics"][0]["f1"] != altered["metrics"][0]["f1"]


def test_upload_rejects_duplicate_and_oversized_expanded_arrays():
    validate_uploaded_splits(upload_bytes())
    with pytest.raises(ValueError, match="Identical"):
        validate_uploaded_splits(upload_bytes(duplicate=True))
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", b"0" * (25 * 1024 * 1024))
    with pytest.raises(ValueError, match="expanded"):
        validate_uploaded_splits(data.getvalue())
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"shape": (10**12, 10**12), "descr": "<f8", "fortran_order": False})
    malformed = io.BytesIO()
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("train_before.npy", header.getvalue())
    with pytest.raises(ValueError, match="headers"):
        validate_uploaded_splits(malformed.getvalue())


def test_negative_calibration_selects_no_false_alarms():
    scores = np.arange(100).reshape(10, 10)
    settings, _ = calibrate([scores], [np.zeros_like(scores, bool)])
    assert not spatial_mask(scores, **settings).any()


def test_portable_bundle_preserves_every_background_prediction(tmp_path):
    config = ExperimentConfig(seed=545, bits=2, side=16, quantum_steps=10, neural_steps=10)
    result, fitted = run_experiment(config)
    save_result(result, tmp_path, fitted)
    restored = restore_fitted(tmp_path)
    before = split_scenes(984, .7, 16)[2][0].before
    for method in METHOD_NAMES:
        np.testing.assert_allclose(predict_background(fitted, method, before), predict_background(restored, method, before), rtol=1e-6, atol=1e-6)


def test_api_rejects_unknown_tools_resource_overruns_and_cross_origin_writes():
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/runs", json={"manual_recipe":"shell"}).status_code == 422
        assert client.post("/api/runs", json={"config":{"bits":20}}).status_code == 422
        assert client.post("/api/runs", json={"config":{"command":"arbitrary"}}).status_code == 422
        assert client.post("/api/runs", json={}, headers={"Origin":"https://unrelated.invalid"}).status_code == 403
        assert client.get("/api/status", headers={"Host":"unrelated.invalid"}).status_code == 400
        assert client.post("/api/runs", content=(chunk for chunk in [b"x" * 20000]), headers={"Content-Type":"application/json"}).status_code == 413
        assert client.get("/api/runs/not-a-capability").status_code == 404


def test_api_executes_archives_and_exports_a_real_procedure():
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/api/runs", json={"manual_recipe":"compare", "project":"Integration test",
            "config":{"seed":321, "bits":2,"side":16,"quantum_steps":10,"neural_steps":10}})
        assert response.status_code == 202
        key = response.json()["id"]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = client.get(f"/api/runs/{key}").json()
            if result["status"] in {"complete", "failed"}:
                break
            time.sleep(.1)
        assert result["status"] == "complete", result["events"]
        assert result["plan"]["origin"] == "explicit-user-selection"
        assert len(result["latest_result"]["metrics"]) == 9
        bundle = client.get(f"/api/runs/{key}/download")
        assert bundle.status_code == 200
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            assert "run-0/conditional-network.safetensors" in archive.namelist()
            assert "run-0/result.json" in archive.namelist()
            assert json.loads(archive.read("run-0/result.json"))["config"]["seed"] == 321
