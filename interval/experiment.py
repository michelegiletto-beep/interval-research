from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import platform
import sys
import numpy as np
import torch
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from .data import split_scenes
from .estimators import METHOD_NAMES, fit_estimators, change_score, update_measurements
from .evaluation import calibrate, mask_metrics, continuous_metrics, spatial_mask

IMPLEMENTATION_SHA256 = sha256(b"".join((Path(__file__).parent / name).read_bytes()
    for name in ("data.py", "quantum.py", "estimators.py", "evaluation.py", "experiment.py"))).hexdigest()


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(7, ge=0, le=999999)
    bits: int = Field(4, ge=2, le=5)
    tail: float = Field(1.1, ge=.2, le=2.)
    side: int = Field(32, ge=16, le=64)
    quantum_steps: int = Field(100, ge=10, le=200)
    neural_steps: int = Field(140, ge=10, le=300)
    shots: int = Field(5000, ge=100, le=100000)
    readout_error: float = Field(.01, ge=0, le=.1)
    test_shift: float = Field(0., ge=-.3, le=.3)
    training_pixels: int = Field(3072, ge=32, le=12288)
    score_mode: Literal["absolute", "log"] = "absolute"


def compact_image(values, max_side=64):
    x = np.asarray(values)
    stride = max(1, int(np.ceil(max(x.shape) / max_side)))
    return x[::stride, ::stride].tolist()


def run_experiment(config: ExperimentConfig, progress=None, supplied_splits=None, reuse_fitted=None):
    """Fit once, calibrate once, test once. No optimizer sees test observations."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    start = perf_counter()
    emit = progress or (lambda *a, **k: None)
    train, calibration, test = supplied_splits or split_scenes(config.seed, config.tail, config.side, config.test_shift)
    identities = {part: [s.identity() for s in scenes]
                  for part, scenes in (("train", train), ("calibration", calibration), ("test", test))}
    flattened = sum(identities.values(), [])
    if len(set(flattened)) != len(flattened):
        raise ValueError("Duplicate observations in independent splits")
    emit("inspect", "Validated independent observation splits", {"split_hashes": identities})
    x = np.concatenate([s.before.ravel() for s in train])
    y = np.concatenate([s.after.ravel() for s in train])
    available_training_pixels = len(x)
    if config.training_pixels < len(x):
        selected = np.random.default_rng(config.seed + 8731).choice(len(x), config.training_pixels, replace=False)
        x, y = x[selected], y[selected]
    emit("fit", "Fitting classical estimators, neural network and quantum copula", {})
    fitted = reuse_fitted if reuse_fitted is not None else fit_estimators(x, y, config)
    if reuse_fitted is not None:
        update_measurements(fitted, config)
    emit("calibrate", "Selecting thresholds on calibration scenes only", fitted["statistics"])
    rows_by_score, preview_masks, preview_scores = {}, {}, {}
    labels = [s.changed for s in calibration]
    for score_mode in ("absolute", "log"):
        rows = []
        for method, name in METHOD_NAMES.items():
            cal_scores = [change_score(fitted, method, s, score_mode)[0] for s in calibration]
            settings, cal_f1 = calibrate(cal_scores, labels)
            scene_rows = []
            for i, scene in enumerate(test):
                score, background = change_score(fitted, method, scene, score_mode)
                predicted = spatial_mask(score, **settings)
                metrics = mask_metrics(scene.changed, predicted)
                metrics.update(continuous_metrics(scene.changed, score))
                if scene.expected_background is not None:
                    metrics["background_nrmse"] = float(np.sqrt(np.mean((background - scene.expected_background) ** 2)) / np.mean(scene.expected_background))
                scene_rows.append(metrics)
                if i == 0 and score_mode == config.score_mode:
                    preview_masks[method] = compact_image(predicted.astype(int))
                    preview_scores[method] = compact_image(score)
            aggregate = {key: float(np.mean([r[key] for r in scene_rows if r.get(key) is not None]))
                         for key in scene_rows[0] if any(r.get(key) is not None for r in scene_rows)}
            rows.append({"method": method, "name": name, "scorer": score_mode,
                         "calibration": settings, "calibration_f1": cal_f1,
                         **aggregate, "scenes": scene_rows})
        rows_by_score[score_mode] = rows
    elapsed = perf_counter() - start
    emit("evaluate", "Evaluated frozen settings on independent test scenes", {"seconds": elapsed})
    qmetadata = fitted["quantum"].copy()
    # Parameters are available for exact reproduction; no opaque model labels.
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "implementation_sha256": IMPLEMENTATION_SHA256,
              "config": config.model_dump(), "source": "uploaded-independent-splits" if supplied_splits else "synthetic-lognormal-v1",
              "split_hashes": identities, "statistics": fitted["statistics"], "quantum": qmetadata,
              "seconds": elapsed, "metrics": rows_by_score[config.score_mode], "metrics_by_scorer": rows_by_score,
              "available_training_pixels": available_training_pixels,
              "effective_training_sha256": sha256(x.tobytes() + y.tobytes()).hexdigest(),
              "neural_loss": fitted["neural_loss"], "fit_reused": reuse_fitted is not None,
              "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()},
              "preview": {"before": compact_image(test[0].before), "after": compact_image(test[0].after),
                          "truth": compact_image(test[0].changed.astype(int)), "predictions": preview_masks,
                          "scores": preview_scores, "observed_joint": fitted["joint"].tolist(),
                          "quantum_joint": fitted["quantum_joint"].tolist()},
              "limits": ["Synthetic observations are not real SAR or medical validation." if not supplied_splits else "Uploaded data identity is recorded; provenance and annotations remain the dataset owner's responsibility.",
                         "Pure-state simulation with terminal bit-flip readout model; no physical QPU execution.",
                         "Threshold and spatial-window choices use calibration labels only.",
                         "Exact discretized conditional integration is simulator-only; hardware rejection cost is separate.",
                         "Small circuit and optimizer differ from the IonQ study; this is not an exact reproduction."]}
    stable = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    result["content_sha256"] = sha256(stable).hexdigest()
    return result, fitted


def save_result(result, destination, fitted=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if fitted is not None:
        from safetensors.torch import save_file
        save_file({k: v.detach().contiguous() for k, v in fitted["network"].state_dict().items()}, str(destination / "conditional-network.safetensors"))
        np.savez_compressed(destination / "distributions.npz", empirical=fitted["joint"], quantum=fitted["quantum_joint"])
        np.savez_compressed(destination / "inference-arrays.npz", marginal_x=fitted["marginal_x"].values,
                            marginal_y=fitted["marginal_y"].values,
                            **{f"lookup_{key}": value for key, value in fitted["lookup"].items()})
        (destination / "inference.json").write_text(json.dumps({"schema_version": 1, "scale": fitted["scale"],
            "offset": fitted["offset"], "config": result["config"], "result_sha256": result["content_sha256"],
            "network": "1-32-32-1-tanh"}, indent=2))
