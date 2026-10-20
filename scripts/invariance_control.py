"""A strictly monotone power transform must preserve rank-space observations."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np
from interval.data import Scene, split_scenes, Marginal, bins
from interval.experiment import ExperimentConfig, run_experiment, save_result


root = Path("evidence/invariance")
root.mkdir(parents=True, exist_ok=True)
records = []
for seed in range(51, 61):
    train, calibration, test = split_scenes(seed, 1.1)
    reference_counts = None
    quantum_reference = None
    for power in (.5, 1., 2.):
        groups = []
        for scenes in (train, calibration, test):
            transformed = []
            for scene in scenes:
                oracle = np.exp(power * (.08 + .9 * np.log(scene.before)) + .5 * power ** 2 * .22 ** 2)
                transformed.append(Scene(scene.before ** power, scene.after ** power, scene.changed, oracle))
            groups.append(transformed)
        result, fitted = run_experiment(ExperimentConfig(seed=seed,bits=4,quantum_steps=80), supplied_splits=tuple(groups))
        if reference_counts is None:
            reference_counts, quantum_reference = fitted["joint"], fitted["quantum_joint"]
        ranks_identical = bool(np.array_equal(fitted["joint"], reference_counts))
        circuit_identical = bool(np.allclose(fitted["quantum_joint"], quantum_reference, atol=1e-7))
        records.append({"seed":seed,"power":power,"rank_joint_identical":ranks_identical,
                        "quantum_distribution_identical":circuit_identical,
                        "metrics":result["metrics_by_scorer"]})
        save_result(result, root / "runs" / f"seed-{seed}-power-{power}")
        print(f"invariance seed={seed} power={power}: ranks={ranks_identical} circuit={circuit_identical}",flush=True)
(root / "summary.json").write_text(json.dumps({"records":records,"all_rank_invariant":all(r["rank_joint_identical"] for r in records),
    "all_circuit_invariant":all(r["quantum_distribution_identical"] for r in records),
    "implementation_sha256":result["implementation_sha256"],
    "interpretation":"Fixed observations under strictly monotone powers; changing inverse marginals and residual scores can still change detector outcomes."},indent=2))
