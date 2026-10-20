"""Frozen factorial benchmark. Seeds, settings and methods precede measurement."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from interval.experiment import ExperimentConfig, run_experiment, save_result
from interval.evaluation import bootstrap_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/reference"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": list(range(31, 31 + args.seeds)), "bits": [3, 4, 5], "tails": [1.1] if args.sparse else [.45, 1.1, 1.8],
                "training_pixels": [64, 256, 1024] if args.sparse else [3072], "scorers": ["absolute", "log"],
                "role": "sparsity extension" if args.sparse else "dense control",
                "test_shifts": [0.], "steps": args.steps, "protocol_written_before_execution": True,
                "unit": "independent training/calibration/test scene groups; average test metrics per seed",
                "inference": "synthetic benchmark, not evidence of clinical or satellite performance"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    rows, start = [], perf_counter()
    for tail in protocol["tails"]:
        for bits in protocol["bits"]:
          for training_pixels in protocol["training_pixels"]:
            for seed in protocol["seeds"]:
                config = ExperimentConfig(seed=seed, bits=bits, tail=tail, quantum_steps=args.steps, training_pixels=training_pixels)
                result, fitted = run_experiment(config)
                for scorer, metrics in result["metrics_by_scorer"].items():
                  for row in metrics:
                    rows.append({"tail": tail, "bits": bits, "training_pixels": training_pixels, "scorer": scorer,
                                 "seed": seed, "method": row["method"],
                                 **{key: row.get(key) for key in ("f1", "average_precision", "recall_at_1pct_fpr", "background_nrmse")},
                                 "experiment_seconds": result["seconds"], "occupancy": result["statistics"]["occupancy"],
                                 "median_observations_per_condition": result["statistics"]["median_observations_per_condition"]})
                run_dir = args.output / "runs" / f"tail-{tail}-bits-{bits}-n-{training_pixels}-seed-{seed}"
                save_result(result, run_dir)
                print(f"tail={tail} bits={bits} n={training_pixels} seed={seed}: {result['seconds']:.2f}s", flush=True)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for tail in protocol["tails"]:
        for bits in protocol["bits"]:
          for training_pixels in protocol["training_pixels"]:
           for scorer in protocol["scorers"]:
            group = [r for r in rows if r["tail"] == tail and r["bits"] == bits and r["training_pixels"] == training_pixels and r["scorer"] == scorer]
            for method in sorted({r["method"] for r in group}):
                selected = [r for r in group if r["method"] == method]
                summaries.append({"tail": tail, "bits": bits, "training_pixels": training_pixels, "scorer": scorer, "method": method,
                                  **{key: bootstrap_mean([r[key] for r in selected])
                                     for key in ("f1", "average_precision", "background_nrmse")}})
    paired = []
    for tail in protocol["tails"]:
        for bits in protocol["bits"]:
          for training_pixels in protocol["training_pixels"]:
           for scorer in protocol["scorers"]:
            for left, right in [("copula_intensity_mean", "copula_rank_mean"), ("qcbm_exact", "smoothed_copula"),
                                ("qcbm_shots", "qcbm_exact"), ("neural_regression", "smoothed_copula")]:
                differences = []
                for seed in protocol["seeds"]:
                    group = {r["method"]: r for r in rows if r["tail"] == tail and r["bits"] == bits and r["seed"] == seed and r["training_pixels"] == training_pixels and r["scorer"] == scorer}
                    differences.append(group[left]["f1"] - group[right]["f1"])
                paired.append({"tail": tail, "bits": bits, "training_pixels": training_pixels, "scorer": scorer, "left": left, "right": right,
                               "delta_f1": bootstrap_mean(differences)})
    summary = {"protocol": protocol, "seconds": perf_counter() - start, "summaries": summaries,
               "implementation_sha256": result["implementation_sha256"],
               "paired_comparisons": paired, "multiple_comparisons": "Exploratory intervals; no familywise error correction. No confirmatory significance claims."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"completed": len(rows) // 18, "seconds": summary["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
