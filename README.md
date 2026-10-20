# Interval

Local research software for conditional background estimation and change detection. Interval fits classical estimators, a neural conditional-mean network and a small quantum copula simulator; calibrates detection settings on separate observations; and archives measured results and reusable inference bundles.

An embedded open-weight language model selects one of four research procedures. Native code validates and executes the selection. A locally trained LoRA adapter, synthetic instruction data and evaluation receipt are supplied. No paid inference API or API key is required.

**Status:** working single-user research reference. Included results use synthetic observations. No satellite, clinical, physical-QPU or quantum-advantage claim is made. Project profiles are application starting points, not validated medical or field products.

## Links

- [System Labs](https://giletto-systems-lab.michele-giletto.chatgpt.site)
- [Source](https://github.com/michelegiletto-beep/interval-research)
- [Executable public demonstration](https://huggingface.co/spaces/Magic51/interval-research)
- [Research protocol](docs/research-protocol.md)
- [Results and manuscript draft](docs/manuscript.md)
- [Model card](docs/model-card.md)
- [Three review cycles](docs/review-cycles.md)
- [Product and application plans](docs/business-development.md)
- [LinkedIn article and post — drafts only](publication/linkedin-article.md)

## Local installation

Python 3.11–3.13 is supported by the source. Create an isolated environment and use the PyTorch wheel appropriate to your hardware. The measured development environment used Python 3.13, PyTorch 2.6 and an RTX 2080 for planner adaptation; numerical experiments ran on CPU.

```sh
python -m venv .venv
# Activate .venv using your operating system's normal command.
python -m pip install -e ".[planner,dev]"
python scripts/download_model.py
python -m interval.cli serve --port 8765
```

Open http://127.0.0.1:8765. The released adapter is in `models/planner`; the downloader obtains the pinned upstream base weights and needs no authentication. Base weights execute locally after installation. To reproduce adaptation rather than use the supplied adapter:

```sh
python -m interval.cli train-planner
```

The planner supports estimator comparison, tail/dependence sensitivity, finite-measurement sensitivity and neural-estimator training. Manual procedure selection is available and is explicitly identified in the execution record.

## Reproduce numerical evidence

```sh
python -m pytest tests -q
python scripts/benchmark.py --seeds 10 --steps 80 --output evidence/final-dense
python scripts/benchmark.py --seeds 10 --steps 80 --sparse --output evidence/final-sparse
python scripts/invariance_control.py
python scripts/build_report.py
```

The dense control and sparse extension use separate fitting runs across 90 configurations each. Every fit is evaluated with absolute and log residuals. The monotone-transform control uses 30 runs. Ten independent scene groups per cell support exploratory bootstrap intervals. No test annotation is used to select detector settings.

## Bring independent data

The local interface accepts an NPZ containing numeric 2D arrays:

```text
train_before, train_after
calibration_before, calibration_after, calibration_mask
test_before, test_after, test_mask
```

Use strictly positive, finite intensities and binary masks. Limits: 12 MB compressed, 24 MB expanded, 8–512 pixels per side, 150,000 total pixels. Imported pairs must have different identities across splits. File identity checks cannot establish that different patients, regions or dates are independent; that remains part of the study design. The public demonstration does not accept user datasets.

Completed procedures export JSON evidence, distributions, Safetensors network weights, fitted marginals and normalization/lookup arrays. `interval.estimators.restore_fitted` restores local inference without pickle. Real-data bundles contain fitted intensity distributions; handle them under the dataset's permissions.

## Adapt the instruction router

Create separate JSONL training and evaluation files with rows like:

```json
{"request":"Compare the background estimators on this study","recipe":"compare"}
```

Allowed recipes are `compare`, `tails`, `measurements`, `neural`. Each split must include all four. Exact instruction overlap is rejected.

```sh
python -m interval.cli train-planner --train-data local-train.jsonl --test-data local-test.jsonl --steps 120
```

This adapts language-to-procedure selection; it does not add new tools or qualify a new domain. Numerical estimator training uses the independent observation arrays, not the instruction examples. See the [deployment guide](docs/deployment.md).

## Existing System Labs work

Interval's independently implemented contracts, procedure validation and execution records connect to the published [multiagentic architecture](https://github.com/michelegiletto-beep/multiagentic-platform/blob/main/docs/MULTIAGENTIC_SYSTEMS.md), [constrained orchestration research](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-agent-orchestration.html) and [Safety Mirrors](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-safety-mirrors.html). [VESPER](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-vesper.html) concerns inference under memory constraints; it is not an integrated runtime in this release. Existing proprietary product sources and model assets have not been copied or relicensed.

## Rights and attribution

New Interval code and synthetic data: Apache License 2.0, copyright 2026 Michele Giletto. The upstream language model retains its Apache 2.0 license and attribution. See [NOTICE](NOTICE), [LICENSE](LICENSE), and [CITATION.cff](CITATION.cff). Third-party datasets retain their own licenses. No affiliation with IonQ or Capella is asserted.
