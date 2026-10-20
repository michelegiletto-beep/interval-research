"""Build tables and figures directly from measured, fingerprinted evidence."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
evidence = ROOT / "evidence"
figure_dir = evidence / "figures"
figure_dir.mkdir(exist_ok=True)
sets = {}
for label in ("final-dense", "final-sparse"):
    folder = evidence / label
    summary = json.loads((folder / "summary.json").read_text())
    rows = list(csv.DictReader((folder / "metrics.csv").open()))
    sets[label] = (summary, rows)
invariance = json.loads((evidence / "invariance" / "summary.json").read_text())
assert len({s[0]["implementation_sha256"] for s in sets.values()} | {invariance["implementation_sha256"]}) == 1
methods = ["log_ratio","copula_rank_mean","copula_intensity_mean","smoothed_copula","gaussian_copula","neural_regression","qcbm_exact","qcbm_shots","qcbm_readout"]
names = ["Log ratio","Copula / rank mean","Copula / intensity mean","Smoothed copula","Gaussian / midpoint","Neural mean","QCBM / exact","QCBM / finite shots","QCBM / readout"]


def average(rows, method, scorer):
    values = [float(r["f1"]) for r in rows if r["method"] == method and r["scorer"] == scorer]
    return float(np.mean(values))


plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
fig, axes = plt.subplots(1,2,figsize=(11.5,5.4),sharey=True)
y = np.arange(len(methods))
for ax,(label,(summary,rows)) in zip(axes,sets.items()):
    ax.barh(y-.16,[average(rows,m,"absolute") for m in methods],height=.29,color="#9bab9b",label="Absolute residual")
    ax.barh(y+.16,[average(rows,m,"log") for m in methods],height=.29,color="#2b615d",label="Log residual")
    ax.set(yticks=y,yticklabels=names,xlim=(0,1),xlabel="Mean test F1 across protocol cells",title="Dense control" if label=="final-dense" else "Sparse-observation extension")
    ax.grid(axis="x",alpha=.18)
axes[0].invert_yaxis()
axes[1].legend(frameon=False,loc="lower right")
fig.suptitle("Estimator and residual-scoring controls · synthetic observations",fontsize=11,y=.98)
fig.tight_layout()
for suffix in ("png","svg"):
    fig.savefig(figure_dir/f"scorer-comparison.{suffix}",dpi=180,bbox_inches="tight")
plt.close(fig)

sparse_summary,sparse_rows=sets["final-sparse"]
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for method,color in [("copula_intensity_mean","#789075"),("neural_regression","#a17a4c"),("qcbm_exact","#2b615d")]:
    for bits,style in [(3,"--"),(5,"-")]:
        selected=[r for r in sparse_summary["summaries"] if r["method"]==method and r["bits"]==bits and r["scorer"]=="log"]
        selected.sort(key=lambda r:r["training_pixels"])
        axes[0].plot([r["training_pixels"] for r in selected],[r["f1"]["mean"] for r in selected],style,marker="o",ms=3,color=color,label=f"{dict(zip(methods,names))[method]} / {bits} bits")
        axes[0].fill_between([r["training_pixels"] for r in selected],[r["f1"]["low"] for r in selected],[r["f1"]["high"] for r in selected],color=color,alpha=.07)
axes[0].set(xscale="log",xlabel="Training observations",ylabel="Test F1 / log residual",ylim=(0,1),title="Sparse-observation control")
axes[0].legend(frameon=False,fontsize=6)
receipt=json.loads((ROOT/"models"/"training-receipt.json").read_text())
axes[1].plot(receipt["training_loss"],color="#2b615d",lw=1.2)
axes[1].set(xlabel="Local LoRA step",ylabel="Training cross entropy",title="Embedded router adaptation")
fig.tight_layout()
for suffix in ("png","svg"):
    fig.savefig(figure_dir/f"sparsity-and-adaptation.{suffix}",dpi=180,bbox_inches="tight")
plt.close(fig)

table="| Detector / fitted background | Dense / absolute | Dense / log | Sparse / absolute | Sparse / log |\n| --- | ---: | ---: | ---: | ---: |\n"
for method,name in zip(methods,names):
    values=[average(sets[label][1],method,score) for label in sets for score in ("absolute","log")]
    table+=f"| {name} | "+" | ".join(f"{v:.4f}" for v in values)+" |\n"

manuscript=f'''# Conditional Estimands, Residual Scoring, and Sampling in Change Detection

Michele Giletto · System Labs · 20 October 2026

**Status: manuscript draft accompanying software version 0.1.0. Not peer reviewed.**

## Abstract

Generative conditional background estimation combines a learned dependence model, marginal transformations, a conditional functional and a detector score. These choices can affect performance independently. We introduce a reproducible simulation benchmark that holds several of them fixed while comparing empirical, smoothed, Gaussian, neural and quantum-circuit estimators. A dense control and a sparse-observation extension each contain 90 fitted configurations; each fit is evaluated with two residual scorers. A separate 30-run monotone-transform control preserves rank-space observations and the fitted quantum distribution in every tested case. Thresholds and spatial windows are selected using calibration scenes and frozen before testing. In the included multiplicative lognormal generator, a log-ratio detector is a strong reference, and the small quantum circuit does not establish an advantage over classical alternatives. The accompanying application executes bounded research procedures using a locally adapted open-weight language model and exports complete inference bundles. These results concern synthetic observations and do not establish satellite, medical or physical-QPU performance.

## 1. Motivation and prior work

Sekwao et al. [1] report QCBM-based background estimation on Capella SAR/coherence data and IonQ hardware. Their result motivates investigating which parts of a generative-estimation pipeline produce useful changes in detection. Interval implements a smaller independent simulator, rather than reproducing the original hardware experiment or its headline scores. Quantum-versus-classical generative comparisons have prior methodological literature [2], and neural copula models also predate this work [3].

The written classical and quantum methods in [1] appear to describe different averaging orders around the inverse marginal transform. This is an inference from the manuscript, not a verified implementation error. The present study makes the distinction explicit and tests it under common observations and joint distributions.

## 2. Estimands

Let U=F_X(X) and V=F_Y(Y). The rank-space estimator F_Y^(-1)(E[V|U]) is generally different from E[F_Y^(-1)(V)|U]. The latter is the original-intensity conditional mean. A discrete copula model is evaluated using the same inverse-quantile bin representatives and row-wise intensity expectation for every fitted joint distribution. The rank-mean alternative is an explicit ablation of the empirical copula only.

The experiment also separates absolute-intensity and absolute-log residual scoring. Log ratio uses the unmodified before image as its reference background and remains a separate detector baseline. Changing a scorer can change a detector's ranking without changing its fitted dependence model.

## 3. Protocol

The complete generator, fitting, calibration and interval definitions are in [research-protocol.md](research-protocol.md). Each seed group has independent training, calibration and test scenes. No training change label is provided to fitting. Ten seed groups are used per protocol cell. The dense control varies three tail/dependence regimes and three discretizations with 3,072 training observations. The sparse extension fixes one regime and varies 64, 256 and 1,024 training observations at the same three discretizations. These regimes have different numbers of observations per conditioning bin; sparse joint occupancy and empty row support are reported separately.

Tail strength also changes dependence because additive log noise is held fixed; the tail sweep is not an isolated causal test of marginal tails. The supplementary fixed-observation control applies strictly monotone powers 0.5, 1 and 2. Its transformed analytic background expectation includes the transformed noise variance rather than raising the old expectation to a power.

The QCBM uses 6, 8 or 10 qubits, two local-unitary layers per register and 80 Adam steps with exact probability gradients. Inference controls use finite multinomial measurements and an explicitly limited terminal bit-flip model. This is distinct from the optimizer, shot budgets, circuit size and physical processors in [1].

## 4. Measured results

The table contains descriptive averages of test F1 over the 90 configurations in each suite. Cells reuse fixed seed groups, so the 90 values are not treated as 90 independent population samples. Per-cell seed-group bootstrap intervals and paired differences are retained in the machine-readable evidence. Intervals are exploratory and uncorrected for multiple comparisons.

{table}

![Detector scoring comparison](../evidence/figures/scorer-comparison.png)

The log-ratio reference is particularly effective because changes in this generator are multiplicative and the unchanged relation is nearly linear in log intensity. This is a property of the selected simulation, not a universal recommendation. For the dense control, changing the empirical copula's averaging order changes mean absolute-residual F1 from {average(sets['final-dense'][1],'copula_rank_mean','absolute'):.4f} to {average(sets['final-dense'][1],'copula_intensity_mean','absolute'):.4f}; with log residuals the corresponding values are {average(sets['final-dense'][1],'copula_rank_mean','log'):.4f} and {average(sets['final-dense'][1],'copula_intensity_mean','log'):.4f}. Thus the intensity-mean choice does not uniformly improve detection, despite its different estimation interpretation.

All {len(invariance['records'])} monotone-transform runs retain identical empirical rank joint distributions and fitted quantum probabilities within the declared numerical tolerance. Detection and intensity-estimation outputs can still change because inverse marginal representatives and scoring change. The control therefore demonstrates why an unchanged copula alone cannot establish detector invariance.

The small-circuit QCBM does not dominate the included baselines. This negative result does not refute [1]: datasets, circuit expressivity, optimizers, preprocessing and evaluation all differ. It does establish that this software does not substantiate a quantum-advantage claim.

![Sparse observations and router adaptation](../evidence/figures/sparsity-and-adaptation.png)

## 5. Model-assisted execution

SmolLM2-135M-Instruct [4], at a pinned revision, selects one of four predefined procedures using length-normalized candidate likelihood. A rank-8 LoRA adapter was trained on 96 synthetic requests for 120 steps. On 32 template-held-out requests containing action phrases already seen during training, selection accuracy changed from {receipt['base_evaluation']['accuracy']*100:.3f}% to {receipt['adapted_evaluation']['accuracy']*100:.1f}%. This is bounded intent selection; valid schema is guaranteed by the candidate set. No general planning, discovery or unrestricted tool-use claim follows.

Native execution validates the recipe, fits models, calibrates detectors and archives observations of completion. Numerical estimator performance is evaluated separately from language-to-procedure routing. A complete inference export contains network weights, marginal arrays, normalization and lookup tables. This release supports a single local user; institutional authentication and tenant qualification are separate development requirements.

## 6. Limitations and next studies

The generator omits radar physics and biology. Its change process favors multiplicative detectors. Training and test share the same generative family. The network is small and the Gaussian baseline uses conditioning-bin midpoints. No broad hyperparameter search is performed, and compute budgets differ in meaning between models. Quantum training uses exact gradients; only inference measurements are varied. The readout channel is not a hardware noise emulator. Whole-experiment timing cannot support per-method speed claims.

Next studies should add independent real acquisitions or patient-level image data, low-change negative controls, alternative dependence families, matched model-search budgets, transfer across acquisition/site conditions and physically justified noise. A clinical or geospatial conclusion requires its own data provenance and specialist validation. No priority claim is made for the estimator identity, copula modeling or orchestration design. The contribution at this stage is the inspectable implementation, controlled ablations and negative/positive simulation evidence.

## 7. Availability

[Source](https://github.com/michelegiletto-beep/interval-research) · [Executable demonstration](https://huggingface.co/spaces/Magic51/interval-research) · [System Labs](https://giletto-systems-lab.michele-giletto.chatgpt.site).

New code and synthetic data are Apache 2.0. Upstream weights retain their own attribution. No original Capella images or clinical observations are redistributed. The numerical implementation fingerprint is `{sets['final-dense'][0]['implementation_sha256']}`.

## References

1. Sekwao et al. (2026). SAR and InSAR Change Detection with Quantum Generative Models. https://arxiv.org/abs/2609.05313
2. Hibat-Allah et al. (2024). A framework for demonstrating practical quantum advantage: comparing quantum against classical generative models. https://www.nature.com/articles/s42005-024-01552-6
3. Copula-based normalizing-flow research. https://arxiv.org/abs/2107.07352
4. Allal et al. (2025). SmolLM2: When Smol Goes Big. https://arxiv.org/abs/2502.02737
'''
(ROOT/"docs"/"manuscript.md").write_text(manuscript,encoding="utf-8")
(ROOT/"publication").mkdir(exist_ok=True)
article=f'''# A reproducible benchmark for conditional background estimation

**LinkedIn article draft. Not published.**

IonQ and Capella's SAR change-detection study prompted a narrower question: how much of a detection result comes from the dependence model, the averaging operation and the residual score?

I built Interval to separate those choices. The current release runs classical estimators, a neural conditional-mean network and a small quantum copula simulator. It uses separate training, calibration and test scenes, and exports the fitted transformations together with model weights.

The numerical release contains 180 fitted benchmark configurations evaluated with two scorers, plus 30 monotone-transform controls. All observations in this release are synthetic. In the selected multiplicative lognormal generator, the log-ratio baseline remains strong; the small quantum circuit does not demonstrate an advantage.

One useful control is already clear: a strictly monotone transformation preserves the empirical rank-space joint distribution and, in these runs, the fitted circuit distribution. The detector can still change because inverse marginal values and scoring change. Averaging ranks before inversion also differs from averaging original intensities. Neither choice uniformly wins in the measured detection results.

Interval also includes a local open-weight language model adapted to select among four executable research procedures. On a small template-held-out synthetic evaluation, its routing accuracy changed from {receipt['base_evaluation']['accuracy']*100:.1f}% to {receipt['adapted_evaluation']['accuracy']*100:.1f}%. That evaluates the router, not scientific reasoning or domain performance.

The application has tissue-research, infrastructure-monitoring and editable client profiles. They share software, while their domain validation remains separate. The current medical profile has no clinical validation; the infrastructure profile has no field-performance claim.

The code, training examples, adapter and measured evidence are open. Work on independent real datasets is the next research requirement. Commercial work would concern deployment, data integration, local adaptation and independently specified validation services.

Source: https://github.com/michelegiletto-beep/interval-research
Demonstration: https://huggingface.co/spaces/Magic51/interval-research
Prior study: https://arxiv.org/abs/2609.05313
System Labs: https://giletto-systems-lab.michele-giletto.chatgpt.site
'''
(ROOT/"publication"/"linkedin-article.md").write_text(article,encoding="utf-8")
(ROOT/"publication"/"linkedin-post.md").write_text('''# LinkedIn post draft — not published

I built Interval, an open-source local workspace for conditional background-estimation experiments.

The current evidence includes 180 synthetic benchmark configurations, two residual scorers, 30 monotone-transform controls, a neural estimator and a small quantum copula simulator. Calibration observations select thresholds; test scenes remain separate.

The included quantum model does not establish an advantage over the classical baselines. The release makes the estimator, score and sampling controls inspectable.

A locally adapted open-weight language model selects among four executable procedures. It has a separate, limited routing evaluation. Tissue-research and infrastructure profiles remain unvalidated domain starting points.

Source: https://github.com/michelegiletto-beep/interval-research
Demo: https://huggingface.co/spaces/Magic51/interval-research
''',encoding="utf-8")
(evidence/"report-summary.json").write_text(json.dumps({"fits":180,"scorers":2,"monotone_controls":len(invariance["records"]),
    "base_router_accuracy":receipt["base_evaluation"]["accuracy"],"adapted_router_accuracy":receipt["adapted_evaluation"]["accuracy"],
    "macro_f1":{label:{score:{method:average(rows,method,score) for method in methods} for score in ("absolute","log")} for label,(_,rows) in sets.items()},
    "implementation_sha256":sets["final-dense"][0]["implementation_sha256"]},indent=2))
print(json.dumps({"figures":4,"manuscript":str(ROOT/'docs'/'manuscript.md'),"reference_fits":180,"monotone_controls":len(invariance['records'])}))
