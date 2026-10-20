# Related work and implementation boundaries

Version 0.1.0 · 20 October 2026. This is a focused research map, not an exhaustive novelty search. Published results from another project are not evidence of Interval's performance.

## Conditional background estimation and quantum copulas

[Sekwao et al., *SAR and InSAR Change Detection with Quantum Generative Models*, arXiv:2609.05313v1](https://arxiv.org/html/2609.05313v1) is the direct application reference. It studies a QCBM-sampled copula background estimator using Capella imagery and IonQ hardware. Interval independently implements a smaller simulator and different training method. It does not reproduce the original acquisitions, annotations, hardware or reported scores.

The motivation for Interval is to make the target conditional mean, discretization, score, calibration and sampling budget separately inspectable. A concern inferred from the written methods is that averaging ranks before inverse transformation and averaging inverse-transformed samples target different quantities. That inference is a hypothesis for an ablation, not a verified defect in the original code. The manuscript also motivates matched sampling controls and stronger independent evaluation. See the [Interval protocol](research-protocol.md) for the implemented comparison and its limits.

Historical classical references include [Carlotto, *Nonlinear background estimation and change detection for wide-area search* (2000)](https://doi.org/10.1117/1.602496) and [Mercier, Moser and Serpico, *Conditional Copulas for Change Detection in Heterogeneous Remote Sensing Images* (2008)](https://doi.org/10.1109/TGRS.2008.916476). Conditional background modeling and copulas for change detection therefore precede this project. Interval's histogram and smoothing controls are reference implementations, not a claim to reproduce all details of those algorithms.

[Zhu et al., *Generative Quantum Learning of Joint Probability Distribution Functions* (2022)](https://arxiv.org/abs/2109.06315) develops quantum copula modeling and trapped-ion demonstrations on financial observations. A maximally entangled starting state followed by local register operations is prior work. Using that construction does not establish a new quantum model or an application-independent advantage.

## Neural comparisons and generalization

[Hibat-Allah et al., *A framework for demonstrating practical quantum advantage: comparing quantum against classical generative models* (2024)](https://www.nature.com/articles/s42005-024-01552-6) compares QCBMs with classical generative models under a defined evaluation framework. It provides precedent for competitive model selection and explicit definitions of advantage. An advantage on one task and resource regime cannot be transferred to image change detection by citation.

[Gili, Mauri and Perdomo-Ortiz, *Generalization Metrics for Practical Quantum Advantage in Generative Models*](https://arxiv.org/abs/2201.08770) evaluates generation beyond training samples in constrained discrete spaces. Its metrics are relevant to thinking about sample quality and generalization, but they are not interchangeable with held-out image-mask performance.

[Laszkiewicz, Lederer and Fischer, *Copula-Based Normalizing Flows* (2021)](https://arxiv.org/abs/2107.07352) is a neural density-modeling precedent involving copulas and heavy-tailed observations. A neural copula or conditional density estimator is therefore an appropriate future comparison family, not an inherently new combination.

The released Interval neural estimator is a small MSE-trained conditional-mean network. It is not a normalizing flow or a comprehensive neural generative benchmark. Fixed architectures and training budgets in a synthetic reference study cannot support superiority over the best available classical or neural methods. Further work should tune viable models under a declared development budget and preserve an untouched test partition.

## Embedded language model

The upstream [SmolLM2-135M-Instruct model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) and [SmolLM2 paper](https://arxiv.org/abs/2502.02737) supply the language model and its pretraining provenance. Interval adapts existing weights using [LoRA](https://huggingface.co/docs/peft/en/developer_guides/lora); it does not train a foundation model from scratch.

The implementation ranks four admitted candidate answers and runs the selected native procedure. Valid output membership follows from the candidate set. The supplied evaluation measures template-held-out intent selection, with action phrases shared between training and evaluation. It does not establish general autonomous planning, scientific discovery or domain expertise. The [model card](model-card.md) separates language routing from numerical estimator training.

## Existing System Labs publications

The following publications connect this project to the existing research programme. They are cited as related work, with implementation status kept separate.

| Publication | Relevant design concern | Boundary in this release |
| --- | --- | --- |
| [Multiagentic systems at System Labs](https://github.com/michelegiletto-beep/multiagentic-platform/blob/main/docs/MULTIAGENTIC_SYSTEMS.md) | Bounded component responsibilities, capability contracts, state and execution receipts | Interval implements a small independent procedure runner; the existing application framework and domain adapters are not integrated |
| [Quantum-Assisted Dynamic Orchestration of Heterogeneous AI Agents, v1.1](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-agent-orchestration.html) | Explicit scheduling constraints, classical references and evidence-based backend selection | Interval does not embed its scheduling solver, QUBO formulation or QAOA experiment; numerical outcomes remain specific to the earlier synthetic study |
| [Safety Mirrors for Dynamic Multi-Agent Systems, Paper III v1.0](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-safety-mirrors.html) | Separation of observation, semantic review, authority and enforcement; state-bound decisions | Interval's validation and execution records are independently implemented; it does not deliver the proposed mirror architecture or inherit its structural simulation results |
| [VESPER: Resource-Constrained Local Inference, RFC 0001 v0.1.0](https://giletto-systems-lab.michele-giletto.chatgpt.site/work-vesper.html) | Memory admission, data movement and explicit numerical/resource contracts | VESPER is a research design, not an integrated inference runtime; this release uses ordinary local model execution |

The Safety Mirrors page preserves the earlier prospective v0.1 protocol alongside its later formal and structural study. Those are distinct evidence records. Neither establishes production safety for Interval. VESPER's RFC does not establish measured full-model inference speed or a larger physical memory capacity.

These links do not transfer rights to existing proprietary implementations. Their private product sources, training assets and model artifacts have not been copied into or relicensed as Interval. New Interval implementation and synthetic materials have their own release terms; third-party weights retain their upstream attribution and license. A published architectural relationship is not an installed software dependency.

## What would justify a further contribution

The current contribution is an executable reference with explicit controls and measured synthetic outcomes. The combination of a copula estimator, neural network, local language model and interface is insufficient to establish scientific novelty.

A stronger result would identify a reproducible effect that survives fair conditional-mean definitions, matched scorers, discretization controls, viable tuned baselines and independent datasets. A negative result that explains when a simpler detector is sufficient is also publishable evidence, subject to a venue's assessment. Scientific priority requires a broader literature search; commercial usefulness requires a separate customer and workflow evaluation.

No hardware speedup, clinical benefit, satellite operational performance or inherited safety result is asserted. Application-specific evidence gates are set out in the [development plan](business-development.md).
