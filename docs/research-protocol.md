# Interval research protocol

Version 0.1.0. This document specifies a simulation study, not a satellite, clinical or QPU performance claim.

## Question and relationship to prior work

Sekwao et al., *SAR and InSAR Change Detection with Quantum Generative Models*, arXiv:2609.05313v1, replace empirical conditional background estimates with a quantum generative copula. Interval studies controls needed to interpret that comparison. It independently implements a smaller circuit and a different optimizer; it is not an exact replication of their scores or hardware.

The written methods appear to describe averaging in discretized copula space before inverse transformation for the classical copula, and inverse-transforming samples before averaging for QCBM. These are generally different estimands:

    F_Y^(-1)( E[V | U] ) != E[ F_Y^(-1)(V) | U ].

This is a methodological inference from the manuscript, not a verified defect in the authors' implementation. Interval explicitly evaluates both choices using the same empirical joint distribution. All other copula models use the same intensity-space conditional-mean functional.

## Observation model

Before intensities are exp(tail * Z), where Z is a standardized smooth Gaussian random field. Unchanged after intensity is exp(0.08 + 0.9 log(X) + shift + epsilon), epsilon independent N(0, 0.22²). An independent smooth field defines approximately 8% contiguous changed pixels; changed values are multiplied by 2.8 or 0.28. The unchanged conditional mean is analytically exp(0.08 + 0.9 log(X) + shift + 0.22²/2).

This generator favors multiplicative-change detectors such as log ratio. It is a controlled null/reference case, not a representative SAR simulator. Complex phase, radar geometry, speckle formation, sensor calibration and medical biology are absent. A background NRMSE measures distance to the unchanged analytic conditional mean, while training observations contain change contamination.

## Independent observations

Each seed group contains three training scenes, two calibration scenes and two test scenes, each 32 by 32 pixels. Training labels are never passed to fitting. Marginal CDFs, joint histograms and network weights use training observations only. The same training observations, marginal transforms and conditional intensity representatives are used by the copula estimators. Separate scenes have separate deterministic seeds and recorded hashes. Independence is at the scene-group level; pixels within a scene are correlated.

The dense control uses 3,072 training pixels and varies tail strength (0.45, 1.1, 1.8) and bits per variable (3, 4, 5), with ten seed groups (31–40). The sparsity extension fixes tail strength 1.1 and varies training observations (64, 256, 1,024), bits (3, 4, 5), and the same ten seed groups. Subsampling is without replacement, fixed by seed and independent of labels. Each cell's 10 groups are the units for bootstrap summaries. Cells sharing a seed are paired, not independent replications of the entire study.

Rank discretization tends to balance conditioning rows. Low joint occupancy is distinct from empty conditioning rows; the evidence records both occupancy and observations per conditioning row. This study tests few-observation and sparse-joint regimes, not arbitrary high-dimensional rare-event conditioning.

## Estimators and scores

Methods: log-ratio detector; empirical copula with rank-space mean; empirical copula with intensity-space mean; Gaussian-smoothed empirical copula; Gaussian copula; a 1–32–32–1 Tanh neural conditional-mean network; QCBM with exact probabilities, finite measurements, and finite measurements after terminal bit-flip readout noise.

The common copula functional is the row-wise arithmetic mean of original-intensity bin representatives. Bin centers are inverse empirical marginal quantiles. This is a discretized approximation to the original conditional expectation. Finite measurement rows with zero mass use the global training intensity mean; this fallback is declared and counted.

Every fitted background is evaluated with both absolute-intensity residual and absolute log-intensity residual. Log ratio is a separate detector baseline and is identical in both score tables. Scorer comparisons reuse fitted weights. Differences between detector/scorer combinations must not be called estimator-only effects.

## Quantum simulation

The state begins as a maximally entangled pair of equal-size registers. Two layers of local Rx, Rz and nearest-neighbor RZZ gates in each register preserve uniform register marginals. Total circuit sizes are 6, 8 and 10 qubits in the reference sweeps. PyTorch complex pure-state autodiff with Adam minimizes cross entropy for 80 steps. A 1e-9 probability floor stabilizes the objective. Training is deterministic-probability simulation; finite measurements affect inference only. This differs from the paper's shot-based optimizer configurations.

Inference uses 5,000 multinomial measurements. The noise control is an independent terminal measurement-bit-flip channel with probability 0.01. It is not a trapped-ion gate-noise model and does not validate hardware robustness. Exact conditional integration is available in simulation; its computational availability is not assumed for a QPU.

## Calibration and evaluation

Each detector selects a threshold from 45 calibration-score quantiles between 0.50 and 0.995, plus all-change and no-change endpoints, and a spatial window from {1,3}; a 3-pixel window uses local changed fraction >=0.5. Ties in calibration F1 prefer fewer false positives. All-negative calibration can therefore select no detections. The same finite calibration grid is used for all methods. Selected settings are frozen before test evaluation. Both test scenes contribute equally to each group's F1, IoU, precision, recall, false-positive rate and average precision.

Recall at 1% false-positive rate is read from the test ROC curve and is a diagnostic summary, not an operationally calibrated threshold. Group means and 2,000 bootstrap-resample 95% intervals are exploratory. There is no familywise multiple-comparison correction and no confirmatory significance claim. Timing is whole-experiment wall time in the recorded environment, not a quantum/classical speed comparison.

## External validation gate

Before a remote-sensing claim: select licensed paired acquisitions, independently establish change annotations, preserve acquisition geometry metadata, and test on held-out regions/dates. Before a medical claim: select an intended research task, obtain appropriate data permissions, and evaluate across independent patients/sites with domain specialists. Same-day RIDER Lung CT data can support repeatability research; it cannot establish treatment-response accuracy.

Novelty review must extend beyond one paper. The initial contribution is an inspectable benchmark and measured controls; no priority or quantum advantage is claimed.

## Sources

- https://arxiv.org/html/2609.05313v1
- https://www.nature.com/articles/s42005-024-01552-6
- https://arxiv.org/abs/2107.07352
- https://www.capellaspace.com/legal/data-licensing
- https://datacommons.cancer.gov/repository/imaging-data-commons
- https://wiki.cancerimagingarchive.net/display/Public/RIDER+Lung+CT
