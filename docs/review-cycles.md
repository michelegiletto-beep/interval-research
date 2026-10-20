# Pre-publication review record

Version 0.1.0 · 20 October 2026. Three review-and-improvement cycles precede publication of the source and executable demonstration. This is an engineering/research review record, not external peer review or regulatory certification.

## Cycle 1 — numerical implementation

Reviewed the independent quantum-copula simulator and estimator functional. Initial checks covered register marginal normalization, differentiable optimization, finite measurement counts, readout channel normalization, scene identities, rank ties and noncommutation of inverse transforms with expectation. Six numerical tests passed.

Improvement: retain the quantum parameter snapshot corresponding to the recorded objective before the optimizer update. Use the same original-intensity conditional-mean functional for every fitted copula distribution, with a separate rank-mean ablation. Define zero-row fallbacks and probability floors explicitly.

## Cycle 2 — scientific controls and executable workflow

A separate read-only audit found detector-score confounding, insufficiently sparse conditioning observations in the initial dense benchmark, missing inverse transforms in model exports, weak negative-control calibration and overly broad wording for the language-model evaluation.

Improvements: evaluate both absolute and log residuals on the same fitted backgrounds; add the 64/256/1,024-observation extension; add 30 fixed-observation monotone-transform controls; export marginals, normalization and lookup arrays; include all/none thresholds with a fewer-false-positives tie-break. Document Gaussian midpoint approximation, tail/dependence confounding and the limited template-held-out router test. Distinct procedures now implement different numerical work; the measurement sweep reuses fitted weights.

Expanded integration checks exercise actual fitting, calibration, archiving, downloadable evidence and inference restoration. They also reject unknown tools, excessive configuration, cross-origin writes, duplicate data identities and oversized expanded archives.

## Cycle 3 — release evidence and delivery

Final read-only audit recomputed manuscript tables from CSV, counted 90 dense and 90 sparse fitted configurations with two scorers, verified all 30 rank/circuit invariance controls, checked numerical fingerprints and verified the adapter hash against its training receipt. Restored lookup predictions matched exactly; restored neural scores differed by less than 2e-6.

Improvements: validate NPY headers and byte counts before loading; enforce request-body budgets, admitted Host names, public compute admission/retention and guaranteed queue-slot release; make adapter metadata portable; include an explicit Docker reference and a source package manifest that includes the adapted weights. The complete test suite passed 12 tests after these changes, including real API execution and portable inference restoration. Desktop interface execution also completed a model-selected procedure; narrow-screen and final package checks are recorded in release evidence.

## Hosted integration verification

The final publication checks additionally exercised the actual Hugging Face runtime. Its default Gradio version conflicted with the pinned Transformers dependency, so the transport pins Gradio 5.49.1. Adapter loading now explicitly starts on CPU. A separate loopback CPU service isolates numerical optimization from ZeroGPU's CUDA emulation; the numerical implementation fingerprint is unchanged.

The remote API completed a model-selected comparison of all nine methods, rejected a cross-origin write with HTTP 403, rejected an oversized body with HTTP 413 and exported a reusable inference bundle. A second comparison completed through the public browser interface. The optional Gradio GPU diagnostic returned a procedure proposal on CUDA. Receipts and a screenshot are supplied in `evidence/hosted-smoke` and `evidence/interface` in the complete release archive. These are delivery checks, separate from the 210 scientific fitting runs.

## Accepted scope

The release is a working single-user research reference with synthetic numerical evidence and a bounded, locally adapted procedure selector. Domain profiles configure the same software. Satellite, clinical, general autonomous planning, physical-QPU and quantum-advantage performance remain unestablished. Timing records whole experiments and does not support per-method speed comparisons.

See the [protocol](research-protocol.md), [model card](model-card.md), [deployment definition](deployment.md) and machine-readable evidence for the exact operating envelope. The release manifest and publication verification record identify the files and final checks used for distribution.
