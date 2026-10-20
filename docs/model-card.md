# Embedded procedure-selection model

## Identity

Base: HuggingFaceTB/SmolLM2-135M-Instruct, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache 2.0. Architecture and pretraining are upstream work. Interval supplies a rank-8 LoRA adaptation of the query/value attention projections, with 460,800 trainable parameters.

The model executes locally. It scores four complete candidate JSON answers with mean suffix-token conditional log likelihood and selects the highest score. Candidate answer lengths are not identical; scores are length-normalized. Valid schema and tool membership follow from the candidate set, not a learned guarantee.

## Training and evaluation

The recorded local run used 96 synthetic training instructions, batch size 4, 120 AdamW steps, learning rate 0.0004, LoRA alpha 16 and dropout 0.05, on an NVIDIA RTX 2080. Training took approximately 59 seconds in the recorded environment. The adapter identity and complete loss curve are in `models/training-receipt.json`.

The evaluation contains 32 instructions. Their sentence wrappers were not used in training, but all 16 action phrases were seen inside training wrappers. The unadapted model selected the correct recipe in 9/32 examples (28.125%); the adapted model selected correctly in 32/32 (100%). This is template-held-out synthetic intent selection, not broad unseen-language generalization, autonomous planning or scientific discovery.

The 32 cases are one small deterministic evaluation set. No population-level accuracy, robustness to arbitrary prompts or clinical/geospatial validity follows. New customer terminology requires independent evaluation and may require adaptation. Report abstention, ambiguous-request behavior and out-of-distribution cases before expanding scope.

## Execution authority

The model chooses a procedure. Native Python validates that choice and executes known tools. There is no generated Python, shell execution, remote publication tool, email tool, arbitrary URL access or unrestricted plugin execution. The four recipes have different execution behavior: standard comparison; three tail/dependence regimes; three inference measurement counts with reused fitted weights; or an extended neural fit.

The runtime has one worker, a bounded queue, durable job records and explicit failure states. It is a single-user research service. Authentication, tenant isolation and institutional production qualification are not implemented.

## Local adaptation

Training scripts and synthetic train/test JSONL are public. Users can provide independent instruction files covering the four allowed labels. This changes phrase-to-procedure routing; adding new procedures requires code, tool contracts and tests. Numerical models are trained separately on observation arrays.

## Separate neural estimator

The conditional-mean network has 1,153 trainable parameters: Linear(1,32), Tanh, Linear(32,32), Tanh, Linear(32,1). It predicts standardized intensity from the training empirical CDF of before intensity, using MSE. Its weights, normalization, marginals and lookup arrays are exported with each completed experiment. It is not the language model.

## Sources

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- https://arxiv.org/abs/2502.02737
- https://huggingface.co/docs/peft/en/developer_guides/lora
