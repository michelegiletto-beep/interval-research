# Community publication drafts

Drafts only. Nothing has been posted by this release task. Recheck the selected community's current rules and self-promotion policy before posting. Use one appropriate discussion, disclose authorship and respond to technical questions. Do not duplicate posts, use promotional comment bots or imply independent endorsement.

## Research-software discussion

Suggested title: Open-source benchmark separating conditional estimation from detector scoring

I built Interval, a local research workspace inspired by a recent SAR quantum-generative-model paper. I wanted to check whether averaging before versus after an inverse marginal transform, and changing the detector residual, could affect comparisons independently of the generative model.

The current release has 180 synthetic fitted configurations, each evaluated under two scorers, plus 30 monotone-transform controls. Calibration and test scenes are separate. The small quantum circuit does not beat the strong log-ratio baseline in this generator. The raw evidence and limitations are included.

The application also runs a locally adapted small language model that selects one of four bounded procedures. That router has its own limited evaluation and is not presented as an autonomous researcher.

I'm the author. I would value criticism of the experimental controls and suggestions for independently annotated, licensed repeat-acquisition datasets.

Source: https://github.com/michelegiletto-beep/interval-research
Executable synthetic demo: https://huggingface.co/spaces/Magic51/interval-research
Prior paper: https://arxiv.org/abs/2609.05313

## Maintainer notes

Potential venues must be selected by fit and current rules, not posting volume. A research community may prefer the method and negative result; an open-source community may prefer installation, tests and reproducibility. Publish the same work in additional venues only where permitted and with a distinct relevant discussion. Keep medical and field-performance claims within the evidence. Keep LinkedIn materials as drafts until separately authorized.
