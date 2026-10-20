# A reproducible benchmark for conditional background estimation

**LinkedIn article draft. Not published.**

IonQ and Capella's SAR change-detection study prompted a narrower question: how much of a detection result comes from the dependence model, the averaging operation and the residual score?

I built Interval to separate those choices. The current release runs classical estimators, a neural conditional-mean network and a small quantum copula simulator. It uses separate training, calibration and test scenes, and exports the fitted transformations together with model weights.

The numerical release contains 180 fitted benchmark configurations evaluated with two scorers, plus 30 monotone-transform controls. All observations in this release are synthetic. In the selected multiplicative lognormal generator, the log-ratio baseline remains strong; the small quantum circuit does not demonstrate an advantage.

One useful control is already clear: a strictly monotone transformation preserves the empirical rank-space joint distribution and, in these runs, the fitted circuit distribution. The detector can still change because inverse marginal values and scoring change. Averaging ranks before inversion also differs from averaging original intensities. Neither choice uniformly wins in the measured detection results.

Interval also includes a local open-weight language model adapted to select among four executable research procedures. On a small template-held-out synthetic evaluation, its routing accuracy changed from 28.1% to 100.0%. That evaluates the router, not scientific reasoning or domain performance.

The application has tissue-research, infrastructure-monitoring and editable client profiles. They share software, while their domain validation remains separate. The current medical profile has no clinical validation; the infrastructure profile has no field-performance claim.

The code, training examples, adapter and measured evidence are open. Work on independent real datasets is the next research requirement. Commercial work would concern deployment, data integration, local adaptation and independently specified validation services.

Source: https://github.com/michelegiletto-beep/interval-research
Demonstration: https://huggingface.co/spaces/Magic51/interval-research
Prior study: https://arxiv.org/abs/2609.05313
System Labs: https://giletto-systems-lab.michele-giletto.chatgpt.site
