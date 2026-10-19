# Conditional mean response protocol

This auxiliary protocol is fixed before evaluating the new mean-response runs.
It does not change the primary experiment in `CONFIRMATORY_PROTOCOL.md`.
The configuration is `configs/inference/mean_v1.json`.

Use the saved primary-profile checkpoints at step 1000, retaining the parent's
200 calibration and 200 evaluation identities. The same checkpoint, input,
known decoder, and target probes are used for all compared estimators. There is
no added observation noise. Set eta=1e-5, perturbation sigma=1e-4, and one constant
update multiplier of one. A random normalization estimated from a finite bank
would change the estimator's mean and is therefore omitted in this analysis.

Five coupled estimators use independent isotropic directions across target
probes: an exactly unbiased directional estimate of the true gradient, the same
estimate with fixed positive diagonal parameter preconditioning, the physical
central-difference node estimator from the main experiment, and two biased
estimators with the same directional noise and a fixed mean equal to the 10%
or fully random feedback direction. True gradients generate simulation nulls;
inference uses only the known-readout gradient, responses, and sample moments.
The biased noisy estimators are controls for mean bias, not a claim that noisy
feedback alignment is a biological mechanism.

At R=8,32,128,512,2048,8192, form both an average of R responses from independently
restored updates and a response to one averaged weight direction. These differ
at finite step sizes and across ReLU gates. Store and report their discrepancy.
The two orders share the same random draws. Nested R values are paired within
each independent network, not independent replicates.

For a response mean V, the test statistic is the maximum absolute reciprocity
contrast in G^T V divided by its estimated Monte Carlo standard error. Sample
moments of individual restored responses supply this normalizer. For the
averaged-direction response it is only a normalizer, not an exact finite-step
standard error. Independent empirical calibration is performed separately for
each averaging order and R, so neither a normal approximation nor a Student
distribution is assumed. The cutoff is the larger of the separate split
conformal cutoffs for the unbiased directional and preconditioned nulls, using
the same order statistic and strict rejection convention as the main study.
These two simulated null distributions do not cover all stochastic PSD rules.

A deliberately naive comparator applies the raw operator score to each finite
mean, but calibrates it only on deterministic GD and preconditioned GD responses
at eta=1e-5. Its rejection rate measures the consequence of neglecting estimator
variability; it is not the calibrated main experiment's decision rule.

Report rejection counts, Wilson 95% intervals, and undefined values across 200
held-out runs, conditional on the calibration sample. Also report median
relative error to the corresponding deterministic mean-direction step,
Monte Carlo variability scale, and the averaging-order gap, with independent
network bootstrap intervals for medians. For physical node perturbation, the
reference is the GD step: finite sigma and gate crossings can produce residual
bias. Nonrejection of reciprocity is a necessary-condition result and does not
establish a complete PSD fit, unbiasedness, or a unique learning rule.
