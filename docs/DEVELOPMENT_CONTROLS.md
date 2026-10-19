# Completing the bounded mechanism and scale checks

An audit after the main CPU study identified two controls from the research
plan that were not yet represented in the manuscript: task versus synthetic
layer-loss gradients at a matched state, and raw versus magnitude-matched update
banks. This follow-up uses only the 20 primary development networks at step
1000. It cannot select a new primary statistic or recalibrate held-out results.
The configuration is `configs/inference/mechanism_controls_v1.json`.

For the first control, reconstruct the original teacher target at the saved
input. Compare its gradient through the fixed decoder with one isotropic
gradient of the same norm on the complete last hidden layer. A quadratic loss
directly on that layer realizes the synthetic gradient; this is an oracle
mechanism control, outside the main experiment's accessible gradient span.
Hold checkpoint, input, and each gradient fixed while sweeping eta from 1e-5
to 1e-2. Compare all hidden parameters, only the observed layer's incoming
parameters, or only its upstream parameters. Record layerwise prediction error,
activity-gradient alignment, gate changes in both directions, and actual loss
change. Preserve individual signed self/interference terms and coordinate loss
contributions in the raw arrays, rather than just their population norms.

For the second control, compare every main rule at eta=1e-4 and sigma=1e-4,
using the exact same checkpoint and perturbation draws for raw multiplier one
and the separate-bank RMS parameter-norm multiplier. Verify that replaying the
same scaled directions reproduces the matched response. The homogeneous
operator score must be invariant under a common positive response scale to
first order; quantify finite-step discrepancies separately. Report development
score distributions and step residuals, not newly tuned rejection rates.

Each network contributes one observation to a selected condition; layers,
neurons, step sizes, and paired interventions are repeated measures. Summaries
retain counts and undefined values. These controls extend the development
diagnostics and leave the locked confirmatory configurations unchanged.

## Fit conditioning and regularization

The same audit identified the planned regularization-sensitivity record as
incomplete. `configs/inference/fit_sensitivity_v1.json` specifies a separate
development-only analysis of GD, both feedback constructions, and shared R=1
node perturbation. It uses the saved basis probes for fitting and the same
reserved mixtures for prediction. Three pseudoinverse tolerances (1e-7, 1e-9,
1e-11) check the analytic unregularized reduction's numerical sensitivity.

The positive-ridge objective adds lambda/2 times the squared Frobenius norm of
K, with lambda equal to 0.001, 0.01, or 0.1 times the largest gradient-Gram
eigenvalue. Orthogonal compression to span(G,V) preserves PSD and cannot
increase the objective. The supporting solver uses projected accelerated
gradient with known strong-convexity constant lambda; it does not modify the
frozen confirmatory optimizer. Three anisotropic diagonal cases with independent
closed-form optima check it before the experiment. Record stationarity,
convergence, operator norms, fit residuals, and reserved-probe errors, retaining
unattained unregularized models as undefined actual PSD predictions.

The positive-ridge models illustrate the effect of a chosen regularizer. They
are not replacements for the primary compatibility score and are not used to
choose a favorable rejection threshold.
