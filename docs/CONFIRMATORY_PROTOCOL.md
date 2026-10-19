# Confirmatory protocol for activity operator inference

The configuration `configs/inference/confirmatory_v1.json` fixes the main
experiment before independent calibration or evaluation runs. Development used
the original 20-seed pilot and separate 20-seed matched runs. No evaluation data
were used to choose the statistic, conditions, or comparison below.

The primary contrast is the shared-operator score against Pearson alignment for
the 10% fixed-feedback mixture at the 1000-step GD-trained checkpoint, width 64,
depth two, eight outputs, and no added observation noise. The primary score is
the maximum of normalized reciprocity, negative-quadratic-form, and linear
consistency residuals. The nullspace residual is descriptive because its exact
rank condition is discontinuous. PSD fitting and prediction are supporting
analyses and distinguish an attained solution from an unattained infimum.

A secondary control retains only the positive and negative version of the first
output-error direction from the same observed bank. It uses the same operator
score and its own null calibration. Comparing it with the full bank isolates the
value of independent intervention directions at identical network states.

Each profile has 200 independent calibration runs and 200 independent evaluation
runs. One bank has a positive and negative spanning output-error basis and 16
independent unit mixtures reserved for prediction. One fixed input is observed
per checkpoint, with the complete last hidden layer and its known linear
decoder. The inference gradient is computed directly from the decoder and
chosen output error, without upstream weights or Jacobians.

The step is 1e-4 and the node perturbation magnitude is 1e-4. Ordinary GD, fixed
positive diagonal preconditioning, 10% and fully random fixed feedback, and
layerwise central node perturbation at 1, 32, and 512 samples are compared at
identical checkpoints. Shared perturbation directions at 1 and 512 samples are
equivalence controls. Every rule has one positive magnitude scale per run and
checkpoint, estimated from a separate bank of 16 target probes and then held
constant. Weight norms are available only to the simulator for this matching
control; they are not inference features.

GD and PSD-preconditioned GD are the two calibration null controls. For each
profile, checkpoint, noise level, and statistic, take the larger of their two
split-conformal upper cutoffs at alpha=0.05. With n finite calibration values,
the order statistic is ceil((n+1)*0.95); reject values strictly above the cutoff.
The nominal rate applies per studied condition, not simultaneously across the
entire set of secondary analyses or every possible PSD operator. Scalar
alignment baselines use the same calibration protocol.

Evaluation reports rejection counts, rates, and Wilson 95% intervals across
independent runs. Comparisons retain pairing by checkpoint. Power differences
use a conservative bound built from 97.5% Wilson intervals for the two paired
discordance probabilities, with a paired bootstrap as a sensitivity analysis.
This avoids a zero-width uncertainty claim when every observed decision agrees.

The primary architecture is also trained with preconditioned GD, fixed feedback,
and 32-sample node perturbation. Measurements at steps 0, 200, and 1000 show both
equal-step comparisons and a prespecified comparison near loss 0.4. For that
comparison select the closest available checkpoint by absolute log loss and
retain runs with actual loss between 0.2 and 0.8. Apply this same selection to
calibration and evaluation data; report actual losses and exclusions. The table
uses checkpoint step -1 for this comparison.

Targeted controls use two and sixteen readout dimensions, width 32, and depth
three. An optional response-noise condition adds independent Gaussian noise with
RMS 1% of the GD response RMS, using the same noise realization across paired
rules. This is a controlled signal-to-noise benchmark, not a claim about an
absolute biological recording-noise level. Gradients, responses, scales, gates,
prediction support, checkpoint states, source revisions, and array hashes are
retained for verification.

Finite-sample stochastic update banks are tested for a *single common operator*
by the main statistic. Their rejection does not establish a biased conditional
mean or identify node perturbation uniquely. A separate analysis of mean
responses must account for Monte Carlo uncertainty and distinguish averaging
weight directions before an update from averaging activity responses after
restored updates. That analysis does not change the locked primary test.
