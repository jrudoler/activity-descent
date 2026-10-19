# Related work audit for learning rule inference

This audit compares the intervention study with the closest primary papers. The
candidate contribution is a calibrated test of a shared activity-learning
operator, accompanied by equivalences and limits of inference. Kernel
representations, activity-based rule discrimination, known-decoder experiments,
and PSD least-squares fitting all have substantial prior work.

## Neural observables across learning trajectories

[Nayebi et al. 2020, Sections 3 to 6 and Appendix B](https://arxiv.org/html/2010.11765v2)
train classifiers on statistics of weights, activities, and layerwise input-output
sensitivities across learning. They evaluate held-out conditions and limited,
noisy observations. Their instantaneous layerwise feature differentiates outputs
with respect to inputs; it is not a single learning-induced activity increment.

The proposed experiment instead varies task gradients at a restored state and
tests one shared operator without a classifier trained on alternative-rule
labels. It assumes a known loss and decoder, making its observation model more
restrictive. Independent calibration and evaluation runs remain necessary. Neural
observables identifying learning rules is an established research direction.

## Known decoder and biased credit assignment

[Portes et al. 2022, Sections 2 and 3 and Appendix A](https://arxiv.org/html/2206.13448v2)
already infer learning rules from activity with a known behavior decoder. Their
main setting compares flow-field changes during recurrent learning under biased
credit assignment and a policy-gradient rule. Appendix A derives hidden-activity
changes in a fixed-readout linear feedforward model and compares predicted
directions. They also examine noise covariance and its effect on inference.

The proposed distinction is to test multiple target interventions against an
unknown shared symmetric PSD operator, preserving gradient-subspace and
nullspace information and reporting expected equivalences. A feedforward BMI-like
simulation, use of a known decoder, or a comparison between random feedback and
node perturbation does not establish novelty. Direct comparison against their
directional-alignment approach is needed in addition to the existing paper's
Pearson diagnostic.

## Internal state kernels and rank

[Hazelden et al. 2026, Sections 3 and 4 and Appendix A](https://arxiv.org/html/2605.08746v1)
define the empirical kernel on global internal states, factor it into direct
parameter interactions and state propagation, and analyze structural rank
bottlenecks across architectures. The internal Gram representation and its
propagation decomposition are therefore established background.

This study's accessible gradient rank is a different question: a fixed decoder
limits which directions can probe an unknown operator even when that operator
has high rank. The distinction must be stated explicitly. New inference claims
depend on the intervention protocol and its calibration, rather than the
existence of an internal kernel or finite-width effects.

## Activation response recursions

[Nordio 2026, Sections 4 to 7 and 11](https://arxiv.org/html/2606.09744v4)
derive first-order activation and conjugate-field dynamics for fixed-readout
ReLU networks. Their layerwise responses include a direct update and propagated
upstream effects, and include cross-example kernels. The fixed-activation-region
and finite-step limitations are explicit.

Our self/cross and own/upstream diagnostics support the inference protocol, but
the corresponding response recursion cannot carry a claim of novelty. The
experiment should retain finite-step and gate measurements rather than treating
a linear expansion's agreement as evidence identifying the learning rule.

## Positive semidefinite operator fitting

[Gillis and Sharma 2018, Sections 2 to 4](https://arxiv.org/html/1612.01354v2)
analyze the PSD Procrustes problem, minimizing the residual of AX-B over PSD A.
They characterize solutions, identify when the infimum is not attained, and use
dimension reduction and accelerated gradient methods.

Our convex operator fit is an application of this established problem. A small
least-squares residual need not imply that an exact PSD interpolant exists in a
singular problem. The finite-probe proof in the methods note makes assumptions
transparent; neither it nor the optimizer should be presented as a new general
matrix-fitting method. Check solution conditioning and nullspaces separately,
and distinguish exact compatibility from a noisy-data rejection test.

## Stochastic gradient estimation

[Dalm et al. 2024, Section 2 and Appendices B and D](https://arxiv.org/html/2310.00965v4)
distinguish traditional, iterative layerwise, and activity-based node perturbation.
The layerwise directional formulation has an explicit gradient relationship;
other variants need separate analysis. Our central-difference estimator is a
specified variant of their layerwise construction.

The simulation must name the variant and test its mean. The derived
shared-direction PSD-preconditioning control explains an observational
equivalence; single-sample variability or its reduction under averaging cannot
by itself uniquely identify node perturbation.

## Claims to earn with experiments

The study should establish whether shared-operator constraints improve detection
over alignment at a calibrated false rejection rate, how available probe rank
changes that result, and which preconditioned or stochastic controls remain
observationally compatible. It must also measure finite-step and numerical
conditions that invalidate its tests. These are proposed empirical contributions,
not conclusions from the literature audit.

The current literature supports treating this as a focused follow-up to prior
activity-based inference and kernel geometry. Broad claims of first identifying
learning rules from activity, discovering internal kernels, or uniquely
identifying GD would be unsupported.
