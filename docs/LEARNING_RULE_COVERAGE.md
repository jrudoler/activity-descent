# Learning-rule coverage and the global/local distinction

Audited 2026-10-21 against the actual update equations, frozen results, and
primary papers. This is a local research note; the original manuscript is
unchanged.

**We implement only part of the relevant rule families. The completed study
does not establish a general gradient-based/global versus nongradient/local
classification.** It tests a more specific response constraint and distinguishes
some particular update constructions under controlled conditions.

## Coverage of the literature

[Nayebi et al., Section 3 and Appendix A](https://arxiv.org/html/2010.11765v2)
compare SGDM, Adam, feedback alignment (FA), and information alignment (IA).
Their SGDM uses Nesterov momentum 0.9; their FA experiments use Adam on the
pseudogradients. They classify learning trajectories, and explicitly exclude
pure Hebbian and weight/node perturbation training from that benchmark.

| Candidate | Current implementation | Coverage limit |
| --- | --- | --- |
| Backpropagation with SGD | Exact sample gradients; one sampled training example per step | Plain online SGD, no momentum, decay, or adaptive state; not Nayebi's SGDM |
| SGDM / Nesterov | Absent from the campaign | Fixed PSD preconditioning is not momentum |
| Adam | Absent from the campaign | No first/second moments or adaptive normalization |
| Feedback alignment | Fixed random feedback (`feedback-1`) and an initialization-aligned blend (`feedback-0.1`) | Same broad FA mechanism, but fixed-decoder ReLU MLPs with plain updates, not Nayebi's FA-plus-Adam implementation |
| Information alignment | Absent | Feedback matrices never learn; no alignment regularizer |
| Layerwise node perturbation | Central finite differences of global loss, perturbing one layer at a time, with independent/shared draws | A specified INP-like variant, not every perturbation or policy-gradient algorithm |
| Target-independent Hebbian/Oja/BCM/STDP | Absent | No comparison with a purely local, task-error-independent family |

The dispatch in [core.py](../analysis/learning_rule_inference/core.py) supports
exactly four update families: `gd`, `preconditioned`, `feedback`, and `node`.
The nine evaluated labels are parameterizations of these families, not nine
independently developed algorithms: `gd`, `preconditioned`, `feedback-0.1`,
`feedback-1`, `node-1`, `node-32`, `node-512`, `node-shared-1`, and
`node-shared-512`. `State` has no optimizer moments or local
plasticity state. Every family uses the chosen task objective: gradient or
vector output error for the first three, scalar loss differences for node
perturbation. The preconditioner is fixed positive diagonal in parameter space.

In `campaign.align_feedback`, mixture 0.1 means
`B = 0.9 W_initial^T + 0.1 c B_random`, with `c` matching matrix norms. The
feedback stays fixed while forward weights train. It is not a guaranteed 10%
angular or magnitude error relative to the current gradient. `feedback-1` is
fully random. This follows the fixed-feedback principle of
[Lillicrap et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5105169/), with our
specified architecture and probe protocol.

[Kunin et al., Sections 3–4](https://proceedings.mlr.press/v119/kunin20a/kunin20a.pdf)
define IA through local regularization of learned feedback connections,
including Hebbian and stabilizing components. Its forward learning still uses
task-error pseudogradients. A fixed feedback blend does not implement IA.

[Portes et al.](https://arxiv.org/html/2206.13448v2) compare biased supervised
credit assignment (an RFLO variant, with biased BPTT controls) and a
REINFORCE-related node-perturbation rule in recurrent BMI networks. Both use
task feedback; the distinction concerns credit-assignment bias and expected
updates. We have related feedforward families, but no recurrent dynamics,
RFLO eligibility traces, or their reward-baseline policy-gradient algorithm.

[Dalm et al., Sections 2.1–2.2, v4](https://arxiv.org/html/2310.00965v4) distinguish
traditional simultaneous perturbation, iterative layerwise perturbation (INP),
and activity-based perturbation (ANP), with decorrelation variants. Our version
is closest to INP, using central differences and repeat averaging. We do not
implement their whole-network ANP update or input-decorrelation mechanism.

## Three meanings that must stay separate

1. **Exact task-gradient computation:** backpropagation computes the gradient;
   fixed feedback usually supplies a pseudogradient; our node rule estimates
   the gradient in expectation in the small-perturbation limit.
2. **Global task information:** FA uses a propagated output-error vector. Node
   perturbation uses a scalar global loss difference. Both depend on task
   feedback despite avoiding exact backpropagation.
3. **Local synaptic implementation:** presynaptic/postsynaptic variables and
   available modulatory signals can support a local calculation without exact
   downstream weight transport. Such locality can coexist with global error or
   reward signals and gradient-following behavior in expectation.

Absence of backpropagation does not imply absence of a global objective.
Even a task-independent local rule can optimize a different local objective.
Rejecting a model relative to the experimenter's chosen loss cannot establish
that a learner optimizes no objective.

## The comparisons we actually ran

At fixed input, network state, decoder, and updated parameter set, we vary
targets and measure `g=dL/dA` and `v=-Delta A/eta`. The tested first-order model
is `v(g)=K g` with one symmetric PSD `K`. It includes GD (`K=J J^T`) and fixed
PSD-preconditioned GD (`K=J P J^T`). The primary score combines normalized
reciprocity, negativity, and linear-consistency residuals, calibrated on
separate GD and fixed-preconditioner runs.

It is not a multiclass rule-identity classifier. Rejection does not label a
response as FA, node perturbation, local learning, or nongradient learning.
Passing checks necessary constraints, not the full singular PSD condition.

Primary counts at step 1000 without added observation noise:

| Probed update | Operator rejection / 200 | Pearson rejection / 200 | Meaning |
| --- | ---: | ---: | --- |
| GD | 11 | 6 | False rejection of the specified GD null |
| Fixed PSD preconditioning | 11 | 8 | Expected equivalence |
| Feedback mixture 0.1 | 200 | 6 | Detects this biased-feedback construction |
| Fully random feedback | 200 | 200 | Both methods detect this alternative |
| Independent node perturbation, R=512 | 200 | 9 | Common deterministic operator fails; this does not establish a nongradient mean |
| Shared node perturbation, R=1 | 24 | 200 | First-order equivalence control has excess finite-step rejection |

These primary states were trained with GD. Separate profiles trained with
preconditioned GD, feedback mixture 0.1, and node R=32 also exist. Probing
feedback-0.1 at its own trained checkpoints gives operator 200/200 versus
Pearson 4/200. However, there is no separate trajectory trained with fully
random FA plus its literature optimizer, nor SGDM/Adam/IA trajectories. Each
saved state is probed with every candidate rule: these are controlled
counterfactual updates, not a blind classification of the historical learner.

In the conditional-mean study, at 8,192 repeats, the uncertainty-aware
reciprocity test rejects physical node perturbation in 9/200 and the biased
feedback-mean controls in 200/200. Those feedback controls add centered
directional-estimator noise to a feedback mean; they are synthetic stochastic
controls. This separates sampling variability from persistent mean bias under
that protocol, without certifying a full PSD model or unbiasedness.

The two calibration nulls do not cover every gradient optimizer or PSD
construction. Feedback detection drops to 30/200 at depth three and 69/200
with 1% GD-response-RMS observation noise. Shared node R=1 is rejected in
176/200 under that noise condition.

## Why momentum and Adam require different null models

The following are scope checks, not additions to the completed campaign.
For fixed prior momentum `m`, classical momentum gives
`v(g)=J(beta m)+J J^T g`. Its first term is independent of the current target.
The uncentered test's requirement `v(-g)=-v(g)` can fail despite exact task
gradients. Nesterov has the same issue with different coefficients. Weight
decay can likewise contribute target-independent drift.

Adam's updated denominator depends on the current probe's squared gradient.
It need not produce one linear map across probes even with restored optimizer
state. At its first step it maps a gradient component approximately to its
sign. A fixed-preconditioner control does not establish coverage for Adam.

A tiny executed check used the actual PyTorch 2.11.0 optimizers with `A=W` in
two dimensions and `L_g(W)=0.5||W+g||^2` at `W=0`. Six probes were
`+e1,+e2,-e1,-e2,+(e1+e2)/sqrt(2),-(e1+e2)/sqrt(2)`. Parameters and optimizer
state were independently restored. Step size was 1e-4; Nesterov used momentum
0.9 and prior buffer `(0.3,-0.2)`; Adam used standard defaults and epsilon
1e-8. The existing `operators.compatibility` returned:

| Exact-gradient optimizer | Normalized linearity residual |
| --- | ---: |
| Plain SGD | 5.23e-16 |
| Nesterov with prior buffer | 0.151926 |
| Nesterov with known offset removed | 4.65e-16 |
| Adam, first step | 0.146447 |

The Nesterov response also matched `v(g)=1.9g+0.81*(0.3,-0.2)` numerically.
These examples demonstrate model mismatch, not false-positive rates in the
network campaign; no campaign threshold was applied to them. The execution
record is `/tmp/activity-descent-rule-coverage-check.json`.

## Requirements for a broader future comparison

- Implement and validate SGDM/Nesterov, Adam, and learned IA feedback with
  explicit optimizer-state handling. Match the literature's FA optimizer when
  claiming replication, and include training under each candidate.
- Include specified task-independent local controls and task-modulated local
  controls if the question concerns dependence on a global task signal.
- Separate target-independent drift from target-dependent response, for example
  with zero-error and paired-error probes. Drift subtraction alone does not
  solve identification: a task-independent response becomes zero, compatible
  with `K=0` and uninformative about gradient descent.
- Define and calibrate the null for the intended claim, accounting for
  momentum, adaptive state, finite steps, and stochastic means. Freeze a new
  protocol before held-out evaluation; retain the existing study unchanged.

These are proposed extensions, not completed experiments. The present audit
establishes which comparisons exist and why they do not answer the broader
binary question.

## Verification record

Inspected `core.py`, `campaign.py`, `operators.py`, `mean_response.py`, the
locked configuration, and main/mean/GPU evaluation tables. All four frozen
science-module hashes match `inference_evaluation.provenance.json`, which
records 160 completed shards. The nine labels above exhaust the main CSV's
rule field. The main manuscript remains at
`85d280b4c1576dc08564245477bf8cf83fed56de`.
