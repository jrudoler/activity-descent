# Learning rule identification assumptions and operator tests

This protocol tests compatibility with a class of learning operators. The true
upstream weights and their Jacobians are unavailable to the inference method.
The experimenter knows the task loss and a fixed linear readout, observes the
complete layer, and can restore the same pre-update state before each probe.

## Exact finite probe compatibility

Let the columns of G contain activity gradients g_k and the columns of V contain
responses v_k = -Delta A_k / eta, with a common known positive step size. In the
first-order noiseless model, a symmetric positive-semidefinite K satisfying
KG = V exists if and only if

1. S = G^T V is symmetric positive semidefinite.
2. ker(S) = ker(V).

Necessity follows from S = G^T K G = (K^(1/2) G)^T (K^(1/2) G). Therefore
ker(S) = ker(K^(1/2) G) = ker(K G) = ker(V).

For sufficiency, let S^+ be the Moore-Penrose inverse and set K_* = V S^+ V^T.
This matrix is symmetric PSD. Because V^T G = S and the assumed nullspaces agree,

```text
K_* G = V S^+ S = V.
```

This proof also handles repeated or linearly dependent probes and zero-response
directions. Reciprocity g_a^T v_b = g_b^T v_a tests symmetry, and nonnegative
quadratic forms test positivity. Neither condition supplies the nullspace
requirement. For example, G = (1, 0)^T and V = (0, 1)^T give S = 0, yet a PSD K
cannot have a zero first diagonal entry and a nonzero first off-diagonal entry.

Conversely, responses outside the span of the gradients can be compatible:
G = (1, 0)^T and V = (1, 2)^T admit K_* = ((1, 2), (2, 4)). Retain the complete
observed response rather than projecting it into the gradient span.

For new probes in span(G), an exact common linear operator has a determined
response: if g_new = G a, then v_new = V a. Outside this span, finite probe data
generally do not identify the operator. A canonical extension is a fitting choice
and does not justify extrapolation. Rank diagnostics must accompany every fit.

## Relation to weight updates

Let J = dA/dW for all updated parameters upstream of the observed layer. Under
ordinary gradient descent, dW = -eta J^T g and v = J J^T g to first order.
For a fixed symmetric PSD parameter preconditioner P, dW = -eta P J^T g induces
v = J P J^T g. Both satisfy the same observable constraints. A parameter update
in ker(J) is invisible on the current input. Acceptance of an activity operator
does not uniquely identify ordinary GD or determine the full weight update.

The parameter set, preconditioner, input, state, and decoder must remain fixed
throughout a probe bank. In particular, normalizing each response to its own
length changes the map between probes and can violate reciprocity even for GD.
Use one shared positive scale for a bank; its magnitude does not change the
symmetry or PSD class.

With a known linear readout y = C A and squared error, g = C^T (C A - target).
The accessible gradient span has dimension at most rank(C), regardless of the
number of targets. The original two-output network supplies at most two
independent directions at a fixed input and state. Eight- and sixteen-output
readouts expand the available span without identifying an entire wide-layer
kernel.

## Layerwise perturbation and observational equivalence

The implemented stochastic control perturbs one layer's preactivations at a
time, using the same clean presynaptic activity for eligibility. For an isotropic
unit direction u in a layer of width n, the central directional estimator tends
to n u u^T delta as perturbation magnitude approaches zero, where delta is the
true preactivation gradient. Its expectation is delta because E[n u u^T] = I.
The finite perturbation estimator uses

```text
signal = n [L(a + eps) - L(a - eps)] eps / (2 ||eps||^2)
weight descent direction = signal * clean_presynaptic_activity^T.
```

This is a central-difference variant of the layerwise directional formulation
described by [Dalm et al. 2024](https://arxiv.org/html/2310.00965v4). Perturbations
at distinct layers are evaluated from the clean checkpoint and their estimated
weight directions are applied together. ReLU gate crossings and finite
perturbation size can bias the estimate.

If the same sampled directions are reused across every target probe, the local
gradient transformation H = mean(n u u^T) is fixed, symmetric, and PSD. Applied
to the rows of a layer's parameter gradient, it defines a PSD block parameter
preconditioner. Thus shared-direction node perturbation satisfies the operator
null to first order even with one direction per layer. Independent draws across
targets instead produce different stochastic operators. Their responses need
not fit one common K, although their conditional means approach GD in the
validated small-perturbation limit. Comparing independent and shared directions
is necessary to interpret a rejection caused by stochastic variability.

## Finite measurements and calibration

Finite updates contain Taylor remainders, finite perturbations contain estimator
bias, and measurements can contain noise. The exact rank and nullspace conditions
are discontinuous near singular matrices. Numerical tolerance is part of the
protocol and must be calibrated; an exact algebraic incompatibility alone is not
a statistically meaningful rejection.

Record reciprocity, negative quadratic forms, linear consistency, nullspace
residuals, and held-out prediction. Use convex least squares over symmetric PSD
operators as a complementary fit, retain solver convergence diagnostics, and
compare it with unrestricted linear and nonnegative diagonal models. An
unconverged fit cannot establish incompatibility.

Calibrate the chosen decision statistic on independent known-null simulations
and evaluate on held-out runs. A probe bank contributes one decision at the
selected operating point; probe pairs and neurons are dependent measurements.
Report rejection separately for GD, fixed preconditioned GD, stochastic single
updates, stochastic averages, and shared-direction perturbation controls. State
the tested rank, step-size range, numerical precision, and noise conditions with
the resulting false rejection and detection power.
