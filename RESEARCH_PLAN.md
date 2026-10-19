# Learning rule inference research plan

Updated 2026-10-19. This is the implementation plan selected after the research
handoff. Diagnostics and a small set of mechanism checks support the main work:
designing an identification protocol and comparing learning rules. All new
experiments below are planned.

The intended paper asks which classes of synaptic update rules can be
distinguished through controlled changes in neural activity, with measured false
rejection rates, detection power, and explicit observational equivalences. A
useful negative result is a completed outcome if it establishes where the
protocol lacks information. Positive discrimination is a hypothesis to test.

## Scientific question and scope

At a fixed network state and input, gradient descent on parameters upstream of a
complete observed layer predicts, to first order,

```text
g = dL/dA
v = -Delta A / eta
v = K g + finite-step error
K = J J^T
```

The identification problem treats the upstream weights and K as unknown. It uses
activity observations and gradients supplied by a known, fixed downstream
mapping. One common symmetric positive-semidefinite (PSD) K must explain multiple
probe directions from the same restored state. Agreement with this class is not
unique identification of ordinary GD: a fixed symmetric PSD parameter
preconditioner P induces K = J P J^T, and weight changes invisible through J cannot
be recovered from these observations.

The primary experiment observes the complete last hidden layer through a known
linear readout held fixed throughout each probe bank. Earlier-layer gradients
computed using all downstream weights are an explicitly labeled oracle control.
The inference algorithm does not receive the true upstream Jacobian, kernel, or
learning-rule label. Those quantities are available for simulation validation.

Broader partial-recording studies, cross-stimulus kernels, long intervals between
observations, expanded dropout studies, and large image benchmarks are deferred.
Small measurement-noise controls needed to calibrate the identification test are
in scope.

## Milestones and order

| Milestone | Deliverable | Completion criterion |
| --- | --- | --- |
| M0 Experiment contract and GPU setup | Mathematical assumptions, probe design, locked runtime, Betty launch configuration | CPU reference checks and a measured B200 smoke run pass |
| M1 Diagnostics | Layerwise alignment, prediction error, interference, and gate measurements | Degenerate cases have defined behavior and independent-run summaries reproduce the existing baseline |
| M2 Limited mechanism checks | Learning-rate and parameter-freezing experiments | First-order interference is distinguished from finite-step error over a documented usable step range |
| M3 Identification pilot | Shared-operator tests, rank audit, and held-out prediction | Known null controls are calibrated and identification limits are measured before scaling |
| M4 Rule comparison | Locked evaluation of GD, PSD-preconditioned GD, feedback alignment, and node perturbation | False rejection, power, baseline comparisons, and stochastic averaging results have uncertainty estimates |
| M5 Paper and reproducibility | Figures, tables, methods, theory, manuscript, and execution instructions | Claims match completed experiments and the workflow reproduces the reported artifacts |

M1 and M2 are bounded prerequisites. M3 and M4 receive the main experimental
effort. After the small baseline checks, run M3 promptly rather than completing
every possible architecture sweep first.

## M0 Experiment contract and GPU setup

Write down the observation model, the known task loss, the updated parameter set,
the fixed downstream mapping, and the shared step-size convention. Derive the
finite-probe compatibility conditions, including linear consistency and the
nullspace conditions needed for a PSD extension. Reciprocity by itself is only a
necessary test. Verify the derivations on analytic matrices with compatible,
incompatible, and rank-deficient responses.

Use Betty's B200 GPUs for simulation and repeated probing. A read-only Slurm check
on 2026-10-19 confirmed the active `dgx-b200` partition with eight B200 GPUs per
node. The current Slurm association for `jrudoler` confirms account
`wharton-phd-students` and GPU QoS `wharton-phd-student-dgx-b200`.
Start with one GPU requested through Slurm. The Wharton PhD GPU group/partition
has a scheduler-enforced limit of three concurrent jobs. Account for this job
limit and existing usage when scheduling the campaign. Benchmark CPU and GPU
throughput and consider queue time when assigning work. Small independent
experiments can run concurrently as CPU shards on `genoa-std-mem` with QoS
`wharton-phd-student-genoa`; reserve B200 batches for work that benefits from
them. Increase array concurrency only when measurements justify it. Both paths
use the same saved inputs, scientific implementation, and numerical checks. Run
substantial experiments inside allocations.

Add a CUDA PyTorch implementation that batches independent seeds, probes, and
perturbation repeats. Keep tensors on the device through the simulation and
aggregate results in batches. Use layer-kernel recursions or Jacobian-vector
products instead of materializing unnecessary parameter Jacobians. A small NumPy
reference and analytic examples establish correctness. Extract only the reusable
network and update routines needed for import-safe experiments; avoid importing
the existing figure script in a way that executes its sweeps.

Validate GPU and CPU calculations using identical saved weights, inputs, and
probe vectors. Use float64 references for finite differences, reciprocity,
eigenvalues, and small-step activity differences. Benchmark float32 and float64
on the actual B200 and retain a faster mode only if it preserves the relevant
scores and decisions. Record matrix-multiplication precision settings; low
precision must not determine near-zero sign or PSD decisions. A CUDA run must
fail clearly if its requested GPU is unavailable rather than silently executing
the campaign on CPU.

Check compatibility between CUDA-enabled PyTorch and the repository's Python
>=3.14 environment before selecting versions. If a separate GPU runtime is
necessary, lock it separately and connect it explicitly to the workflow. Record
Python, PyTorch, CUDA, driver, GPU model, dtype, seeds, source hashes, Slurm job ID,
runtime, and peak memory. Benchmark representative batched work before fixing
array sizes and wall times; checkpoint and resume completed work.

## M1 Diagnostics

Retain Pearson correlation for continuity and add the following for each hidden
layer and the output separately:

- Cosine similarity with the negative activity gradient and each prediction.
- Fractions moving downhill, uphill, or by an uninformatively small amount.
- Relative vector prediction errors for the full-kernel and diagonal models.
- The population quantity g^T Delta A and its individual-neuron contributions.
- Self-terms K_ii g_i and signed interference h_i = sum(j != i) K_ij g_j.
- Observed Taylor residuals and ReLU gates switching in both directions.
- Counts of inactive units, zero gradients, tiny gradients, and excluded values.

Define absolute and relative tolerances from numerical validation and freeze them
before evaluation. Undefined correlation or cosine values are missing values with
counts, not artificial zeros. Present the complete-layer population calculation;
label any active-neuron subset summaries. For finite updates, g^T Delta A is a
first-order loss contribution through that layer, not the full loss change after
all parameters move.

Begin with small versions of the current bar-image experiments at widths 32 and
64. Summarize within each independent run first, then quantify variability across
runs. Inputs, probes, snapshots, and neurons within a run are repeated measures.
Include analytic checks of the Appendix C counterexample and cases where
correlation conceals incorrect scale or coordinate signs.

## M2 Limited mechanism checks

At identical saved checkpoints and inputs, compute one pre-update gradient and
restore the checkpoint before each intervention:

1. Sweep a small logarithmic learning-rate grid, initially 1e-5 to 1e-2. Identify
   the range above the numerical noise floor where normalized Taylor residuals
   shrink as expected, and separately record gate crossings. Do not retrain a
   different network for each diagnostic step size.
2. Compare updates to the observed layer's own incoming weights only, upstream
   weights only, and both together. For the current unshared dense layers, the
   own-layer contribution is diagonal; upstream updates isolate shared-pathway
   interference. Compare the first-order decomposition and finite-step residual.
3. Compare task-generated gradients with synthetic layer-loss probes at the same
   state. Label synthetic layer losses as mechanism controls distinct from probes
   accessible through the task readout.

Independent upstream-width sweeps are optional follow-up if they resolve an
ambiguity in the identification results. Select a small-step operating range
using development runs before the main identification evaluation.

## M3 Identification pilot

### Task and probe rank

Keep the original two-output task as a continuity and limited-rank baseline.
Use a small teacher-regression task on 16-dimensional inputs with a fixed known
linear readout as the primary identification setting. Start with eight outputs;
use two and sixteen outputs as rank controls. Begin with two hidden layers of
width 32 for development. Width 64 is the initial main-study setting, with width
32 and depth three as targeted robustness conditions rather than a full factorial
sweep. Keep the task small enough that independent runs and repeated interventions
are inexpensive on GPU.

At every checkpoint and input, build a bank of output errors by varying targets
under squared loss. For a fixed linear readout C, g = C^T (C A - target), so its
rank is at most the readout rank. Measure the singular values and numerical rank
of the actual gradient bank. Use independent directions, opposite directions,
repeats, and mixtures. Reserve mixtures and additional directions within the
supported span for prediction tests. Never claim prediction in an unprobed
subspace. More targets in the same two-dimensional span do not supply new
independent dimensions.

Restore the same parameters and rule-specific state before every probe. Fix the
input, decoder, parameter mask, and preconditioner throughout the bank. Random
perturbation draws are controlled experimental replicates, not sequential
training steps. A different input or checkpoint starts a new operator fit.

### Tests and baselines

Compare one shared operator across all probes using:

- Linearity and consistency under repeated, opposite, and mixed gradients.
- Normalized reciprocity residuals g_a^T v_b - g_b^T v_a.
- PSD constraints on observable quadratic forms and their combinations.
- A shared symmetric PSD fit evaluated on held-out probes, compared with an
  unrestricted linear fit and a diagonal nonnegative fit.

Keep measured responses outside the gradient span; silently projecting them away
could hide incompatibility. Use a convex fitting formulation and validate its
solver on analytic cases. Record convergence, conditioning, residuals, and
regularization sensitivity so solver failure or an arbitrary extension of an
underidentified operator is not mistaken for rejection.

Compare against the existing Pearson correlation, cosine alignment, downhill
fraction, and population descent statistics. Set rejection thresholds for all
baselines using the same calibration protocol. Measure whether joint probe
constraints add power beyond single-update alignment.

### Calibration and progression

Use 20 independent development seeds initially. Separate development, calibration,
and evaluation seeds, including data, initialization, feedback, and perturbation
randomness. Choose score normalization, probe count, fitting settings, and the
primary statistic on development data. Calibrate a single decision per
checkpoint; if multiple layers or statistics contribute, calibrate their joint
decision rather than counting each probe pair as an independent test.

Target a 5% false rejection rate using known GD and fixed PSD-preconditioned GD
null controls. Initially plan 200 calibration runs and 200 independent evaluation
runs for each primary protocol, with sample sizes finalized from measured pilot
cost and the desired binomial confidence interval before evaluation. Calibration
must cover the studied null conditions; report rejection separately by null
condition and do not claim uniform validity for every possible PSD operator.
Keep architecture, step-size, and noise changes explicit when thresholds are
transferred.

Proceed to M4 when null behavior is understood, rank is adequate, solver checks
pass, and the protocol can either detect a controlled incompatible response or
explain why it cannot. Include a synthetic nonsymmetric response as a detector
sanity check, labeled separately from a biological learning rule. Fix inadequate
probe rank or numerical calibration before adding broad sweeps. If valid probes
still lack power, pursue the inference-limit result with the same controls.

## M4 Learning rule comparisons

| Rule | Role | Required interpretation |
| --- | --- | --- |
| Ordinary GD | Reference null | Finite steps and gates can cause deviations from its first-order prediction |
| Fixed symmetric PSD-preconditioned GD | Equivalence control | The shared-operator class should accept it in its validated regime |
| Feedback alignment | Biased credit-assignment alternative | Test detectability; acceptance can occur for particular states or probe spans |
| Layerwise node perturbation | Stochastic alternative | Analyze single updates and conditional mean responses separately |

Use fixed random feedback matrices for feedback alignment and true task gradients
for evaluating every rule. For node perturbation, implement a documented
layerwise formulation with a clear small-perturbation gradient limit, including
the perturbation location, loss baseline, presynaptic eligibility, and estimator
normalization. Verify its mean against the true gradient as perturbation size and
sample count vary. Do not substitute weight perturbation or assume all node
perturbation variants have the same expectation.

First apply all four rules to the exact same checkpoints. This isolates the
effect of the update rule from training history. Then repeat at checkpoints
trained with each rule, reporting learning curves and comparing both equal-step
and overlapping loss ranges. Document failures to learn or lack of comparable
stages rather than silently dropping those runs. Match update magnitudes using a
single positive scale per rule and checkpoint, estimated from a separate
development probe bank and held constant across evaluation probes. Keep raw-scale
results as a sensitivity check. Per-probe normalization would change the tested
operator and can create false reciprocity violations.

For perturbation rules, measure individual updates and averages over increasing
numbers of independent perturbations from the same state. Initial repeat counts
are 1, 8, 32, 128, and 512, adjusted on development data for variance and GPU
throughput. Distinguish averaging update directions before one step from
averaging observed responses after separate restored steps. Their finite-step
remainders need not coincide. An averaged response that becomes compatible with
GD is an informative equivalence result. A finite-sample rejection alone does
not identify node perturbation specifically.

Also reuse identical perturbation directions across probes as an equivalence
control. In the small-perturbation limit, fixed isotropic directions give a fixed
PSD parameter preconditioner. Compare this control with independent directions
to determine whether a rejection reflects variation between stochastic updates.

Lock the main protocol before evaluation. Report held-out false rejection, power
at the calibrated operating point, score distributions, prediction errors, and
binomial or run-level uncertainty. Bootstrap independent runs for comparisons;
retain pairing when rules share checkpoints. Summarize which rule classes are
distinguishable and where the available observations admit several explanations.

## M5 Paper and reproducibility

Organize the manuscript around the intervention protocol and its inference
limits. Use the activity-kernel identity and neuronwise diagnostics as supporting
theory. Include a derivation of the tested constraints, rank limitations,
preconditioned equivalence, and the role of stochastic averaging. Explain what
the experimenter knows about the loss, decoder, state restoration, and step size.

Planned figures cover the protocol and probe rank; diagnostic and finite-step
validation; calibrated rule comparisons against alignment baselines; and
perturbation sample requirements and equivalences. Keep successful and failed
conditions in the reported results. Audit title, abstract, diagonal-dominance
language, and biological measurement claims. Preserve
`paper/archive/neurreps_extended_abstract.tex` and the separate COSYNE submission.

The related-work comparison must establish the distinction from existing
activity-based rule inference and internal-kernel analyses. The current plan
targets an interpretable intervention test with validity conditions and measured
error rates; its novelty must be checked against full papers before final claims.

Keep analysis implementation and this plan in the parent repository. Add explicit
Snakemake entrypoints for diagnostics, pilot, comparison, and figure aggregation.
Store resumable arrays and manifests in `data/generated/`, scientific tables in
`results/data/`, and final plots in `results/figures/`. Store launch scripts and
configuration in the repository, and avoid treating successful submission as a
completed experiment. Record analysis revision and configuration with every
result; collect and inspect Slurm exit status and output completeness.

Pull collaborators' manuscript changes before editing or collecting assets.
Copy publication figures into `paper/generated/figures/` and record analysis
provenance there. Keep analysis and manuscript Git histories separate. Compile
and inspect the final manuscript with citations and references resolved; the
current login environment does not expose `latexmk`, so provision an appropriate
build environment or supported compiler during execution.

## Completion criteria

- The GPU implementation passes reference and analytic validation, and Betty
  B200 jobs complete with recorded runtime and device provenance.
- Layerwise diagnostics and the limited mechanism experiments establish the
  validated conditions for the identification tests.
- The fixed-state pilot quantifies probe rank, compatibility, and prediction
  limits without exposing oracle kernels to the inference method.
- Held-out comparisons cover all four learning rules, both null controls, simple
  alignment baselines, stochastic single updates, and mean responses.
- False rejection and detection power have independent-run uncertainty, and
  equivalences or failures of identification are reported explicitly.
- Reproducible artifacts and a revised, compiled manuscript support every
  empirical claim. A scientifically honest bounded or negative result satisfies
  the goal; preparing code or launching pending jobs alone does not.

## Starting references

- [Nayebi et al. 2020](https://arxiv.org/abs/2010.11765): inference from neural
  observables and learning trajectories, including limited observations.
- [Portes et al. 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/a6d94c38506f16fb50894a5b555f2c9a-Abstract-Conference.html): learning-rule
  discrimination from activity changes with a known behavior mapping.
- [Hazelden et al. 2026](https://arxiv.org/abs/2605.08746): internal-state kernels,
  their structure, and rank constraints.
- [Nordio 2026](https://arxiv.org/abs/2606.09744): layerwise response recursions
  under fixed readout and quadratic loss.
- [Lillicrap et al. 2016](https://www.nature.com/articles/ncomms13276): feedback
  alignment using fixed random feedback connections.
- [Dalm et al. 2024](https://arxiv.org/html/2310.00965v4): node perturbation variants
  and a layerwise formulation related to directional derivatives.
