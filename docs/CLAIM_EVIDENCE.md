# Manuscript claim and evidence review

Reviewed 2026-10-20 against the frozen CPU summaries, completed matched
CPU/B200 MIG preflights, and executed primary CUDA replication. This review checks interpretation and reported values;
it does not replace the strict raw-shard verification in the scientific evaluators.

## Main claims

Unless specified otherwise, primary rows in `results/data/inference_evaluation.csv`
have `profile=primary`, `checkpoint_step=1000`, and `response_noise=0.0`.
Every quoted rejection count has 200 independent evaluation networks, with
candidate rules paired at the same state and separate calibration networks.

| Manuscript claim | Exact evidence | Scope retained in the draft |
| --- | --- | --- |
| Shared constraints detect the partially biased feedback construction when scalar alignment often accepts it | `rule=feedback-0.1`: operator 200, Pearson 6, cosine 7, single_direction 2 | Known decoder, complete layer, restored states, this checkpoint and construction; 0.1 is a feedback mixture, not a measured angle |
| Both specified deterministic null controls have low observed rejection | Operator: `gd=11`, `preconditioned=11` | Wilson intervals are conditional on the realized calibration; conformal coverage is marginal for the two specified exchangeable null distributions |
| Primary paired detection gain is 97.0 percentage points | `inference_paired_comparison.csv`, same primary condition, `rule=feedback-0.1`, `baseline=pearson`: 0.97, interval [0.9048983, 0.9875641] | A per-condition approximate conservative interval from discordance probabilities, not a simultaneous guarantee across secondary analyses |
| Power depends materially on depth and observation noise | Operator on `feedback-0.1`: `profile=depth-three` gives 30; primary `response_noise=0.01` gives 69 | Targeted architecture controls and noise referenced to GD response RMS, without uniform generalization claims |
| Calibrating on GD and one preconditioner does not cover all PSD constructions | Primary operator on `node-shared-1`: 24 without noise, 176 at `response_noise=0.01` | Shared-direction equivalence is first order; finite-step errors and response signal-to-noise can change rejection |
| A stochastic bank can violate a common operator while a sampling-aware mean test accepts its limiting control | Independent `node-1/32/512` operator: 200 each; `inference_mean_evaluation.csv`, `mode=responses-first`, `samples=8192`, `method=uncertainty_aware`: directional-null 9, preconditioned-null 11, node 9, feedback-0.1 200 | Different steps and protocols; mean reciprocity is necessary, not a full PSD or unbiasedness certificate |
| A PSD restriction changes held-out prediction relative to a general linear map | `inference_prediction_summary.csv`, primary `feedback-0.1`: PSD median about 0.0278 versus linear about 8.4e-6 | Mixtures lie inside the fitted gradient span; actual PSD prediction requires an attained optimum and its stated denominator |
| Measured update simulation benefits from GPU batching | `inference_hardware_benchmark.csv`, float64, n=20: CPU/MIG ratios 54.6 GD, 56.2 feedback, 90.2 node | One B200 MIG 2g.45gb slice versus one CPU core; synchronized bank timings, not end-to-end scientific campaign speedups |
| The primary detection contrast persists in CUDA retraining | `inference_gpu_evaluation.csv`, step 1000, noise 0: feedback-0.1 operator 200, Pearson 11; operator null counts GD 12, preconditioned 10 | Original CPU cutoffs, no recalibration; same initialization/data identities but different training RNG trajectories, analyzed separately |
| CUDA agrees numerically at identical saved CPU states | `inference_backend_validation.csv`: maximum relative response error 1.6454434e-12, maximum absolute score difference 1.9504664e-13, zero operator decision disagreements | 200 saved networks, three checkpoints, four deterministic rules; 2400 banks are repeated measures, not independent networks |

## Mathematical and statistical boundaries

For exact observations, the PSD interpolation condition is symmetry and
positivity of `S=G^T V` together with `ker(S)=ker(V)`. The stated construction
`K=V S^+ V^T` supplies a PSD interpolant when those conditions hold. Responses
outside `span(G)` are retained. Fixed PSD parameter preconditioning can induce
the same activity class as GD, and weight changes in the activity Jacobian's
nullspace are unobservable at the current input.

The continuous primary score tests necessary constraints. It does not robustly
implement the discontinuous exact nullspace condition. An arbitrarily small
PSD fitting residual can approach an unattained infimum. Fit nonattainment,
undefined predictions, and numerical failures cannot be added to the rejection
count. The manuscript theorem and PSD-fitting appendix make these distinctions.

With 200 finite calibration values, the 95% conformal order is 191. The maximum
of the two null cutoffs retains marginal coverage for either specified null;
Wilson evaluation intervals do not integrate calibration randomness. Secondary
conditions reuse networks or nested draws and do not provide independent
replications of the primary contrast. The single-direction control uses fewer
probes; its comparison does not isolate rank from measurement budget under noise.

## GPU evidence boundary and reproducibility

MIG job 9158857 supplies actual CUDA scientific validation and precision/timing
measurements. Job 9160192 subsequently passed the same 18 checks, executed the
same-state deterministic comparison, and completed all ten primary CUDA shards.
The strict evaluator verified all 432 conditions with the original CPU cutoffs.
The noisy secondary CUDA comparison detects feedback-0.1 in 74/200 and rejects
node-shared-1 in 177/200, preserving the observed calibration limitation.
Retraining uses a different device RNG stream; these results are distinct from
the numerical comparison at saved CPU states. The full-device timing job remains
queued and supplies no evidence yet. All completed GPU results name the MIG slice.

The first CPU benchmark predates the current operator implementation. New CPU
job 9169837 removes that source mismatch. The hardware summary verifies matched
sources, workloads, seeds, runtime locks, and passed checks before writing its
table. Small original reports and Slurm allocation records are versioned, so
`python tools/summarize_inference_hardware.py` reproduces this evidence without
another allocation. Operator timings concern a single compatibility call and
the generic 500-iteration helper, while the scientific prediction study uses
the analytic span reduction. There is one measured allocation per backend;
queue waits and timing repetitions do not establish allocation-level uncertainty.

The archived abstract and separate COSYNE work remain outside these revisions.
The existing related-work audit bounds the novelty claim to the calibrated
shared-constraint protocol and its measured limits.
