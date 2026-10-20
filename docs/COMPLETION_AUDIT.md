# Learning-rule inference goal audit

Audited 2026-10-20 against the active objective, `RESEARCH_PLAN.md`, the user's
CPU-parallelism authorization, and the clarified limit of three concurrent GPU
jobs. B200 MIG job 9160192 completed all 18 checks, exact-checkpoint numerical
agreement, and the primary CUDA retraining replication. The strict evaluator
verified ten shards and 432 conditions with the original CPU cutoffs. The goal
is **not complete** while the separate full-device timing preflight 9156748
remains queued and the corresponding final hardware audit is outstanding.

The earlier allocation blocker persisted across three consecutive goal turns,
and the goal was marked blocked. The replication subsequently completed on
October 19 from 22:33:55 to 22:36:28 Eastern. Only 9156748 remains pending
Priority. Its existing handle is retained; no full-device execution is claimed.

The previous goal turn made concrete progress by completing the CPU study and
compiled draft. This continuation identified and executed omitted development
controls and fit sensitivity, rather than treating a plausible draft as proof
that every planned control had been completed.

| Requirement | Authoritative evidence in this checkout | Audit result |
| --- | --- | --- |
| Focus on inference classes, with diagnostics and bounded mechanisms | Main manuscript sections 3--6, protocol/configuration files, separate supporting appendix | CPU evidence complete; final hardware evidence pending |
| Known loss/readout, complete layer, restored states | `campaign.py` functional updates and saved checkpoints/probe banks; `core.py` frozen decoder | Implemented and exercised |
| Inference receives G and V without upstream kernels | `campaign.score_observation` and `operators.py` inputs; kernels confined to simulator diagnostics | Verified by source scope and analytic checks |
| Shared PSD compatibility, linearity, rank/nullspace and observational limits | `docs/learning_rule_identification.md`; CPU scientific checks; manuscript theorem and singular counterexample | Verified in stated first-order noiseless scope |
| Retain responses outside the gradient span | Extension checks and explicit PSD example with an outside-span response | Verified; no unsupported extrapolation claimed |
| PSD fitting, convergence, conditioning and nonattainment | Semi-analytic main fit; `inference_prediction_summary.csv`; analytic solver checks | Executed; unattained optima are not scientific rejections |
| Numerical tolerance and regularization sensitivity | Job 9161259; `inference_fit_sensitivity.csv`; three analytic ridge checks; 240 converged positive-ridge fits | Executed on development data only |
| Original two-output continuity and per-layer/output diagnostics | Job 9158478; `inference_original_baseline.csv` and provenance; original source SHA matches current source | 20 original runs reproduced to 1e-12 for pooled correlations |
| Sign, cosine/Pearson, self/interference, gate and degeneracy diagnostics | `diagnostics.py`, original layer summaries, saved signed arrays from 9161103; 18 CPU checks | Executed with undefined values/counts retained |
| Matched-state step and parameter-freezing study | Original pilots plus job 9161103, eta 1e-5--1e-2, all/own/upstream masks | Executed; no retraining per diagnostic step |
| Task versus synthetic complete-layer gradients | Job 9161103 and `inference_mechanism_controls.csv`; explicitly labeled oracle mechanism control | Executed at the same development checkpoints and inputs |
| Adequate accessible directions, limited-rank controls and reserved prediction | Fixed q=2/8/16 readouts, actual gradient ranks, +/- basis, 16 reserved mixtures; one-direction comparison | Executed within the accessible span |
| Four rules at identical states and rule-trained checkpoints | Main locked config; 160 completed CPU shards at steps 0/200/1000 for all eight profiles | Executed; no failed training states dropped |
| Equal-step and overlapping-loss comparisons | `inference_loss_matched_stages.csv`; actual losses, selected steps and exclusions | Executed; not represented as exact loss equality |
| One positive multiplier per bank from a separate probe bank | `matched_observation`, saved scale controls, matching validation; same-draw replay in 9161103 | Verified; no per-probe normalization |
| Raw-scale sensitivity | Job 9161103 and `inference_scale_sensitivity.csv` | Executed; finite-step effects reported separately from first-order homogeneity |
| Node estimator specification and small-perturbation limit | `core.directions`, Dalm v4 variant description, independent estimator checks and sigma sensitivity | Verified for the implemented layerwise central estimator, not every node variant |
| Independent versus shared perturbation equivalence | Main shared R=1/512 conditions; exact shared-PSD sensitivity and proof | Executed; elevated finite-step/noise rejection is reported |
| Conditional means, sampling uncertainty and both averaging orders | 20 completed mean shards, `mean_v1.json`, mean evaluation/accuracy CSVs, naive-cutoff comparison | Executed with 200 calibration and 200 evaluation networks |
| Independent development/calibration/evaluation identities | Locked split ranges, strict evaluator identity checks and manifests | Verified; repeated measurements are not counted as independent networks |
| FPR, power and baseline comparisons with run-level uncertainty | 4,608 held-out conditions, paired comparisons, Wilson intervals and paired sensitivity | Executed; coverage limited to specified null families and per-condition claims |
| Noise, depth and failed-identification conditions reported | Robustness figure; shared-control false rejection; depth-three power and loss-selection exclusions | Included; no uniform PSD-family or biological-validity claim |
| Full prior-work and novelty audit | `docs/related_work_audit.md`, primary citations and manuscript background | Complete for the closest stated comparisons; no broad first-inference/kernel/optimizer claim |
| CPU reference and analytic validation | `validation-final-cpu.json`: 18 checks, unchanged checked modules; independent NumPy reference | Passed on CPU |
| Actual B200/CUDA scientific validation and precision benchmarks | Completed MIG45 job 9158857: all 18 checks pass, float32/64 timings at n=1/20/64, actual GPU/driver/runtime/memory metadata | Achieved on MIG slice; full-device 9156748 remains pending |
| Same saved inputs/checkpoints CPU versus CUDA | Executed 9160192, `inference_backend_validation.csv`, 200 saved networks x 3 stages x 4 deterministic rules | Achieved; max relative response error 1.65e-12, zero operator decision disagreements |
| Executed B200 comparison with provenance | Completed 9160192 and verified `inference_gpu_evaluation.csv`, all ten shards and 432 conditions | Achieved on B200 MIG; primary operator 200/200 versus Pearson 11/200; null counts 12/200 and 10/200 |
| Measured CPU/GPU throughput and scheduling tradeoff | Matched-source CPU job 9169837 and MIG45 job 9158857, 24 timing conditions, original archived reports and Slurm waits | Achieved for slice benchmark; scope distinguishes generic helper from main fit and timing from training |
| Claims checked against available evidence | `docs/CLAIM_EVIDENCE.md`, exact primary/noise/depth/mean/CUDA rows, theorem and calibration review | Completed CPU and MIG evidence distinguished; no pooling of device runs or broader null claims |
| Configurations, immutable jobs, workflow and final assets | `Snakefile`, Slurm modes, locked configs, strict manifests, five figures/six generated TeX assets, provenance-aware collector | CPU and scientific MIG outputs incorporated; full-device timing preflight pending |
| Compiled manuscript with resolved citations and inspected artifacts | Tectonic build and source/PDF hashes, supporting/hardware/CUDA appendices | Draft incorporates executed CUDA replication; final full-device statement and audit remain |
| Preserve archived abstract/COSYNE and separate histories | Independent paper Git repository; archive/COSYNE diff against 85d280b is empty | Verified; local commits in each repository |

The full-device `dgx-b200` preflight and MIG45 fallback are distinct resource
shapes. A MIG result must identify the slice and must not be described as a
full-device benchmark. CPU and CUDA retraining use different RNG streams;
only the saved-checkpoint comparison establishes paired numerical agreement.
The CUDA replication uses the already fixed CPU cutoffs, without tuning from
its evaluation outcomes. Its seed identities overlap the CPU evaluation, so
the two device samples are not pooled as 400 independent networks.

The remaining completion work is the executed full-device timing preflight,
its measured hardware statement in the manuscript, and the final cross-repository
audit. Keep its existing queue handle; do not restart work merely because a queue
estimate or observation timeout changes.
