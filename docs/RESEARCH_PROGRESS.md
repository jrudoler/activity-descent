# Learning rule inference research progress

Updated 2026-10-19. The goal is active. The main CPU study, conditional-mean
study, five publication figures, supporting tables, and revised compiled draft
are complete. Actual B200 validation, benchmarking, and replication remain
queued; they must succeed before the full goal can be completed. No permission
or scientific choice is currently required from the user.

## Completed evidence

The analysis and manuscript are separate repositories on
`codex/learning-rule-inference`. The main experiment was locked at analysis
revision `69021a5`; the mean-response protocol at `7bde9b9`. The current workflow
and CUDA comparison entrypoints were first committed at `8a9a8e3`; the added
development controls and asset workflow are at `bd2ed9a`. Scientific source
hashes are retained in the result manifests. Changes to publication code do not
alter the frozen experiment implementations.

The runtime is Python 3.14.0 and PyTorch 2.11.0+cu128, with the official CUDA 12.8
wheel index explicitly locked in `pyproject.toml` and `uv.lock`. Eighteen CPU
scientific checks pass. They cover independent NumPy Jacobians and backprop,
autograd and finite differences, restored-state steps and Taylor scaling,
preconditioning, own/upstream parameter masks, exact PSD interpolation and
nullspaces, isotropic PSD Procrustes and nonattainment, unsupported prediction,
constant bank scaling, shared perturbations, explicit-versus-factored updates,
and undefined/tiny-value handling.

| Milestone | Current status |
| --- | --- |
| M0 Setup and GPU validation | CPU checks complete; actual B200 checks and benchmarks queued |
| M1 Layerwise diagnostics | Original 20-run classification simulation reproduced; pooled correlations match the untouched original to 1e-12; hidden/output statistics separated |
| M2 Bounded mechanism checks | Matched-state task/synthetic layer losses, step sizes 1e-5 through 1e-2, parameter freezing, signed coordinate arrays, exact shared PSD, and raw/matched scales executed |
| M3 Identification | Exact conditions, rank/nullspace audit, empirical cutoffs, held-out prediction, nonattainment, tolerance checks, and ridge sensitivity executed |
| M4 Rule comparison | 3,200-network main study and 400-network conditional-mean analysis complete on CPU, with independent calibration/evaluation and uncertainty |
| M5 Paper and workflow | Revised full draft compiles with resolved references; four figures and generated tables included; GPU evidence and final cross-repository audit remain |

## Main held-out results

Eight profiles each have 200 calibration and 200 evaluation networks. Candidate
rules share restored checkpoints at steps 0, 200, and 1000. The default observes
one complete width-64 hidden layer with a fixed eight-output known decoder.
The primary comparison was locked before evaluation data generation. No training
failures occurred. All 160 shards and their raw arrays passed strict identity,
configuration, scientific-source, completeness, and checksum checks.

At the primary GD-trained 1000-step checkpoint, without observation noise:

| Rule | Shared operator | Pearson | One direction |
| --- | ---: | ---: | ---: |
| GD | 11/200 | 6/200 | 4/200 |
| Fixed diagonal PSD preconditioning | 11/200 | 8/200 | 2/200 |
| Fixed feedback, mixture 0.1 | 200/200 | 6/200 | 2/200 |
| Fixed feedback, mixture 1 | 200/200 | 200/200 | 103/200 |
| Independent node R=1 | 200/200 | 200/200 | 200/200 |
| Independent node R=32 | 200/200 | 152/200 | 200/200 |
| Independent node R=512 | 200/200 | 9/200 | 200/200 |
| Shared node R=1 | 24/200 | 200/200 | 7/200 |
| Shared node R=512 | 19/200 | 9/200 | 6/200 |

The primary paired operator-minus-Pearson detection improvement is 97.0
percentage points, with conservative 95% interval [90.5, 98.8]. Intervals are
conditional on the realized calibration sample and are per condition, not
simultaneous across the secondary grid. Cutoffs are the maximum of separate
split-conformal GD and P-GD cutoffs; this covers the two specified exchangeable
null distributions marginally, not all possible PSD update distributions.

Depth three reduces 10% feedback detection to 30/200; preconditioned-trained
checkpoints give 155/200. At 1% GD-response-RMS noise, primary 10% feedback
detection falls to 69/200. Shared R=1 rejection rises to 176/200: its response
can have lower signal than GD despite parameter-norm matching. Even noiseless
shared-node rejection exceeds nominal calibration. These results are explicit
limits of GD/P-GD nuisance calibration, not evidence against the algebraic PSD
equivalence. Do not claim uniform false-rejection control over the PSD class.

The supported prediction comparison fits only the +/- output basis and predicts
16 reserved mixtures within its span. Primary median relative PSD prediction
errors are 9.8e-6 for GD, 1.0e-5 for P-GD, and 0.0278 for 10% feedback. The
unrestricted linear error for 10% feedback is 8.4e-6. The best nonnegative
diagonal GD fit has error 0.186. Actual PSD prediction is undefined when the fit
infimum is unattained: primary fully random feedback and independent R=1 node
have no attained optimum; shared R=1 has 191/200 attained fits. Fit flags are
not additional scientific rejections.

Loss matching is a prespecified secondary selection near loss 0.4. Eligible
held-out counts are 200 primary, 199 P-trained, 197 FA-trained, 200 NP-trained,
149 rank-two, 195 rank-sixteen, 198 width-32, and 176 depth-three. The CSV retains
actual selected steps/losses and exclusions, rather than claiming exact equality.

## Conditional means and finite steps

The separate mean protocol reuses the primary saved checkpoints with disjoint
200-run calibration and evaluation sets, eta=1e-5, sigma=1e-4, constant multiplier
one, and nested R=8 through 8192. It calibrates a studentized reciprocity statistic
against exactly unbiased directional GD and its fixed preconditioned version.
Both orders of averaging are retained: independent restored responses first,
or weight directions first followed by one step.

At R=8192, restored-response mean rejection is 9/200 for directional GD, 11/200
for its preconditioned version, and 9/200 for physical node perturbation. The
10% biased mean is detected in 200/200 runs, versus 8/200 at R=8. Applying the
naive deterministic cutoff rejects all 200 unbiased controls at every studied
R. Median averaged-direction response errors at R=8192 are 4.59% for directional
GD and 4.60% for physical node perturbation. These necessary reciprocity tests
do not prove a full PSD model or uniquely identify a stochastic algorithm.

The post-evaluation sensitivity experiment uses only 20 development checkpoints
and is labeled exploratory. An exact shared PSD construction has median
first-order operator residual about 8e-16; its finite-step score decreases from
5.0e-6 at eta=1e-4 to 5.0e-8 at eta=1e-6. Thus finite-step incompatibility occurs
even without estimator bias. This explains a mechanism, not every individual
shared-control rejection or a revised main cutoff.

## Additional audited development controls

The completion audit found planned task/synthetic-loss and raw-scale controls
missing from the first full draft. Jobs 9161103 and 9161259 executed these
follow-ups on the existing 20 development checkpoints; they do not alter the
locked primary test or its calibration/evaluation splits.

At the same width-64 checkpoint and input, task and synthetic complete-layer
gradients are norm matched and held fixed over eta=1e-5,1e-4,1e-3,1e-2, with
all/own/upstream parameter masks. At eta=1e-4, median full prediction errors
are 2.10e-6 and 1.84e-6, while diagonal errors remain 0.192 and 0.185. The raw
arrays retain signed self/interference and individual coordinate loss
contributions for hidden layers and output, including both gate directions.
The synthetic layer loss is an oracle mechanism control outside the actual
decoder's accessible gradient span.

Same-draw raw and magnitude-matched banks verify exact replay of the common
scalar. First-order operator-score differences are at most 4.45e-16. Finite-step
shared R=1 node scores differ: median raw 5.70e-5 versus matched 5.82e-6. The 10%
feedback score is about 0.00905 for either scale. This preserves the feedback
contrast while quantifying scale-dependent errors in equivalence controls.

PSD-fit sensitivity uses four rules, three pseudoinverse tolerances, and three
positive ridge fractions. The unregularized attained counts and prediction
medians are unchanged across tolerances 1e-7 through 1e-11 in these 20 networks.
All 240 positive-ridge fits meet the recorded stationarity threshold. Three
independent diagonal analytic cases validate the auxiliary solver to maximum
absolute error below 2.1e-11. Ridge biases prediction even for GD: its median
reserved-probe error is 1.14e-5 without ridge and 0.00536 at lambda/c=0.001.
These are supporting optimizer diagnostics, never additional rejections.

Results are in `inference_mechanism_controls.csv`,
`inference_scale_sensitivity.csv`, and `inference_fit_sensitivity.csv` under
`results/data/`. The fifth figure and regularization table appear in the
manuscript's additional-development-controls appendix. The requirement audit is
in `docs/COMPLETION_AUDIT.md`.


## Jobs and scheduling

The user's Wharton PhD GPU group/partition limit is **three concurrent GPU
jobs**. Each request here uses one GPU. This is a concurrent-job limit, not a
claim about the cluster's total hardware. CPU shards were selected because they
became available and completed while B200 requests remained queued.

| Job | Work | Verified state |
| --- | --- | --- |
| 9157408 | Current-runtime CPU preflight and benchmark | Completed 0:0 |
| 9157417, 9157426, 9157428, 9157431 | Initial rank-two/rank-eight development and mechanism pilots | Completed 0:0 |
| 9157970--9157973 | Matched development training-source profiles | Completed 0:0 |
| 9158474--9158477 | Matched development rank/width/depth profiles | Completed 0:0 |
| 9158478 | Original classification continuity, 20 seeds | Completed 0:0 |
| 9158833 | Locked main CPU array, 160 shards, concurrency 12 | All 160 completed 0:0 |
| 9159329 | Locked mean CPU array, 20 shards, concurrency 6 | All 20 completed 0:0 |
| 9159778 | Development-only shared-PSD sensitivity | Completed 0:0 |
| 9161103 | Development task/synthetic losses and same-draw raw/matched banks | Completed 0:0 in 31s |
| 9161259 | Development PSD tolerance/ridge sensitivity and analytic solver checks | Completed 0:0 in 11s |
| 9156748 | Full B200 preflight | Pending; preserve handle |
| 9158857 | B200 MIG45 preflight, 6 CPUs, 48G, 10 min | Pending; preserve handle |
| 9160192 | B200 MIG45 validation, exact-checkpoint comparison, primary CUDA replication | Pending afterok:9158857; preserve handle |

9159779 was canceled before starting and replaced by 9160192 to add the explicit
same-checkpoint CPU/CUDA comparison. A retraining run alone is not that check:
CPU and CUDA random-number streams differ. Do not restart the current jobs
merely because the queue estimate changes. Do not infer GPU execution from a
queued request. The full-GPU and MIG preflights measure distinct resource shapes.

Both job wrappers run immutable Git source snapshots. CPU shards use one CPU
and 5632M per CPU on genoa-std-mem. Full B200 requests require 28 CPUs and 224G.
MIG45 accepts six CPUs and 48G despite the filter's displayed 56G table entry.

## Artifacts and reproduction

- `results/data/inference_evaluation.csv`: 4,608 held-out method/rule conditions.
- `results/data/inference_paired_comparison.csv`: paired improvement and intervals.
- `results/data/inference_loss_matched_stages.csv`: actual selection and exclusions.
- `results/data/inference_prediction_summary.csv`: support, attainment, and errors.
- `results/data/inference_mean_evaluation.csv`, `inference_mean_accuracy.csv`: mean decisions and convergence.
- `results/data/inference_original_baseline.csv`, `inference_layerwise_summary.csv`: original-task continuity and diagnostics.
- Matching provenance JSONs retain raw file identities, source hashes, and job metadata.
- Five `results/figures/fig_inference_*.pdf`/`.png` figures and four `results/tables/inference_*.tex` files supply the manuscript.
- `paper/activity_dynamics.tex` is the revised full draft; its PDF compiles using pinned Tectonic 0.17.0 with resolved references. `paper/generated/build_provenance.json` records source/asset/PDF hashes.

The parent `Snakefile` has explicit CPU generation, aggregation, publication,
asset-copy, and compilation rules; the original figures remain under
`legacy_figures`. See the root README for allocation-aware execution and the
frozen-summary artifact rebuild. The nested paper still compiles independently
using its tracked generated assets. The archive source and COSYNE files retain
the handoff revision's hashes.

## Remaining work for the active goal

1. Observe successful actual B200 preflight: CUDA scientific checks, float64 and
   float32 benchmarks at batch sizes 1/20/64, hardware/driver/runtime/memory logs.
2. Let 9160192 execute after preflight. Verify deterministic CUDA responses
   against the same saved CPU checkpoints; inspect numerical differences and
   decision disagreements. Evaluate the CUDA retraining replication against the
   fixed CPU calibration cutoffs, identifying its different RNG trajectory.
   The prepared entrypoint is `PYTHONPATH=. .venv/bin/python -P -m
   analysis.learning_rule_inference.evaluate_gpu`; it refuses missing shards,
   non-CUDA or non-B200 provenance, or changed scientific sources.
3. Compare measured CPU/GPU throughput including queue time. Record the chosen
   scheduling tradeoff and any meaningful numerical limitations.
4. Update the manuscript's pending-GPU implementation paragraph with measured
   evidence, refresh the build/provenance, and finish the cross-repository audit.
5. Mark the goal complete only after these executed results and final manuscript
   checks succeed. The completed CPU evidence and compiled draft alone do not
   satisfy the goal's actual-GPU requirement.
