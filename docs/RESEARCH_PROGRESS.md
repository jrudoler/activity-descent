# Learning rule inference research progress

Updated 2026-10-19. The goal remains active. The previous goal turn made progress
by saving the research plan and establishing the goal; the execution recorded
here adds implemented methods, scientific validation, and completed development
experiments. Confirmatory evaluation and the revised manuscript remain ahead.

## Current implementation and evidence

The analysis branch is `codex/learning-rule-inference`. The scientific core and
completed pilot inputs use revision `a2c495e`. The manuscript repository remains
unmodified at its handoff revision. The runtime is Python 3.14 with optional
PyTorch 2.11.0+cu128; Linux wheels come from the explicit official CUDA 12.8 index
in `pyproject.toml` and `uv.lock`.

Eleven CPU checks pass, including an independent NumPy forward/backprop/Jacobian
reference, automatic differentiation, finite differences, preconditioned
kernels, restored-state probes, own/upstream decomposition, output diagnostics,
exact PSD interpolation and nullspace counterexamples, zero-gradient handling,
and shared-direction perturbation. The theory and primary-paper audit are in
`docs/learning_rule_identification.md` and `docs/related_work_audit.md`.

Completed CPU jobs:

| Job | Work | Scheduler outcome |
| --- | --- | --- |
| 9156749 | Initial PyTorch 2.14 CPU validation and benchmark | Completed 0:0 in 2m20s; superseded runtime for comparison |
| 9157408 | Current runtime CPU validation and benchmark | Completed 0:0 in 54s; all 11 checks passed |
| 9157417 | Two outputs, initial checkpoints, 20 development seeds | Completed 0:0 in 35s |
| 9157426 | Eight outputs, initial checkpoints, 20 development seeds | Completed 0:0 in 39s |
| 9157428 | Two outputs, after 1000 GD training steps, 20 seeds | Completed 0:0 in 34s |
| 9157431 | Eight outputs, after 1000 GD training steps, 20 seeds | Completed 0:0 in 42s |

GPU preflight job `9156748` was verified pending for priority after its runtime
maintenance hold was released. It is the original submitted job, not a replacement.
Recheck `squeue`/`sacct` and its log before taking further action. No B200 execution
or GPU speedup has yet been demonstrated. This first job predates the snapshot
shell wrapper, so its source can still follow the checkout until it starts. New
jobs run committed code from per-job archives under
`data/generated/inference/source-JOBID/`.

Betty requires 28 CPUs and 224 GB host memory per B200 allocation. CPU requests
must use 5632 MB per CPU on `genoa-std-mem`. The Wharton GPU job limit must be
accounted for when scheduling; all submitted GPU jobs request one GPU. CPU
pilot shards use one CPU each and can run concurrently.

## Development results

The four pilots each contain 27 conditions: three learning rates, GD, fixed
diagonal PSD-preconditioned GD, fixed random feedback, and layerwise node
perturbation at 1, 32, and 512 repeats with independent or shared directions.
Each condition has 20 network realizations. The same seed identities are used
across tasks/stages; these are paired development conditions, not 80 independent
evaluation seeds. All checkpoints and raw gradient/response arrays were saved,
and their recorded SHA256 values were checked.

`results/data/inference_pilot_summary.csv` has 108 condition summaries.
`results/figures/fig_inference_pilot.pdf` and `.png` show alignment and reciprocity.
The ranges are descriptive development quantiles, not confidence intervals.

For eight outputs at initialization and eta=1e-4, median reciprocity residuals
were about 3.35e-6 for GD, 4.45e-6 for PSD-preconditioned GD, and 0.118 for pure
random feedback. Independent node perturbation gave approximately 0.119, 0.055,
and 0.015 at 1, 32, and 512 repeats. Shared directions gave residuals near the GD
scale even when cosine alignment was much lower. This supports retaining the
shared-direction equivalence control; thresholds and detection power are not yet
calibrated.

The GD reciprocity residual decreased approximately tenfold for each tenfold
step reduction from 1e-3 to 1e-5, consistent with finite-step error. The pilot
also saves parameter-freezing, interference, and gate-change diagnostics for
each layer and the output. Kernels in these diagnostics are simulation oracles;
the inference functions use only gradients and responses.

The current CPU benchmark completes a one-network deterministic bank in about
2.5 ms. At width 64 and batch size 64, a float64 32-repeat node bank takes about
1.07s and a 500-iteration PSD fit about 6.4s. Float32 activity-response errors
reach roughly 6.8e-4 relative to float64 at eta=1e-4, so retain float64 for the
inference study until decision-level precision checks justify a change.

## Issues to resolve before confirmatory evaluation

1. The generic projected-gradient PSD fit sometimes fails its stationarity
   tolerance, including some compatible or nearly compatible null conditions.
   Do not count this as scientific rejection. The spanning +/- output probes
   give an isotropic gradient design, permitting a reduced semi-analytical
   PSD Procrustes fit. Use the Gillis-Sharma reduction, distinguish unattained
   infima, and test the response outside the probe span. Retain the algebraic
   compatibility tests independently of the fitting implementation.
2. Pure random feedback has near-zero or negative alignment and is an easy
   alternative. Add partially aligned fixed feedback at matched checkpoints to
   test whether joint constraints improve on alignment for a harder alternative.
   Use one positive magnitude-matching scale per rule/checkpoint, estimated from
   a separate probe bank and fixed across evaluation probes. The present pilots
   use raw scales and are not the final matched comparison.
3. Finalize and lock the primary statistic using development data, then run
   independent calibration and evaluation seeds. The planned initial targets
   are 200 runs each at a nominal 5% false rejection rate, with both GD and
   preconditioned-GD null controls. Multiple probes within a run are dependent.
4. A finite Monte Carlo mean can still fail a very precise deterministic-null
   threshold. Mean-rule inference needs sampling uncertainty, such as replicate
   blocks and a bootstrap of reciprocity contrasts; do not interpret finite-R
   rejection as biased expectation. Separately compare averaged weight directions
   before a step and averages of responses after restored single steps.
5. The existing training in the pilots is GD only. Add checkpoints trained by
   each rule, learning curves, and overlapping-loss comparisons. Preserve failed
   learning outcomes and report stochastic single-update/mean equivalences.
6. Complete the original task's layerwise continuity checks and targeted
   architecture/rank controls. The current two-output pilot uses teacher
   regression on the bar inputs, not a replication of the old pooled-neuron,
   trainable-readout bar-classification figures.
7. Complete GPU validation and benchmark, route large perturbation batches to
   B200 when beneficial, and run remaining CPU shards in parallel. Do not infer
   job termination from a polling timeout; inspect the exact existing handle.
8. Add workflow entrypoints and final result verification, revise the manuscript
   using measured results, collect assets with analysis provenance, and compile
   with references resolved. Preserve the archived abstract and COSYNE work.

## Reproduce and inspect

Use the documented `uv sync --frozen --extra gpu` environment. For current job
states, inspect the IDs above. Completed local raw results and scheduler logs are
under the ignored `data/generated/inference/` directory; publication summaries
and the development figure are tracked under `results/`. Their provenance JSON
records source revisions and implementation hashes. Recreate the pilot summary:

```bash
.venv/bin/python -m analysis.summarize_inference.run \
  data/generated/inference/pilot-q2-initial.json \
  data/generated/inference/pilot-q8-initial.json \
  data/generated/inference/pilot-q2-trained.json \
  data/generated/inference/pilot-q8-trained.json
```
