# Inferring learning rules from activity responses

Controlled target interventions test whether learning-induced activity changes
share an unknown symmetric positive-semidefinite operator. The study compares
this constraint with scalar alignment, measures observational equivalences, and
separates finite stochastic update banks from conditional mean responses.

The completed CPU campaign contains 3,200 independent network realizations
across eight profiles, with separate calibration and evaluation splits. The
primary operator test detects a partially biased fixed-feedback construction in
200/200 held-out runs, compared with 6/200 for Pearson alignment. This result is
bounded by the observation model, finite-step effects, and the chosen null
calibration. See the [progress record](docs/RESEARCH_PROGRESS.md),
[research plan](RESEARCH_PLAN.md), [primary protocol](docs/CONFIRMATORY_PROTOCOL.md),
[mean-response protocol](docs/MEAN_RESPONSE_PROTOCOL.md),
[supporting development controls](docs/DEVELOPMENT_CONTROLS.md),
[derivation](docs/learning_rule_identification.md), and
[related-work audit](docs/related_work_audit.md).

## Setup and checks

The manuscript is an independent repository at `paper/`, not a submodule. A new
analysis checkout also needs the corresponding manuscript checkout:

```bash
git clone git@github.com:jrudoler/activity-descent-manuscript.git paper
uv sync --frozen --extra gpu
PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.validate --device cpu
```

Use the corresponding research branch in both repositories. The optional `gpu`
extra installs PyTorch for both CPU and CUDA execution. The locked Linux runtime
is Python 3.14 and PyTorch 2.11.0+cu128. CUDA requests fail if no GPU is available.
The 18 scientific checks compare independent NumPy Jacobians, differentiation,
restored updates, PSD examples, and perturbation implementations.

## Betty experiments

Submit from the repository root after committing the scientific code. Jobs run
from immutable per-job source archives, retain source/configuration/array hashes,
and verify completed shards before resuming. Run substantial work in allocations.

```bash
mkdir -p data/generated/inference
sbatch --cpus-per-task=1 cluster/inference_cpu.sbatch validate
sbatch cluster/inference_gpu.sbatch preflight
sbatch --array=0-159%12 --cpus-per-task=1 cluster/inference_cpu.sbatch campaign-array \
  --configuration configs/inference/confirmatory_v1.json \
  --output-dir data/generated/inference/confirmatory-v1
```

After the primary profile's 20 calibration/evaluation shards finish:

```bash
sbatch --array=0-19%6 --cpus-per-task=1 cluster/inference_cpu.sbatch mean-array \
  --configuration configs/inference/mean_v1.json \
  --input-dir data/generated/inference/confirmatory-v1 \
  --output-dir data/generated/inference/mean-v1
```

The Wharton PhD GPU group/partition permits **three concurrent GPU jobs**. Each
GPU job here requests one GPU. CPU arrays run concurrently with that queue;
choose CPU/GPU work using measured execution time plus queue delay. A B200 MIG
preflight can use `--partition=b200-mig45 --cpus-per-task=6 --mem=48G`; full B200
jobs use the script's 28 CPUs and 224G. The scheduler's displayed MIG45 table says
56G, but the accepted allocation is 48G. Preserve existing job handles and check
the progress record before submitting overlapping work.

The `replicate-primary` GPU mode validates CUDA, compares four deterministic
probe banks at the exact saved CPU checkpoints, then repeats the primary
200-network evaluation on CUDA. Device random-number streams differ, so CUDA
retraining is a distributional replication, not a paired numerical check. GPU
execution remains pending in the current draft.

After all ten CUDA replication shards complete, evaluate them with the original
CPU cutoffs, without recalibration:

```bash
PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.evaluate_gpu
```

## Reproduce summaries, figures, and the manuscript

The Snakemake graph contains CPU shard generation, strict aggregation, the
original-task continuity check, bounded development diagnostics, publication
figures, asset collection, and compilation. Once raw experiments are present:

```bash
XDG_CACHE_HOME=/tmp/activity-descent-cache .venv/bin/snakemake --cores 1
```

For a fresh experiment directory this graph can generate all CPU experiments;
run it inside a CPU allocation with `--cores` matched to the allocation. The
Slurm arrays above usually make better use of independently available CPUs.
The main evaluator refuses incomplete splits, duplicate identities, changed
scientific sources, or corrupted arrays. It retains undefined results and fit
nonattainment. Final CSVs, figures, and provenance are versioned in `results/`;
large checkpoints and job logs under `data/generated/inference/` are ignored.

The bounded development follow-ups have explicit `mechanism-controls` and
`fit-sensitivity` CPU job modes, as well as Snakemake rules. They compare task
and synthetic layer losses, raw and matched scales, and PSD-fit regularization
without changing the locked main evaluation.

To update publication artifacts from already verified, frozen summaries without
re-running experiments:

```bash
XDG_CACHE_HOME=/tmp/activity-descent-cache .venv/bin/snakemake collect_inference_assets \
  --cores 1 --allowed-rules inference_publication collect_inference_assets
python tools/build_paper.py
```

Compilation uses latexmk when available, otherwise the pinned Tectonic binary:

```bash
python tools/install_tectonic.py
python tools/build_paper.py --compiler tectonic
```

The installer verifies the official release archive's SHA-256. The build fails
on unresolved references and records source, figure, compiler, and PDF hashes
in `paper/generated/build_provenance.json`. Compiled PDFs are ignored by Git.
`paper/activity_dynamics.pdf` is the full draft; the preserved extended abstract
is `paper/archive/neurreps_extended_abstract.tex`. The separate COSYNE work is
unchanged.

## Original activity-descent study

The original figure entrypoints and browser demonstration remain available:

```bash
XDG_CACHE_HOME=/tmp/activity-descent-cache .venv/bin/snakemake legacy_figures --cores 1
```

[Interactive demo](https://koerding.github.io/ADescent/) explores kernel and
diagonal approximations. It illustrates the original mechanism and is not the
new inference experiment. Manuscript changes and analysis changes have separate
Git histories; review and commit each repository independently.
