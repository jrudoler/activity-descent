# Activity descent and staged inference research

The main manuscript is preserved at its original handoff version in the separate
`paper/` repository. New learning-rule inference results remain in local
Markdown notes and separate appendix assets until the user chooses promotion.

Start with [the research index](docs/RESEARCH_INDEX.md) and
[the concise results note](docs/INFERENCE_RESULTS.md). The
[research plan](RESEARCH_PLAN.md), [execution record](docs/RESEARCH_PROGRESS.md),
and [claim/evidence review](docs/CLAIM_EVIDENCE.md) retain technical detail.

## Setup and original manuscript

A new analysis checkout needs the independent manuscript repository:

```bash
git clone git@github.com:jrudoler/activity-descent-manuscript.git paper
uv sync --frozen --extra gpu
python tools/install_tectonic.py
python tools/build_paper.py
```

`paper/activity_dynamics.tex`, its bibliography, and original figures match the
handoff at manuscript commit `85d280b`. `paper/manuscript_baseline.json` records
their hashes. The builder checks them before compiling. With Tectonic, legacy
watermark units are adjusted in a temporary build copy; canonical files remain
unchanged. The default workflow builds only this preserved manuscript:

```bash
XDG_CACHE_HOME=/tmp/activity-descent-cache .venv/bin/snakemake --cores 1
```

The separate COSYNE work and archived abstract are preserved. The superseded
19-page inference draft is an archival snapshot under
`paper/research/archive/inference_draft_2026-10-20/`, not the main paper.

## Research artifacts

New figures, tables, and provenance are staged under
`paper/research/learning_rule_inference/generated/`. They are not automatically
included in the main manuscript. To refresh staging from verified results:

```bash
XDG_CACHE_HOME=/tmp/activity-descent-cache .venv/bin/snakemake collect_inference_assets --cores 1 --allowed-rules inference_publication inference_hardware collect_inference_assets
```

`inference_research` is the explicit full research-artifact target. Raw arrays
under `data/generated/inference/` are ignored; final CSVs, configurations,
provenance, and original small hardware reports are versioned. The evaluators
reject incomplete or mismatched source/configuration/seed identities and
corrupted arrays. Retain undefined values, fit nonattainment, and the stated
calibration limits.

The locked CPU study has 3,200 network realizations over eight profiles. The
primary operator detects the partially biased feedback construction in 200/200
held-out networks versus Pearson's 6/200. The primary CUDA replication gives
200/200 versus 11/200 using the original CPU cutoffs. These device evaluations
share initialization/data identities and are not pooled as 400 independent
networks. All 18 scientific checks pass on CPU, B200 MIG, and a full B200.

Re-evaluate the completed CUDA replication:

```bash
PYTHONPATH=. .venv/bin/python -P -m analysis.learning_rule_inference.evaluate_gpu
```

Reproduce the small archived hardware summaries without new allocations:

```bash
python tools/summarize_inference_hardware.py
python tools/summarize_inference_hardware.py --configuration configs/inference/hardware_b200_v1.json --output results/data/inference_hardware_b200.csv --table results/tables/inference_hardware_b200.tex
```

## Compute and original analyses

Betty GPU jobs use account `wharton-phd-students` and QoS
`wharton-phd-student-dgx-b200`. Respect the user's three-concurrent-GPU-job
limit; each job here uses one GPU. Full B200 requests use 28 host CPUs and 224G;
MIG45 accepts six CPUs and 48G. Independent CPU arrays use `genoa-std-mem` with
`wharton-phd-student-genoa`. Choose workloads using measured runtime and queue
delay. Existing experiment handles are recorded in the execution record.

The original simulation entrypoints remain available under `legacy_figures`.
Regenerated legacy plots are staged in `paper/research/legacy_figures/`, so this
workflow also preserves the original manuscript assets. The browser demo is at
https://koerding.github.io/ADescent/ . Keep analysis and manuscript Git histories
separate. Main-paper promotion requires an explicit user decision.
