# Gradient Descent in Weight Space Is Kernel Descent in Activity Space

A short paper showing that gradient descent on network weights induces **kernel descent** in the space of neural activities, governed by a neural-tangent-kernel-style Gram matrix on internal neurons.

**Key result:** Weight gradient descent induces kernel-mediated activity changes. Neuronwise descent is guaranteed when its self-term dominates gradient-weighted interference and finite-step error; matrix diagonal dominance alone is insufficient. See Appendix C of the manuscript.

## Reproduce the results

Install `uv`, then run the complete dependency graph from the repository root:

The manuscript is an independent Git repository checked out at `paper/`. New
analysis clones must also obtain the manuscript repository at that path before
running the complete workflow:

```bash
git clone git@github.com:jrudoler/activity-descent-manuscript.git paper
```

See `paper/HANDOFF.md` for research status and cluster setup.

```bash
uv run snakemake --cores 1
```

This regenerates seeded simulation outputs, publication-ready figures,
manuscript assets, and the compiled paper. The Snakemake dependency graph is
the authoritative provenance record.
To rebuild one step, use `uv run snakemake --cores 1 --forcerun <rule_name>`.

## Repository layout

- [`Snakefile`](Snakefile) — the executable workflow and artifact lineage
- [`analysis/`](analysis/) — one explicit entrypoint per workflow step
- [`data/generated/`](data/generated/) — machine-generated intermediates
- [`results/`](results/) — final scientific tables and figures
- [`paper/activity_dynamics.tex`](paper/activity_dynamics.tex) — manuscript source
- `paper/activity_dynamics.pdf` — locally compiled paper
- [`index.html`](index.html) — static interactive demo for GitHub Pages

## Manuscript collaboration

`paper/` has its own Git history and is ignored by this analysis repository.
Commit manuscript and figure-asset changes from inside `paper/` (or use
`git -C paper ...`). See `paper/README.md` for standalone compilation and Overleaf
setup. The analysis repository's existing history retains the earlier paper
versions.

`collect_paper_assets` copies analysis figures into `paper/generated/figures/`;
review and commit those updates in the manuscript repository after regeneration.
The complete workflow compiles the paper using latexmk from its own directory.

## Interactive Demo

**[Launch the interactive demo](https://koerding.github.io/ADescent/)** — the runnable demo is the static browser page in [`index.html`](index.html), served directly by GitHub Pages with no install or build step. Adjust width, depth, and learning rate to see how the kernel and diagonal approximation behave.
