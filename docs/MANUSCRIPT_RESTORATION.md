# Exact manuscript restoration and rendering check

Verified 2026-10-21 against the user-specified manuscript commit
`85d280b4c1576dc08564245477bf8cf83fed56de` in the separate `paper/` repository.

## What was wrong

The earlier restoration put the correct source and figures back, but the PDF
was compiled with Tectonic/XeTeX instead of the pdfLaTeX engine specified by
the source and original README. The builder also replaced the watermark's
`px` units with `bp` in a temporary input. That was not a faithful rendering
restoration: it used scalable Latin Modern fonts, rendered the watermark at
153 PDF points, and produced 11 pages.

The corrected build uses pdfLaTeX on an exact copy of the original source,
without any substitutions. It produces 9 pages. The original Computer Modern
font setup renders the watermark at approximately 24.787 PDF points. The
associated font-size substitution warning also occurs in a clean build of
the original commit; the source and its font configuration remain unchanged.

## Verification

- Compared all 30 original manuscript, bibliography, figure, template, archive,
  and COSYNE files directly against their Git blobs at the full commit above:
  no differences. Repository README and Git-ignore policy changes are separate.
- Extracted an independent complete source tree using `git archive` at that
  exact commit. Built both this tree and the current manuscript from fresh
  auxiliary files with the same pdfLaTeX toolchain.
- Compared every page as an RGB raster at 108 dpi: all 9 pages are
  pixel-for-pixel identical. Visually inspected the first page and all-page
  contact sheet, including the figure, references, and appendices.
- Both builds resolve citations and cross-references with no overfull boxes.
- Verified that temporary fixtures with a changed main source or class file
  are rejected before compilation. The real source files were not changed by
  those checks.

Compiler: `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)`.
Build driver: `Latexmk 4.88`.
The corrected PDF's SHA-256 at this check is
`123587a2cda8ea00fd85764f8ece5283ce3738d32eb2deb189028157531f097c`.
PDF metadata/IDs can vary across builds; the equivalence check above compares
rendered pages, not PDF container bytes. No archived original compiled PDF was
available at the specified commit, so this verifies against a clean build of
that commit with the documented engine.

The review files are local under `/tmp/activity-descent-render-review/`:
`baseline-85d280b/`, `render-comparison.json`, `restored-page-1.png` through
`restored-page-9.png`, and `restored-all-pages.png`. The earlier incorrect PDF
and log were retained there as `previous-tectonic.pdf` and `.log`.

## Builds going forward

Run `python tools/build_paper.py` from the analysis repository, or use the
default Snakemake target. The builder requires pdfLaTeX and latexmk on PATH or
under `.tools/texlive/bin/x86_64-linux`. It creates a fresh build directory,
copies source bytes unchanged, verifies the resulting engine and references,
and replaces the output PDF only after a successful build. It does not fall
back to Tectonic. `paper/manuscript_baseline.json` now also protects `jmlr.cls`,
`jmlrutils.sty`, and `latexmkrc`, alongside the text, bibliography, and figures.
`paper/generated/build_provenance.json` records the exact compiled-source hash,
compiler, build driver, inputs, and current output PDF hash.

This workspace has an isolated TeX Live installation under `.tools/texlive`.
It was installed using the official [TeX Live installer](https://tug.org/texlive/quickinstall.html)
with `--scheme=basic --portable --no-doc-install --no-src-install`, followed by
`tlmgr install latexmk xkeyval etoolbox placeins natbib xcolor algorithm2e
draftwatermark relsize ifoddpage xifthen`. No system installation or shell
startup file was changed.

## New research remains preserved separately

The inference draft is retained in manuscript history (through `ed71d21`) and
in `paper/research/archive/inference_draft_2026-10-20/`. Active research notes
remain in `docs/`, with appendix assets under
`paper/research/learning_rule_inference/`. No research was discarded or promoted
into the restored main paper. The user chooses any future promotion.
