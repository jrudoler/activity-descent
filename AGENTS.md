# Manuscript boundary

The user requires preservation of the original manuscript. `paper/` is a
separate repository. Its main text, bibliography, and original figures are the
handoff version at manuscript commit `85d280b`.

Keep new inference experiments, results, theory, and hardware measurements in
local Markdown notes under `docs/`, or in separate appendix material under
`paper/research/`. Do not rewrite the title, abstract, main text, bibliography,
or original figure assets to promote research results unless the user explicitly
chooses that promotion. A completed experiment does not authorize promotion.

The default manuscript build must compile the preserved original. Research
generation and collection must use explicit separate targets and write only to
research staging paths. Preserve the archived abstract and separate COSYNE work.

Use the current `RESEARCH_PLAN.md` for the revised goal scope. Keep the analysis
and manuscript histories separate. Do not start sub-agents unless the user
explicitly requests delegation or parallel agent work.
