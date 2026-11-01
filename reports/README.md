# Reports

Written reports, kept apart from the code.

| Folder | Content |
|---|---|
| `proposal/` | Research proposal (LaTeX, `hcmut-report.cls`, `proposal.md`) |
| `phase1/` | Phase 1 report (`main.tex`) |
| `thesis/` | Final thesis report (not started) |
| `figures/` | Figures exported from `runs/` once a result is final. Never raw data |

Build a report outside the repo so the build files stay out of git:

```bash
latexmk -pdf -outdir="$TMPDIR/phase1" -cd reports/phase1/main.tex
```
