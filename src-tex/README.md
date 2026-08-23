# LaTeX report

This directory contains an original, reproducible draft of the diabetes-classifier report. It follows the supplied paper's broad organization while using the official Springer Nature author template and newly generated text, tables, and figures.

Before submission, replace or confirm the author and institutional details near the top of `report.tex`. Read the whole report, verify every claim, and revise the wording so the final document accurately reflects your own work and understanding. The PDF includes a transparent AI-assisted drafting statement.

## Build and audit

Requirements:

- the project dependencies installed through `uv`;
- Tectonic 0.17 or a compatible LaTeX installation; and
- Poppler's `pdftotext` for the local similarity audit.

From the repository root, run:

```bash
make -C src-tex
```

This command:

1. reruns the exact train/test experiment;
2. verifies the expected metrics;
3. regenerates all seven PDF figures and the results table;
4. compiles `src-tex/report.pdf`; and
5. checks citations, numerical consistency, render warnings, placeholders, and exact phrase overlap with the supplied sample and instructions.

The phrase-overlap check is a small local safeguard. It is not a commercial plagiarism service or an AI detector, and it cannot guarantee an institutional checker result. No report text is uploaded to a third-party service.

## Main files

```text
report.tex                 Main manuscript
references.bib             Cited sources
report.pdf                 Reviewed rendered report
generate_figures.py        Reproducible experiment and plots
check_report.py            Local citation/consistency/similarity audit
figures/                    Generated vector figures
generated/results.json     Machine-readable rerun results
generated/model_metrics.*  Generated results table
sn-jnl.cls                 Official Springer Nature class
sn-vancouver-num.bst       Official numbered reference style
```

The template assets are unchanged files from Springer Nature's LaTeX package, version 3.1 (December 2024): <https://www.springernature.com/gp/authors/campaigns/latex-author-support>. The committed report is suitable for the course project, but a real Springer submission would also need to follow the target journal's current packaging rules.
