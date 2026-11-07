# The ARC-AGI-3 Lottery: code, data and sealed files

Everything needed to rebuild the numbers, tables and figures of the paper "The ARC-AGI-3 Lottery: a testable model of
LLM-agent scores under Kaggle compute" (ARC Prize 2026 Paper Track). Every number in the paper is traced to a script
here and checked by `paper/check_numbers.py`.

## Rebuild and check

Run from the repo root; the steps are in `paper/README.md`. In short:

1. Number registry: `python3 paper/build_results.py`, then `python3 paper/check_numbers.py` (must report 0 unmatched).
2. Figures and tables: the scripts in `analysis/s37` (see its README).
3. PDF: `node paper/build_pdf.cjs --final` writes `paper/build/paper.pdf`.

## What is where

- `paper.pdf`: the paper (final build of `paper/draft/`).
- `paper/`: text (`draft/`), figures, tables, the Kaggle summary (`writeup/`), the references and the number registry.
- `paper/notebook/how_many_runs.ipynb`: how many reruns a team needs before it can trust a score difference (CPU only); it also compares two agents on crack cost, checks the crack model on a team's own runs, and gives the cap that permanent walls would set.
- `analysis/s37`, `analysis/s7`: the analysis scripts and their outputs; `analysis/s37/sealed/` holds the sealed
  predictions with their OpenTimestamps proofs. Sealed files and the scripts that made them are never edited.
- `analysis/s38`: the SWE-Bench-Verified check of Section 4.8. It uses the resolved-task ids of the public runs of
  Bjarnason et al. (huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts, CC BY 4.0), kept unchanged otherwise.
- `data/`: per-run benchmark files of the development runs.
- `kaggle/`: the notebooks as they were run on Kaggle, their builders and patches.
- `state/EXPERIMENTS.csv`, `state/DECISIONS.md`: the experiment ledger and the decision log with pre-registrations.

Links: primary notebook https://www.kaggle.com/code/shreekumarshah/arc3-b12-x6dprime ; sealed prediction post
https://www.kaggle.com/competitions/arc-prize-2026-paper-track/discussion/755123

## Licences

Code written for this project: MIT (`LICENSE`), and also available under CC BY 4.0. Data, text, tables and figures: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`).
Material derived from other authors keeps its original terms: forks of other authors' public Kaggle notebooks (Kaggle
releases public notebooks under Apache 2.0 by default), model chat templates (their providers' terms) and the quoted
excerpts in `paper/sources/` (their authors).
