# Paper: how to rebuild and check (4 Nov 2026)

Run from the repo root.

1. Figures: `cd analysis/s37 && python3 fig_noise_and_hazard.py && python3 fig_survival_fit.py && python3 fig_pergame_lottery.py && python3 fig_noise_time.py && python3 fig_v5_dose.py && python3 fig_arc2_plateau.py` -> `paper/figs/`.
2. Tables: `python3 table_submissions.py` (Table 1), `python3 table_submissions_arc2.py` (Table 2), `python3 table_ledger_negatives.py` (Appendix A), `python3 table_dev_runs.py` (Appendix B) -> `paper/tables/`.
3. Number registry: `python3 paper/build_results.py` -> `paper/results.json`; then `python3 paper/check_numbers.py` must report 0 unmatched. Add a registry entry for every new number in `paper/draft/`.
4. PDF: `node paper/build_pdf.cjs --final` -> `paper/build/paper.html` and `paper.pdf` (gitignored; the folder is created if missing; without `--final` the PDF carries a DRAFT banner). It orders sections by number, places figures at first citation, lists only cited references from `REFERENCES.md`, and warns about missing references or out-of-order figures.

5. Scheduler simulation (Section 4.4, exploratory; about 15 minutes in all): `cd analysis/s37 && python3 sched_endgame_sim.py 40 10 > out/sched_endgame_sim.out && python3 sched_sealedcalc_check.py 100 > out/sched_sealedcalc_check.out && python3 sched_sealedcalc_check.py 40 40000 > out/sched_sealedcalc_check_drain40k.out && python3 sched_sealedcalc_check.py 40 80000 > out/sched_sealedcalc_check_drain80k.out && python3 sched_guard_supply.py > out/sched_guard_supply.out`.

Figure numbers (order of appearance): 1 noise_and_hazard, 2 survival_fit, 3 pergame_lottery, 4 noise_over_time, 5 v5_dose_response, 6 arc2_plateau.
No placeholder is left in the text: the two private-board outcomes in Section 4.4 read "pending" until the hosts publish the private board, then they are scored with the sealed scorers. `[V5]` was filled on 4 Nov 2026 from `analysis/s37/out/v5_readout.out`.
Sealed files in `analysis/s37/sealed/` and the scripts that produced them are never edited.
