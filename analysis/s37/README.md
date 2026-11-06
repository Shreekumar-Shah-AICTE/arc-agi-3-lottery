# analysis/s37: paper-track evidence (2 Nov 2026)

Reproduces the numbers and figures in paper/S37_PAPER_PLAN.md. Run from anywhere: `python3 analysis/s37/<script>.py`.

- `variance_decomp.py`: lottery test on 7 replicate dev runs (4 control, 3 spec2). Prints run means, predicted SD of the 25-game mean from per-game variance (3.86) vs observed pooled SD (4.53), and per-game swings.
- `fig_noise_and_hazard.py`: Figure 1 (5 identical B12 leaderboard runs placed on the 2 Nov public board; level-clear hazard from analysis/s7/level_attempts.json).
- `fig_pergame_lottery.py`: Figure 2 (levels completed per game across the 7 runs).

Data (gzip, VERIFIED pulls on 2 Nov 2026):
- `data/runs/<run>.benchmark.json.gz`: per-game results of dev save runs, pulled with `kaggle kernels output <slug> -p . --file-pattern benchmark` (slug in kaggle/<dir>/kernel-metadata.json). b12d25 and b12d25_r2 are included but unused so far; b12d25_r3 failed to download.
- `data/lb/*_public_2026-11-02T1136Z.csv.gz`: public leaderboard snapshots (ARC-AGI-3 5,047 teams, ARC-AGI-2 3,608 teams), from `kaggle competitions leaderboard -c <comp> --download`.

Next (plan V1 to V4): pull the remaining dev runs, build the crack table, bootstrap intervals for the variance decomposition, and compare hazard in token time vs action time.
- V6 (2 Nov): `lb_cluster.py` reads `data/lb/arc-agi-3_lb_20261102T1307.csv.gz` (public LB at 13:07 UTC; named so `common.load_lb` keeps reading the 11:36 snapshot) and `data/external/dfranzen_milestone2_4pass.benchmark.json.gz` (public notebook output, `kaggle kernels output dfranzen/arc-agi-3-milestone-2-solution --file-pattern benchmark`).
- V7 (2 Nov): `discovery_split.py` (first level vs later levels: clear rates and actions per human baseline; imports v3_hazard, which prints its own table first).
