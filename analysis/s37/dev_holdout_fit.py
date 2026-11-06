"""Crack model fitted separately to the 10 tuned (dev) games and the 15 untuned (holdout) games (exploratory, 5 Nov 2026).
The 10 dev games are the ones our early development runs played (B1 to X9, 25 minutes, 10 games); the other 15 were
held out until the FB25 runs. Same cure-lognormal fit as the sealed V5 calculation (imported read-only), on the four
controls (the sealed fit's data) and on all eight runs of this agent. Run: python3 analysis/s37/dev_holdout_fit.py >
analysis/s37/out/dev_holdout_fit.out"""
import math
from common import CTL, SPEC2_VALID
from v5_budget_prediction import load, attempts, fit
G = {r: load(r) for r in CTL + SPEC2_VALID + ['t1c_fb25_ctl', 'v5_fb50']}
DEV = {'ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33'}
for lab, runs in (('four controls', CTL), ('eight runs (7 at 121 min + V5)', list(G))):
    for part, keep in (('dev (10 games)', lambda g: g in DEV), ('holdout (15 games)', lambda g: g not in DEV), ('all 25', lambda g: True)):
        att = [a for r in runs for g in G[r] if keep(g['game_id'][:4]) for a in attempts(g)[0]]
        ll, (q, mu, s) = fit(att, 'lognormal')
        lev = sum(g['levels_completed'] for r in runs for g in G[r] if keep(g['game_id'][:4])) / len(runs)
        nl = sum(g['number_of_levels'] for g in G[runs[0]] if keep(g['game_id'][:4]))
        print(f'{lab}, {part}: attempts {len(att)}; q {q:.3f} median {math.exp(mu) / 1e3:.1f}k s {s:.2f}; '
              f'levels per run {lev:.1f} of {nl} ({100 * lev / nl:.0f}%)')
