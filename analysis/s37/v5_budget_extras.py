"""V5 extras (written 4 Nov 2026 ~10:13 UTC, before the run): secondary-prediction ranges for the sealed V5 file.
Reuses the crack model of v5_budget_prediction.py (lognormal, full fit). Per simulated 2x run: games completed, levels,
and Spearman correlation between the model's predicted per-game gain and that run's per-game gain (games not completed
in all four FB25 runs). Also the combined 90% interval: simulation spread plus parameter-bootstrap spread (normal approx)."""
import numpy as np
from v5_budget_prediction import FB, load, attempts, fit, states, surv, tinv
def sim_games(sts, par, kind, rng, R):
    out = {}
    for gid, done, nl, c, delta, fin in sts:
        lv = np.full(R, done, float)
        if not fin:
            sc = float(surv(c, par, kind))
            for rr in range(R):
                budget, first, n = delta, True, done
                while n < nl:
                    u = rng.random() * sc if first else rng.random()
                    if u <= par[0]: break
                    need = tinv(u, par, kind) - (c if first else 0.0)
                    if need > budget: break
                    budget -= need; n += 1; first = False
                lv[rr] = n
        out[gid] = (lv, nl)
    return out
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])
rng = np.random.default_rng(20261104 + 1)
runs = {r: load(r) for r in FB}
A = [a for r in FB for g in runs[r] for a in attempts(g)[0]]; par = fit(A, 'lognormal')[1]
R = 600
sims = {r: sim_games(states(runs[r]), par, 'lognormal', rng, R) for r in FB}
gids = sorted(sims[FB[0]])
one = {gid: np.mean([next(g['levels_completed'] for g in runs[r] if g['game_id'].startswith(gid)) for r in FB]) for gid in gids}
nl = {gid: sims[FB[0]][gid][1] for gid in gids}
pred2 = {gid: np.mean([sims[r][gid][0].mean() for r in FB]) for gid in gids}
open_g = [g for g in gids if one[g] < nl[g]]  # not completed in all four runs
pg = np.array([pred2[g] - one[g] for g in open_g])
tot, wins, rho = [], [], []
for r in FB:
    for k in range(R):
        lv = {g: sims[r][g][0][k] for g in gids}
        tot.append(sum(lv.values())); wins.append(sum(lv[g] >= nl[g] for g in gids))
        rho.append(spearman(pg, np.array([lv[g] - one[g] for g in open_g])))
tot, wins, rho = np.array(tot), np.array(wins), np.array(rho)
print('params', np.round(par, 3), 'open games', len(open_g), open_g)
print('total 5/50/95', np.quantile(tot, [.05, .5, .95]).round(1), 'mean %.1f SD %.1f' % (tot.mean(), tot.std()))
print('games completed 5/50/95', np.quantile(wins, [.05, .5, .95]), 'mean %.2f' % wins.mean(), 'P(>=9) %.2f' % (wins >= 9).mean())
print('spearman(pred gain, sim gain) 5/50/95', np.nanquantile(rho, [.05, .5, .95]).round(2), 'P(rho>0) %.2f' % np.nanmean(rho > 0))
boot_sd = (142.3 - 120.3) / (2 * 1.645)  # bootstrap 5-95% of the mean, from v5_budget_prediction.out
comb_sd = float(np.sqrt(tot.std() ** 2 + boot_sd ** 2))
print('combined SD %.1f -> 90%% PI %.0f-%.0f (centre 127.4)' % (comb_sd, 127.4 - 1.645 * comb_sd, 127.4 + 1.645 * comb_sd))
print('per game: id levels 1x_mean pred_2x')
for g in gids: print(f'  {g} {nl[g]} {one[g]:.2f} {pred2[g]:.1f}')
