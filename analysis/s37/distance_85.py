#!/usr/bin/env python3
"""What would 85% take under the crack model? (4 Nov 2026; INFERENCE, not part of any sealed prediction.)

Model as fitted on the four control runs (analysis/s37/out/v5_budget_prediction.out, full fit): each level is a wall
with probability q (never cracked within the budgets observed), otherwise its crack wait in generated tokens is
lognormal (median M, log-sd S). Levels are played in order, so a wall ends the game. A game's score is approximated by
its weighted completion share, sum of l over completed levels / sum of l over all levels (section 3.3: the efficiency
term costs 2.9-5.2% on our runs and is ignored here). Levels are independent here (no per-level difficulty that persists
across runs), so the finite-budget columns are optimistic: at k=2 this gives 134 levels where the sealed V5 model, which
conditions on each control run's state, gives 127. The k=inf column depends only on q and the level counts. Budget per game: the median generated tokens of a game that did not
finish, in the control runs, times a multiplier k. Output: expected levels and score over the 25 public games (local
set; the leaderboard games are different and our public score there is lower, 33.37 against about 47 locally).
"""
import gzip, json, re, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CTL = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
Q, M, S = 0.06, 22e3, 1.05          # full fit, lognormal (v5_budget_prediction.out)

def runs(r):
    return json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']

games, budgets, obs_levels, obs_score = {}, [], [], []
for r in CTL:
    g = runs(r)
    obs_levels.append(sum(x['levels_completed'] for x in g)); obs_score.append(statistics.mean(x['final_score'] for x in g))   # final_score is already in percent
    for x in g:
        games[x['game_id'].split('-')[0]] = x['number_of_levels']
        m = re.search(r'tokens=(\d+)', x.get('solver_note') or '')
        if m and x['levels_completed'] < x['number_of_levels']:
            budgets.append(int(m.group(1)))
B1 = statistics.median(budgets)
L = np.array([games[k] for k in sorted(games)])

def expect(q, k, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    lv = np.zeros(n); sc = np.zeros(n)
    for Lg in L:
        wall = rng.random((n, Lg)) < q
        t = np.exp(np.log(M) + S * rng.standard_normal((n, Lg)))
        t[wall] = np.inf
        done = (np.cumsum(t, axis=1) <= k * B1)
        done = np.cumprod(done, axis=1).astype(bool)      # in order: stop at the first level not cleared
        c = done.sum(axis=1)
        lv += c; sc += c * (c + 1) / (Lg * (Lg + 1))
    return lv.mean(), 100 * sc.mean() / len(L)

print(f'{len(L)} games, {L.sum()} levels; budget per game at 1x (median tokens, unfinished games, n={len(budgets)}): {B1:,.0f}')
print(f'observed controls: levels {obs_levels} (mean {statistics.mean(obs_levels):.1f}); local score mean {statistics.mean(obs_score):.1f}')
lv, sc = expect(Q, 1)
print(f'model at q={Q}, k=1: levels {lv:.1f}, completion score {sc:.1f}  (calibration check against the line above)')
print('\nexpected completion score (levels) by wall rate q and budget multiplier k')
ks = [1, 2, 4, 10, 1e9]
print('q      ' + ''.join(f'{("k=" + str(k)) if k < 1e9 else "k=inf":>16}' for k in ks))
for q in [0.06, 0.045, 0.03, 0.02, 0.01, 0.0]:
    cells = [expect(q, k) for k in ks]
    print(f'{q:<6} ' + ''.join(f'{s:>9.1f} ({l:5.1f})' for l, s in cells))
# smallest q that reaches 85 with unlimited budget; smallest k that reaches 85 with no walls
lo, hi = 0.0, 0.06
for _ in range(20):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if expect(mid, 1e9)[1] >= 85 else (lo, mid)
print(f'\nunlimited budget: 85% needs q <= {lo:.3f} (fitted q {Q})')
lo, hi = 1.0, 10.0          # bisection (5 Nov 2026; the first version stepped k by 1.25 and printed the step above, 2.0x)
for _ in range(20):
    mid = (lo + hi) / 2
    lo, hi = (lo, mid) if expect(0.0, mid)[1] >= 85 else (mid, hi)
print(f'no walls: 85% needs a budget multiplier of about {hi:.1f}x')

# Caps by game set (added 6 Nov 2026 after outside review round 2, D102). The k=inf cap depends only on q and the level
# counts, so it has an exact form: levels in order, a wall ends the game, P(c = j) = (1-q)^j q for j < L and (1-q)^L
# for j = L, and the completion share of a game is c(c+1)/(L(L+1)). q values: analysis/s37/out/dev_holdout_fit.out.
DEV = {'ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33'}

def cap_exact(q, Ls):
    tot = 0.0
    for Lg in Ls:
        e = sum(((1 - q) ** j) * q * j * (j + 1) for j in range(Lg)) + ((1 - q) ** Lg) * Lg * (Lg + 1)
        tot += e / (Lg * (Lg + 1))
    return 100 * tot / len(Ls)

L_hold = [games[k] for k in sorted(games) if k not in DEV]
L_dev = [games[k] for k in sorted(games) if k in DEV]
print(f'\nunlimited-play cap by game set (exact; {len(L_dev)} dev games, {len(L_hold)} held-out games, {sum(L_hold)} held-out levels)')
for label, q in (('pooled, four controls', 0.06), ('pooled, eight runs', 0.04), ('held-out, four controls', 0.110), ('held-out, eight runs', 0.070)):
    print(f'  {label:<24} q {q:.3f}: {cap_exact(q, list(L)):.1f} on the 25 public level counts, {cap_exact(q, L_hold):.1f} on the 15 held-out games')
for label, qh in (('four controls', 0.110), ('eight runs', 0.070)):
    mix = (cap_exact(0.0, L_dev) * len(L_dev) + cap_exact(qh, L_hold) * len(L_hold)) / len(L)
    print(f'  25-game mix, dev q 0.000 and held-out q {qh:.3f} ({label}): {mix:.1f}')
lo, hi = 0.0, 0.2
for _ in range(40):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if cap_exact(mid, L_hold) >= 85 else (lo, mid)
print(f'  held-out games: 85% needs q <= {lo:.3f}')
