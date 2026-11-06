"""Tier 2, walls (exploratory; written 5 Nov 2026 ~13:45 UTC, after V5, T1, T1 rerun and T1c were read out).
Reviewer question: which levels are walls, and is the fitted wall share q (0.06) right, given that unlimited play at
q = 0.06 caps the 25 public games at about 142 of 183 levels while the doubled run V5 reached 141 and was still climbing?
1. Refit the cure-lognormal crack model (same code as the sealed V5 fit, imported read-only) on more data:
   the four controls (as sealed), + V5 (242 min), + T1c (a fifth control), + the two valid serving-only runs.
   For each fit: the unlimited-budget cap if walls strike each attempt independently, sum over games of
   sum_k (1-q)^k, and the same cap if walls are fixed per level (the persistent-wall model H2 of Section 3.4).
2. Wall census: pool every attempt at each (game, level) over the same-agent runs and V5; for levels never cleared,
   the posterior wall probability under the pooled fit, q / (q + (1-q) prod_i S0(c_i)), with S0 the non-wall
   survival at each attempt's censored token count c_i (independent waits, so it overstates P(wall) for hard levels).
3. Start times: every game's 121 minutes count from the start of the run and at most 10 games play at once, so a late
   start means less play. Per game: start minute (first logged step), last step, levels, over the same runs.
Run: python3 analysis/s37/walls.py > analysis/s37/out/walls.out"""
import gzip, json, math, re, statistics
import numpy as np
from common import HERE, CTL, SPEC2_VALID
from v5_budget_prediction import load, attempts, fit, surv
ONEX = CTL + SPEC2_VALID + ['t1c_fb25_ctl']            # seven valid same-agent runs at 121 min
RUNS = ONEX + ['v5_fb50']
G = {r: load(r) for r in RUNS}
NLEV = {g['game_id'][:4]: g['number_of_levels'] for g in G['b12_fb25']}
def A(runs): return [a for r in runs for g in G[r] for a in attempts(g)[0]]
def cap_indep(q): return sum(sum((1 - q) ** k for k in range(1, n + 1)) for n in NLEV.values())
fits = {}
for name, runs in (('4 controls (sealed fit)', CTL), ('4 controls + V5', CTL + ['v5_fb50']),
                   ('5 controls (with T1c) + V5', CTL + ['t1c_fb25_ctl', 'v5_fb50']),
                   ('7 same-agent runs + V5', RUNS)):
    att = A(runs); ll, par = fit(att, 'lognormal'); fits[name] = par
    n_open = sum(1 for a in att if not a[2])
    print(f'fit {name}: attempts {len(att)} (open {n_open}); q {par[0]:.3f} median {math.exp(par[1]) / 1e3:.1f}k s {par[2]:.2f}; '
          f'unlimited cap, independent walls: {cap_indep(par[0]):.0f} of {sum(NLEV.values())}')
print(f'V5 observed: {sum(g["levels_completed"] for g in G["v5_fb50"])} levels at 242 min')
# 2. wall census on the pooled fit
q, mu, s = fits['7 same-agent runs + V5']
def S0(c):
    return (float(surv(max(c, 200), (q, mu, s), 'lognormal')) - q) / (1 - q)
cells = {}
for r in RUNS:
    for g in G[r]:
        for k, tok, ok in attempts(g)[0]:
            cells.setdefault((g['game_id'][:4], k), []).append((r, tok, ok))
never = []
for (gm, k), L in cells.items():
    if any(ok for _, _, ok in L): continue
    prod = math.prod(max(S0(t), 1e-12) for _, t, _ in L)
    pw = q / (q + (1 - q) * prod)
    never.append((pw, gm, k, len(L), max(t for _, t, _ in L), sum(t for _, t, _ in L), [t for r, t, _ in L if r == 'v5_fb50']))
never.sort(reverse=True)
print(f'\nwall census (pooled fit q {q:.3f}): {len(never)} levels reached but never cleared in any of {len(RUNS)} runs')
print('  game level  runs-reaching  max-tokens  total-tokens  V5-tokens  P(wall)')
for pw, gm, k, n, mx, tot, v5 in never:
    print(f'  {gm} L{k + 1:<2d}  {n:2d}  {mx / 1e3:7.0f}k  {tot / 1e3:8.0f}k  {(v5[0] / 1e3 if v5 else float("nan")):7.0f}k  {pw:.2f}')
likely = [x for x in never if x[0] >= 0.5]
fixed_cap = sum(min([k for pw, gm2, k, *_ in likely if gm2 == gm], default=n) for gm, n in NLEV.items())
print(f'levels with P(wall) >= 0.5: {len(likely)}; cap if those are fixed walls: {fixed_cap} of {sum(NLEV.values())}')
# levels cleared in some runs but not others: how many runs reached / cleared
mixed = [(gm, k, sum(ok for *_, ok in L), len(L)) for (gm, k), L in cells.items() if any(ok for *_, ok in L) and not all(ok for *_, ok in L)]
print(f'levels cleared in some runs and open in others: {len(mixed)}')
# 3. start times
print('\nper game over the 7 runs at 121 min: start minute (mean, range), levels (mean), games starting after minute 30')
rows = []
for gm in NLEV:
    st = [g['history'][0]['wallclock_seconds'] / 60 for r in ONEX for g in G[r] if g['game_id'][:4] == gm and g.get('history')]
    lv = [g['levels_completed'] for r in ONEX for g in G[r] if g['game_id'][:4] == gm]
    rows.append((statistics.mean(st), min(st), max(st), statistics.mean(lv), gm))
rows.sort()
for m, lo, hi, lv, gm in rows:
    print(f'  {gm}: start {m:5.1f} min ({lo:4.0f}-{hi:4.0f}), levels {lv:4.1f} of {NLEV[gm]}')
late = [x for x in rows if x[0] > 30]
print(f'games starting after minute 30 on average: {len(late)} ({", ".join(x[4] for x in late)}); '
      f'their mean share of levels cleared {statistics.mean(x[3] / NLEV[x[4]] for x in late):.2f} vs '
      f'{statistics.mean(x[3] / NLEV[x[4]] for x in rows if x[0] <= 30):.2f} for the rest')
# zero-level game runs and early stops
print('\ngame runs with 0 levels in the 7 runs at 121 min: run, game, start min, last step min, tokens')
for r in ONEX:
    for g in G[r]:
        if g['levels_completed'] == 0:
            h = g.get('history') or []
            tok = int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1))
            print(f'  {r} {g["game_id"][:4]}: start {h[0]["wallclock_seconds"] / 60 if h else float("nan"):.0f}, '
                  f'last {h[-1]["wallclock_seconds"] / 60 if h else float("nan"):.0f}, tokens {tok / 1e3:.0f}k, actions {len(h)}')
# 4. shelving: the slot priority (D' rule) multiplies a game's value by a "chance" factor that halves at 62k tokens on
# the current level, so a game stuck on a level loses its slot to waiting games and may not play again (Appendix C).
print('\nshelved game runs at 121 min (last logged step more than 30 minutes before the end of the budget):')
shelved = []
for r in ONEX:
    for g in G[r]:
        h = g.get('history') or []
        if h and h[-1]['wallclock_seconds'] / 60 < 121 - 30:
            k = g['levels_completed']; att = attempts(g)[0]
            shelved.append((r, g['game_id'][:4], k, h[-1]['wallclock_seconds'] / 60, att[-1][1] if att and not att[-1][2] else 0))
n_runs = len(ONEX) * len(NLEV)
print(f'  {len(shelved)} of {n_runs} game runs; at level 1 (0 levels cleared): {sum(1 for x in shelved if x[2] == 0)}; '
      f'all-level games finished early are excluded (they stop because they are done)')
z = [x for x in shelved if x[2] == 0]
print(f'  0-level shelved game runs: tokens on level 1 when shelved: median {statistics.median(x[4] for x in z) / 1e3:.0f}k '
      f'(range {min(x[4] for x in z) / 1e3:.0f}k-{max(x[4] for x in z) / 1e3:.0f}k)')
zero_all = [(r, g['game_id'][:4]) for r in ONEX for g in G[r] if g['levels_completed'] == 0]
print(f'  0-level game runs in total: {len(zero_all)}, of which shelved: {len(z)}; '
      f'shelved by minute 20 (last logged step at or before minute 20): {sum(1 for x in z if x[3] <= 20)}')
done_early = [(r, g['game_id'][:4]) for r in ONEX for g in G[r] if g['levels_completed'] >= g['number_of_levels']]
print(f'  (games completed: {len(done_early)} game runs)')
print('\nthe 0-level games at 121 min, in V5 (242 min): levels, tokens on level 1, start and last step')
for gm in sorted({x[1] for x in zero_all}):
    g = next(g for g in G['v5_fb50'] if g['game_id'][:4] == gm); att = attempts(g)[0]; h = g['history']
    n0 = sum(1 for r, x in zero_all if x == gm)
    print(f'  {gm}: 0 levels in {n0} of {len(ONEX)} runs at 121 min; V5 {g["levels_completed"]} of {NLEV[gm]}, '
          f'level 1 took {att[0][1] / 1e3:.0f}k tokens ({"cleared" if att[0][2] else "open"}), start {h[0]["wallclock_seconds"] / 60:.0f}, '
          f'last step {h[-1]["wallclock_seconds"] / 60:.0f} min')
v5l1 = [attempts(next(g for g in G['v5_fb50'] if g['game_id'][:4] == gm))[0][0] for gm in sorted({x[1] for x in zero_all})]
print(f'  all {len(v5l1)} cleared level 1 in V5: {all(a[2] for a in v5l1)}; tokens on level 1 from {min(a[1] for a in v5l1) / 1e3:.0f}k '
      f'to {max(a[1] for a in v5l1) / 1e3:.0f}k')
# 5. share of open attempts that are walls, sealed fit vs refit (Section 4.3 quotes 42% for the sealed fit)
def wall_share(att, par):
    op = [c for _, c, ok in att if not ok]
    return statistics.mean(par[0] / max(float(surv(max(c, 200), par, 'lognormal')), 1e-12) for c in op), len(op)
print('\nshare of open attempts that are walls (posterior mean), by fit')
for runs, lab in ((CTL, 'four controls at 121 min'), (['v5_fb50'], 'V5 at 242 min'), (ONEX, '7 same-agent runs at 121 min')):
    for name in ('4 controls (sealed fit)', '7 same-agent runs + V5'):
        sh, n = wall_share(A(runs), fits[name])
        print(f'  {lab}, {name}: {100 * sh:.0f}% of {n} open attempts')
# 6. distance to 85% under the refit (same simulation as distance_85.py, with the refit's median and s)
CTL_BUDGETS = []
for r in CTL:
    for g in G[r]:
        m = re.search(r'tokens=(\d+)', g.get('solver_note') or '')
        if m and g['levels_completed'] < g['number_of_levels']: CTL_BUDGETS.append(int(m.group(1)))
B1 = statistics.median(CTL_BUDGETS); LV = np.array([NLEV[k] for k in sorted(NLEV)])
def expect(q, M, S, k, n=4000, seed=0):
    rng = np.random.default_rng(seed); lv = np.zeros(n); sc = np.zeros(n)
    for Lg in LV:
        wall = rng.random((n, Lg)) < q
        t = np.exp(np.log(M) + S * rng.standard_normal((n, Lg))); t[wall] = np.inf
        done = np.cumprod(np.cumsum(t, axis=1) <= k * B1, axis=1).astype(bool); c = done.sum(axis=1)
        lv += c; sc += c * (c + 1) / (Lg * (Lg + 1))
    return lv.mean(), 100 * sc.mean() / len(LV)
print(f'\ndistance to 85% (completion score, independent levels; budget per game at 1x {B1:,.0f} tokens as in distance_85.py)')
for name in ('4 controls (sealed fit)', '7 same-agent runs + V5'):
    q_, mu_, s_ = fits[name]; M_ = math.exp(mu_)
    cells = [expect(q_, M_, s_, k) for k in (1, 2, 1e9)]
    lo, k = 1.0, 10.0
    for _ in range(20):
        mid = (lo + k) / 2; lo, k = (lo, mid) if expect(0.0, M_, s_, mid)[1] >= 85 else (mid, k)
    print(f'  {name} (q {q_:.3f}, median {M_ / 1e3:.1f}k, s {s_:.2f}): k=1 {cells[0][1]:.1f} ({cells[0][0]:.1f} levels); '
          f'k=2 {cells[1][1]:.1f} ({cells[1][0]:.1f}); k=inf {cells[2][1]:.1f} ({cells[2][0]:.1f}); no walls: 85% at about {k:.1f}x')
# 7. how much of the per-game variance is the 0-level outcome (population variances over the 7 runs at 121 min)
tot = btw = 0.0; per = []
for gm in NLEV:
    lv = np.array([g['levels_completed'] for r in ONEX for g in G[r] if g['game_id'][:4] == gm], float)
    v = lv.var(); p0 = (lv == 0).mean()
    b = p0 * (1 - p0) * (lv[lv > 0].mean() ** 2) if 0 < p0 < 1 else 0.0
    tot += v; btw += b; per.append((b, gm, v))
per.sort(reverse=True)
LVM = np.array([[next(g for g in G[r] if g['game_id'][:4] == gm)['levels_completed'] for gm in NLEV] for r in ONEX], float)
def zshare(M):
    t = b = 0.0
    for j in range(M.shape[1]):
        lv = M[:, j]; p0 = (lv == 0).mean(); t += lv.var()
        if 0 < p0 < 1: b += p0 * (1 - p0) * lv[lv > 0].mean() ** 2
    return b / t if t else float('nan')
rng = np.random.default_rng(2); bs = [zshare(LVM[rng.integers(0, len(ONEX), len(ONEX))]) for _ in range(4000)]
print(f'\nsummed per-game variance over the 7 runs: {tot:.1f}; part due to 0 levels vs some: {btw:.1f} ({100 * btw / tot:.0f}%; '
      f'bootstrap over runs 90% interval {100 * np.nanpercentile(bs, 5):.0f}% to {100 * np.nanpercentile(bs, 95):.0f}%)')
print('  ' + ', '.join(f'{gm} {b:.1f} of {v:.1f}' for b, gm, v in per if b > 0))
# 8. start time within game: levels against start minute with game fixed effects (permutation within game)
X, Y, GI = [], [], []
for gi, gm in enumerate(NLEV):
    for r in ONEX:
        g = next(g for g in G[r] if g['game_id'][:4] == gm)
        if g.get('history'): X.append(g['history'][0]['wallclock_seconds'] / 60); Y.append(g['levels_completed']); GI.append(gi)
X, Y, GI = np.array(X), np.array(Y, float), np.array(GI)
def dm(v): return v - np.array([v[GI == i].mean() for i in GI])
def slope(x): xd = dm(x); return float((xd * dm(Y)).sum() / (xd * xd).sum())
b0 = slope(X); rng = np.random.default_rng(1); perm = []
for _ in range(4000):
    xp = X.copy()
    for i in set(GI): idx = np.where(GI == i)[0]; xp[idx] = rng.permutation(xp[idx])
    perm.append(slope(xp))
p = float(np.mean(np.abs(perm) >= abs(b0)))
print(f'\nwithin-game effect of a later start (7 runs, {len(X)} game runs): {10 * b0:+.2f} levels per 10 minutes later; '
      f'permutation p = {p:.3f} (two-sided, start minutes shuffled within game)')
sdx = float(np.sqrt((dm(X) ** 2).mean()))
print(f'  within-game SD of the start minute: {sdx:.1f} min')
# 9. the late starters at 242 min (V5): start and levels against their 121-min mean
print('\nlate starters (mean start after minute 30) in V5: start minute, levels at 242 min vs mean at 121 min')
for m, lo, hi, lv, gm in rows:
    if m > 30:
        g = next(g for g in G['v5_fb50'] if g['game_id'][:4] == gm)
        print(f'  {gm}: V5 start {g["history"][0]["wallclock_seconds"] / 60:5.1f}, levels {g["levels_completed"]} of {NLEV[gm]} (121-min mean {lv:.1f})')
# 10. the slot rule D' at the median state of the shelved 0-level game runs (formula copied from the B12 notebook's
# ours_form_priority.py, d_parts; Appendix C). A game that has not started has level 1 and a = t = 0.
def d_parts(level, a, t, pace, n):
    nc = min(10, max(6, n)); norm = 55 / (nc * (nc + 1) / 2)
    A = (1.0 + 0.5 * (level - 1)) * norm * (300.0 / (300.0 + a)) ** 2.5
    if pace > 0: A *= min(4.0, max(0.25, (30000.0 / max(1.0, pace)) ** 0.4))
    T = 225000.0 * min(2.0, max(0.5, (pace if pace > 0 else 30000.0) / 30000.0)) ** 0.5
    C = 0.1 * max(0.1, 1 - a / 115.0) + 0.9 * max(0.1, 1 - t / T)
    B = (0.0, 10.0, 14.0, 16.0)[min(max(0, n - level), 3)]
    return A * C, B, norm
sh0 = [(len(g['history']), attempts(g)[0][0][1]) for r in ONEX for g in G[r]
       if g['levels_completed'] == 0 and g.get('history') and g['history'][-1]['wallclock_seconds'] / 60 < 121 - 30]
ma, mt = statistics.median(a for a, _ in sh0), statistics.median(t for _, t in sh0)
u, _, nm = d_parts(1, ma, mt, 0, 8)
print(f"\nslot rule D' at the median shelved 0-level state ({len(sh0)} game runs: {ma:.0f} actions, {mt / 1e3:.0f}k tokens on level 1): "
      f"A*M*C = {u / nm:.2f} x norm(N); a game that has not started: 1.00 x norm(N)")
print('  priority of a game that has not started (norm(N) + 16): ' + ', '.join(f'N={n}: {d_parts(1, 0, 0, 0, n)[2] + 16:.2f}' for n in range(6, 11)))
