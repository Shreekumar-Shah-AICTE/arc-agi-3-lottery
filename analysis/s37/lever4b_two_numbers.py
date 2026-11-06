"""Lever 4b (4 Nov 2026, exploratory, CPU only, written before the V5 readout).

Idea from lever 3c (crack effort is mostly the level's own, ICC 0.73): compare agents on two numbers instead of scores.
  1. token supply S  = total generated tokens in the run (throughput; between-run CV about 2.5%),
  2. crack cost  b   = log multiplier of tokens-to-crack relative to the control agent, estimated on the SAME levels.
Model: log T(run r, level l) = a_l + b_r + e, e ~ N(0, s^2), with open attempts right-censored at the tokens spent
(lognormal AFT with level fixed effects, fitted by EM). Levels never cracked in any run carry no information on b.
Then: effective budget multiplier m_r = (S_r / S_ctl) * exp(-b_r), and predicted levels = control's token-truncation
curve L_ctl(m) (m <= 1 read off the control runs' logs; m > 1 extrapolated from the end slope, to be replaced by V5).
Outputs: b per run with an A/A spread (same-agent runs), predicted vs observed levels per run, and how many runs each
comparison method needs to detect a 10% efficiency change.
"""
import gzip, json, math
from collections import defaultdict
import numpy as np
from common import HERE

FB25 = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3',            # control (CTL)
        'b31_spec2_fb25', 'b32_spec2_r2', 'b32_spec2_r3',                     # spec depth 2 (D87 NOISE)
        'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b',   # harness / prompt variants
        'b15c_fb25', 'b28_tc_fb25', 'b29_occ_fb25', 'b31_spec1_fb25']        # serving / model variants
# Appendix A rows 31-32: these two runs did not play all 25 games (lf52, and lf52 + su15, never admitted), so their
# level totals are not comparable with full runs. Their played attempts stay in the level-level fit (token time is the
# clock, so a missing game removes data but does not bias the others); every run-level comparison below leaves them out.
INVALID = ('b31_spec2_fb25', 'b31_spec1_fb25', 'b29_occ_fb25')   # + the occupancy run (row 30, lf52 never admitted, D84)
CTL = FB25[:4]
SAME = [r for r in FB25[:7] if r not in INVALID]                               # 6 valid same-agent runs
SAME7 = FB25[:7]                                                               # the review copy's cohort (incl. spec2 A)

def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def phi(z): return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

obs = []            # (run, level_key, log_tokens, cracked)
supply, levels, pergame = {}, {}, {}
for run in FB25:
    d = json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt'))
    S = 0; L = 0; pg = {}
    for g in d['game_runs']:
        game = g['game_id'].split('-')[0]; h = g.get('history') or []; apl = g.get('actions_per_level') or []
        lc = g.get('levels_completed') or 0; cum = np.cumsum([0] + list(apl))
        tok = [x.get('generated_tokens') or 0 for x in h]; S += sum(tok); L += lc
        costs = []
        for i in range(min(len(apl), lc + 1)):
            t = sum(tok[cum[i]:cum[i + 1]])
            if i < lc:
                obs.append((run, (game, i), math.log(max(t, 1)), True)); costs.append(t)
            elif t > 0:
                obs.append((run, (game, i), math.log(t), False)); costs.append(-t)   # negative = open
        pg[game] = (costs, sum(tok))
    supply[run], levels[run], pergame[run] = S, L, pg

def fit(rows, runs, ref, iters=200):
    lev = sorted({o[1] for o in rows}); li = {k: j for j, k in enumerate(lev)}; ri = {r: j for j, r in enumerate(runs)}
    R = np.array([ri[o[0]] for o in rows]); Lx = np.array([li[o[1]] for o in rows])
    y = np.array([o[2] for o in rows]); c = np.array([o[3] for o in rows])
    # drop levels with no crack at all (pure walls carry no information on b)
    good = np.zeros(len(lev), bool); good[np.unique(Lx[c])] = True; keep = good[Lx]
    R, Lx, y, c = R[keep], Lx[keep], y[keep], c[keep]
    a = np.zeros(len(lev)); b = np.zeros(len(runs)); s = 1.0
    for _ in range(iters):
        mu = a[Lx] + b[R]
        z = (y - mu) / s
        lam = np.array([phi(v) / max(1 - Phi(v), 1e-12) for v in z])
        ey = np.where(c, y, mu + s * lam)
        ev = np.where(c, (y - mu) ** 2, s * s * (1 + z * lam))         # E[(Y - mu)^2]
        for _ in range(3):
            a = np.bincount(Lx, weights=ey - b[R], minlength=len(lev)) / np.maximum(np.bincount(Lx, minlength=len(lev)), 1)
            b = np.bincount(R, weights=ey - a[Lx], minlength=len(runs)) / np.maximum(np.bincount(R, minlength=len(runs)), 1)
            b -= np.mean([b[ri[r]] for r in ref])
        # variance: E[(Y - mu_new)^2] = E[(Y-mu)^2] + 2 (mu - mu_new) (E[Y] - mu) + (mu - mu_new)^2
        mun = a[Lx] + b[R]
        s = math.sqrt(np.mean(ev + 2 * (mu - mun) * (ey - mu) + (mu - mun) ** 2))
    return {r: b[ri[r]] for r in runs}, s, int(keep.sum()), int(c.sum())

bhat, s, n_obs, n_crk = fit(obs, FB25, CTL)
print(f'AFT fit on {len(FB25)} FB25 runs: {n_obs} level attempts ({n_crk} cracked) on levels cracked at least once; residual s {s:.2f}')

# control curve: levels cleared by wall-clock time m x 121 min in the control runs (the clock-bound counterfactual:
# an agent with m times the effective budget behaves like the control run stopped at m x 121 min)
clears = {}
for run in CTL:
    d = json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt')); ts = []
    for g in d['game_runs']:
        h = g.get('history') or []; apl = g.get('actions_per_level') or []; lc = g.get('levels_completed') or 0
        cum = np.cumsum([0] + list(apl)); wc = [x.get('wallclock_seconds') or 0 for x in h]
        ts += [wc[cum[i + 1] - 1] for i in range(lc) if 0 <= cum[i + 1] - 1 < len(wc)]
    clears[run] = np.array(ts)
T_END = 121 * 60
ms = np.linspace(0.5, 1.0, 11)
curve = np.array([np.mean([(clears[r] <= m * T_END).sum() for r in CTL]) for m in ms])
slope = (curve[-1] - curve[-4]) / math.log(ms[-1] / ms[-4])                 # levels per unit log budget, m 0.85-1.0
print('control clock curve L(m): ' + ', '.join(f'{m:.2f}:{v:.1f}' for m, v in zip(ms, curve)))
print(f'end slope dL/dlog m (0.85-1.00) = {slope:.1f} levels (sealed V5 implies (127-103)/ln 2 = {(127 - 103) / math.log(2):.1f} on average over 1x-2x)')
def L_pred(m):
    if m <= 1: return float(np.interp(m, ms, curve))
    return float(curve[-1] + slope * math.log(m))

S_ctl = np.mean([supply[r] for r in CTL])
print('\nrun              levels  supply(M)  b (log cost)  m_eff   predicted  observed-predicted')
res = {}
for r in FB25:
    m = supply[r] / S_ctl * math.exp(-bhat[r]); p = L_pred(m); res[r] = levels[r] - p
    print(f'{r:16s} {levels[r]:5d}   {supply[r] / 1e6:6.2f}    {bhat[r]:+.3f}       {m:5.3f}   {p:7.1f}    {levels[r] - p:+6.1f}')
same_b = [bhat[r] for r in SAME]
valid = [r for r in FB25 if r not in INVALID]
print(f'\nA/A spread of b over the {len(SAME)} valid same-agent runs (ctl + spec2 B, C): SD {np.std(same_b, ddof=1):.3f}')
print(f'levels SD over the same {len(SAME)} runs: {np.std([levels[r] for r in SAME], ddof=1):.1f}; residual SD after the two numbers: '
      f'{np.std([res[r] for r in SAME], ddof=1):.1f} (same-agent) / {np.std([res[r] for r in valid], ddof=1):.1f} (all {len(valid)} valid)')
others = [r for r in FB25 if r not in SAME7 and r not in INVALID]
obs_d = np.array([levels[r] - np.mean([levels[x] for x in CTL]) for r in others])
pred_d = obs_d - np.array([res[r] for r in others])
print(f'variant runs ({len(others)}): corr(predicted, observed level difference vs ctl) = {np.corrcoef(pred_d, obs_d)[0, 1]:.2f}')

# runs needed per arm to detect a 10% efficiency change (log 1.1), 80% power, two-sided 5%
eff = math.log(1.1); zz = (1.96 + 0.84) ** 2
sd_lv = np.std([levels[r] for r in SAME], ddof=1); dL = slope * eff
sd_b = np.std(same_b, ddof=1)
def n_arm(sd, delta): return 2 * zz * sd ** 2 / delta ** 2
print(f'\n10% efficiency change: +{dL:.1f} levels on the score scale vs {eff:.3f} on the b scale')
print(f'runs per arm, raw level totals: {n_arm(sd_lv, dL):.1f}; crack-cost b on shared levels: {n_arm(sd_b, eff):.1f}')
print(f'ratio (raw levels / b): {n_arm(sd_lv, dL) / n_arm(sd_b, eff):.1f}x on the same {len(SAME)} runs')
# how fragile is the ratio? leave one of the valid same-agent runs out at a time (both SDs from the same 5 runs)
jk = []
for drop in SAME:
    keep = [r for r in SAME if r != drop]
    jk.append((drop, n_arm(np.std([levels[r] for r in keep], ddof=1), dL), n_arm(np.std([bhat[r] for r in keep], ddof=1), eff)))
for drop, a, b_ in jk:
    print(f'  leave out {drop:15s}: raw levels {a:5.1f}, b {b_:5.1f}, ratio {a / b_:5.1f}x')
rat = [a / b_ for _, a, b_ in jk]
print(f'leave-one-out ratio range: {min(rat):.1f}x to {max(rat):.1f}x')
sd_lv7, sd_b7 = np.std([levels[r] for r in SAME7], ddof=1), np.std([bhat[r] for r in SAME7], ddof=1)
print(f'review-copy cohort (7 runs incl. invalid spec2 A): levels SD {sd_lv7:.2f}, b SD {sd_b7:.3f} -> '
      f'{n_arm(sd_lv7, dL):.1f} and {n_arm(sd_b7, eff):.1f} runs per arm ({n_arm(sd_lv7, dL) / n_arm(sd_b7, eff):.1f}x)')

# which variant comes closest to a real change in crack cost? (valid runs; repeated variants pooled)
groups = {'ht (harness)': ['b12_ht', 'b12_ht2'], 'sdt (harness)': ['b12_sdt', 'b12_sdt_r2'], 'len (prompt)': ['b12_len'],
          'x7b (prompt)': ['b12_x7b'], 'b15c (serving)': ['b15c_fb25'], 'ThinkingCap (model)': ['b28_tc_fb25'],
          'occ (serving)': ['b29_occ_fb25']}
groups = {g: [r for r in rs if r not in INVALID] for g, rs in groups.items()}
print('variant groups left out (partial runs only):', ', '.join(g for g, rs in groups.items() if not rs) or 'none')
groups = {g: rs for g, rs in groups.items() if rs}
b_ctl = np.mean([bhat[r] for r in CTL]); zs = {}
for g, rs in groups.items():
    zs[g] = (np.mean([bhat[r] for r in rs]) - b_ctl) / (sd_b * math.sqrt(1 / len(rs) + 1 / len(CTL)))
from statistics import NormalDist
zb = NormalDist().inv_cdf(1 - 0.05 / 2 / len(groups))
print(f'\nvariant crack cost vs control (A/A SD {sd_b:.3f}; {len(groups)} variants, Bonferroni |z| >= {zb:.2f}):')
for g, rs in groups.items():
    print(f'  {g:20s} runs {len(rs)}  mean b {np.mean([bhat[r] for r in rs]):+.3f}  z {zs[g]:+.1f}')
gmax = max(zs, key=lambda g: abs(zs[g]))
print(f'largest |z|: {gmax} {abs(zs[gmax]):.1f} ({abs(zs[gmax]):.3f} vs line {zb:.3f}); variants past the Bonferroni line: {sum(abs(z) >= zb for z in zs.values())}')
