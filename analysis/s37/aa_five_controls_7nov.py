"""A/A check on the five byte-identical control runs (7 Nov 2026, exploratory, CPU only; written after the 7 Nov reviews).

Review item: the six 'same-agent' runs behind the A/A SD of b (0.051, Section 4.5) include two spec2 runs (two-step
speculative decoding, judged noise in D87), b32_spec2_r3 sets most of that SD, and the valid T1c control of 4 Nov
(t1c_fb25_ctl, the control config run again on the day of T1; Appendix D) was left out. Here we refit Section 3.5's model
on the 17 FB25 runs plus t1c_fb25_ctl (loader and fit copied from lever4b_two_numbers.py, unchanged logic; b relative to
the mean of the 4 original controls), then report, for the 4 and the 5 true controls without spec2, the A/A SD of b, the
level SD, the runs-per-arm ratio of Section 4.6 and the variants' z, with spec2 now tested as a variant.
Run: python3 analysis/s37/aa_five_controls_7nov.py > analysis/s37/out/aa_five_controls_7nov.out
"""
import gzip, json, math
from statistics import NormalDist
import numpy as np
from common import HERE, CTL, SPEC2_VALID, INVALID
FB25 = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b31_spec2_fb25', 'b32_spec2_r2', 'b32_spec2_r3',
        'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b', 'b15c_fb25', 'b28_tc_fb25', 'b29_occ_fb25',
        'b31_spec1_fb25']
T1C = 't1c_fb25_ctl'; RUNS = FB25 + [T1C]; CTL5 = CTL + [T1C]
def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def phi(z): return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
obs, supply, levels, nobs = [], {}, {}, {}
for run in RUNS:
    d = json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt')); S = L = 0
    for g in d['game_runs']:
        game = g['game_id'].split('-')[0]; h = g.get('history') or []; apl = g.get('actions_per_level') or []
        lc = g.get('levels_completed') or 0; cum = np.cumsum([0] + list(apl))
        tok = [x.get('generated_tokens') or 0 for x in h]; S += sum(tok); L += lc
        for i in range(min(len(apl), lc + 1)):
            t = sum(tok[cum[i]:cum[i + 1]])
            if i < lc: obs.append((run, (game, i), math.log(max(t, 1)), True))
            elif t > 0: obs.append((run, (game, i), math.log(t), False))
    supply[run], levels[run] = S, L
def fit(rows, runs, ref, iters=200):
    lev = sorted({o[1] for o in rows}); li = {k: j for j, k in enumerate(lev)}; ri = {r: j for j, r in enumerate(runs)}
    R = np.array([ri[o[0]] for o in rows]); Lx = np.array([li[o[1]] for o in rows])
    y = np.array([o[2] for o in rows]); c = np.array([o[3] for o in rows])
    good = np.zeros(len(lev), bool); good[np.unique(Lx[c])] = True; keep = good[Lx]
    R, Lx, y, c = R[keep], Lx[keep], y[keep], c[keep]
    a = np.zeros(len(lev)); b = np.zeros(len(runs)); s = 1.0
    for _ in range(iters):
        mu = a[Lx] + b[R]; z = (y - mu) / s
        lam = np.array([phi(v) / max(1 - Phi(v), 1e-12) for v in z])
        ey = np.where(c, y, mu + s * lam); ev = np.where(c, (y - mu) ** 2, s * s * (1 + z * lam))
        for _ in range(3):
            a = np.bincount(Lx, weights=ey - b[R], minlength=len(lev)) / np.maximum(np.bincount(Lx, minlength=len(lev)), 1)
            b = np.bincount(R, weights=ey - a[Lx], minlength=len(runs)) / np.maximum(np.bincount(R, minlength=len(runs)), 1)
            b -= np.mean([b[ri[r]] for r in ref])
        mun = a[Lx] + b[R]; s = math.sqrt(np.mean(ev + 2 * (mu - mun) * (ey - mu) + (mu - mun) ** 2))
    return {r: b[ri[r]] for r in runs}, s, int(keep.sum()), int(c.sum()), np.bincount(R, minlength=len(runs))
b17, s17, _, _, _ = fit([o for o in obs if o[0] != T1C], FB25, CTL)
bh, s, n, nc, per_run = fit(obs, RUNS, CTL)
n_r = dict(zip(RUNS, per_run))
print(f'AFT refit on the 17 FB25 runs plus {T1C}: {n} attempts ({nc} cracked), residual s {s:.2f} (17-run fit: s {s17:.2f})')
print('largest change of a 17-run b from adding T1c: %.4f' % max(abs(bh[r] - b17[r]) for r in FB25))
for r in CTL5 + SPEC2_VALID: print(f'  {r:15s} b {bh[r]:+.3f}  levels {levels[r]:3d}  supply {supply[r] / 1e6:.2f}M  attempts {n_r[r]}')
def sd(v): return float(np.std(v, ddof=1))
# Section 4.6's power calculation: 10% efficiency change, 80% power, two-sided 5%; slope from the 4 controls' clock curve
clears = []
for run in CTL:
    d = json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt')); ts = []
    for g in d['game_runs']:
        h = g.get('history') or []; apl = g.get('actions_per_level') or []; lc = g.get('levels_completed') or 0
        cum = np.cumsum([0] + list(apl)); wc = [x.get('wallclock_seconds') or 0 for x in h]
        ts += [wc[cum[i + 1] - 1] for i in range(lc) if 0 <= cum[i + 1] - 1 < len(wc)]
    clears.append(np.array(ts))
ms = np.linspace(0.5, 1.0, 11); T_END = 121 * 60
curve = np.array([np.mean([(c <= m * T_END).sum() for c in clears]) for m in ms])
slope = (curve[-1] - curve[-4]) / math.log(ms[-1] / ms[-4]); eff = math.log(1.1); dL = slope * eff; zz = (1.96 + 0.84) ** 2
def n_arm(v, delta): return 2 * zz * v ** 2 / delta ** 2
# 90% interval of an SD from k runs (chi-square, df = k - 1), and how often k identical runs would spread this little
def gammainc_lower(a, x, terms=400):          # regularized lower incomplete gamma P(a, x), series
    t = s_ = 1.0 / a
    for k in range(1, terms): t *= x / (a + k); s_ += t
    return s_ * math.exp(-x + a * math.log(x) - math.lgamma(a))
def chi2_cdf(x, df): return gammainc_lower(df / 2, x / 2)
def chi2_q(p, df):
    lo, hi = 1e-9, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if chi2_cdf(mid, df) < p: lo = mid
        else: hi = mid
    return (lo + hi) / 2
print(f'\n10% efficiency change: +{dL:.1f} levels (end slope {slope:.1f}) vs {eff:.3f} on the b scale')
cohorts = {'paper (4 ctl + spec2 B, C)': CTL + SPEC2_VALID, '4 controls': CTL, '5 controls (with T1c)': CTL5,
           '5 controls + spec2 B, C': CTL5 + SPEC2_VALID}
print('cohort                        k  b SD   90% interval    levels SD  runs/arm levels  runs/arm b  ratio')
for name, rs in cohorts.items():
    k = len(rs); sb = sd([bh[r] for r in rs]); sl = sd([levels[r] for r in rs]); df = k - 1
    lo, hi = sb * math.sqrt(df / chi2_q(0.95, df)), sb * math.sqrt(df / chi2_q(0.05, df))
    print(f'{name:28s} {k:2d}  {sb:.3f}  {lo:.3f} to {hi:.3f}  {sl:6.2f}     {n_arm(sl, dL):7.1f}        {n_arm(sb, eff):5.2f}  {n_arm(sl, dL) / n_arm(sb, eff):6.1f}x')
# what independent attempts would give: b_r is a mean over the run's attempts, so SD about s / sqrt(attempts)
exp_sd = s * math.sqrt(np.mean([1 / n_r[r] for r in CTL5]))
sb5 = sd([bh[r] for r in CTL5])
print(f'\nif attempts were independent given the level, a run\'s b would have SD about s / sqrt(attempts) = {exp_sd:.3f}; '
      f'the chance that 5 such runs spread {sb5:.3f} or less is {chi2_cdf(4 * sb5 ** 2 / exp_sd ** 2, 4):.4f}')
# leave-one-out over the 5 controls: the ratio and each control tested as a one-run variant against the other four
print('\nleave one of the 5 controls out:')
rat = []
for drop in CTL5:
    keep = [r for r in CTL5 if r != drop]; sb = sd([bh[r] for r in keep]); sl = sd([levels[r] for r in keep])
    m = np.mean([bh[r] for r in keep]); z = (bh[drop] - m) / (sb * math.sqrt(1 + 1 / len(keep))); rat.append(n_arm(sl, dL) / n_arm(sb, eff))
    print(f'  {drop:15s}: b SD of the other 4 {sb:.3f}, levels SD {sl:5.2f}, ratio {rat[-1]:6.1f}x; tested as a variant z {z:+.2f}')
print(f'leave-one-out ratio range: {min(rat):.1f}x to {max(rat):.1f}x')
groups = {'ht': ['b12_ht', 'b12_ht2'], 'sdt': ['b12_sdt', 'b12_sdt_r2'], 'len': ['b12_len'], 'x7b': ['b12_x7b'],
          'b15c': ['b15c_fb25'], 'ThinkingCap': ['b28_tc_fb25'], 'spec2': SPEC2_VALID}
groups = {g: [r for r in rs if r not in INVALID] for g, rs in groups.items()}
for name, rs_ctl in (('4 controls', CTL), ('5 controls', CTL5)):
    sb = sd([bh[r] for r in rs_ctl]); m = np.mean([bh[r] for r in rs_ctl]); zb = NormalDist().inv_cdf(1 - 0.05 / 2 / len(groups))
    zs = {g: (np.mean([bh[r] for r in rs]) - m) / (sb * math.sqrt(1 / len(rs) + 1 / len(rs_ctl))) for g, rs in groups.items()}
    print(f'\nvariants against the {name} (A/A SD {sb:.3f}, control mean {m:+.3f}; {len(groups)} variants incl. spec2, Bonferroni |z| >= {zb:.2f}):')
    for g, rs in groups.items():
        print(f'  {g:12s} runs {len(rs)}  mean b {np.mean([bh[r] for r in rs]):+.3f}  z {zs[g]:+.1f}')
    print(f'  past the line: {sum(abs(z) >= zb for z in zs.values())} of {len(zs)} ({", ".join(g for g in zs if abs(zs[g]) >= zb)})')
