"""B12 against the base notebook B1 with our own method: crack cost b on shared levels (exploratory, 5 Nov 2026).
Section 3.5's model log T(run, level) = a_l + b_r + e (lognormal AFT, level fixed effects, open attempts censored, EM),
fitted on the levels of the 10 dev games that the B1 runs played (25 minutes, 10 games, one stream each), together with
the same levels in the seven valid 121-minute B12 runs (25 games on 10 streams). b is relative to the mean of the two B1
runs. Token supply S is not compared: the two setups share streams differently. Loader and fit copied from
lever4b_two_numbers.py (unchanged logic). Run: python3 analysis/s37/b1_vs_b12_aft.py > analysis/s37/out/b1_vs_b12_aft.out"""
import gzip, json, math, statistics
import numpy as np
from common import HERE, CTL, SPEC2_VALID
B1 = ['b1_franzen', 'b1c_ctrl']; X6 = ['b4_x6']; OTHER10 = ['b3_x2', 'b8_x7', 'b9_x8', 'b10_x9']
B12 = CTL + SPEC2_VALID + ['t1c_fb25_ctl']
def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))
def phi(z): return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
def load(run):
    return json.load(gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt'))['game_runs']
GAMES10 = {g['game_id'].split('-')[0] for g in load('b1_franzen')}
def rows(run):
    out = []
    for g in load(run):
        game = g['game_id'].split('-')[0]
        if game not in GAMES10: continue
        h = g.get('history') or []; apl = g.get('actions_per_level') or []; lc = g.get('levels_completed') or 0
        cum = np.cumsum([0] + list(apl)); tok = [x.get('generated_tokens') or 0 for x in h]
        for i in range(min(len(apl), lc + 1)):
            t = sum(tok[cum[i]:cum[i + 1]])
            if i < lc: out.append((run, (game, i), math.log(max(t, 1)), True))
            elif t > 0: out.append((run, (game, i), math.log(t), False))
    return out
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
    return {r: b[ri[r]] for r in runs}, s, int(keep.sum()), int(c.sum())
RUNS = B1 + X6 + OTHER10 + B12
obs = [o for r in RUNS for o in rows(r)]
bh, s, n, nc = fit(obs, RUNS, B1)
print(f'AFT fit on the {len(GAMES10)} dev games of B1 ({", ".join(sorted(GAMES10))}): {len(RUNS)} runs, {n} attempts ({nc} cracked), s {s:.2f}')
for r in RUNS: print(f'  b {r}: {bh[r]:+.3f}')
bb = [bh[r] for r in B12]; m = statistics.mean(bb); sd = statistics.stdev(bb)
b1sd = abs(bh[B1[0]] - bh[B1[1]]) / math.sqrt(2)
print(f'B12 mean b vs B1: {m:+.3f} (SD over the 7 B12 runs {sd:.3f}); tokens per crack x{math.exp(m):.2f}; '
      f'B1 A/A: the two B1 runs differ by {abs(bh[B1[0]] - bh[B1[1]]):.3f}')
# bootstrap over runs within each arm (refit each time)
rng = np.random.default_rng(3); bs = []
by = {r: [o for o in obs if o[0] == r] for r in RUNS}
for k in range(200):
    pick1 = [B1[i] for i in rng.integers(0, 2, 2)]; pick12 = [B12[i] for i in rng.integers(0, len(B12), len(B12))]
    names = [f'{r}#{j}' for j, r in enumerate(pick1 + pick12)]
    rr = [(nm, o[1], o[2], o[3]) for nm, r in zip(names, pick1 + pick12) for o in by[r]]
    b_, *_ = fit(rr, names, names[:2], iters=60)
    bs.append(statistics.mean(b_[nm] for nm in names[2:]))
print(f'bootstrap over runs (200 refits): B12 - B1 in b, 90% interval {np.percentile(bs, 5):+.3f} to {np.percentile(bs, 95):+.3f}')
print(f'X6 (one run) vs B1: {bh["b4_x6"]:+.3f}')
