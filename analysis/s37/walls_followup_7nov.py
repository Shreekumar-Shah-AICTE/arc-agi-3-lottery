"""Do the walls level off with longer play, and how sure are the caps? (7 Nov 2026, exploratory, CPU only; written after
the V5 outcome and the 7 Nov outside reviews, so nothing here is a test.)

A cure model's wall share q is identified only if follow-up is long enough for the survival curve to flatten (the
sufficient follow-up problem; Maller and Zhou). One review asked to refit q on logs cut at 0.5x and 1x the budget, to
show whether it levels off, and for intervals on the unlimited-play caps and the 85% threshold.
Model and data as in the sealed V5 fit (v5_budget_prediction.py, imported read-only: same attempts, same 200-token
floor, same wall-clock cuts); maximum likelihood by Nelder-Mead instead of the sealed grid (the grid steps q by 0.01).
(1) Follow-up refits: the crack model (lognormal + walls) fitted on runs cut at a share of the 121-minute play time:
    the 4 controls and the 7 runs at 121 minutes cut at 0.25, 0.5, 0.75 and 1; the doubled run (V5) alone cut at 0.5,
    1, 1.5 and 2; the 7 runs plus V5 with V5 cut at 1 and at 2; the same for the 15 held-out games alone.
    Each line gives q and the unlimited-play cap it implies (exact form of distance_85.py).
(2) Game bootstrap (1,000 resamples of games with all their attempts, so a wall that recurs in every run counts once):
    q and the cap for the pooled fit (cap on the 25 public level counts) and the held-out fit (cap on the 15 held-out
    games), on the 4 controls and on the 7 runs plus V5; and how often q falls at or below the wall rate that 85% needs.
(3) Follow-up per attempt: the same attempts (7 runs plus V5), each censored at a token cap c, refitted for c from 50k to
    300k tokens; the Kaplan-Meier survival of attempts at those waits with the number still at risk; and a profile
    likelihood for q (attempts treated as independent, so narrower than the game bootstrap). Walls that are real show
    as a q that settles as c grows; walls that are slow cracks show as a q that keeps falling.
The likely walls by game and level are listed in analysis/s37/out/walls.out (wall census).
Run: python3 analysis/s37/walls_followup_7nov.py > analysis/s37/out/walls_followup_7nov.out (about 3 minutes)."""
import math
import numpy as np
from common import CTL, SPEC2_VALID
from v5_budget_prediction import load, attempts
RUNS7 = CTL + SPEC2_VALID + ['t1c_fb25_ctl']; V5 = 'v5_fb50'
G = {r: load(r) for r in RUNS7 + [V5]}
DEV = {'ar25', 'ft09', 'lp85', 'r11l', 're86', 'sb26', 'sc25', 'tr87', 'tu93', 'vc33'}
LEV = {}
for g in G[CTL[0]]: LEV[g['game_id'][:4]] = g['number_of_levels']
L_ALL = [LEV[k] for k in sorted(LEV)]; L_HOLD = [LEV[k] for k in sorted(LEV) if k not in DEV]
def cap_exact(q, Ls):   # distance_85.py: levels in order, a wall ends the game, share c(c+1)/(L(L+1))
    tot = 0.0
    for Lg in Ls:
        tot += (sum(((1 - q) ** j) * q * j * (j + 1) for j in range(Lg)) + ((1 - q) ** Lg) * Lg * (Lg + 1)) / (Lg * (Lg + 1))
    return 100 * tot / len(Ls)
def q85(Ls):
    lo, hi = 0.0, 0.3
    for _ in range(50):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if cap_exact(mid, Ls) >= 85 else (lo, mid)
    return lo
def per_game(runs, frac_of=lambda r: None, keep=lambda g: True):
    """{game: (t array, cleared array)} pooled over runs; frac_of(run) gives that run's cut (None = full run)"""
    out = {}
    for r in runs:
        for g in G[r]:
            k = g['game_id'][:4]
            if not keep(k): continue
            A = attempts(g, frac_of(r))[0]
            t = [max(a[1], 200) for a in A]; d = [a[2] for a in A]
            T, D = out.get(k, ([], [])); out[k] = (T + t, D + d)
    return {k: (np.array(v[0], float), np.array(v[1], bool)) for k, v in out.items()}
def log_sf_normal(z):   # log of the upper normal tail, vectorised (Numerical Recipes erfc, fractional error < 1.2e-7)
    x = z / math.sqrt(2); a = np.abs(x); t = 1 / (1 + 0.5 * a)
    p = -a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (
        -1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    log_erfc_pos = np.log(t) + p
    return np.where(x >= 0, math.log(0.5) + log_erfc_pos, np.log1p(-0.5 * np.exp(log_erfc_pos)))
LOG2PI = math.log(2 * math.pi)
def nll(p, lt, d):
    m, ls, lq = p; s = math.exp(ls)
    if lq > 30: return 1e12
    q = 1 / (1 + math.exp(-lq)); z = (lt - m) / s
    lf = -0.5 * z * z - 0.5 * LOG2PI - ls - lt
    l1q = math.log1p(-q) if q < 1 else -1e300
    ll = np.where(d, l1q + lf, np.logaddexp(math.log(max(q, 1e-300)), l1q + log_sf_normal(z)))
    return -float(ll.sum())
def nelder_mead(f, x0, step=0.4, iters=1500, tol=1e-10):
    n = len(x0); X = [np.array(x0, float)] + [np.array(x0, float) + step * np.eye(n)[i] for i in range(n)]; F = [f(x) for x in X]
    for _ in range(iters):
        o = np.argsort(F); X = [X[i] for i in o]; F = [F[i] for i in o]
        if abs(F[-1] - F[0]) < tol: break
        c = np.mean(X[:-1], axis=0); xr = c + (c - X[-1]); fr = f(xr)
        if fr < F[0]:
            xe = c + 2 * (c - X[-1]); fe = f(xe); X[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]: X[-1], F[-1] = xr, fr
        else:
            xc = c + 0.5 * (X[-1] - c); fc = f(xc)
            if fc < F[-1]: X[-1], F[-1] = xc, fc
            else: X = [X[0]] + [X[0] + 0.5 * (x - X[0]) for x in X[1:]]; F = [F[0]] + [f(x) for x in X[1:]]
    i = int(np.argmin(F)); return X[i], -F[i]
def fit(t, d, starts=None):
    lt = np.log(t); best = None
    for x0 in (starts or [[m0, ls0, lq0] for m0 in (math.log(15e3), math.log(30e3)) for ls0 in (-0.2, 0.2) for lq0 in (-3.0, -1.5)]):
        x, ll = nelder_mead(lambda p: nll(p, lt, d), x0)
        if best is None or ll > best[1]: best = (x, ll)
    return best
def qof(x): return 1 / (1 + math.exp(-x[2]))
def pooled(pg): return np.concatenate([v[0] for v in pg.values()]), np.concatenate([v[1] for v in pg.values()])
def line(label, pg, Ls):
    t, d = pooled(pg); x, ll = fit(t, d); q = qof(x)
    print(f'  {label:<44} attempts {len(t):4d} (open {int((~d).sum()):3d}); longest wait seen {t.max() / 1e3:5.0f}k; '
          f'q {q:.3f} median {math.exp(x[0]) / 1e3:5.1f}k s {math.exp(x[1]):.2f}; cap {cap_exact(q, Ls):.1f}%')
    return x
hold = lambda k: k not in DEV
print(f'85% needs q <= {q85(L_ALL):.3f} on the 25 public level counts and q <= {q85(L_HOLD):.3f} on the 15 held-out games (exact cap form)')
print('\n(1) follow-up refits (crack model, Nelder-Mead MLE); cut = share of the 121-minute play time; cap = unlimited-play cap')
print(' all 25 games (cap on the 25 public level counts)')
for f in (0.25, 0.5, 0.75, 1.0): line(f'4 controls cut at {f:g}', per_game(CTL, lambda r: None if f == 1.0 else f), L_ALL)
for f in (0.25, 0.5, 0.75, 1.0): line(f'7 runs at 121 min cut at {f:g}', per_game(RUNS7, lambda r: None if f == 1.0 else f), L_ALL)
for f in (0.5, 1.0, 1.5, 2.0): line(f'V5 alone cut at {f:g}', per_game([V5], lambda r: None if f == 2.0 else f), L_ALL)
for f in (1.0, 2.0): line(f'7 runs + V5, V5 cut at {f:g}', per_game(RUNS7 + [V5], lambda r: (None if f == 2.0 else f) if r == V5 else None), L_ALL)
print(' 15 held-out games (cap on the 15 held-out games)')
for f in (0.5, 1.0): line(f'4 controls cut at {f:g}', per_game(CTL, lambda r: None if f == 1.0 else f, hold), L_HOLD)
for f in (0.5, 1.0): line(f'7 runs at 121 min cut at {f:g}', per_game(RUNS7, lambda r: None if f == 1.0 else f, hold), L_HOLD)
for f in (1.0, 2.0): line(f'V5 alone cut at {f:g}', per_game([V5], lambda r: None if f == 2.0 else f, hold), L_HOLD)
for f in (1.0, 2.0): line(f'7 runs + V5, V5 cut at {f:g}', per_game(RUNS7 + [V5], lambda r: (None if f == 2.0 else f) if r == V5 else None, hold), L_HOLD)

print('\n(2) game bootstrap, 1,000 resamples of games (all their attempts in all runs); 90% intervals are the 5th to 95th percentiles')
rng = np.random.default_rng(20261107)
for label, runs, keep, Ls in (('pooled fit, 4 controls', CTL, lambda k: True, L_ALL), ('pooled fit, 7 runs + V5', RUNS7 + [V5], lambda k: True, L_ALL),
                              ('held-out fit, 4 controls', CTL, hold, L_HOLD), ('held-out fit, 7 runs + V5', RUNS7 + [V5], hold, L_HOLD)):
    pg = per_game(runs, keep=keep); keys = sorted(pg); t, d = pooled(pg); x0, _ = fit(t, d); qq, cc = [], []
    for b in range(1000):
        pick = rng.integers(0, len(keys), len(keys))
        tb = np.concatenate([pg[keys[i]][0] for i in pick]); db = np.concatenate([pg[keys[i]][1] for i in pick])
        if db.all(): qq.append(0.0); cc.append(cap_exact(0.0, Ls)); continue
        x, _ = fit(tb, db, starts=[list(x0), [x0[0], x0[1], -4.0]]); q = qof(x); qq.append(q); cc.append(cap_exact(q, Ls))
    qq, cc = np.array(qq), np.array(cc); thr = q85(Ls)
    print(f'  {label:<26} q {qof(x0):.3f} (90% {np.percentile(qq, 5):.3f} to {np.percentile(qq, 95):.3f}); cap {cap_exact(qof(x0), Ls):.1f}% '
          f'(90% {np.percentile(cc, 5):.2f} to {np.percentile(cc, 95):.2f}); q <= {thr:.3f} (85% reachable) in {100 * np.mean(qq <= thr):.0f}% of resamples; '
          f'games {len(keys)}')

print('\n(3) follow-up per attempt: the 7 runs + V5 with every attempt censored at a token cap c (same attempts each time)')
for label, keep, Ls in (('all 25 games', lambda k: True, L_ALL), ('15 held-out games', hold, L_HOLD)):
    t, d = pooled(per_game(RUNS7 + [V5], keep=keep))
    print(f' {label} (cap on {"the 25 public level counts" if Ls is L_ALL else "the 15 held-out games"})')
    for c in (50e3, 100e3, 150e3, 200e3, 250e3, 300e3, None):
        tc = t if c is None else np.minimum(t, c); dc = d if c is None else (d & (t <= c))
        x, _ = fit(tc, dc); q = qof(x)
        print(f'  censored at {"none" if c is None else f"{c / 1e3:.0f}k":>5}: clears {int(dc.sum()):4d} of {len(t)}; q {q:.3f} median {math.exp(x[0]) / 1e3:5.1f}k s {math.exp(x[1]):.2f}; cap {cap_exact(q, Ls):.1f}%')
    o = np.argsort(t); ts, ds = t[o], d[o]; S, n, i, rows = 1.0, len(ts), 0, []
    marks = [50e3, 100e3, 150e3, 200e3, 250e3, 300e3]; out = {}
    while i < len(ts):
        j = i
        while j < len(ts) and ts[j] == ts[i]: j += 1
        dd = int(ds[i:j].sum())
        for mk in marks:
            if mk not in out and ts[i] > mk: out[mk] = (S, n)
        if dd: S *= 1 - dd / n
        n -= j - i; i = j
    for mk in marks:
        if mk not in out: out[mk] = (S, n)
    print('  Kaplan-Meier survival of attempts (still at risk): ' + '; '.join(f'{mk / 1e3:.0f}k {out[mk][0]:.3f} ({out[mk][1]})' for mk in marks))
    lt = np.log(t); xb, llb = fit(t, d); prof = []
    for qv in np.arange(0.0, 0.2001, 0.005):
        lq = math.log(max(qv, 1e-9) / (1 - qv))
        x2, ll2 = nelder_mead(lambda p: nll([p[0], p[1], lq], lt, d), [xb[0], xb[1]])
        prof.append((qv, ll2))
    inside = [qv for qv, l2 in prof if llb - l2 <= 1.3528]
    print(f'  profile likelihood for q: best {qof(xb):.3f}; 90% likelihood-ratio interval about {min(inside):.3f} to {max(inside):.3f} '
          f'(attempts treated as independent); log-likelihood at q 0: {dict(prof)[0.0] - llb:+.2f}')
