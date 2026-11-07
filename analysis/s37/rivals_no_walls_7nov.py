"""Walls or a heavy tail? Rival waiting-time models without walls (7 Nov 2026, exploratory, CPU only; written after the
V5 outcome and the 7 Nov outside reviews, so nothing here is a test).

The crack model is a mixture cure model: a share q of level attempts never crack (walls), the rest wait a lognormal time
in generated tokens. The reviews asked whether a heavy-tailed wait with no walls (q = 0) explains the same data. Four
families, each with q fixed at 0 and with q free: lognormal (the crack model when q is free), log-logistic, Weibull and
Lomax (Pareto II, a power-law tail). Data and censoring as in the sealed V5 fit (v5_budget_prediction.py, imported
read-only: same attempts, same 200-token floor); maximum likelihood by Nelder-Mead instead of the sealed grid.
(1) In-sample fit (log-likelihood, AIC, BIC) on the 4 controls (the sealed fit's 480 attempts), on the 7 runs of this agent
    at 121 minutes (4 controls, spec2 B and C, T1c) and on those 7 plus V5.
(2) Out of sample: fitted on the 7 runs at 121 minutes, the log-likelihood of the 155 attempts of the doubled run (V5),
    whose long waits the 121-minute runs cannot see, and V5's clears past 150k tokens against each model's expectation
    (cumulative hazard over each attempt's exposure, as in v5_followup2.py).
(3) Leave one game out: fitted on the 7 runs plus V5 without one game, the log-likelihood of that game's attempts (25 folds).
(4) The sealed V5 calculation redone with each model fitted on the 4 controls (same continuation simulator as the seal):
    predicted total levels at twice the play time, against 141 observed and the sealed 127.
(5) The same continuation at four times the play time, where walls and a long tail part more clearly.
Run: python3 analysis/s37/rivals_no_walls_7nov.py > analysis/s37/out/rivals_no_walls_7nov.out (about 3 minutes)."""
import math
import numpy as np
from common import CTL, SPEC2_VALID
from v5_budget_prediction import load, attempts, states, ninv
RUNS7 = CTL + SPEC2_VALID + ['t1c_fb25_ctl']; V5 = 'v5_fb50'
G = {r: load(r) for r in RUNS7 + [V5]}
def att(runs, keep=lambda g: True):
    A = [a for r in runs for g in G[r] if keep(g['game_id'][:4]) for a in attempts(g)[0]]
    return np.array([max(a[1], 200) for a in A], float), np.array([a[2] for a in A], bool)
def Phi(z): return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
LOG2PI = math.log(2 * math.pi)
# each family: log S0(t) and log f0(t) for parameters (m, log s); m is a log time scale
def fam(kind, t, m, ls):
    s = math.exp(ls); lt = np.log(t); z = (lt - m) / s
    if kind == 'lognormal':
        lS = np.log(np.maximum(0.5 * np.vectorize(math.erfc)(z / math.sqrt(2)), 1e-300)); lf = -0.5 * z * z - 0.5 * LOG2PI - ls - lt
    elif kind == 'loglogistic':
        lS = -np.logaddexp(0, z); lf = z - ls - lt - 2 * np.logaddexp(0, z)
    elif kind == 'weibull':               # S = exp(-(t / e^m)^k), k = 1 / s
        k = 1 / s; x = np.exp(k * (lt - m)); lS = -x; lf = math.log(k) + (k - 1) * (lt - m) - m - x
    else:                                 # lomax: S = (1 + t / e^m)^-a, a = 1 / s
        a = 1 / s; l1 = np.log1p(t / math.exp(m)); lS = -a * l1; lf = math.log(a) - m - (a + 1) * l1
    return lS, lf
def loglik(p, kind, cure, t, d):
    m, ls = p[0], p[1]; lS, lf = fam(kind, t, m, ls)
    if not cure: return float(np.sum(np.where(d, lf, lS)))
    q = 1 / (1 + math.exp(-p[2])); return float(np.sum(np.where(d, math.log(1 - q) + lf, np.logaddexp(math.log(q), math.log(1 - q) + lS))))
def nelder_mead(f, x0, step=0.4, iters=900, tol=1e-9):
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
MODELS = [(k, c) for k in ('lognormal', 'loglogistic', 'weibull', 'lomax') for c in (True, False)]
def name(k, c): return f'{k}{" + walls" if c else ", q = 0"}'
def fit(kind, cure, t, d):
    best = None
    for m0 in (math.log(10e3), math.log(22e3), math.log(50e3)):
        for ls0 in (-0.5, 0.0, 0.5):
            x0 = [m0, ls0] + ([-2.5] if cure else [])
            x, ll = nelder_mead(lambda p: -loglik(p, kind, cure, t, d), x0)
            if best is None or ll > best[1]: best = (x, ll)
    return best
def qof(x, cure): return 1 / (1 + math.exp(-x[2])) if cure else 0.0
def S_full(tt, x, kind, cure):
    lS, _ = fam(kind, np.atleast_1d(np.asarray(tt, float)), x[0], x[1]); q = qof(x, cure); return q + (1 - q) * np.exp(lS)
def H(tt, x, kind, cure): return -np.log(np.maximum(S_full(tt, x, kind, cure), 1e-300))
def desc(x, kind, cure):
    s = math.exp(x[1]); q = qof(x, cure); med = float(np.exp(x[0]))
    shape = {'lognormal': f's {s:.2f}', 'loglogistic': f's {s:.2f}', 'weibull': f'k {1 / s:.2f}', 'lomax': f'a {1 / s:.2f}'}[kind]
    return f'q {q:.3f} scale {med / 1e3:.1f}k {shape}'
print('(1) in-sample fit; AIC = 2k - 2 loglik, BIC = k ln n - 2 loglik; dAIC and dBIC against the crack model (lognormal + walls)')
sets = {'4 controls (sealed fit data)': CTL, '7 runs at 121 min': RUNS7, '7 runs + V5': RUNS7 + [V5]}
FITS = {}
for lab, runs in sets.items():
    t, d = att(runs); n = len(t); res = {}
    for kind, cure in MODELS:
        x, ll = fit(kind, cure, t, d); kpar = 3 if cure else 2; res[(kind, cure)] = (x, ll, 2 * kpar - 2 * ll, kpar * math.log(n) - 2 * ll)
    FITS[lab] = res; base = res[('lognormal', True)]
    print(f'  {lab}: {n} attempts ({int(d.sum())} clears)')
    for (kind, cure), (x, ll, aic, bic) in sorted(res.items(), key=lambda kv: kv[1][2]):
        print(f'    {name(kind, cure):24s} loglik {ll:9.1f}  dAIC {aic - base[2]:+6.1f}  dBIC {bic - base[3]:+6.1f}  {desc(x, kind, cure)}')
print('\n(2) fitted on the 7 runs at 121 min, scored on the doubled run V5')
tv, dv = att([V5]); res7 = FITS['7 runs at 121 min']
print(f'  V5: {len(tv)} attempts ({int(dv.sum())} clears); clears past 150k tokens {int((dv & (tv >= 150e3)).sum())} of {int((tv > 150e3).sum())} attempts that got there')
outv = {}
for kind, cure in MODELS:
    x = res7[(kind, cure)][0]; llv = loglik(x, kind, cure, tv, dv)
    sel = tv > 150e3; E = float(np.sum(H(tv[sel], x, kind, cure) - H(150e3, x, kind, cure)))
    Eall = float(np.sum(H(tv, x, kind, cure))); outv[(kind, cure)] = llv
    print(f'    {name(kind, cure):24s} V5 loglik {llv:8.1f}  clears past 150k expected {E:4.1f}; all clears expected {Eall:5.1f}')
b = outv[('lognormal', True)]
print('  V5 loglik relative to the crack model: ' + '; '.join(f'{name(k, c)} {outv[(k, c)] - b:+.1f}' for k, c in MODELS))
print('\n(3) leave one game out (7 runs + V5): summed held-out log-likelihood of each game\'s attempts, relative to the crack model')
games = sorted({g['game_id'][:4] for g in G[CTL[0]]}); cv = {m: 0.0 for m in MODELS}
for gm in games:
    tr_t, tr_d = att(RUNS7 + [V5], lambda g, gm=gm: g != gm); te_t, te_d = att(RUNS7 + [V5], lambda g, gm=gm: g == gm)
    for kind, cure in MODELS:
        x, _ = fit(kind, cure, tr_t, tr_d); cv[(kind, cure)] += loglik(x, kind, cure, te_t, te_d)
b = cv[('lognormal', True)]
for m in sorted(cv, key=lambda m: -cv[m]): print(f'    {name(*m):24s} held-out loglik {cv[m]:9.1f}  ({cv[m] - b:+.1f})')
print('\n(4) the sealed V5 calculation with each model fitted on the 4 controls (1,000 continuations per control run)')
def tinv(u, x, kind, cure):
    q = qof(x, cure); p = (u - q) / (1 - q); m, s = x[0], math.exp(x[1])
    if kind == 'lognormal': return math.exp(m + s * ninv(1 - p))
    if kind == 'loglogistic': return math.exp(m + s * math.log((1 - p) / p))
    if kind == 'weibull': return math.exp(m) * (-math.log(p)) ** s
    return math.exp(m) * (p ** (-s) - 1)
def simulate(sts, x, kind, cure, rng, R):
    tot = np.zeros(R); q = qof(x, cure)
    for gid, done, nl, c, delta, fin in sts:
        lv = np.full(R, done, float)
        if not fin:
            sc = float(S_full(max(c, 1.0), x, kind, cure)[0])
            for rr in range(R):
                budget, first, n = delta, True, done
                while n < nl:
                    u = rng.random() * sc if first else rng.random()
                    if u <= q or u <= 0: break
                    need = tinv(u, x, kind, cure) - (c if first else 0.0)
                    if need > budget: break
                    budget -= need; n += 1; first = False
                lv[rr] = n
        tot += lv
    return tot
rng = np.random.default_rng(20261107); res4 = FITS['4 controls (sealed fit data)']
for kind, cure in MODELS:
    x = res4[(kind, cure)][0]; tots = [simulate(states(G[r]), x, kind, cure, rng, 1000) for r in CTL]
    allt = np.concatenate(tots)
    print(f'    {name(kind, cure):24s} 2x total levels {np.mean([s.mean() for s in tots]):6.1f} (model-only 5-95% {np.quantile(allt, .05):.0f}-{np.quantile(allt, .95):.0f}); observed 141, sealed 127')
print('\n(5) where the two readings part: the same continuation with three more 121-minute budgets (four times the play time;'
      ' stream redistribution as in the seal), each model fitted on the 4 controls')
rng = np.random.default_rng(20261108)
for kind, cure in MODELS[:4]:
    x = res4[(kind, cure)][0]
    tots = [simulate([(a, b_, c_, d_, 3 * e_, f_) for a, b_, c_, d_, e_, f_ in states(G[r])], x, kind, cure, rng, 1000) for r in CTL]
    allt = np.concatenate(tots)
    print(f'    {name(kind, cure):24s} 4x total levels {np.mean([s.mean() for s in tots]):6.1f} (model-only 5-95% {np.quantile(allt, .05):.0f}-{np.quantile(allt, .95):.0f}) of 183')
