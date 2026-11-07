"""Builds paper/notebook/how_many_runs.ipynb (4 Nov 2026): the "how many runs do you need" notebook for ARC-AGI-3.
Stand-alone (numpy + standard library), so it runs on Kaggle as is. Numbers match analysis/s37/out/lever4_runcount.out.
Not published: posting it on Kaggle is the author's call."""
import nbformat as nbf
from pathlib import Path
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# How many runs do you need? Run-to-run noise on ARC-AGI-3

Identical code does not give identical ARC-AGI-3 scores. Five byte-identical submissions of one of our agents scored with
an SD of **2.87 points** on the public leaderboard, and six runs of one agent on the 25 public games varied by
**5.3 levels** (SD; mean 102.5 levels). This notebook turns a noise level into seven practical answers:

1. how many runs per arm a comparison needs,
2. how large a difference one run per arm can show,
3. how uncertain the noise estimate itself is, and
4. how much keeping the best of *n* identical resubmissions lifts a public score,
5. which of two agents needs fewer tokens per crack on the same levels (section 7, from run logs), and
6. whether the crack model forecasts your own runs: fit on the first half, predict the second (section 8).
7. what that fit implies if walls are permanent: the most that unlimited play can reach, with a bootstrap over games
   (section 9).

The numbers come from our ARC Prize 2026 paper (Shreekumar Shah). Change `SIGMA_LB` and `SIGMA_DEV`, or paste your own
scores in the last section."""),
code("""import math
from statistics import NormalDist
import numpy as np

SIGMA_LB = 2.87    # SD of one leaderboard submission, points (5 byte-identical submissions)
SIGMA_DEV = 5.3    # SD of one dev run on the 25 public games, levels (6 runs of one agent)
DEV_MEAN = 102.5   # mean levels of those 6 runs
ALPHA, POWER = 0.05, 0.80
Z = NormalDist().inv_cdf(1 - ALPHA / 2) + NormalDist().inv_cdf(POWER)

def runs_per_arm(sigma, delta):
    \"\"\"Normal approximation: n = 2 (z_a + z_b)^2 sigma^2 / delta^2 runs in each arm.\"\"\"
    return math.ceil(2 * Z * Z * sigma * sigma / (delta * delta))

def smallest_difference(sigma, n):
    \"\"\"Smallest true difference that n runs per arm detect with the chosen power.\"\"\"
    return Z * sigma * math.sqrt(2 / n)"""),
md("## 1. Runs per arm (two-sided test at 0.05, power 0.8)"),
code("""print(f'Leaderboard, SD {SIGMA_LB} points per submission')
for d in (1, 2, 3, 5, 8):
    print(f'  true gain +{d} points: {runs_per_arm(SIGMA_LB, d):4d} submissions per arm')
print(f'Development runs on the 25 public games, SD {SIGMA_DEV} levels per run (mean {DEV_MEAN})')
for pct in (2, 5, 10, 20):
    d = DEV_MEAN * pct / 100
    print(f'  true gain +{pct}% levels (+{d:.1f}): {runs_per_arm(SIGMA_DEV, d):4d} runs per arm')"""),
md("## 2. What a small comparison can show"),
code("""for n in (1, 2, 3, 5, 10):
    lb, dev = smallest_difference(SIGMA_LB, n), smallest_difference(SIGMA_DEV, n)
    print(f'{n:2d} run(s) per arm: leaderboard {lb:5.1f} points | dev {dev:5.1f} levels ({100 * dev / DEV_MEAN:.0f}%)')"""),
md("""## 3. With few runs, check the formula by simulation

The formula assumes the SD is known. In practice it is estimated from the same runs, so the test is a t-test and the
real power at small *n* is a little lower. A quick simulation finds the smallest *n* that reaches power 0.8."""),
code("""rng = np.random.default_rng(0)

def t_stats(sigma, delta, n, R):
    a, b = rng.normal(0, sigma, (R, n)), rng.normal(delta, sigma, (R, n))
    return (b.mean(1) - a.mean(1)) / np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / n)

def t_power(sigma, delta, n, R=40000):
    crit = np.quantile(np.abs(t_stats(sigma, 0.0, n, R)), 1 - ALPHA)   # simulated critical value
    return float((np.abs(t_stats(sigma, delta, n, R)) > crit).mean())

for label, sigma, delta in (('leaderboard +3 points', SIGMA_LB, 3), ('leaderboard +5 points', SIGMA_LB, 5),
                            ('dev +10% levels', SIGMA_DEV, 0.10 * DEV_MEAN), ('dev +20% levels', SIGMA_DEV, 0.20 * DEV_MEAN)):
    n0 = max(2, runs_per_arm(sigma, delta)); n = n0
    while t_power(sigma, delta, n) < POWER:
        n += 1
    print(f'{label}: formula {n0} per arm (simulated power {t_power(sigma, delta, n0):.2f}); t-test needs {n}')"""),
md("""## 4. How sure are you of the noise?

An SD from five or six runs is itself loose. The 90% interval below uses the chi-square distribution of the sample
variance (normal runs assumed), computed exactly. Plan with the upper end if a wrong "no difference" would be costly."""),
code("""def p_lower(a, x):   # regularized lower incomplete gamma (series), for chi-square quantiles without scipy
    s = term = 1.0 / a; k = 0
    while abs(term) > 1e-15 * abs(s):
        k += 1; term *= x / (a + k); s += term
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))

def chi2_q(p, df):
    lo, hi = 1e-9, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_lower(df / 2, mid / 2) < p else (lo, mid)
    return (lo + hi) / 2

for label, s, k, d in (('leaderboard', SIGMA_LB, 5, 3.0), ('dev', SIGMA_DEV, 6, 0.10 * DEV_MEAN)):
    lo, hi = s * math.sqrt((k - 1) / chi2_q(0.95, k - 1)), s * math.sqrt((k - 1) / chi2_q(0.05, k - 1))
    print(f'{label}: SD {s} from {k} runs -> 90% interval {lo:.2f} to {hi:.2f}; '
          f'runs per arm for +{d:g}: {runs_per_arm(lo, d)} to {runs_per_arm(hi, d)} (point estimate {runs_per_arm(s, d)})')"""),
md("""## 5. Best of *n* identical resubmissions

If a team resubmits identical code and keeps its best public score, the expected best is the mean plus
SD x e_n, where e_n is the expected maximum of n standard normal draws (computed exactly below)."""),
code("""def e_max(n):
    x = np.linspace(-12, 12, 240001); phi = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    return float(np.sum(x * n * phi * Phi ** (n - 1)) * (x[1] - x[0]))

for n in (1, 2, 5, 10, 20, 50, 100, 200):
    e = e_max(n)
    sim = rng.standard_normal((20000, n)).max(1).mean()
    print(f'best of {n:3d}: +{SIGMA_LB * e:.2f} points above the mean (e_n {e:.3f}; simulation {sim:.3f})')"""),
md("## 6. Your own numbers\n\nPaste the public scores of identical resubmissions of your own code (at least two)."),
code("""my_scores = []   # for example [24.1, 27.3, 22.8, 25.0]
if len(my_scores) >= 2:
    s = float(np.std(my_scores, ddof=1))
    print(f'your SD from {len(my_scores)} submissions: {s:.2f} points')
    for d in (2, 3, 5):
        print(f'  +{d} points: {runs_per_arm(s, d)} submissions per arm')
    print(f'  best of 5 adds about {s * e_max(5):.2f} points to your mean')
else:
    print('add at least two scores to my_scores')"""),
] + []
# --- 5 Nov 2026: sections 7-8, the two-number comparison and a fit-half, predict-half check, with example data embedded
# from our runs (analysis/s37/data/runs; same loaders as analysis/s37/b1_vs_b12_aft.py and v5_budget_prediction.py).
import sys, json, math
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis' / 's37'))
import b1_vs_b12_aft as BA                     # prints its own result while importing; harmless here
from v5_budget_prediction import load as vload, attempts as vatt
from common import CTL as VCTL
_runs = BA.B1 + BA.B12
_rows = [[o[0], o[1][0], int(o[1][1]), int(round(math.exp(o[2]))), int(o[3])] for o in BA.obs if o[0] in _runs]
_games, _obs = {}, {}
for r in VCTL:
    _games[r] = []
    for g in vload(r):
        at, used = vatt(g, 0.5)                    # attempts cut at half the clock (60.5 of 121 minutes), as in Section 4.2
        _games[r].append([g['game_id'][:4], g['number_of_levels'], [[int(t), int(ok)] for _, t, ok in at], int(used)])
    _obs[r] = sum(g['levels_completed'] for g in vload(r))
cells += [
md("""## 7. Compare two agents on crack cost (two numbers)

A level total mixes two things: how many tokens an agent generates (supply S) and how many tokens it needs per crack
(crack cost b). On the **same levels**, b can be estimated far more precisely than a level total, because each level's
own difficulty is fitted and removed: log T(run, level) = a_level + b_run + noise, with attempts still open at the end
counted as censored (a lognormal accelerated-failure-time model, fitted by EM). b is a log multiplier: b = +0.1 means
about 10% more tokens per crack than the reference arm.

The example compares dfranzen's Milestone 2 notebook (B1, two 25-minute runs on 10 public games) with our B12 (seven
121-minute runs) on the levels of those 10 games. Replace `ROWS`, `ARM_A` and `ARM_B` with your own attempts: one row per
level attempt, `[run, game, level, generated tokens, 1 if cleared else 0]`."""),
code("ARM_A = " + json.dumps(BA.B1) + "\nARM_B = " + json.dumps(BA.B12) + "\nROWS = " + json.dumps(_rows, separators=(',', ':'))),
code("""def fit_aft(rows, ref, iters=200):
    runs = sorted({r[0] for r in rows}); levels = sorted({(r[1], r[2]) for r in rows})
    ri = {r: i for i, r in enumerate(runs)}; li = {l: i for i, l in enumerate(levels)}
    R = np.array([ri[r[0]] for r in rows]); L = np.array([li[(r[1], r[2])] for r in rows])
    y = np.log(np.maximum([r[3] for r in rows], 1)); c = np.array([r[4] for r in rows], bool)
    keep = np.isin(L, np.unique(L[c]))            # levels never cleared carry no information on b
    R, L, y, c = R[keep], L[keep], y[keep], c[keep]
    a, b, s = np.zeros(len(levels)), np.zeros(len(runs)), 1.0
    Phi = np.vectorize(lambda z: 0.5 * math.erfc(-z / math.sqrt(2)))
    for _ in range(iters):
        mu = a[L] + b[R]; z = (y - mu) / s
        lam = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / np.maximum(1 - Phi(z), 1e-12)
        ey = np.where(c, y, mu + s * lam); ev = np.where(c, (y - mu) ** 2, s * s * (1 + z * lam))
        for _ in range(3):
            a = np.bincount(L, ey - b[R], len(levels)) / np.maximum(np.bincount(L, minlength=len(levels)), 1)
            b = np.bincount(R, ey - a[L], len(runs)) / np.maximum(np.bincount(R, minlength=len(runs)), 1)
            b -= np.mean([b[ri[r]] for r in ref])
        mun = a[L] + b[R]; s = math.sqrt(np.mean(ev + 2 * (mu - mun) * (ey - mu) + (mu - mun) ** 2))
    return {r: float(b[ri[r]]) for r in runs}, s

b, s = fit_aft(ROWS, ARM_A)
d = np.mean([b[r] for r in ARM_B])
print(f'residual s {s:.2f}; b per run: ' + ', '.join(f'{r} {b[r]:+.3f}' for r in ARM_A + ARM_B))
print(f'arm B - arm A: b {d:+.3f}, so arm B needs x{math.exp(d):.2f} the tokens per crack')
rng = np.random.default_rng(0); boot = []
by_run = {r: [x for x in ROWS if x[0] == r] for r in ARM_A + ARM_B}
for _ in range(100):                               # bootstrap over runs within each arm (a refit each time)
    pa = [ARM_A[i] for i in rng.integers(0, len(ARM_A), len(ARM_A))]
    pb = [ARM_B[i] for i in rng.integers(0, len(ARM_B), len(ARM_B))]
    names = [f'{r}#{j}' for j, r in enumerate(pa + pb)]
    rows = [[n] + x[1:] for n, r in zip(names, pa + pb) for x in by_run[r]]
    bb, _ = fit_aft(rows, names[:len(pa)], iters=60)
    boot.append(np.mean([bb[n] for n in names[len(pa):]]))
print(f'bootstrap 90% interval for the difference: {np.percentile(boot, 5):+.3f} to {np.percentile(boot, 95):+.3f}')"""),
md("""## 8. Fit half, predict half: check the crack model on your own runs

Before trusting the model for an agent, check that it forecasts. Cut each run at half its clock, fit the three numbers
(wall share q, median wait, spread s) on the attempts seen so far, then predict each run's final level total. The second
half is assumed to generate as many tokens as the first, shared among the games still playing in proportion to their
first-half tokens. Levels are played in order; an attempt open at the cut continues from the tokens it already has. This
is the check of Section 4.2 of the paper (there: 101.1 levels predicted on average, 102.75 observed).

The example uses our four control runs: `GAMES[run]` lists, per game, its name, its number of levels, its attempts up to
the cut as `[generated tokens, 1 if cleared]`, and its generated tokens up to the cut; `OBSERVED[run]` is the final
level total."""),
code("GAMES = " + json.dumps(_games, separators=(',', ':')) + "\nOBSERVED = " + json.dumps(_obs)),
code("""from math import erfc, sqrt
ERFC = np.vectorize(erfc); INV = np.vectorize(NormalDist().inv_cdf)
def S0(c, mu, s): return 0.5 * ERFC((np.log(np.maximum(c, 1)) - mu) / (s * sqrt(2)))
clr = np.array([t for r in GAMES for _, n, att, _ in GAMES[r] for t, ok in att if ok], float)
opn = np.array([att[-1][0] for r in GAMES for _, n, att, _ in GAMES[r] if att and not att[-1][1]], float)
best = (-np.inf,)
for mu in np.log(np.linspace(5e3, 60e3, 56)):     # grid search for the maximum likelihood (numpy only)
    for s in np.linspace(0.6, 1.8, 25):
        z = (np.log(clr) - mu) / s
        base = (-0.5 * z * z - np.log(s * clr * sqrt(2 * np.pi))).sum(); so = S0(opn, mu, s)
        for q in np.linspace(0, 0.3, 31):
            ll = base + len(clr) * np.log(1 - q) + np.log(q + (1 - q) * so).sum()
            if ll > best[0]: best = (ll, q, mu, s)
_, q, mu, s = best
print(f'fit on the first halves: q {q:.2f}, median {math.exp(mu) / 1e3:.1f}k tokens, s {s:.2f}')
rng = np.random.default_rng(1); N = 2000
def waits(c, n):                                   # tokens still needed to crack, given c tokens already spent (inf = wall)
    sc = float(S0(c, mu, s)) if c > 0 else 1.0
    wall = rng.random(n) < q / (q + (1 - q) * sc)
    z = INV(1 - rng.random(n) * sc)                # lognormal draw above c (inverse CDF)
    return np.where(wall, np.inf, np.exp(mu + s * z) - c)
pred = []
for r in GAMES:
    done_all = [sum(ok for _, ok in att) >= n for _, n, att, _ in GAMES[r]]
    tot_used = sum(u for *_, u in GAMES[r]); open_used = sum(u for (*_, u), d in zip(GAMES[r], done_all) if not d)
    ratio = tot_used / max(1, open_used); total = 0.0
    for (name, n, att, used), d in zip(GAMES[r], done_all):
        k = sum(ok for _, ok in att)
        if d: total += n; continue
        left = np.full(N, used * ratio); lv = np.full(N, k, float); alive = np.ones(N, bool)
        w = waits(att[-1][0] if att and not att[-1][1] else 0.0, N)
        for _ in range(k, n):
            ok = alive & (w <= left); left = np.where(ok, left - w, left); lv += ok; alive = ok
            if not alive.any(): break
            w = waits(0.0, N)
        total += lv.mean()
    pred.append(total)
    print(f'{r}: predicted {total:.1f} levels, observed {OBSERVED[r]}')
print(f'mean: predicted {np.mean(pred):.1f}, observed {np.mean(list(OBSERVED.values())):.2f}')"""),
md("""## 9. If walls are permanent: the most that unlimited play can reach

If each level is a wall with probability q and walls never crack, a game with L levels clears c levels before its first
wall, with P(c = j) = (1 - q)^j q for j < L and (1 - q)^L for c = L, whatever the budget. The score weighs level l by l,
so the game's completion share is c(c + 1) / (L(L + 1)), and the cap is its mean over your games. The table reproduces
the paper's caps on the 25 public games (73% at q = 0.06, 81% at 0.04; 85% needs 3.1% or less). Below, q is refitted with
the grid of section 8 on all games and then on games drawn with replacement, which shows how loose the cap is.

Two warnings. The fitted q depends on how long open attempts were followed: on these first halves it is 0.02, since few
open attempts have run long enough to look like walls; on the full runs the paper finds 0.06, and with longer play it
falls again (paper, Section 4.3). And the cap holds only if walls are permanent: a lognormal wait without walls fits our
runs almost as well. Fit your full runs, and read the cap as a model extrapolation, not a measurement."""),
code("""LEVELS = {name: n for r in GAMES for name, n, _, _ in GAMES[r]}   # level count of each game in the runs above
def cap(q, levels):   # completion score (percent) that unlimited play reaches if each level is a wall with probability q
    tot = 0.0
    for L in levels:
        j = np.arange(L + 1); p = np.where(j < L, (1 - q) ** j * q, (1 - q) ** L)
        tot += float((p * j * (j + 1)).sum()) / (L * (L + 1))
    return 100 * tot / len(levels)
def q_for(target, levels):                          # largest wall share that still allows `target` percent
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if cap(mid, levels) >= target else (lo, mid)
    return lo
L_ALL = list(LEVELS.values())
print(f'{len(L_ALL)} games, {sum(L_ALL)} levels')
for qq in (0.0, 0.031, 0.04, 0.06, 0.11):
    print(f'  q {qq:.3f}: unlimited play reaches at most {cap(qq, L_ALL):.0f}% of the completion score')
print(f'  85% needs q of {100 * q_for(85, L_ALL):.1f}% or less on these games')
ATT = {r: {name: att for name, _, att, _ in GAMES[r]} for r in GAMES}
def fit_q(pairs):                                   # the grid fit of section 8, vectorized, on (run, game) pairs
    clr = np.array([t for r, g in pairs for t, ok in ATT[r][g] if ok], float)
    opn = np.array([ATT[r][g][-1][0] for r, g in pairs if ATT[r][g] and not ATT[r][g][-1][1]], float)
    MU = np.log(np.linspace(5e3, 60e3, 56))[:, None, None]; SS = np.linspace(0.6, 1.8, 25)[None, :, None]
    Q = np.linspace(0, 0.3, 31)
    z = (np.log(clr) - MU) / SS
    base = (-0.5 * z * z - np.log(SS * clr * sqrt(2 * np.pi))).sum(-1)
    so = 0.5 * ERFC((np.log(np.maximum(opn, 1)) - MU) / (SS * sqrt(2)))
    ll = base[..., None] + len(clr) * np.log(1 - Q) + np.log(Q + (1 - Q) * so[..., None]).sum(-2)
    return float(Q[np.unravel_index(np.argmax(ll), ll.shape)[2]])
names = sorted(LEVELS)
q_all = fit_q([(r, g) for r in GAMES for g in names if g in ATT[r]])
print(f'fit on all games again: q {q_all:.2f} (section 8: {q:.2f}); cap {cap(q_all, L_ALL):.0f}%')
rng = np.random.default_rng(3); bq, bc = [], []
for _ in range(200):                                # bootstrap over games: resample games, refit q, recompute the cap
    pick = rng.choice(names, len(names))
    bq.append(fit_q([(r, g) for g in pick for r in GAMES if g in ATT[r]])); bc.append(cap(bq[-1], [LEVELS[g] for g in pick]))
bq = np.array(bq)
print(f'bootstrap over games (200 resamples): q {np.percentile(bq, 5):.2f} to {np.percentile(bq, 95):.2f}, '
      f'cap {np.percentile(bc, 5):.0f}% to {np.percentile(bc, 95):.0f}% (90% intervals); q at or below 3.1% in {np.mean(bq <= 0.031):.0%}')"""),
]
nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = Path(__file__).resolve().parent / 'how_many_runs.ipynb'
nbf.write(nb, str(out)); print('wrote', out)
