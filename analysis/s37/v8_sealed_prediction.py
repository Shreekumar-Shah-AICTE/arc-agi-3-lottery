"""V8 (pre-registered; written 2 Nov 2026 ~20:05 UTC, before the ARC-AGI-3 private leaderboard exists).
Lottery model fitted to the public ARC-AGI-3 board (snapshot 19:58 UTC), then used to predict the private board.
Model H (the theory): in the 20-40 cluster a team's best public score = skill + the best of k noisy draws,
  skill ~ N(mu, tau^2) independent of how often the team submitted, k ~ Binomial(n, f) given k >= 1
  (n = SubmissionCount, f = share of submissions that are comparable scored draws), noise SD sigma per half-board.
  Private = the same skill + the better of the 2 selected runs' private-half draws (1 draw if k = 1); the public and
  private halves of one run are independent given skill (no shared run factor, V2).
Model S (the alternative): the submission-count gradient is real skill (skill = mu + beta*log2 n), little noise.
Prints fitted parameters and the posterior-predictive distribution of shift-invariant private-board statistics."""
import csv, gzip, math, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
LB = HERE / 'data' / 'lb' / 'arc-agi-3_lb_20261102T1958.csv.gz'
FAMILY = [33.37, 29.69, 28.24, 26.76, 26.21, 25.76, 23.82, 25.54]   # 8 scored B12-family runs (ours)
OURS = 'Shreekumar Shah'
rng = np.random.default_rng(20261102)

def Phi(x):
    x = np.asarray(x, float); z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * y)
def phi(x): return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
def max_pdf(z, n, f):
    q = 1.0 - f
    return phi(z) * n * f * (q + f * Phi(z)) ** (n - 1) / (1.0 - q ** n)
def max_cdf(z, n, f):
    q = 1.0 - f
    return ((q + f * Phi(z)) ** n - q ** n) / (1.0 - q ** n)

with gzip.open(LB, 'rt', encoding='utf-8-sig') as fh:
    rows = [r for r in csv.DictReader(fh) if r.get('Score')]
S_all = np.array([float(r['Score']) for r in rows]); N_all = np.array([int(r['SubmissionCount']) for r in rows])
names = [r['TeamName'] for r in rows]
m = (S_all >= 20) & (S_all < 40)
S, N = S_all[m], N_all[m].astype(float); NAMES = [x for x, k in zip(names, m) if k]
iours = NAMES.index(OURS)
L2 = np.log2(N) - np.log2(N).mean()
T = np.arange(12.0, 42.0, 0.2)
_cache = {}
def mats(sigma, f):
    key = (sigma, f)
    if key not in _cache:
        Z = (S[:, None] - T[None, :]) / sigma
        D = max_pdf(Z, N[:, None], f) / sigma
        C = max_cdf((40 - T[None, :]) / sigma, N[:, None], f) - max_cdf((20 - T[None, :]) / sigma, N[:, None], f)
        _cache.clear(); _cache[key] = (D, C)
    return _cache[key]
def prior(mu, tau, beta):
    c = mu + beta * L2
    W = np.exp(-0.5 * ((T[None, :] - c[:, None]) / tau) ** 2)
    return W / W.sum(1, keepdims=True)
def loglik(mu, tau, f, sigma, beta=0.0):
    D, C = mats(sigma, f); W = prior(mu, tau, beta)
    return float(np.sum(np.log((W * D).sum(1) + 1e-300) - np.log((W * C).sum(1) + 1e-300)))
def fit(sigma, beta_grid=(0.0,), taus=(0.3, 0.75, 1.25, 1.75, 2.25, 2.75), fs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    best = None
    for f in fs:
        for beta in beta_grid:
            for tau in taus:
                lo, hi = 18.0, 34.0
                for _ in range(16):                      # golden-section search on mu
                    a = hi - (hi - lo) / 1.618; b = lo + (hi - lo) / 1.618
                    if loglik(a, tau, f, sigma, beta) > loglik(b, tau, f, sigma, beta): hi = b
                    else: lo = a
                mu = (lo + hi) / 2; ll = loglik(mu, tau, f, sigma, beta)
                if best is None or ll > best[0]: best = (ll, mu, tau, f, beta)
    return best

def rank(v):
    o = np.argsort(v, kind='mergesort'); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])
TOP = S >= np.quantile(S, 0.9)
G1, G64 = N == 1, N >= 64
def stats(P):
    k = TOP.sum(); topP = rank(P) >= len(P) - k
    return dict(spearman=spearman(S, P), retain=float((TOP & topP).sum() / k),
                grad=float(P[G64].mean() - P[G1].mean()),
                excess_drop=float((S - P)[TOP].mean() - (S - P).mean()))
def predictive(mu, tau, f, sigma, beta=0.0, R=1500):
    D, _ = mats(sigma, f); post = prior(mu, tau, beta) * D; post /= post.sum(1, keepdims=True)
    cdf = np.cumsum(post, 1); out = []; ours = []
    fam_mu, fam_sd = np.mean(FAMILY), np.std(FAMILY, ddof=1)
    for _ in range(R):
        u = rng.random(len(S))[:, None]
        j = np.minimum((cdf < u).sum(1), len(T) - 1)
        th = T[j] + rng.uniform(-0.1, 0.1, len(S))
        z = (S - th) / sigma
        p1 = np.where(N > 1, ((1 - f) / (1 - f + f * Phi(z))) ** (N - 1), 1.0)
        k1 = rng.random(len(S)) < p1
        w = rng.standard_normal((2, len(S)))
        P = th + sigma * np.where(k1, w[0], w.max(0))
        mo = fam_mu + rng.standard_normal() * fam_sd / math.sqrt(len(FAMILY))
        P[iours] = mo + fam_sd * rng.standard_normal(2).max()          # our 2 finals: 2 fresh family draws
        st = stats(P); st['ours_pct'] = float((P < P[iours]).mean()); out.append(st)
    return {k: np.array([o[k] for o in out]) for k in out[0]}

if __name__ == '__main__':
    print(f'snapshot {LB.name}: teams {len(S_all)}; cluster [20,40): {len(S)}, mean {S.mean():.2f}, SD {S.std(ddof=1):.2f}; '
          f'public gradient mean(n>=64) - mean(n=1) = {S[G64].mean() - S[G1].mean():.2f} ({G64.sum()} vs {G1.sum()} teams); '
          f'top-decile cut {np.quantile(S, 0.9):.2f}; ours = {S[iours]} (n={int(N[iours])})', flush=True)
    res = {}
    BETAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
    for sigma in (1.0, 2.3, 2.87, 3.3):
        for label, bg in (('H', (0.0,)), ('HB', BETAS)):
            ll, mu, tau, f, beta = fit(sigma, beta_grid=bg)
            res[(label, sigma)] = (ll, mu, tau, f, beta)
            print(f'{label:2s} sigma={sigma}: loglik {ll:.1f} mu {mu:.2f} tau {tau} f {f} beta {beta}', flush=True)
    q = lambda a: f'{np.mean(a):.3f} [{np.quantile(a, 0.05):.3f}, {np.quantile(a, 0.95):.3f}] (2.5-97.5%: {np.quantile(a, 0.025):.3f}, {np.quantile(a, 0.975):.3f})'
    for key, (ll, mu, tau, f, beta) in res.items():
        pr = predictive(mu, tau, f, key[1], beta, R=800)
        print(f'--- {key[0]} sigma={key[1]} (mu {mu:.2f}, tau {tau}, f {f}, beta {beta})', flush=True)
        for k, v in pr.items(): print(f'  {k:12s} {q(v)}', flush=True)
