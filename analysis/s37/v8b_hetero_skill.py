"""V8b (pre-registered extension, 2 Nov 2026 ~20:20 UTC): V8 with skill spread that may grow with submission count,
tau_i = max(0.3, tau0 + tau1 * (log2 n_i - mean)). Same likelihood, predictive and statistics as V8."""
import numpy as np, importlib.util, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v8', HERE / 'v8_sealed_prediction.py'); v8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)
S, N, L2, T = v8.S, v8.N, v8.L2, v8.T
def prior(mu, tau0, tau1, beta):
    tau = np.maximum(0.3, tau0 + tau1 * L2)
    W = np.exp(-0.5 * ((T[None, :] - (mu + beta * L2)[:, None]) / tau[:, None]) ** 2)
    return W / W.sum(1, keepdims=True)
def loglik(mu, tau0, tau1, f, sigma, beta):
    D, C = v8.mats(sigma, f); W = prior(mu, tau0, tau1, beta)
    return float(np.sum(np.log((W * D).sum(1) + 1e-300) - np.log((W * C).sum(1) + 1e-300)))
def fit(sigma, betas=(0.0,), fs=(0.5, 0.75, 1.0), t0s=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0), t1s=(-0.3, 0.0, 0.2, 0.4, 0.6, 0.8)):
    best = None
    for f in fs:
        for beta in betas:
            for t0 in t0s:
                for t1 in t1s:
                    lo, hi = 18.0, 34.0
                    for _ in range(16):
                        a = hi - (hi - lo) / 1.618; b = lo + (hi - lo) / 1.618
                        if loglik(a, t0, t1, f, sigma, beta) > loglik(b, t0, t1, f, sigma, beta): hi = b
                        else: lo = a
                    mu = (lo + hi) / 2; ll = loglik(mu, t0, t1, f, sigma, beta)
                    if best is None or ll > best[0]: best = (ll, mu, t0, t1, f, beta)
    return best
def predictive(mu, t0, t1, f, sigma, beta, R=800, seed=11):
    rng = np.random.default_rng(seed)
    D, _ = v8.mats(sigma, f); post = prior(mu, t0, t1, beta) * D; post /= post.sum(1, keepdims=True); cdf = np.cumsum(post, 1)
    fam = np.array(v8.FAMILY); fm, fsd = fam.mean(), fam.std(ddof=1); out = []
    for _ in range(R):
        j = np.minimum((cdf < rng.random(len(S))[:, None]).sum(1), len(T) - 1); th = T[j] + rng.uniform(-0.1, 0.1, len(S))
        z = (S - th) / sigma
        p1 = np.where(N > 1, ((1 - f) / (1 - f + f * v8.Phi(z))) ** (N - 1), 1.0)
        k1 = rng.random(len(S)) < p1; w = rng.standard_normal((2, len(S)))
        P = th + sigma * np.where(k1, w[0], w.max(0))
        P[v8.iours] = fm + rng.standard_normal() * fsd / math.sqrt(len(fam)) + fsd * rng.standard_normal(2).max()
        st = v8.stats(P); st['ours_pct'] = float((P < P[v8.iours]).mean()); st['D_hat'] = float((P - S)[v8.G1].mean()); out.append(st)
    return {k: np.array([o[k] for o in out]) for k in out[0]}
if __name__ == '__main__':
    q = lambda a: f'{np.mean(a):.3f} [{np.quantile(a, 0.05):.3f}, {np.quantile(a, 0.95):.3f}]'
    for label, sigma, betas in (('H', 2.3, (0.0,)), ('H', 2.87, (0.0,)), ('H', 3.3, (0.0,)), ('HB', 2.87, (0.0, 0.25, 0.5)), ('S', 1.0, (0.25, 0.5, 0.75))):
        ll, mu, t0, t1, f, beta = fit(sigma, betas)
        print(f'{label} sigma={sigma}: loglik {ll:.1f} mu {mu:.2f} tau0 {t0} tau1 {t1} f {f} beta {beta} '
              f'(tau at n=1: {max(0.3, t0 + t1 * L2.min()):.2f}, n=100: {max(0.3, t0 + t1 * (np.log2(100) - np.log2(N).mean())):.2f})', flush=True)
        pr = predictive(mu, t0, t1, f, sigma, beta)
        print('   ' + '; '.join(f'{k} {q(v)}' for k, v in pr.items()), flush=True)
