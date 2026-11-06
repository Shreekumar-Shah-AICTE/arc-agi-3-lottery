"""V9 extras (S37, 4 Nov 2026, before the ARC-AGI-2 private board): alternative S (low noise, real skill gradient),
our public percentile in C, and the top group (public >= 40) rank stability under H."""
import numpy as np, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v9', HERE / 'v9_arc2_sealed_model.py'); v9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v9)
q = lambda a: f'{np.mean(a):.3f} [{np.quantile(a, 0.05):.3f}, {np.quantile(a, 0.95):.3f}]'
print(f'ours: public {v9.S[v9.iours]} (n {int(v9.N[v9.iours])}), public percentile within C {(v9.S < v9.S[v9.iours]).mean():.3f}')
for sigma in (0.3, 0.45):
    ll, mu, tau, f, beta = v9.fit(sigma, beta_grid=(0.3, 0.4, 0.5, 0.6))
    print(f'S sigma={sigma}: loglik {ll:.1f} mu {mu:.2f} tau {tau} f {f} beta {beta}')
    pr = v9.predictive(mu, tau, f, sigma, beta, R=400)
    print('   ' + '; '.join(f'{k} {q(v)}' for k, v in pr.items()))
# top group: teams with final public >= 40; private = public - (selection gain) + noise; rank stability only
top = v9.S_all >= 40
print(f'teams with public >= 40: {top.sum()}')
rng = np.random.default_rng(5)
for sigma in (0.81, 1.0, 1.3):
    rs = []
    for _ in range(2000):
        P = v9.S_all[top] + sigma * np.sqrt(2) * rng.standard_normal(top.sum())   # public and private draws independent
        rs.append(v9.spearman(v9.S_all[top], P))
    print(f'  sigma {sigma}: Spearman(public, private) among public >= 40: {q(np.array(rs))}')
