"""Does a model without walls also reproduce the run-to-run noise? (7 Nov 2026, exploratory, CPU only; written after the
7 Nov outside reviews and after rivals_no_walls_7nov.py found that a lognormal wait without walls fits almost as well.)
Section 4.3 says the crack model reproduces the run-to-run SD of levels (5.6 against 5.3 observed) only if level difficulty
and walls persist across runs (lever3d_noise_model.py: H0 12.3, H1 11.1, H2 5.6). Here the same simulator (imported
read-only: 25 games, FB25 level counts, 10 streams, 4.5M tokens, fair sharing) runs two models with no walls at all,
both centred on the no-wall lognormal fitted to the four controls (median 25.7k tokens, s 1.17; rivals_no_walls_7nov.out):
N0 every level the same (median 25.7k, s 1.17 drawn afresh in every run), N1 level difficulty that persists across runs:
each level's log-median fixed per 'world' with SD sqrt(1.17^2 - 0.49^2) = 1.06 around log 25.7k, and run luck s 0.49 as in
H1 and H2 (lever3c). 20 worlds x 30 runs each, as in lever3d. If N1 also gives an SD near 5.3, then what the noise needs is
persistence, not walls.
Run: python3 analysis/s37/noise_no_walls_7nov.py > analysis/s37/out/noise_no_walls_7nov.out (a few minutes)."""
import math
import numpy as np
from lever3d_noise_model import sim, NL
MU0, S0, LUCK = math.log(25.7e3), 1.17, 0.49
SB = math.sqrt(S0 ** 2 - LUCK ** 2)
rng = np.random.default_rng(20261107)
NOWALL = [[False] * n for n in NL]
for name in ('N0', 'N1'):
    means, sds = [], []
    for w in range(20):
        if name == 'N0': med = [[MU0] * n for n in NL]; s = S0
        else: med = [[MU0 + SB * rng.standard_normal() for _ in range(n)] for n in NL]; s = LUCK
        tot = np.array([sim(med, NOWALL, s, rng) for _ in range(30)])
        means.append(tot.mean()); sds.append(tot.std(ddof=1))
    print(f'{name}: mean total {np.mean(means):6.1f} (between worlds SD {np.std(means):4.1f}); run-to-run SD within a world: '
          f'mean {np.mean(sds):4.1f}, 10-90% {np.quantile(sds, .1):4.1f}-{np.quantile(sds, .9):4.1f}')
print(f'(between-level SD in N1: {SB:.2f}; crack model for comparison, lever3d_noise_model.out: H0 12.3, H1 11.1, H2 5.6)')
print('observed: mean 102.5, run-to-run SD 5.3 (six runs; 6.0 with T1c)')
