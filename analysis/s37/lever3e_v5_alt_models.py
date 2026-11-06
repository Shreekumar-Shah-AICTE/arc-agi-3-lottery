"""Lever 3e (written 4 Nov 2026 ~10:50 UTC: AFTER the V5 seal (10:16 UTC), BEFORE any V5 output exists; the run
ends ~14:45 UTC). Exploratory second opinion on V5, not part of the sealed test: total levels at 121 vs 242 minutes under
H0 (homogeneous), H1 (level-specific medians, random walls) and H2 (level-specific medians and walls, the only version
that reproduces the observed run-to-run SD, lever3d_noise_model.out). Same simulator, FAIR sharing of 10 streams; tokens
4.5M per 121 minutes. Reported as: mean total at 1x, at 2x, the 2x/1x ratio, and 102.75 x ratio."""
import math
import numpy as np
from lever3d_noise_model import NL, sim, STEP, STREAMS
import lever3d_noise_model as M
rng = np.random.default_rng(8)
MU = math.log(22.1e3)
for name in ('H0', 'H1', 'H2'):
    r1, r2 = [], []
    for w in range(24):
        if name == 'H0': med = [[MU] * n for n in NL]; wall = None; s = 1.05
        else:
            med = [[MU + 0.80 * rng.standard_normal() for _ in range(n)] for n in NL]; s = 0.49
            wall = None if name == 'H1' else [[rng.random() < 0.06 for _ in range(n)] for n in NL]
        M.STEPS = int(4.5e6 / STREAMS / STEP); a = np.mean([sim(med, wall, s, rng) for _ in range(12)])
        M.STEPS = int(9.0e6 / STREAMS / STEP); b = np.mean([sim(med, wall, s, rng) for _ in range(12)])
        r1.append(a); r2.append(b)
    r1, r2 = np.array(r1), np.array(r2); ratio = r2.mean() / r1.mean()
    print(f'{name}: 1x {r1.mean():6.1f}  2x {r2.mean():6.1f}  ratio {ratio:.3f} -> 102.75 x ratio = {102.75 * ratio:6.1f} '
          f'(world-to-world 2x-1x gain SD {np.std(r2 - r1):4.1f})')
