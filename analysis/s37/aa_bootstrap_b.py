"""Level bootstrap of the crack-cost b (6 Nov 2026, exploratory, CPU only; follow-up to aa_calibration.py).

Is b's run-to-run noise nearer the six-run A/A SD (0.051) or the five-run SD without b32_spec2_r3 (0.015)? This
resamples levels with replacement (each copy keeps its own level effect), refits the same lognormal AFT model of
lever4b_two_numbers.py (its code is executed unchanged up to the main fit), and reports each run's bootstrap SE of b.
Seeded (numpy default_rng(7)); B refits as set below.
"""
import math, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
src = (HERE / 'lever4b_two_numbers.py').read_text().split("bhat, s, n_obs, n_crk = fit(obs, FB25, CTL)")[0]
ns = {'__name__': 'lever4b_import', '__file__': str(HERE / 'lever4b_two_numbers.py')}
exec(compile(src, 'lever4b_two_numbers.py', 'exec'), ns)
obs, fit, FB25, CTL = ns['obs'], ns['fit'], ns['FB25'], ns['CTL']
SAME = CTL + ['b32_spec2_r2', 'b32_spec2_r3']
B = int(sys.argv[1]) if len(sys.argv) > 1 else 60
t0 = time.time(); b0, s0, n0, c0 = fit(obs, FB25, CTL, iters=120)
print(f'full fit (120 EM iterations): {n0} attempts, residual s {s0:.3f}; ' + ', '.join(f'{r} {b0[r]:+.3f}' for r in SAME) + f'  [{time.time()-t0:.1f} s]')
bylev = defaultdict(list)
for o in obs: bylev[o[1]].append(o)
keys = sorted(bylev); rng = np.random.default_rng(7); draws = defaultdict(list)
for i in range(B):
    pick = rng.integers(0, len(keys), len(keys)); rows = []
    for j, k in enumerate(pick):
        rows += [(o[0], (keys[k], j), o[2], o[3]) for o in bylev[keys[k]]]
    bb, _, _, _ = fit(rows, FB25, CTL, iters=120)
    for r in FB25: draws[r].append(bb[r])
se = {r: float(np.std(draws[r], ddof=1)) for r in FB25}
print(f'level bootstrap, {B} refits: bootstrap SE of b per run (control mean fixed at 0 in every refit)')
for r in SAME: print(f'  {r:14s} b {b0[r]:+.3f}  SE {se[r]:.3f}')
print(f'median SE over the 6 same-agent runs {np.median([se[r] for r in SAME]):.3f}; over all {len(FB25)} runs {np.median(list(se.values())):.3f}')
z = b0['b32_spec2_r3'] / se['b32_spec2_r3']
print(f'b32_spec2_r3 against its own bootstrap SE: z {z:+.2f}  [{time.time()-t0:.0f} s total]')
# Variant tests of Section 4.5 with the bootstrap SE of each variant's mean b (draws are paired across runs, so the SE
# of a two-run mean is taken over the draws themselves); Bonferroni line |z| >= 2.64 over the 6 variants.
VAR = {'ht': ['b12_ht', 'b12_ht2'], 'sdt': ['b12_sdt', 'b12_sdt_r2'], 'len': ['b12_len'], 'x7b': ['b12_x7b'],
       'b15c': ['b15c_fb25'], 'ThinkingCap': ['b28_tc_fb25']}
past = 0
print('variant tests with bootstrap SEs:')
for v, rs in VAR.items():
    m = np.mean([b0[r] for r in rs]); sev = float(np.std(np.mean([draws[r] for r in rs], axis=0), ddof=1)); zv = m / sev
    past += abs(zv) >= 2.64
    print(f'  {v:12s} runs {len(rs)}  mean b {m:+.3f}  bootstrap SE {sev:.3f}  z {zv:+.2f}')
print(f'variants past |z| 2.64 with bootstrap SEs: {past} of {len(VAR)}')
