"""V5 readout (written 4 Nov 2026 ~10:40 UTC, before the V5 output exists). Scores the sealed predictions in
sealed/v5_budget_prediction.txt exactly as written. Usage: python3 v5_readout.py [path to V5 benchmark.json(.gz)]
(default data/runs/v5_fb50.benchmark.json.gz)."""
import gzip, json, re, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
PRED_2X = {'bp35': 2.0, 'cn04': 3.8, 'dc22': 3.1, 'ft09': 5.6, 'g50t': 3.7, 'ka59': 5.9, 'lf52': 2.7, 'ls20': 5.0, 'm0r0': 5.7,
           're86': 7.0, 's5i5': 2.9, 'sc25': 4.5, 'sk48': 1.9, 'sp80': 2.0, 'su15': 6.0, 'tn36': 5.1, 'tu93': 8.7, 'vc33': 6.3,
           'wa30': 3.3}                                   # sealed P-V5-4 table (model expected levels at 2x)
def load(p):
    op = gzip.open if str(p).endswith('.gz') else open
    return json.load(op(p, 'rt'))['game_runs']
def gid(g): return g['game_id'].split('-')[0]
def level_times(g):
    h = g['history']; apl = g['actions_per_level']; out, i = [], 0
    for k in range(g['levels_completed']): i += apl[k]; out.append(float(h[i - 1]['wallclock_seconds']))
    return out
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else float('nan')
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'data' / 'runs' / 'v5_fb50.benchmark.json.gz'
V = load(path)
one = {}
for r in FB:
    for g in load(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz'): one.setdefault(gid(g), []).append(g['levels_completed'])
one = {k: float(np.mean(v)) for k, v in one.items()}
# validity
states = {}
for g in V: states[g['state']] = states.get(g['state'], 0) + 1
tok = [int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1)) for g in V]
wall = [float(g.get('final_wallclock_seconds') or 0) for g in V]
print(f'run: {path.name}; games {len(V)}; states {states}; games with tokens {sum(t > 0 for t in tok)}; total tokens {sum(tok)/1e6:.2f}M; '
      f'max game wall clock {max(wall)/60:.1f} min')
valid = len(V) == 25 and all(t > 0 for t in tok)
print('VALID (25 games, all played):', valid, '(also check the kernel log for server crashes)')
L = sum(g['levels_completed'] for g in V); won = sum(g['levels_completed'] >= g['number_of_levels'] for g in V)
L121 = sum(sum(1 for t in level_times(g) if t <= 121 * 60) for g in V)
v1 = 'PASS' if 110 <= L <= 145 else ('FAIL (stuck side)' if L <= 109 else 'FAIL (out-of-time side)')
print(f'P-V5-1 total levels {L} (sealed 127, 90% 110-145): {v1}')
print(f'P-V5-2 games completed {won} (sealed 13, 90% 10-16): {"inside" if 10 <= won <= 16 else "outside"}')
print(f'P-V5-3 levels by minute 121 {L121} (sealed about 103, 93-113): {"inside" if 93 <= L121 <= 113 else "outside"}')
og = sorted(PRED_2X); act = {gid(g): g['levels_completed'] for g in V}
rho = spearman(np.array([PRED_2X[k] - one[k] for k in og]), np.array([act[k] - one[k] for k in og]))
print(f'P-V5-4 Spearman(predicted gain, actual gain) over {len(og)} open games: {rho:.2f} (sealed median 0.22, 90% -0.23 to 0.54)')
print('alternatives: A saturation ~103, B log-linear ~140, C linear 150-175; log-logistic variant 130.5')
print('per game: id nl FB25-mean predicted-2x V5 (levels by 121 min)')
for g in sorted(V, key=gid):
    k = gid(g); n121 = sum(1 for t in level_times(g) if t <= 121 * 60)
    print(f'  {k} {g["number_of_levels"]:2d} {one.get(k, float("nan")):5.2f} {PRED_2X.get(k, g["number_of_levels"] if one.get(k, 0) >= g["number_of_levels"] else float("nan")):5.1f} {g["levels_completed"]:2d} ({n121})')
