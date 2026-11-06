#!/usr/bin/env python3
"""V5 follow-up (written 4 Nov 2026 ~14:45 UTC, AFTER the V5 readout; exploratory, not part of the sealed test).
Where did the 2x run beat the sealed point prediction? Splits the per-game gain into games that were stuck early at 1x
(FB25 mean <= 1.5 levels) and the rest. Reads the sealed per-game table from v5_readout.py (P-V5-4, PRED_2X)."""
import ast, gzip, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']
src = (HERE / 'v5_readout.py').read_text()
PRED_2X = ast.literal_eval(src[src.index('PRED_2X = ') + len('PRED_2X = '): src.index('}', src.index('PRED_2X = ')) + 1])
def load(r): return json.load(gzip.open(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz', 'rt'))['game_runs']
def gid(g): return g['game_id'].split('-')[0]
one = {}
for r in FB:
    for g in load(r): one.setdefault(gid(g), []).append(g['levels_completed'])
one = {k: sum(v) / len(v) for k, v in one.items()}
V = {gid(g): g for g in load('v5_fb50')}
nl = {k: V[k]['number_of_levels'] for k in V}
pred = {k: PRED_2X.get(k, nl[k] if one[k] >= nl[k] else one[k]) for k in V}   # games solved at 1x stay solved
obs = {k: V[k]['levels_completed'] for k in V}
stuck = sorted(k for k in V if one[k] <= 1.5)
rest = sorted(k for k in V if k not in stuck)
def tot(d, ks): return sum(d[k] for k in ks)
print(f'all 25 games: 1x mean {tot(one, V):.2f}, sealed model 2x {tot(pred, V):.1f}, V5 {tot(obs, V)}; '
      f'gain observed {tot(obs, V) - tot(one, V):.2f} vs predicted {tot(pred, V) - tot(one, V):.2f}')
print(f'{len(stuck)} games stuck early at 1x (FB25 mean <= 1.5: {", ".join(stuck)}): 1x {tot(one, stuck):.2f}, '
      f'predicted {tot(pred, stuck):.1f}, V5 {tot(obs, stuck)}; excess over prediction {tot(obs, stuck) - tot(pred, stuck):.1f}')
print(f'other {len(rest)} games: 1x {tot(one, rest):.2f}, predicted {tot(pred, rest):.1f}, V5 {tot(obs, rest)}; '
      f'excess over prediction {tot(obs, rest) - tot(pred, rest):.1f}')
print(f'games completed: V5 {sum(obs[k] >= nl[k] for k in V)}; solved in every 1x control {sum(one[k] >= nl[k] for k in V)}')
