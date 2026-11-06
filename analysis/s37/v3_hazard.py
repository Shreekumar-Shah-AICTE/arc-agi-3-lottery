"""V3 preview: level-clear hazard in token time and in action time for the 16 B12-family runs,
split into the 10 original dev games and the 15 original holdout games (sets taken from runs b1 and b5)."""
import gzip, json
from collections import defaultdict
from common import HERE
FAM = ['b12_fb25', 'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b', 'b15c_fb25', 'b28_ctl_fb25',
       'b29_occ_fb25', 'b31_spec1_fb25', 'b31_spec2_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3']
def games_of(run):
    with gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt') as f:
        return {g['game_id'].split('-')[0] for g in json.load(f)['game_runs']}
DEV, HOLD = games_of('b1_franzen'), games_of('b5_x2ho')
att = []  # (set, tokens on level, actions on level, cleared, actions / human baseline, tokens on first 10 actions, tokens on later actions, n later)
for run in FAM:
    with gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt') as f:
        d = json.load(f)
    for g in d['game_runs']:
        gid = g['game_id'].split('-')[0]; s = 'dev' if gid in DEV else 'hold' if gid in HOLD else None
        h = g['history']; apl = g['actions_per_level']; lv = g['levels_completed']; i = 0
        for k, n in enumerate(apl):
            if n == 0 and k > lv: break
            seg = h[i:i + n]; i += n
            base = (g.get('base_actions_per_level') or [0] * (k + 1))[k] or None
            tk = [x.get('generated_tokens') or 0 for x in seg]
            att.append((s, sum(tk), n, k < lv, n / base if base else None, sum(tk[:10]), sum(tk[10:]), max(0, n - 10)))
            if k >= lv: break
def hazard(rows, idx, bins, unit):
    out = []
    for lo, hi in zip(bins, bins[1:]):
        risk = [r for r in rows if r[idx] > lo]
        ev = sum(1 for r in risk if r[3] and r[idx] <= hi)
        expo = sum(min(r[idx], hi) - lo for r in risk)
        out.append(round(ev / expo * unit, 3) if expo else None)
    return out
for s in ['dev', 'hold']:
    R = [a for a in att if a[0] == s]
    print(f"{s:4s} attempts {len(R):4d} cleared {sum(a[3] for a in R):4d} | per 10k tokens, bins 0-10-20-40-80k: {hazard(R, 1, [0, 1e4, 2e4, 4e4, 8e4], 1e4)}"
          f" | per 10 actions, bins 0-25-50-100-200: {hazard(R, 2, [0, 25, 50, 100, 200], 10)}")

print('Human-normalized clock (actions / human baseline actions for the level), bins 0-0.5-1-2-4:')
for s in ['dev', 'hold']:
    R = [a for a in att if a[0] == s and a[4] is not None]
    print(f"{s:4s} {hazard(R, 4, [0, 0.5, 1, 2, 4], 1)} clears per baseline-unit")
print('Tokens per action, first 10 actions of a level vs later actions:')
for s in ['dev', 'hold']:
    R = [a for a in att if a[0] == s and a[2] > 10]
    print(f"{s:4s} first10 {sum(a[5] for a in R) / (10 * len(R)):.0f}  later {sum(a[6] for a in R) / max(1, sum(a[7] for a in R)):.0f}  (levels with >10 actions: {len(R)})")
