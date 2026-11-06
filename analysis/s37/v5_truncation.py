"""Budget retrodiction (no GPU): does a 1/3-budget run (B12-D25: 16 stream-min per game) look like a
full-budget run (B12 FB25: 48.4 stream-min per game) cut off at the same per-game token count?
If yes, the score is set by the token budget alone (clock-bound), which is what the lottery model assumes."""
import gzip, json, statistics as st
from common import HERE
FB = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']          # B12 control, full budget
D25 = ['b12d25', 'b12d25_r2']                                            # same notebook, 1/3 budget (D39, D44)
def load(run):
    with gzip.open(HERE / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt') as f:
        return {g['game_id'].split('-')[0]: g for g in json.load(f)['game_runs']}
def clears_within(g, budget):
    """levels cleared within the first `budget` generated tokens of the game's history"""
    h = g['history']; apl = g['actions_per_level']; lv = g['levels_completed']; i = 0; used = 0; n = 0
    for k in range(lv):
        seg = h[i:i + apl[k]]; i += apl[k]; used += sum(x.get('generated_tokens') or 0 for x in seg)
        if used <= budget: n += 1
        else: break
    return n
fb = {r: load(r) for r in FB}; dd = {r: load(r) for r in D25}
games = sorted(set.intersection(*[set(v) for v in list(fb.values()) + list(dd.values())]))
tok_d = {g: st.mean(sum(x.get('generated_tokens') or 0 for x in dd[r][g]['history']) for r in D25) for g in games}
tok_f = {g: st.mean(sum(x.get('generated_tokens') or 0 for x in fb[r][g]['history']) for r in FB) for g in games}
print(f'games {len(games)}; tokens per run: D25 {sum(tok_d.values())/1e6:.2f}M vs FB25 {sum(tok_f.values())/1e6:.2f}M (ratio {sum(tok_f.values())/sum(tok_d.values()):.2f})')
act_d = [sum(dd[r][g]['levels_completed'] for g in games) for r in D25]
trunc = [sum(clears_within(fb[r][g], tok_d[g]) for g in games) for r in FB]
full = [sum(fb[r][g]['levels_completed'] for g in games) for r in FB]
print('levels: D25 actual', act_d, '| FB25 cut at the D25 per-game token count', trunc, '| FB25 full', full)
