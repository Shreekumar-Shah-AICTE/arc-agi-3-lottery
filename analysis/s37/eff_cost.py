"""Cost of the action-efficiency term on our development runs (paper Section 3.3). Exploratory; no fitted parameters.

Official score = the run file's per-game final_score (local scoring with the ARC-AGI-3 formula). Completion-only score
= the weighted completion share alone: sum of l over completed levels / sum of l over all levels, times 100 (levels are
cleared in order, so completed levels are 1..c). Loss = 1 - official / completion-only, on the run's mean over games.
Prints every development run, then the 16 B12-family runs (the paper's number), the four runs first quoted for this in
October (b12d25, b12d25_r2, b15_vllm, b15_vllm_r2), and the range over all runs."""
import gzip, json, statistics as st
from common import dev_run_files, B12_FAMILY as _FAM16, INVALID
B12_FAMILY = [r for r in _FAM16 if r not in INVALID]   # 14 valid runs (the two partial runs left out; same numbers)

FIRST4 = ('b12d25', 'b12d25_r2', 'b15_vllm', 'b15_vllm_r2')
res = {}
for p in dev_run_files():
    run = p.name.split('.')[0]
    gr = json.load(gzip.open(p, 'rt'))['game_runs']
    off = st.mean(g['final_score'] for g in gr)
    comp = st.mean(100 * (c := g['levels_completed'] or 0) * (c + 1) / (n := g['number_of_levels']) / (n + 1) for g in gr)
    res[run] = (len(gr), off, comp, 1 - off / comp if comp else None)
    print(f'{run:15s} games {len(gr):3d}  official {off:6.2f}  completion-only {comp:6.2f}  '
          f'loss {"n/a" if comp == 0 else f"{100 * (1 - off / comp):.1f}%"}')
def span(runs):
    v = sorted(res[r][3] for r in runs)
    return f'median {100 * st.median(v):.1f}%, range {100 * v[0]:.1f}% to {100 * v[-1]:.1f}%'
print(f'B12 family ({len(B12_FAMILY)} runs): loss {span(B12_FAMILY)}')
print(f'four runs first quoted: loss {span(FIRST4)}')
allr = [r for r, v in res.items() if v[3] is not None]
print(f'all {len(allr)} development runs with a completed level: loss {span(allr)}')
# Why the term costs little: on cleared levels, a = actions the scorer counts for the level (actions_per_level in the
# run file), h = the scorer's human baseline for that level (base_actions_per_level). (h/a)^2 >= 1.15 iff a/h <= 0.9325.
from common import HERE as _H
ratios = []
for run in B12_FAMILY:
    for g in json.load(gzip.open(_H / 'data' / 'runs' / f'{run}.benchmark.json.gz', 'rt'))['game_runs']:
        a, h, c = g['actions_per_level'], g['base_actions_per_level'], g['levels_completed'] or 0
        ratios += [a[i] / h[i] for i in range(min(c, len(a), len(h))) if h[i]]
q = st.quantiles(ratios, n=4)
print(f'cleared levels, {len(B12_FAMILY)} family runs, all 25 games: n {len(ratios)}; actions / human baseline median {st.median(ratios):.2f} '
      f'(quartiles {q[0]:.2f} to {q[2]:.2f}); at the 1.15 cap {100 * sum(r <= 1.15 ** -0.5 for r in ratios) / len(ratios):.0f}%; '
      f'slower than the baseline {100 * sum(r > 1 for r in ratios) / len(ratios):.0f}%')
