"""V2: share of run-to-run variance of the 25-game
mean explained by per-game variance, over the 14 valid B12-family dev runs on the 25 public games
(control plus variants whose verdicts were NOISE or null; ThinkingCap b28_tc excluded as a real
negative effect). Prediction registered: share >= 60%. Run crack_table.py first."""
import csv, math, random, statistics as st
from collections import defaultdict
from common import HERE
FAM = ['b12_fb25', 'b12_ht', 'b12_ht2', 'b12_len', 'b12_sdt', 'b12_sdt_r2', 'b12_x7b', 'b15c_fb25', 'b28_ctl_fb25',
       'b29_occ_fb25', 'b31_spec1_fb25', 'b31_spec2_fb25', 'b32_ctl_r2', 'b32_ctl_r3', 'b32_spec2_r2', 'b32_spec2_r3']
# Correction after review: the two runs that did not play all 25 games (Appendix A rows 31-32) entered the first version
# with 0 levels on their unplayed games, which counts a game that was never played as run-to-run variation. Left out now.
from common import INVALID
FAM = [r for r in FAM if r not in INVALID]
S, L = defaultdict(dict), defaultdict(dict)
for r in csv.DictReader(open(HERE / 'out' / 'crack_table.csv')):
    if r['run'] in FAM:
        S[r['run']][r['game']] = float(r['score']); L[r['run']][r['game']] = int(r['levels'])
games = sorted(set.intersection(*[set(v) for v in S.values()])); N = len(games)
def stats(rs):
    m = [st.mean(S[r][g] for g in games) for r in rs]
    return st.stdev(m), math.sqrt(sum(st.variance([S[r][g] for r in rs]) for g in games)) / N, st.mean(m)
obs, pred, mu = stats(FAM)
print(f'{len(FAM)} runs x {N} games: mean {mu:.2f}, observed SD {obs:.2f}, predicted SD {pred:.2f}, share {pred**2 / obs**2:.0%}')
random.seed(0); sh = []
for _ in range(2000):
    rs = [random.choice(FAM) for _ in FAM]
    if len(set(rs)) < 3: continue
    o, p, _ = stats(rs)
    if o > 0: sh.append(p * p / (o * o))
sh.sort(); print(f'bootstrap 90% interval for the share: {sh[int(.05 * len(sh))]:.0%} to {sh[int(.95 * len(sh))]:.0%}')
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b); va = sum((x - ma) ** 2 for x in a); vb = sum((y - mb) ** 2 for y in b)
    return None if va == 0 or vb == 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)
cs = [c for i, g in enumerate(games) for h in games[i + 1:]
      if (c := corr([S[r][g] for r in FAM], [S[r][h] for r in FAM])) is not None]   # pairs where both games vary
print(f'mean pairwise correlation of per-game scores across runs: {st.mean(cs):+.3f} ({len(cs)} game pairs that vary)')
sw = [(g, min(L[r][g] for r in FAM), max(L[r][g] for r in FAM)) for g in games]
print('games whose level count varies:', sum(a != b for _, a, b in sw), 'of', N, '; max swing', max(b - a for _, a, b in sw), 'levels')
