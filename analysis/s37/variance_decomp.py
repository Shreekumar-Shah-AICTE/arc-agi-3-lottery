"""Lottery test: does per-game run-to-run variance explain the SD of the 25-game run mean?"""
import math, statistics as st
from common import CTL, SPEC2, load_run
data = {r: load_run(r) for r in CTL + SPEC2}
games = sorted(set.intersection(*[set(v) for v in data.values()]))
N = len(games)
print('games common', N)
for r in CTL + SPEC2:
    sc = [data[r][g][0] for g in games]
    print(f'{r:16s} mean {st.mean(sc):6.2f}  levels {sum(data[r][g][1] for g in games):3d}  games>0 {sum(x > 0 for x in sc):2d}')
def grp(G):
    return [st.mean([data[r][g][0] for g in games]) for r in G], [st.variance([data[r][g][0] for r in G]) for g in games]
mc, vc = grp(CTL); ms, vs = grp(SPEC2)
vg = [(3 * a + 2 * b) / 5 for a, b in zip(vc, vs)]            # pooled within-group per-game variance (df-weighted)
pred = math.sqrt(sum(vg)) / N
obs = math.sqrt((3 * st.variance(mc) + 2 * st.variance(ms)) / 5)
print(f'control means {[round(x, 2) for x in mc]} SD {st.stdev(mc):.2f}; spec2 means {[round(x, 2) for x in ms]} SD {st.stdev(ms):.2f}')
print(f'predicted SD of 25-game mean from per-game variance (independent games): {pred:.2f}')
print(f'observed pooled within-group SD of run means: {obs:.2f}  -> share of variance explained {pred**2/obs**2:.0%}')
allr = CTL + SPEC2
print('games scoring >0 in at least one of 7 runs', sum(any(data[r][g][0] > 0 for r in allr) for g in games),
      '; in all 7 runs', sum(all(data[r][g][0] > 0 for r in allr) for g in games))
rng = [(g, min(data[r][g][1] for r in allr), max(data[r][g][1] for r in allr)) for g in games]
print('games whose levels_completed varies across runs:', sum(a != b for _, a, b in rng), 'of', N)
print('largest swings:', sorted(rng, key=lambda x: x[1] - x[2])[:6])
