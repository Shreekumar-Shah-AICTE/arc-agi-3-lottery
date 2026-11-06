"""V6 (exploratory, 2 Nov): is the middle of the ARC-AGI-3 public leaderboard inside run-to-run noise?
Inputs: data/lb/arc-agi-3_lb_20261102T1307.csv.gz (public LB, 13:07 UTC), B12_LB (5 scored reruns of one notebook),
data/external/dfranzen_milestone2_4pass.benchmark.json.gz (public notebook output, 4 concurrent passes, 25 public games).
Prints cluster stats, the rank span of our 5 same-code reruns, a reliability estimate and a simulated private-LB
retention rate (the basis for a sealed prediction), plus the external 4-pass variance check."""
import csv, gzip, json, math, random, statistics as st, collections
from common import HERE, B12_LB
with gzip.open(HERE/'data'/'lb'/'arc-agi-3_lb_20261102T1307.csv.gz','rt',encoding='utf-8-sig') as f:
    sc = sorted((float(r['Score']) for r in csv.DictReader(f) if r.get('Score')), reverse=True)
rank = lambda s: sum(x > s for x in sc) + 1
cl = [s for s in sc if 20 <= s < 40]
sd_c, sd_n = st.stdev(cl), st.stdev(B12_LB)
print(f'teams {len(sc)}; cluster 20-40: n={len(cl)} ({len(cl)/len(sc):.0%}), mean {st.mean(cl):.2f}, SD {sd_c:.2f}')
print('our 5 same-code reruns', B12_LB, 'SD', round(sd_n, 2), '-> ranks', [rank(s) for s in B12_LB])
for name, n in [('5 LB reruns', sd_n), ('dev-run SD 3.44', 3.44), ('noise 2.5', 2.5)]:
    R = max(0.0, 1 - n**2 / sd_c**2)
    print(f'reliability if noise SD={n:.2f} ({name}): {R:.2f}')
def sim(noise, reps=200, top=0.10, seed=7):
    rnd = random.Random(seed); keep = []
    sk = math.sqrt(max(sd_c**2 - noise**2, 0.0)); m = len(cl); k = int(m * top)
    for _ in range(reps):
        t = [rnd.gauss(0, sk) for _ in range(m)]
        pub = [x + rnd.gauss(0, noise) for x in t]; pri = [x + rnd.gauss(0, noise) for x in t]
        a = set(sorted(range(m), key=lambda i: -pub[i])[:k]); b = set(sorted(range(m), key=lambda i: -pri[i])[:k])
        keep.append(len(a & b) / k)
    return st.mean(keep)
for n in (sd_n, 3.0, 2.5, 2.0):
    print(f'simulated share of cluster public-top-10% still top-10% on private, noise {n:.2f}: {sim(n):.0%} (10% = pure chance)')
with gzip.open(HERE/'data'/'external'/'dfranzen_milestone2_4pass.benchmark.json.gz','rt') as f:
    b = json.load(f)
by = collections.defaultdict(list)
for r in b['game_runs']: by[r['game_id'].split('-')[0]].append(r['final_score'])
P = min(map(len, by.values())); G = sorted(by); N = len(G)
m = [st.mean(by[g][p] for g in G) for p in range(P)]
pred = math.sqrt(sum(st.variance(by[g][:P]) for g in G)) / N
print(f'external 4-pass check: pass means {[round(x,2) for x in m]}, observed SD {st.stdev(m):.2f} vs predicted {pred:.2f} if games independent')
