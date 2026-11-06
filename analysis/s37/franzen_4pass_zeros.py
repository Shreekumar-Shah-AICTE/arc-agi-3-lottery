"""Level-1 zeros in dfranzen's 4-pass file (6 Nov 2026, exploratory, CPU; outside review round 2 asked whether the slot
rule makes much of the noise). Counts, per pass, the games that started (any action) and the games that ended with 0 levels.
Data: data/external/dfranzen_milestone2_4pass.benchmark.json.gz (pass order = order of appearance, as in lever2b)."""
import gzip, json
from collections import defaultdict
from common import HERE
d = json.load(gzip.open(HERE / 'data' / 'external' / 'dfranzen_milestone2_4pass.benchmark.json.gz', 'rt'))
cnt = defaultdict(int); started = defaultdict(int); zeros = defaultdict(list)
for g in d['game_runs']:
    game = g['game_id'].split('-')[0]; p = cnt[game]; cnt[game] += 1
    h = g.get('history') or []
    if any((x.get('action') or {}).get('id') != 'RESET' for x in h): started[p] += 1
    if (g.get('levels_completed') or 0) == 0: zeros[p].append(game)
P = sorted(cnt[k] for k in cnt)
for p in sorted(started):
    print(f'pass {p}: games started {started[p]} of {sum(1 for k in cnt if cnt[k] > p)}, zero-level games {len(zeros[p])} ({", ".join(sorted(zeros[p]))})')
z = [len(zeros[p]) for p in sorted(started)]
print(f'zero-level games per pass: {", ".join(map(str, z))} (mean {sum(z)/len(z):.2f})')
