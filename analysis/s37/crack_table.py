"""V1: per (run, game) crack table from every downloaded dev-run benchmark.json.

crack = at least one level cleared. Tokens are generated tokens summed over the
history entries (per action) up to the first level clear (actions_per_level[0]).
Writes analysis/s37/out/crack_table.csv and prints per-run and per-game summaries.
"""
import csv, gzip, json, statistics as st
from collections import defaultdict
from common import HERE, dev_run_files
OUT = HERE / 'out'; OUT.mkdir(exist_ok=True)
rows = []
for p in dev_run_files():
    run = p.name.split('.')[0]
    with gzip.open(p, 'rt') as f:
        d = json.load(f)
    for g in d['game_runs']:
        h = g.get('history') or []
        apl = g.get('actions_per_level') or []
        lv = g.get('levels_completed') or 0
        k = apl[0] if (lv >= 1 and apl) else None
        tok = [x.get('generated_tokens') or 0 for x in h]
        wc = [x.get('wallclock_seconds') or 0 for x in h]
        rows.append(dict(run=run, game=g['game_id'].split('-')[0], n_levels=g['number_of_levels'], levels=lv,
                         score=round(g.get('final_score') or 0, 3), cracked=int(lv >= 1), actions=len(h), tokens=sum(tok),
                         wall_s=round(max(wc) if wc else 0), actions_to_first=k if k is not None else '',
                         tokens_to_first=sum(tok[:k]) if k else '', wall_to_first=round(wc[k - 1]) if k and k <= len(wc) else '',
                         state=g.get('state', '')))
with open(OUT / 'crack_table.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by_run = defaultdict(list)
for r in rows: by_run[r['run']].append(r)
print(f'{len(rows)} (run, game) rows from {len(by_run)} runs')
for run, R in by_run.items():
    print(f"{run:16s} games {len(R):2d}  mean score {st.mean(r['score'] for r in R):6.2f}  levels {sum(r['levels'] for r in R):3d}  cracked {sum(r['cracked'] for r in R):2d}")
by_game = defaultdict(list)
for r in rows: by_game[r['game']].append(r)
print('game  runs  P(crack)  mean levels|crack  median tokens to first clear')
for gname, R in sorted(by_game.items(), key=lambda kv: -st.mean(r['cracked'] for r in kv[1])):
    c = [r for r in R if r['cracked']]
    t = [r['tokens_to_first'] for r in c if r['tokens_to_first'] != '']
    print(f"{gname:5s} {len(R):4d}  {st.mean(r['cracked'] for r in R):8.2f}  {(st.mean(r['levels'] for r in c) if c else 0):17.2f}  {(st.median(t) if t else float('nan')):10.0f}")
