import json,gzip,pandas as pd
R={'B1ctl':'/data/arc/data/runs/b1.benchmark.json.gz','B1c':'/data/arc/data/runs/b1c.benchmark.json.gz',
   'X6a':'/data/arc/data/runs/b4.benchmark.json.gz','X6b':'/data/arc/data/runs/b4v2.benchmark.json.gz',
   'X7':'/data/kaggle_ro/out/x7/benchmark.json'}
rows=[]
for run,p in R.items():
    b=json.load(gzip.open(p) if p.endswith('.gz') else open(p))
    for g in b['game_runs']:
        h=g['history']; gen=sum(x.get('generated_tokens',0) or 0 for x in h)
        rows.append(dict(run=run,game=g['game_id'].split('-')[0],levels=g['levels_completed'],nlev=g['number_of_levels'],
             score=g['final_score'],actions=len(h),gen_tok=gen,state=g['state'],wall=g.get('final_wallclock_seconds')))
df=pd.DataFrame(rows)
df.to_csv('/data/analysis/x7_compare.csv',index=False)
lv=df.pivot(index='game',columns='run',values='levels')[list(R)]
print('LEVELS per game'); print(lv.to_string()); print('TOTAL', lv.sum().to_dict())
sc=df.groupby('run').score.mean()[list(R)]; print('MEAN SCORE', sc.round(2).to_dict())
tk=df.groupby('run').gen_tok.sum()[list(R)]; print('GEN TOKENS', tk.to_dict())
ac=df.groupby('run').actions.sum()[list(R)]; print('ACTIONS', ac.to_dict())
best_ctrl=lv[['B1ctl','B1c','X6a','X6b']]
print('X7 below min(others) in', int((lv.X7<best_ctrl.min(axis=1)).sum()),'games; above max(others) in',int((lv.X7>best_ctrl.max(axis=1)).sum()))
