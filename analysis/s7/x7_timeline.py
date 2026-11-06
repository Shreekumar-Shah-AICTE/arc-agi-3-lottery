import json,gzip,collections
R={'X6a':'/data/arc/data/runs/b4.benchmark.json.gz','X6b':'/data/arc/data/runs/b4v2.benchmark.json.gz','X7':'/data/kaggle_ro/out/x7/benchmark.json'}
b=json.load(open(R['X7'])); h=b['game_runs'][0]['history']; print('history keys:', sorted(h[0].keys()))
def level_times(p):
    b=json.load(gzip.open(p) if p.endswith('.gz') else open(p)); res={}
    for g in b['game_runs']:
        t=0; out=[]; prev=0
        apl=g.get('actions_per_level') or []
        # cumulative actions at each level completion
        cum=0; marks=[]
        for a in apl:
            if a is None: break
            cum+=a; marks.append(cum)
        h=g['history']; times=[]; tt=0
        for x in h:
            tt+=x.get('wallclock_seconds',0) or 0; times.append(tt)
        lv=g['levels_completed']
        res[g['game_id'].split('-')[0]]=[round(times[m-1]) if 0<m<=len(times) else None for m in marks[:lv]]
    return res
for k,p in R.items():
    print(k, level_times(p))
