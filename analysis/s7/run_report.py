# Compare new ARC-AGI-3 dev runs against stored baselines. Usage: python3 run_report.py NAME=/data/kaggle_ro/out/x8 [NAME2=dir ...]
import json,gzip,glob,os,re,sys,statistics as st,pandas as pd
BASE={'B1':'/data/arc/data/runs/b1.benchmark.json.gz','B1c':'/data/arc/data/runs/b1c.benchmark.json.gz',
      'X6a':'/data/arc/data/runs/b4.benchmark.json.gz','X6b':'/data/arc/data/runs/b4v2.benchmark.json.gz','X7':'/data/kaggle_ro/out/x7'}
NEW=dict(a.split('=',1) for a in sys.argv[1:]); RUNS={**BASE,**NEW}
def bench(p):
    if os.path.isdir(p): p=os.path.join(p,'benchmark.json')
    return json.load(gzip.open(p) if p.endswith('.gz') else open(p))
rows=[]
for run,p in RUNS.items():
    try: b=bench(p)
    except Exception as e: print('skip',run,e); continue
    for g in b['game_runs']:
        h=g['history']
        rows.append(dict(run=run,game=g['game_id'].split('-')[0],levels=g['levels_completed'],score=g['final_score'],actions=len(h),
                         gen_tok=sum(x.get('generated_tokens',0) or 0 for x in h)))
df=pd.DataFrame(rows); order=[r for r in RUNS if r in set(df.run)]
lv=df.pivot(index='game',columns='run',values='levels')[order]
print(lv.to_string()); print('TOTAL LEVELS',lv.sum().to_dict())
print('MEAN SCORE',df.groupby('run').score.mean()[order].round(2).to_dict())
print('ACTIONS',df.groupby('run').actions.sum()[order].to_dict()); print('GEN TOK',df.groupby('run').gen_tok.sum()[order].to_dict())
base=[r for r in ('B1','B1c','X6a','X6b') if r in lv]
for r in NEW:
    if r in lv: print(r,'below min(B1,B1c,X6a,X6b) in',int((lv[r]<lv[base].min(axis=1)).sum()),'games; above max in',int((lv[r]>lv[base].max(axis=1)).sum()))
for r,d in NEW.items():
    if not os.path.isdir(d): continue
    s=open(f'{d}/summary.txt').read() if os.path.exists(f'{d}/summary.txt') else ''
    m=re.search(r'generated tokens/sec: ([\d.]+)',s); print(r,'tok/s',m.group(1) if m else None,'| gate tok/s >= 555')
    C=[];R=[];F={};req=0
    for f in glob.glob(f'{d}/*_requests.jsonl'):
        for l in open(f):
            o=json.loads(l)
            if o.get('event')=='request': req+=1
            u=o.get('usage') or {}
            if o.get('event')!='response' or not u: continue
            C.append(u.get('completion_tokens',0) or 0); R.append(u.get('reasoning_tokens',0) or 0); F[o.get('finish_reason')]=F.get(o.get('finish_reason'),0)+1
            eff=(o.get('chat_template_kwargs') or {}).get('reasoning_effort'); F['effort='+str(eff)]=F.get('effort='+str(eff),0)+1
    if C: print(r,'requests',req,'responses',len(C),'completion',sum(C),'reasoning share',round(sum(R)/max(1,sum(C)),3),
                'median completion',st.median(C),'median reasoning',st.median(R),'finish/effort',F)
