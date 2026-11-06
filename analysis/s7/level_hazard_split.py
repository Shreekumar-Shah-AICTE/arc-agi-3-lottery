# Clear hazard per 10k tokens by tokens already spent on the level, split dev/holdout and level 1 vs later (level_attempts.json:
# run, game, level, tokens_on_level, actions_on_level, cleared). Earlier finding: holdout shows a learning phase; dev is flat.
import json
A=json.load(open('/data/arc/analysis/s7/level_attempts.json')); B=[0,10e3,20e3,40e3,60e3,90e3]
for nm,runs in [('dev',{'b1','b1c','b3v2','b4','b4v2'}),('holdout',{'b5','b5c','b7','b7c'})]:
  for ln,f in [('all',lambda x:True),('L1',lambda x:x[2]==1),('L>=2',lambda x:x[2]>=2)]:
    S=[x for x in A if x[0] in runs and f(x)]; print(nm,ln,'attempts',len(S),'cleared',sum(x[5] for x in S))
    for a,b in zip(B,B[1:]):
        risk=[x for x in S if x[3]>a]; ev=[x for x in risk if x[5] and x[3]<=b]; expo=sum(min(x[3],b)-a for x in risk)
        if risk: print(f'   {int(a/1e3):2d}-{int(b/1e3):2d}k at_risk {len(risk):3d} cleared {len(ev):3d} hazard/10k {len(ev)/expo*1e4 if expo else 0:.3f}  heuristic {0.5**((a/62000)**2):.2f}')
