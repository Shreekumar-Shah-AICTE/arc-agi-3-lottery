# Empirical clear-probability vs tokens spent on the current level (Kaplan-Meier), vs Franzen's scheduler 'chance' heuristic.
import json,gzip,math,collections
RUNS=['b1','b1c','b3v2','b4','b4v2','b5','b5c','b7','b7c']
att=[]  # (run, game, level_idx(1-based), tokens_on_level, actions_on_level, cleared)
for r in RUNS:
    b=json.load(gzip.open(f'/data/arc/data/runs/{r}.benchmark.json.gz'))
    for g in b['game_runs']:
        h=g['history']; apl=g.get('actions_per_level') or []; lv=g['levels_completed']
        bounds=[]; c=0
        for k in range(lv): c+=apl[k]; bounds.append(c)
        start=0
        for k in range(lv+1):
            end=bounds[k] if k<lv else len(h)
            seg=h[start:end]; tok=sum(x.get('generated_tokens',0) or 0 for x in seg)
            if k==lv: tok+=g.get('final_generated_tokens',0) or 0
            if k<g['number_of_levels']:
                att.append((r,g['game_id'].split('-')[0],k+1,tok,len(seg),k<lv))
            start=end
json.dump(att,open('/data/analysis/level_attempts.json','w'))
print('level attempts',len(att),'cleared',sum(a[5] for a in att),'censored',sum(1 for a in att if not a[5]))
def km(rows,grid):
    # P(cleared by t) via KM on time=tokens, event=cleared
    rows=sorted(rows,key=lambda a:a[3]); n=len(rows); S=1.0; out={}; i=0; atrisk=n
    ev=[(a[3],a[5]) for a in rows]
    j=0; surv=[]
    for t,e in ev:
        if e: S*=(1-1/atrisk)
        atrisk-=1; surv.append((t,S))
    for gt in grid:
        s=1.0
        for t,v in surv:
            if t<=gt: s=v
            else: break
        out[gt]=round(1-s,3)
    return out,len(rows)
grid=[10000,20000,40000,62000,80000,100000,124000]
for name,sel in [('all',lambda a:True),('level1',lambda a:a[2]==1),('level2-3',lambda a:2<=a[2]<=3),('level4+',lambda a:a[2]>=4)]:
    rows=[a for a in att if sel(a)]; o,n=km(rows,grid)
    print(f'{name:9s} n={n:3d} P(cleared by t tokens):',o)
# conditional: of attempts still uncleared at t0, share cleared later (observed), and at-risk counts
for t0 in (20000,40000,62000,80000):
    risk=[a for a in att if a[3]>t0]; cl=[a for a in risk if a[5]]
    print(f'still open after {t0//1000}k tokens: {len(risk)} attempts, later cleared {len(cl)} ({len(cl)/max(1,len(risk)):.0%}); median extra tokens to clear {sorted(a[3]-t0 for a in cl)[len(cl)//2] if cl else None}')
print('heuristic token-chance 0.5^((t/62k)^2):',{t:round(0.5**((t/62000)**2),3) for t in grid})
cl=[a[3] for a in att if a[5]]; cl.sort()
print('tokens-to-clear quantiles (cleared only): p25 %d p50 %d p75 %d p90 %d max %d'%(cl[len(cl)//4],cl[len(cl)//2],cl[3*len(cl)//4],cl[int(.9*len(cl))],cl[-1]))
