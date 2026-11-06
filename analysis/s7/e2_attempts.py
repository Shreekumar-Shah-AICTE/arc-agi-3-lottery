import json,collections,statistics as st
rows=[json.loads(l) for l in open('/data/kaggle_ro/out/e2/e2_raw.jsonl')]
print('rows',len(rows),'keys',sorted(rows[0].keys()))
c=collections.Counter((r['fin'],r.get('natural'),r.get('forced'),r.get('ok')) for r in rows)
for k,v in sorted(c.items(),key=lambda x:-x[1]): print(v,dict(zip(['fin','natural','forced','ok'],k)))
nat=[r for r in rows if r.get('natural')]
print('natural answers',len(nat),'correct',sum(1 for r in nat if r['ok']),'precision',round(sum(1 for r in nat if r['ok'])/max(1,len(nat)),3))
print('natural ctok quantiles', [int(x) for x in st.quantiles([r['ctok'] for r in nat],n=4)], 'max',max(r['ctok'] for r in nat))
for cap in (16000,24000,32000,40000,48000):
    n=sum(1 for r in nat if r['ctok']<=cap); ok=sum(1 for r in nat if r['ctok']<=cap and r['ok'])
    print(f'  natural finished within {cap}: {n} ({ok} correct)')
fr=[r for r in rows if r.get('forced')]
print('forced',len(fr),'ok',sum(1 for r in fr if r['ok']))
# attempt-1 value
by=collections.defaultdict(dict)
for r in rows: by[(r['tid'],r['j'])][r['a']]=r
a1=[(k,v) for k,v in by.items() if 1 in v]
gain=sum(1 for k,v in a1 if v[1]['ok'] and not v.get(0,{}).get('ok'))
print('inputs with attempt-1:',len(a1),'attempt-1 rescued:',gain)
print('mean secs per attempt',round(st.mean(r['secs'] for r in rows)),'mean ctok',round(st.mean(r['ctok'] for r in rows)))
