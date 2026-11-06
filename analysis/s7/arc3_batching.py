import re,collections,statistics as st
for run,p in [('X6v2','/data/kaggle_ro/out/x6v2/serve.log'),('X7','/data/kaggle_ro/out/x7/serve.log')]:
    s=open(p,errors='ignore').read()
    m=re.search(r'max_total_num_tokens[=: ]+(\d+)',s); print(run,'max_total_num_tokens',m.group(1) if m else None)
    for kw in ['max_running_requests','context_len','mem_fraction','KV Cache is allocated','#tokens:','mamba']:
        mm=re.search(r'.{0,80}'+re.escape(kw)+r'.{0,120}',s)
        if mm and kw in ('KV Cache is allocated','max_running_requests','context_len'): print('  ',mm.group(0)[:220])
    dec=re.findall(r'Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)',s)
    if not dec:
        dec2=re.findall(r'Decode batch.*',s); print('  sample decode line:',dec2[0][:300] if dec2 else None); continue
    by=collections.defaultdict(list); us=[]; acc=[]
    for n,tok,u,a,t in dec: by[int(n)].append(float(t)); us.append(float(u)); acc.append(float(a))
    print('  token usage max',max(us),'p90',round(st.quantiles(us,n=10)[-1],3),'| accept len mean',round(st.mean(acc),2))
    for n in sorted(by): print(f'   running={n:2d} lines={len(by[n]):4d} gen_tps median={st.median(by[n]):7.1f} per-req={st.median(by[n])/n:6.1f}')
