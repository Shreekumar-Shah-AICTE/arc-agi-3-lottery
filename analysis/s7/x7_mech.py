import json,re,glob,os,collections,statistics as st
RUNS={'X6v2':'/data/kaggle_ro/out/x6v2','X7':'/data/kaggle_ro/out/x7'}
out={}
for run,d in RUNS.items():
    r={}
    s=open(f'{d}/serve.log',errors='ignore').read()
    pre=re.findall(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)',s)
    r['prefill_batches']=len(pre); r['prefill_new_tok']=sum(int(b) for a,b,c in pre); r['prefill_cached_tok']=sum(int(c) for a,b,c in pre)
    dec=re.findall(r'Decode batch.*?#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)',s)
    if dec:
        r['decode_lines']=len(dec); r['decode_tps_mean']=round(st.mean(float(t) for _,t in dec),1); r['running_mean']=round(st.mean(int(n) for n,_ in dec),2)
    q=re.findall(r'#queue-req: (\d+)',s); r['queue_mean']=round(st.mean(int(x) for x in q),2) if q else None
    L=json.load(open(sorted(glob.glob(f'{d}/arc3-*.log'))[0])); txt=''.join(x['data'] for x in L)
    # log lines are often duplicated; count unique lines
    lines=set(txt.splitlines())
    r['analyzer_timeouts_unique']=sum(1 for l in lines if 'analyzer request timed out' in l)
    r['summary_failed_unique']=sum(1 for l in lines if 'rolling summary request failed' in l)
    # request logs
    nreq=0; pt=0; ct=0; comp=0; fin=collections.Counter(); summ_req=0; summ_resp=0; per_game={}
    for f in sorted(glob.glob(f'{d}/*_p0_requests.jsonl')):
        g=os.path.basename(f).split('-')[0]; gr=dict(req=0,resp=0,summ=0,pt=0,ct=0,comp=0)
        with open(f) as fh:
            for line in fh:
                o=json.loads(line)
                last=o['messages'][-1] if o.get('messages') else {}
                is_summ='Summarize everything above' in str(last.get('content',''))
                if o['event']=='request':
                    gr['req']+=1; gr['summ']+=is_summ
                elif o['event']=='response':
                    gr['resp']+=1; u=o.get('usage') or {}
                    gr['pt']+=u.get('prompt_tokens',0); gr['ct']+=(u.get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0; gr['comp']+=u.get('completion_tokens',0)
                    fin[o.get('finish_reason')]+=1
                    if is_summ: summ_resp+=1
        per_game[g]=gr
    r['requests']=sum(v['req'] for v in per_game.values()); r['responses']=sum(v['resp'] for v in per_game.values())
    r['summary_requests_logged']=sum(v['summ'] for v in per_game.values()); r['summary_responses_logged']=summ_resp
    r['prompt_tok']=sum(v['pt'] for v in per_game.values()); r['cached_tok_reported']=sum(v['ct'] for v in per_game.values()); r['completion_tok']=sum(v['comp'] for v in per_game.values())
    r['finish']=dict(fin)
    r['per_game_req']={g:v['req'] for g,v in per_game.items()}; r['per_game_summ']={g:v['summ'] for g,v in per_game.items()}
    out[run]=r
json.dump(out,open('/data/analysis/x7_mech.json','w'),indent=1)
for k in out['X7']:
    print(f"{k:26s} X6v2={out['X6v2'].get(k)}  |  X7={out['X7'].get(k)}")
