import json,glob,os,collections
for run,d in [('X6v2','/data/kaggle_ro/out/x6v2'),('X7','/data/kaggle_ro/out/x7')]:
    tot=0; per={}; maxpt={}
    for f in sorted(glob.glob(f'{d}/*_p0_requests.jsonl')):
        g=os.path.basename(f).split('-')[0]; prev=None; drops=0; mp=0
        with open(f) as fh:
            for line in fh:
                o=json.loads(line)
                if o['event']=='response':
                    mp=max(mp,(o.get('usage') or {}).get('prompt_tokens',0)); continue
                n=len(o['messages'])
                if prev is not None and n<prev: drops+=1
                prev=n
        per[g]=drops; maxpt[g]=mp; tot+=drops
    print(run,'context drops (message count fell):',tot,per); print('   max prompt tokens per game:',maxpt)
