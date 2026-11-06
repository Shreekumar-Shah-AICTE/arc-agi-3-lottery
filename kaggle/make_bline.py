# Build a B-line notebook = B1 (Franzen fork) + bline patch + env flags. Usage: make_bline.py dir kernel_slug patch KEY=VAL ...
import json,copy,os,sys
d,kid,patchf=sys.argv[1:4]; flags=dict(a.split('=',1) for a in sys.argv[4:]); demo_min=flags.pop('DEMO_MIN',None); holdout=flags.pop('HOLDOUT',None)
nb=json.load(open('/data/arc/kaggle/b1_franzen/b1.ipynb'))
n=copy.deepcopy(nb); patch=open(patchf).read()
cell=copy.deepcopy(n['cells'][2]); cell['source']=('%%writefile /kaggle/bline.patch\n'+patch).splitlines(True)
n['cells'].insert(3,cell)
c=n['cells'][5]; s=''.join(c['source'])
a="print('harness patch applied successfully')\n"; assert s.count(a)==1
s=s.replace(a,a+'subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/bline.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\nprint("bline patch applied")\n')
a="setup_env = {\n"; assert s.count(a)==1
s=s.replace(a,a+''.join(f"    '{k}': '{v}',\n" for k,v in flags.items()))
c['source']=s.splitlines(True)
hk=[cc for cc in n['cells'] if 'bm.solver.max_runtime_s_per_game = 25*60' in ''.join(cc['source'])]; assert len(hk)==1; hs=''.join(hk[0]['source'])
if demo_min:  # per-game demo budget in minutes (hidden set gives ~48 stream-min/game)
    hs=hs.replace('bm.solver.max_runtime_s_per_game = 25*60',f'bm.solver.max_runtime_s_per_game = {int(demo_min)}*60')
if holdout:  # run the 15 normally-excluded public games instead of the 10 dev games
    a="demo_excluded_games = [] if TRUE_SUBMISSION else ["; i=hs.index(a); j=hs.index(']',i+len(a))
    hs=hs[:i]+a+"'ar25','ft09','lp85','r11l','re86','sb26','sc25','tr87','tu93','vc33'"+hs[j:]
hk[0]['source']=hs.splitlines(True)
for cc in n['cells']:
    if cc['cell_type']=='code': cc['outputs']=[]; cc['execution_count']=None
os.makedirs(d,exist_ok=True); json.dump(n,open(f'{d}/{os.path.basename(d)}.ipynb','w'),indent=1)
m=json.load(open('/data/arc/kaggle/b1_franzen/kernel-metadata.json')); m.update(id='shreekumarshah/'+kid,title=kid,code_file=f'{os.path.basename(d)}.ipynb')
json.dump(m,open(f'{d}/kernel-metadata.json','w'),indent=1); print('built',d,kid,flags,'demo_min',demo_min,'holdout',holdout)
