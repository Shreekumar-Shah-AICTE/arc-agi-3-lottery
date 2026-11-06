# exec'd by make_s20.py for arm 'wmk' (n, src, put, rep, ENV_ANCHOR, d, kid, opts, base already defined)
ALL25 = ['ar25', 'bp35', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lf52', 'lp85', 'ls20', 'm0r0', 'r11l',
         're86', 's5i5', 'sb26', 'sc25', 'sk48', 'sp80', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']
games = [g for g in opts.get('games', ','.join(ALL25)).split(',') if g]
assert games and all(g in ALL25 for g in games), games
minutes = int(opts.get('minutes', 121))
excluded = [g for g in ALL25 if g not in games]
patch = open(opts.get('patch', '/data/arc/kaggle/s20_wmk.patch')).read()   # S21: patch= selects a variant (e.g. s21_wmkd.patch)
cell = copy.deepcopy(n['cells'][3]); cell['source'] = ('%%writefile /kaggle/wmk.patch\n' + patch).splitlines(True)
n['cells'].insert(4, cell)   # cells 4.. shift by one: old 5 -> 6, old 17 -> 18
s = src(6)
s = rep(s, 'print("bline patch applied")\n',
        'print("bline patch applied")\n'
        '_wmk_apply = subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/wmk.patch"], cwd=f"{BUNDLE_DIR}/src", capture_output=True, text=True)\n'
        'print(_wmk_apply.stdout[-2000:], _wmk_apply.stderr[-2000:])\n'
        'print("#S20 WMK patch applied" if _wmk_apply.returncode == 0 else f"#S20 WMK patch FAILED rc={_wmk_apply.returncode} (run = B12 replicate)", flush=True)\n')
s = rep(s, ENV_ANCHOR, ENV_ANCHOR.replace("    'EXPOSE_UNDO': 'on',\n}\n",
    "    'EXPOSE_UNDO': 'on',\n\n"
    "    ########## S20 WMK: world-model toolkit (check_model / plan / run_plan in the python sandbox) ##########\n"
    "    'ARC3_WMK': '1',\n"
    "}\n"))
put(6, s)
s = src(18)
s = rep(s, 'demo_excluded_games = []\n', 'demo_excluded_games = %r  # S20 WMK smoke subset: %s\n' % (excluded, ','.join(games)))
s = rep(s, 'bm.solver.max_runtime_s_per_game = 121*60', 'bm.solver.max_runtime_s_per_game = %d*60' % minutes)
put(18, s)
title = ("# B12-WMK%s: world-model toolkit (%d games x %d min)\n\nPrivate experiment (S20). = arc3-b12-fb25 (B12) + a harness "
         "patch (cell 4, /kaggle/wmk.patch, ARC3_WMK=1): the python sandbox preloads check_model (replay this level's "
         "transitions through the model's own simulator), plan (bounded BFS over that simulator) and run_plan (execute one "
         "real action at a time, stop at the first unpredicted observation), plus a 6-line system-prompt note. Save run: "
         "%s, %d min each. Cells 0, 4 (new), 6 (old 5), 18 (old 17) differ from arc3-b12-fb25.\n\n"
         % (opts.get('label', ''), len(games), minutes, ', '.join(games), minutes))
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells', games, minutes)
