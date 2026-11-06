# B14 = LHS-D25: Lord Han Solo's public notebook (lordhansolo/built-on-tufa-labs-duck-harness-milestone-2 v306,
# vLLM main e975732 + primitive-ai mixed NVFP4-FP8 with built-in MTP head, 14 slots, KV pool ~1.42M) run unchanged
# except the inline hook: matched dev probe on the same 25 public games for ~40 min of play
# (14 slots, 20 min per game -> two waves), request logs on. Gate D40. Submission path untouched.
import json, os, sys
d, kid = sys.argv[1:3]
n = json.load(open('/data/arc/kaggle/b14_lhs_src/lhs_v306.ipynb'))
hook = [i for i, c in enumerate(n['cells']) if '# Inline customization hook.' in ''.join(c['source'])]
assert len(hook) == 1, hook
runc = [i for i, c in enumerate(n['cells']) if 'await bm.run(' in ''.join(c['source'])]
assert len(runc) == 1 and runc[0] > hook[0], runc
assert 'soft_end_time=soft_end' in ''.join(n['cells'][runc[0]]['source'])
HOOK = '''# Inline customization hook.
# B14 (LHS-D25, private experiment): matched dev probe. 25 public games, the notebook's own 14 slots,
# 20 min per game (two waves, ~40 min of play after the server is ready), request logs on.
# Submission path (KAGGLE_IS_COMPETITION_RERUN) is untouched.
if not run_as_submission:
    assert len(bm.games) == 25, len(bm.games)
    bm.solver.max_runtime_s_per_game = 20 * 60
    bm.solver.save_request_logs = True
    soft_end = datetime.now() + timedelta(minutes=42)
    print(f"#B14 dev probe: games={len(bm.games)} concurrency={bm.solver.concurrency} "
          f"per_game_s={bm.solver.max_runtime_s_per_game} soft_end={soft_end}", flush=True)
'''
n['cells'][hook[0]]['source'] = HOOK.splitlines(True)
note = ("# B14 (LHS-D25): private dev probe of Lord Han Solo's public ARC-AGI-3 notebook\n\n"
        "Fork of lordhansolo/built-on-tufa-labs-duck-harness-milestone-2 (v306) for a matched serving/levels comparison "
        "against our B12-D25 baseline. Only the inline hook cell changes. All credit for the solver and runtime to the original author.\n\n")
n['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': note.splitlines(True)})
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open('/data/arc/kaggle/b14_lhs_src/lhs_v306_metadata.json'))
m.pop('id_no', None); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn, is_private=True)
assert m['enable_internet'] is False and m['machine_shape'] == 'NvidiaRtxPro6000'
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
