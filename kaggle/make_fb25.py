# FB25 = full-budget dev probe (S16, D47). Same 25 public games as the D25 probe, but each game may run 121 min
# instead of 40, so each game's share of the server matches the real run: 10 streams x 532 min / 110 games =
# 48.4 stream-min per game (real) vs 10 x 121 / 25 = 48.4 (FB25). D25 gave a third of that (16 stream-min), where
# the tokens->score curve is steep (elasticity > 1.5), so a throughput gain looked like a quality gain (B15).
# usage:
#   python3 make_fb25.py b12  /data/arc/kaggle/b12_fb25    arc3-b12-fb25      (B12 unchanged, full budget)
#   python3 make_fb25.py b15c /data/arc/kaggle/b15c_fb25   arc3-b15c-fb25     (B15 serving at 12 streams, production
#                                                                              server flags, /metrics sampler)
#   python3 make_fb25.py b15p /data/arc/kaggle/b15p_fb25   arc3-b15p-fb25     (alternative: exact LB replay of B15 at
#                                                                              14 streams, production flags, sampler)
#   python3 make_fb25.py b15c-submit /data/arc/kaggle/b15c_submit arc3-b15c-submit (submission twin of b15c; only
#                                                                              built, pushed only after a b15c GO)
import json, os, sys
arm, d, kid = sys.argv[1:4]
base = {'b12': 'b12d25', 'b15c': 'b15_vllm', 'b15p': 'b15_vllm', 'b15c-submit': 'b15_submit'}[arm]
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)

if arm != 'b15c-submit':
    s = src(17)
    s = rep(s, 'bm.solver.max_runtime_s_per_game = 40*60  # B12-D25 dev probe: 25 games compete for 10 slots',
               'bm.solver.max_runtime_s_per_game = 121*60  # FB25 (D47): 48.4 stream-min per game, = the real run (10 x 532 / 110)')
    assert 'bm.solver.concurrency = 120' in s and 'bm.solver.max_runtime_s_per_game = 532*60' in s
    put(17, s)

if arm.startswith('b15'):
    s = src(5)
    if arm != 'b15p': s = rep(s, "'ARC3_MAX_ACTIVE_STREAMS': 14,  # B15: = vLLM --max-num-seqs 14 (LHS pool)",
               "'ARC3_MAX_ACTIVE_STREAMS': 12,  # B15c (D47): 12 x ~116K peak contexts fit the 1.42M-token pool; 14 ran it at 97.8-99.5%")
    put(5, s)
    s = src(13)
    if arm != 'b15p': s = rep(s, "for must in ('VLLM_MAX_NUM_SEQS = 14\\n', 'VLLM_MAX_MODEL_LEN = 147072\\n'):\n    assert must in body, must\n"
               "assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == 14\n",
               "for must in ('VLLM_MAX_NUM_SEQS = 14\\n', 'VLLM_MAX_MODEL_LEN = 147072\\n'):\n    assert must in body, must\n"
               "body = _patch(body, 'VLLM_MAX_NUM_SEQS = 14\\n', 'VLLM_MAX_NUM_SEQS = 12  # B15c (D47)\\n')\n"
               "assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == 12\n")
    if arm in ('b15c', 'b15p'):
        anchor = ("body = _patch(body, 'VLLM_MAX_NUM_SEQS = 14\\n', 'VLLM_MAX_NUM_SEQS = 12  # B15c (D47)\\n')\n" if arm == 'b15c'
                  else "assert int(os.environ['ARC3_MAX_ACTIVE_STREAMS']) == 14\n")
        # Production server: the LB run starts vLLM with TAAF_RUN_AS_SUBMISSION=1 (no cache-diagnostics plugin, no
        # --kv-cache-metrics sampling, no smoke test, no host-memory monitor). Every dev run so far used test mode.
        # Keep the two observational flags the analysis needs: periodic stats (and /metrics) and cached-token details.
        s = rep(s, anchor, anchor +
                   "body = _patch(body, \"        cmd += ['--disable-log-stats', '--disable-uvicorn-access-log']\\n\",\n"
                   "              \"        cmd += ['--disable-uvicorn-access-log', '--enable-prompt-tokens-details']  # FB25: keep stats\\n\")\n")
        s = rep(s, "                TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_SETUP_ENV='/tmp/b15_lhs_setup_env.json')\n",
                   "                TAAF_KAGGLE_WORKING_DIR=str(WORKING_DIR), TAAF_KAGGLE_SETUP_ENV='/tmp/b15_lhs_setup_env.json',\n"
                   "                TAAF_RUN_AS_SUBMISSION='1')  # B15c (D47): the LB run's server path (boot process only)\n")
        s += '''

# FB25 (D47): sample vLLM's Prometheus counters once a minute for the whole run (observational; a failed poll is
# logged and skipped). Answers the B15 post-mortem: preemptions, KV usage, prefix-cache hits, queue depth over time.
import threading
FB25_METRICS = WORKING_DIR / 'vllm_metrics.tsv'
_FB25_KEYS = ('vllm:num_preemptions', 'vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc', 'vllm:num_requests_running',
              'vllm:num_requests_waiting', 'vllm:prefix_cache_queries', 'vllm:prefix_cache_hits', 'vllm:prompt_tokens_total',
              'vllm:generation_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total', 'vllm:spec_decode_num_draft_tokens_total')

def _fb25_sampler():
    url = f'http://127.0.0.1:{SERVED_MODEL_PORT}/metrics'
    while True:
        ts = int(time.time() - NOTEBOOK_START_TIME)
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                text = r.read().decode('utf-8', 'replace')
            rows = [f'{ts}\\t{line}' for line in text.splitlines() if line.startswith(_FB25_KEYS)]
        except Exception as exc:
            rows = [f'{ts}\\t#poll failed {exc!r}']
        with open(FB25_METRICS, 'a', encoding='utf-8') as f:
            f.write('\\n'.join(rows) + '\\n')
        time.sleep(60)

threading.Thread(target=_fb25_sampler, name='fb25-metrics', daemon=True).start()
print(f'#FB25 metrics sampler -> {FB25_METRICS}', flush=True)
'''
    put(13, s)

title = {
    'b12': ("# B12-FB25: B12 baseline, full-budget dev probe\n\nPrivate experiment (D47). Code and serving = our B12 notebook "
            "(X6 v2 + D' slot priority, SGLang, 10 streams), byte-identical to B12-D25 except the save run's per-game budget: "
            "25 public games, 121 min each, so each game gets the real run's share of the server (48.4 stream-min).\n\n"),
    'b15c': ("# B15c-FB25: B12 harness on the LHS vLLM runtime at 12 streams, full-budget dev probe\n\nPrivate experiment (D47). "
             "= B15 (arc3-b15-vllm-d25) with three changes: 12 streams instead of 14, the LB run's production server flags "
             "(plus stats and cached-token details) and a once-a-minute /metrics sampler; save run = the FB25 probe "
             "(25 public games, 121 min each). Paired with arc3-b12-fb25.\n\n"),
    'b15p': ("# B15p-FB25: exact replay of the B15 LB configuration at full budget\n\nPrivate experiment (D47, alternative to "
             "B15c-FB25). = B15 (arc3-b15-vllm-d25), 14 streams, with the LB run's production server flags (plus stats and "
             "cached-token details) and a once-a-minute /metrics sampler; save run = the FB25 probe (25 public games, 121 min each). "
             "Answers whether B15 collapses at the LB regime. Paired with arc3-b12-fb25.\n\n"),
    'b15c-submit': ("# B15c-submit: B12 submission notebook on the LHS vLLM runtime at 12 streams\n\nSubmission candidate, built only "
                    "(push and submit only after a B15c-FB25 GO and the operator's go, D47). = kaggle/b15_submit with 12 streams.\n\n"),
}[arm]
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
