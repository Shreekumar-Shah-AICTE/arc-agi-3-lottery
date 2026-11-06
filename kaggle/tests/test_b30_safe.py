"""Offline tests for B30 (D82). Not a Kaggle rerun integration test."""
import contextlib, io, json, os, re, socket, subprocess, sys, time, types
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]; K = ROOT / 'kaggle'
def nb(name): return json.loads(next((K / name).glob('*.ipynb')).read_text())
def cell(n, i): return ''.join(n['cells'][i]['source'])
arms = [a for a in ('b30_safe', 'b30_safe_occ') if (K / a).exists()]
base = nb('b29_safe')
for arm in arms:
    b = nb(arm); c5 = cell(b, 5); c22 = cell(b, 22)
    # 1) branch selection (D81 cases) with the B30 cell-5 block
    blk = c5[c5.index('TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN"'):c5.index('NOTEBOOK_START_EPOCH = time.time()')]
    cases = [('no signals', {}, 'none', False), ('base URL only', {'ARC_BASE_URL': 'http://gateway:8001/'}, 'none', True),
             ('rerun flag only', {'KAGGLE_IS_COMPETITION_RERUN': 'true'}, 'none', True), ('numeric flag', {'KAGGLE_IS_COMPETITION_RERUN': '1'}, 'none', True),
             ('DNS only', {}, 'ok', True), ('explicit false plus URL', {'KAGGLE_IS_COMPETITION_RERUN': 'false', 'ARC_BASE_URL': 'http://gateway:8001/'}, 'none', True),
             ('DNS raises gaierror', {}, 'raise', False)]
    for name, env, dns, expected in cases:
        scope = {'os': os}
        kw = {'side_effect': socket.gaierror(-2, 'Name or service not known')} if dns == 'raise' else {'return_value': [('mock',)] if dns == 'ok' else []}
        err = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch('socket.getaddrinfo', **kw), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            exec(blk, scope); time.sleep(0.05)
        assert scope['TRUE_SUBMISSION'] is expected, (arm, name)
        assert 'Traceback' not in err.getvalue(), (arm, name, err.getvalue()[:200])
        print('PASS:', arm, name, '->', expected)
    # 2) real resolver, unresolvable host, separate process: False and no thread traceback on stderr
    fn = c5[c5.index('def _b29_gateway_resolves'):c5.index('try:\n    _B29_SIG')]
    r = subprocess.run([sys.executable, '-c', fn + "\nprint(_b29_gateway_resolves('no-such-host-b30.invalid'))"], capture_output=True, text=True, timeout=30)
    assert r.stdout.strip() == 'False' and 'Traceback' not in r.stderr, (r.stdout, r.stderr[:300])
    print('PASS:', arm, 'real resolver on unresolvable host -> False, clean stderr')
    # 3) end-time cap
    cap = c22[c22.index('B30_END_KERNEL_MIN = 543'):c22.index('# Play the benchmark;')]
    for sub, start_min, old, want in [(True, 11, 532 * 60, 532 * 60), (True, 40, 532 * 60, 503 * 60), (True, 60, 532 * 60, 483 * 60),
                                      (True, 530, 532 * 60, 30 * 60), (False, 40, 5 * 60, 5 * 60), (False, 11, 5 * 60, 5 * 60)]:
        now = time.time()
        sc = {'time': time, 'TRUE_SUBMISSION': sub, 'NOTEBOOK_START_TIME': now - start_min * 60,
              'bm': types.SimpleNamespace(solver=types.SimpleNamespace(max_runtime_s_per_game=old))}
        with contextlib.redirect_stdout(io.StringIO()): exec(cap, sc)
        got = sc['bm'].solver.max_runtime_s_per_game
        assert abs(got - want) <= 2, (arm, sub, start_min, got, want)
        print('PASS:', arm, f'end cap sub={sub} start={start_min}min -> {got/60:.1f} min')
    # 4) cell diff vs b29_safe, and intended edits only
    diffs = [i for i, (x, y) in enumerate(zip(base['cells'], b['cells'])) if x != y]
    assert len(base['cells']) == len(b['cells']) and diffs == [0, 5, 22], (arm, diffs)
    assert ('ARC3_MAX_ACTIVE_STREAMS\': 12' in c5) == (arm == 'b30_safe_occ'), arm
    assert 'SERVER_STARTUP_TIMEOUT = 50 * 60' in c5
    print('PASS:', arm, 'changed cells vs b29_safe', diffs)
print('LIMIT: mocked/locally-resolved DNS and mocked clock; real Save & Run and scored rerun are unverified.')
