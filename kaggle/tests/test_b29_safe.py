"""Offline branch-selection tests. Not a Kaggle rerun integration test."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.argv = ['make_b29.py', 'none']
fix = runpy.run_path(str(ROOT / 'kaggle/make_b29.py'))['SAFE_FIX']
cases = [
    ('no signals', {}, False, False),
    ('base URL only', {'ARC_BASE_URL': 'http://gateway:8001/'}, False, True),
    ('rerun flag only', {'KAGGLE_IS_COMPETITION_RERUN': 'true'}, False, True),
    ('numeric flag', {'KAGGLE_IS_COMPETITION_RERUN': '1'}, False, True),
    ('DNS only', {}, True, True),
    ('explicit false plus URL', {'KAGGLE_IS_COMPETITION_RERUN': 'false', 'ARC_BASE_URL': 'http://gateway:8001/'}, False, True),
]
for name, env, dns, expected in cases:
    scope = {'os': os}
    with patch.dict(os.environ, env, clear=True), patch('socket.getaddrinfo', return_value=[('mock',)] if dns else []), contextlib.redirect_stdout(io.StringIO()):
        exec(fix, scope)
    assert scope['TRUE_SUBMISSION'] is expected, name
    print('PASS:', name, '->', expected)
for base, arm in [('b28_ctl_fb25', 'b29_occ_fb25'), ('b12_safe', 'b29_safe')]:
    def read(name):
        p = next((ROOT / 'kaggle' / name).glob('*.ipynb'))
        return json.loads(p.read_text())
    a, b = read(base), read(arm)
    assert len(a['cells']) == len(b['cells'])
    diffs = [i for i, (x, y) in enumerate(zip(a['cells'], b['cells'])) if x != y]
    assert diffs == [0, 5], (arm, diffs)
    if arm == 'b29_safe':
        assert fix in ''.join(b['cells'][5]['source'])
    print('PASS:', arm, 'changed cells', diffs)
print('LIMIT: mocked DNS; timeout, real Save & Run, and scored rerun are unverified.')
