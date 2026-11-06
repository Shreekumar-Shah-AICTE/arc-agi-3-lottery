"""Offline checks for b30_safe_specN (built only after a D85 GO): vs b30_safe only cells 0 and 13 differ,
the SPEC_STEPS edit is exact, metadata equals b30_safe except id/title/code_file. Usage: test_b30_spec.py 2|1"""
import json, sys
from pathlib import Path
K = Path(__file__).resolve().parents[1]
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
name = f'b30_safe_spec{n}'
base = json.load(open(K/'b30_safe/b30_safe.ipynb')); bm = json.load(open(K/'b30_safe/kernel-metadata.json'))
nb = json.load(open(K/name/f'{name}.ipynb')); m = json.load(open(K/name/'kernel-metadata.json'))
src = lambda x, i: ''.join(x['cells'][i]['source'])
diff = [i for i in range(len(base['cells'])) if src(nb, i) != src(base, i)]
assert len(nb['cells']) == len(base['cells']) and diff == [0, 13], diff
assert src(base, 13).replace('SPEC_STEPS=3,', f'SPEC_STEPS={n},  # B31 (D85): 3 -> {n}; draft tokens {n + 1}') == src(nb, 13)
assert m['id'] == f'shreekumarshah/arc3-b30-safe-spec{n}' and m['is_private'] is True
assert {k: v for k, v in m.items() if k not in ('id', 'title', 'code_file')} == {k: v for k, v in bm.items() if k not in ('id', 'title', 'code_file')}
print(f'PASS 4/4 {name} offline checks (cells 0/13 only vs b30_safe; exact SPEC_STEPS edit; metadata)')
