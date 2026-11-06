"""B31 speculative-depth arms (D85), built from b28_ctl_fb25 (the current control, D79).
Rationale (D84, VERIFIED serve logs): decode is saturated near 8-10 running requests, where step time grows
with verify tokens (ctl: ~12.5 ms + ~0.62 ms per verify token). Fewer draft tokens per request may raise
aggregate tok/s at batch 9-10. Spec verification is exact (accept thresholds 1.0), so only speed can change.
b31_spec2_fb25 = ctl + SPEC_STEPS 3 -> 2 (num_draft_tokens 4 -> 3)
b31_spec1_fb25 = ctl + SPEC_STEPS 3 -> 1 (num_draft_tokens 4 -> 2)"""
import json, copy, sys
from pathlib import Path
K = Path(__file__).resolve().parent
src = json.load(open(K/'b28_ctl_fb25/b28_ctl_fb25.ipynb')); meta = json.load(open(K/'b28_ctl_fb25/kernel-metadata.json'))
def cell(nb, i): return ''.join(nb['cells'][i]['source'])
def setcell(nb, i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
def sub1(s, a, b):
    assert s.count(a) == 1, (a[:60], s.count(a)); return s.replace(a, b)
def build(name, title_md, steps):
    nb = copy.deepcopy(src); m = copy.deepcopy(meta)
    setcell(nb, 13, sub1(cell(nb, 13), "SPEC_STEPS=3,", f"SPEC_STEPS={steps},  # B31 (D85): 3 -> {steps}; draft tokens {steps + 1}"))
    setcell(nb, 0, title_md + "\n\n" + cell(src, 0))
    m.update(id=f'shreekumarshah/arc3-{name.replace("_", "-")}', title=f'arc3-{name.replace("_", "-")}', code_file=f'{name}.ipynb')
    out = K/name; out.mkdir(exist_ok=True)
    json.dump(nb, open(out/f'{name}.ipynb', 'w'), indent=1); json.dump(m, open(out/'kernel-metadata.json', 'w'), indent=1)
    print('built', name, m['id']); return nb
if __name__ == '__main__':
    args = sys.argv[1:] or ['spec2', 'spec1']
    if 'spec2' in args: build('b31_spec2_fb25', '# B31-SPEC2 (D85): B28 ctl + speculative depth 3 -> 2 (draft tokens 3)', 2)
    if 'spec1' in args: build('b31_spec1_fb25', '# B31-SPEC1 (D85): B28 ctl + speculative depth 3 -> 1 (draft tokens 2)', 1)
