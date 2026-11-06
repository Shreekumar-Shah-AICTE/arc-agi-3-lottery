"""B31 offline checks: only cells 0 and 13 differ from b28_ctl_fb25; SPEC_STEPS edited once; metadata ids/slugs."""
import json
from pathlib import Path
K = Path(__file__).resolve().parents[1]
ctl = json.load(open(K/'b28_ctl_fb25/b28_ctl_fb25.ipynb')); cm = json.load(open(K/'b28_ctl_fb25/kernel-metadata.json'))
src = lambda nb, i: ''.join(nb['cells'][i]['source'])
ok = 0
for name, steps in (('b31_spec2_fb25', 2), ('b31_spec1_fb25', 1)):
    nb = json.load(open(K/name/f'{name}.ipynb')); m = json.load(open(K/name/'kernel-metadata.json'))
    assert len(nb['cells']) == len(ctl['cells'])
    diff = [i for i in range(len(ctl['cells'])) if src(nb, i) != src(ctl, i)]
    assert diff == [0, 13], (name, diff); ok += 1
    c13 = src(nb, 13); assert c13.count(f'SPEC_STEPS={steps},') == 1 and 'SPEC_STEPS=3,' not in c13; ok += 1
    assert src(ctl, 13).replace('SPEC_STEPS=3,', f'SPEC_STEPS={steps},  # B31 (D85): 3 -> {steps}; draft tokens {steps + 1}') == c13; ok += 1
    assert '"--speculative-num-draft-tokens", str(CFG["SPEC_STEPS"]+1)' in c13; ok += 1
    slug = name.replace('_', '-')
    assert m['id'] == f'shreekumarshah/arc3-{slug}' and m['code_file'] == f'{name}.ipynb' and m['is_private'] is True; ok += 1
    assert {k: v for k, v in m.items() if k not in ('id', 'title', 'code_file')} == {k: v for k, v in cm.items() if k not in ('id', 'title', 'code_file')}; ok += 1
print(f'PASS {ok}/12 B31 offline checks (cell diff 0/13 only, SPEC_STEPS edit exact, metadata equal to ctl except id/title/code_file)')
