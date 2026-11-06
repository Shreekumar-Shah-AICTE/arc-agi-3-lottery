"""B28 arms (D77/D78): b28_ctl_fb25 = b12_fb25 + 40-min startup; b28_tc_fb25 = ctl + ThinkingCap target via a metadata-only view.
Inputs for the TC patch (/data/tc/*) were fetched per-file from the Kaggle models API (S30)."""
import json, copy, re, sys
from pathlib import Path
K = Path(__file__).resolve().parent
src = json.load(open(K/'b12_fb25/b12_fb25.ipynb')); meta = json.load(open(K/'b12_fb25/kernel-metadata.json'))
def cell(nb, i): return ''.join(nb['cells'][i]['source'])
def setcell(nb, i, s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
def sub1(s, a, b):
    assert s.count(a) == 1, (a, s.count(a)); return s.replace(a, b)
INTEL_MD = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
TC_MD = "lamdang/thinkingcap-qwen3-8-flash-next-w4a16-autoround/Transformers/default/1"
def build(name, title_md, tc):
    nb = copy.deepcopy(src); m = copy.deepcopy(meta)
    c5 = sub1(cell(nb, 5), "SERVER_STARTUP_TIMEOUT = 12 * 60", "SERVER_STARTUP_TIMEOUT = 40 * 60  # B28: platform mounts load in 12-30 min since ~18 Oct")
    if tc:
        te = json.load(open('/data/tc/lamdang/config.json'))['quantization_config']['extra_config']
        assert te.get('.*mlp\\.gate.*') == {'bits': 16, 'data_type': 'float'}, 'TC lacks the generic router-gate bf16 rule'
        tokcfg = open('/data/tc/dfranzen/tokenizer_config.json').read()
        view = f'''
# B28-TC (D78): ThinkingCap-Qwen3.8-Flash-Next W4A16 AutoRound (lamdang) through a metadata-only view.
# Symlinked from TC: all weights + index (tensor names identical to the Intel mirror, 224,280 keys),
# tokenizer.json (sha 06b95093 = FR-Spec key), chat_template.jinja, processor configs (sha-identical to Intel).
# config.json/quantization_config.json are TC's own: they drop Intel's 48 per-layer router-gate rules but keep the
# generic '.*mlp\\.gate.*' bf16 rule (same effect). Only tokenizer_config.json is swapped to Intel's (TC's is the older format).
B28_INTEL_TOKENIZER_CONFIG = {json.dumps(tokcfg)}
def _b28_tc_view():
    import glob
    hits = [Path(h).parent for h in sorted(glob.glob('/kaggle/input/models/lamdang/thinkingcap-qwen3-8-flash-next-w4a16-autoround/**/model.safetensors.index.json', recursive=True))]
    assert len(hits) == 1, f'TC model not found or ambiguous: {{hits}}'
    tc = hits[0]; view = Path('/tmp/b28-tc-view'); view.mkdir(parents=True, exist_ok=True)
    patched = ('tokenizer_config.json',)
    for p in tc.iterdir():
        d = view / p.name
        if p.name not in patched and not (d.exists() or d.is_symlink()): d.symlink_to(p.resolve())
    (view / 'tokenizer_config.json').write_text(B28_INTEL_TOKENIZER_CONFIG)
    print('B28-TC view', view, '<-', tc, '| entries', len(list(view.iterdir())))
    return str(view)
MODEL_DIR = _b28_tc_view()
'''
        c5 = sub1(c5, "SERVED_MODEL_NAME = 'flashnext'\n", "SERVED_MODEL_NAME = 'flashnext'\n" + view)
        m['model_sources'] = [TC_MD if s == INTEL_MD else s for s in m['model_sources']]
        assert TC_MD in m['model_sources'] and INTEL_MD not in m['model_sources']
    setcell(nb, 5, c5)
    setcell(nb, 0, title_md + "\n\n" + cell(src, 0))
    m.update(id=f'shreekumarshah/arc3-{name.replace("_", "-")}', title=f'arc3-{name.replace("_", "-")}', code_file=f'{name}.ipynb')
    out = K/name; out.mkdir(exist_ok=True)
    json.dump(nb, open(out/f'{name}.ipynb', 'w'), indent=1); json.dump(m, open(out/'kernel-metadata.json', 'w'), indent=1)
    print('built', name, m['id'], m['model_sources'])
build('b28_ctl_fb25', '# B28-CTL-FB25 (D78): fresh control = B12-FB25 + 40-min startup timeout', tc=False)
build('b28_tc_fb25', '# B28-TC-FB25 (D78): control + ThinkingCap W4A16 AutoRound target (lamdang), metadata-only view', tc=True)
