# S17 arms on the FB25 instrument (25 public games x 121 min, D47), both = kaggle/b12_fb25 + one mechanism.
#   python3 make_s17.py x7b /data/arc/kaggle/b12_x7b arc3-b12-x7b-fb25
#       X7b (D26 follow-up): Franzen's dormant rolling-summary machinery, aligned to the drain (interval 54K < drain 58K)
#       + chat-template prefix fix. Stock Qwen3.8 template puts "Reasoning effort is set to xhigh..." at the top of the
#       system turn only when enable_thinking is not false, so every enable_thinking=false summary request diverged at
#       char 19 and re-prefilled the whole context (X7: cache hit 94% -> 69%). Forcing that conditional true makes a
#       summary request share the prefix up to its own user message; normal turns render byte-identically (jinja2 check).
#   python3 make_s17.py len /data/arc/kaggle/b12_len arc3-b12-len-fb25
#       LEN: dead-trace fix. ARC3_STUB_DEAD_REASONING=1 (dormant Franzen flag) + kaggle/s17_len.patch (one notice after a
#       reasoning-only length-capped reply). B12-FB25: 35/38 length caps made no tool call = 1.23M reasoning chars (10.3%).
import json, os, sys, copy
arm, d, kid = sys.argv[1:4]
base = 'b12_fb25'
n = json.load(open(f'/data/arc/kaggle/{base}/{base}.ipynb'))
src = lambda k: ''.join(n['cells'][k]['source'])
def put(k, s): n['cells'][k]['source'] = s.splitlines(True)
def rep(s, a, b, cnt=1):
    assert s.count(a) == cnt, (a, s.count(a)); return s.replace(a, b)
ENV_ANCHOR = "    ########## controls ##########\n    'EXPOSE_UNDO': 'on',\n}\n"
assert src(5).count(ENV_ANCHOR) == 1 and src(13).count('tmpl = Path(MODEL_DIR, "chat_template.jinja")\nif tmpl.is_file(): args += ["--chat-template", str(tmpl)]\n') == 1

if arm == 'x7b':
    put(5, rep(src(5), ENV_ANCHOR, ENV_ANCHOR.replace("    'EXPOSE_UNDO': 'on',\n}\n",
        "    'EXPOSE_UNDO': 'on',\n\n"
        "    ########## S17 X7b: drain-aligned rolling summaries (Franzen machinery, dormant in M2/B12) ##########\n"
        "    # interval just under the 58K drain, so one summary lands inside each drained span and the drain stops on it:\n"
        "    # drains stay ~as large and as rare as B12's (X7 at 16K made them 3.6x smaller and more frequent).\n"
        "    'ARC3_SUMMARY_INTERVAL_TOKENS': 54*1024,\n"
        "    'ARC3_DRAIN_STOP_AT_SUMMARIES': '1',\n"
        "    'ARC3_HIDE_FOLLOWUP_SUMMARIES': '1',   # no prompt change before the first eviction; then only the oldest kept summary is sent\n"
        "    'ARC3_SUMMARY_TURN_CONTEXT': '1',\n"
        "    'ARC3_SUMMARY_ENABLE_THINKING': '0',   # cell 13 forces the template's reasoning-effort sentence on, so this costs no prefill\n"
        "    'ARC3_SUMMARY_MAX_GEN': 6144,          # X7 summaries ran 4.4-7.1K chars (~2K tokens)\n"
        "}\n")))
    put(13, rep(src(13), 'tmpl = Path(MODEL_DIR, "chat_template.jinja")\nif tmpl.is_file(): args += ["--chat-template", str(tmpl)]\n',
        'tmpl = Path(MODEL_DIR, "chat_template.jinja")\n'
        '# S17 X7b: the stock template emits "Reasoning effort is set to xhigh..." at the top of the system turn only when\n'
        '# enable_thinking is undefined/true, so a summary request (enable_thinking=false) diverged at char 19 and re-prefilled\n'
        '# the whole context. Forcing that one conditional true keeps the prefix shared; normal turns render byte-identically.\n'
        '_X7B_ANCHOR = "{%- if enable_thinking is undefined or enable_thinking is true %}"\n'
        'if tmpl.is_file():\n'
        '    _x7b_src = tmpl.read_text()\n'
        '    if _x7b_src.count(_X7B_ANCHOR) == 1:\n'
        '        _x7b_tmpl = Path("/kaggle/working/chat_template_x7b.jinja")\n'
        '        _x7b_tmpl.write_text(_x7b_src.replace(_X7B_ANCHOR, "{%- if true %}"))\n'
        '        tmpl = _x7b_tmpl\n'
        '        print("#X7B template fix applied ->", tmpl, flush=True)\n'
        '    else:\n'
        '        # fallback: thinking-on summaries keep the stock prefix identical (costs reasoning tokens, not prefill)\n'
        '        os.environ["ARC3_SUMMARY_ENABLE_THINKING"] = "1"\n'
        '        print("#X7B template anchor count", _x7b_src.count(_X7B_ANCHOR), "-> no fix; ARC3_SUMMARY_ENABLE_THINKING=1", flush=True)\n'
        '    args += ["--chat-template", str(tmpl)]\n'
        'else:\n'
        '    os.environ["ARC3_SUMMARY_ENABLE_THINKING"] = "1"\n'
        '    print("#X7B no chat_template.jinja -> ARC3_SUMMARY_ENABLE_THINKING=1", flush=True)\n'))
    title = ("# B12-X7b-FB25: drain-aligned rolling summaries + template prefix fix\n\nPrivate experiment (S17). = arc3-b12-fb25 "
             "(B12, 25 public games x 121 min) + Franzen's dormant summary machinery (interval 54K < drain 58K, stop drains at "
             "summaries, hide follow-up summaries, no thinking) + a one-line chat-template fix so summary requests reuse the "
             "prefix cache. Cells 0, 5, 13 differ from arc3-b12-fb25.\n\n")
elif arm == 'len':
    patch = open('/data/arc/kaggle/s17_len.patch').read()
    cell = copy.deepcopy(n['cells'][3]); cell['source'] = ('%%writefile /kaggle/len.patch\n' + patch).splitlines(True)
    n['cells'].insert(4, cell)   # cells 4.. shift by one: old 5 -> 6
    s = src(6)
    s = rep(s, 'print("bline patch applied")\n',
            'print("bline patch applied")\n'
            'subprocess.run(["git", "apply", "--include=ARC3-Inference/*", "-v", "/kaggle/len.patch"], cwd=f"{BUNDLE_DIR}/src", check=True)\n'
            'print("S17 LEN patch applied")\n')
    s = rep(s, ENV_ANCHOR, ENV_ANCHOR.replace("    'EXPOSE_UNDO': 'on',\n}\n",
        "    'EXPOSE_UNDO': 'on',\n\n"
        "    ########## S17 LEN: dead-trace fix ##########\n"
        "    'ARC3_STUB_DEAD_REASONING': '1',  # dormant Franzen flag: a reasoning-only reply is not replayed into context\n"
        "    'ARC3_LEN_CAP_NOTICE': '1',       # s17_len.patch: one notice after a reasoning-only length-capped reply\n"
        "}\n"))
    put(6, s)
    title = ("# B12-LEN-FB25: dead-trace fix\n\nPrivate experiment (S17). = arc3-b12-fb25 (B12, 25 public games x 121 min) + "
             "ARC3_STUB_DEAD_REASONING=1 + a 38-line harness patch (cell 4, /kaggle/len.patch) that tells the model once when "
             "its previous reply hit the output limit while still reasoning. B12-FB25 lost 10.3% of its reasoning chars to such replies. "
             "Cells 0, 4 (new), 6 (old 5) differ from arc3-b12-fb25.\n\n")
else:
    raise SystemExit('arm?')
n['cells'][0]['source'] = title.splitlines(True) + n['cells'][0]['source'][2:]
for c in n['cells']:
    if c['cell_type'] == 'code': c['outputs'] = []; c['execution_count'] = None
os.makedirs(d, exist_ok=True); fn = os.path.basename(d) + '.ipynb'
json.dump(n, open(f'{d}/{fn}', 'w'), indent=1)
m = json.load(open(f'/data/arc/kaggle/{base}/kernel-metadata.json')); m.update(id='shreekumarshah/' + kid, title=kid, code_file=fn)
assert m['is_private'] is True and m['machine_shape'] == 'NvidiaRtxPro6000' and m['enable_internet'] is False
assert m['docker_image'].endswith('32a8be4cb13c') and 'kaggle-private-byod' in m['docker_image']
json.dump(m, open(f'{d}/kernel-metadata.json', 'w'), indent=1); print('built', d, kid, len(n['cells']), 'cells')
