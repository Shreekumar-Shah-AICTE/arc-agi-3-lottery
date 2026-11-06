# S19 (P7): ARC-AGI-2 commit-mode fork of the public NVARC "minimal perfpatch" line
# (defiaudit/arc-agi2-lb33-89-minimal-perfpatch, same as mikelou1's original; sorokin qwen3_4b_grids15_sft139, 4xL4,
# task-time LoRA training + 16-aug inference, 12 h - 10 min budget). Only change: in a save (non-rerun) run, queue just
# the first 2 evaluation tasks instead of all 120, so the save run takes minutes, not ~12 h. The rerun path (hidden
# test set, all tasks, same budget) is byte-identical to the source.
# Usage: python3 make_agi2_ppc.py   Build only. Pushing to Kaggle needs an operator go (BOOT.md).
import json
d = '/data/arc/kaggle/agi2_ppc'
nb = json.load(open(f'{d}/src_defiaudit_v.ipynb'))
c6 = ''.join(nb['cells'][6]['source'])
old = "        if not rerun_mode:\n            pass  # v13.0 INFO RUN: process ALL 120 evaluation tasks interactively\n        queue.put(key)\n"
assert c6.count(old) == 1
new = ("        if not rerun_mode and key not in sorted(data.keys())[:2]:\n"
       "            continue  # S19 commit mode: save run solves 2 evaluation tasks; rerun solves every hidden task\n"
       "        queue.put(key)\n")
nb['cells'][6]['source'] = c6.replace(old, new).splitlines(keepends=True)
md = ''.join(nb['cells'][0]['source'])
nb['cells'][0]['source'] = ("# arc2-perfpatch-commit (S19)\nFork of the public minimal-perfpatch NVARC line. Save run: 2 evaluation tasks "
                            "(commit mode). Hidden rerun: unchanged (all test tasks, 12 h - 10 min).\n\n" + md).splitlines(keepends=True)
json.dump(nb, open(f'{d}/arc2_perfpatch_commit.ipynb', 'w'), indent=1)
meta = json.load(open('/data/agi2/kernel-metadata.json'))
for k in ('id_no',): meta.pop(k, None)
meta.update(id='shreekumarshah/arc2-perfpatch-commit', title='arc2-perfpatch-commit', code_file='arc2_perfpatch_commit.ipynb', is_private=True)
json.dump(meta, open(f'{d}/kernel-metadata.json', 'w'), indent=1)
print('built')
