"""Local ARC-AGI-2 scorer. Kaggle-style pass@2: a test output counts if either attempt matches exactly.
Task score = mean over its test outputs; overall = mean over tasks.
Usage: python tools/score_arc2.py submission.json [data_dir]"""
import json, sys, glob, os
def load_solutions(d):
    sol = {}
    for f in glob.glob(os.path.join(d, '*.json')):
        t = json.load(open(f)); sol[os.path.basename(f)[:-5]] = [x['output'] for x in t['test']]
    return sol
def score(sub, sol):
    per = {}
    for tid, outs in sol.items():
        preds = sub.get(tid, [])
        hits = 0
        for i, gt in enumerate(outs):
            p = preds[i] if i < len(preds) else {}
            if any(p.get(k) == gt for k in ('attempt_1', 'attempt_2')): hits += 1
        per[tid] = hits / len(outs)
    return sum(per.values()) / len(per), per
if __name__ == '__main__':
    d = sys.argv[2] if len(sys.argv) > 2 else '/data/arc/ARC-AGI-2/data/evaluation'
    s, per = score(json.load(open(sys.argv[1])), load_solutions(d))
    print(f'score={s*100:.2f}% tasks={len(per)} solved={sum(1 for v in per.values() if v==1)}')
