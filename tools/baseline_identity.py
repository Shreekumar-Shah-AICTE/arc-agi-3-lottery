"""Sanity baseline: predict test input unchanged (attempt_1) and transpose (attempt_2)."""
import json, glob, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else '/data/arc/ARC-AGI-2/data/evaluation'
sub = {}
for f in glob.glob(os.path.join(d, '*.json')):
    t = json.load(open(f)); tid = os.path.basename(f)[:-5]
    sub[tid] = [{'attempt_1': x['input'], 'attempt_2': [list(r) for r in zip(*x['input'])]} for x in t['test']]
json.dump(sub, open('/tmp/sub_identity.json', 'w')); print('wrote /tmp/sub_identity.json', len(sub))
