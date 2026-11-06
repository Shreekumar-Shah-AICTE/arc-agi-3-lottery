# Summarize a P1 output dir: python3 analyze_p1.py <dir>
import json, sys, statistics as st
d = sys.argv[1]; r = json.load(open(f'{d}/p1_result.json'))
print('build_s', r.get('build_s'), 'best_mode', r.get('best_mode'))
for m, v in (r.get('modes') or {}).items():
    print(m, {k: v.get(k) for k in ('np', 'ctx', 'load_s', 'boot')})
    for row in v.get('sweep', []): print('   ', row)
a = r.get('accuracy') or {}
print('accuracy', {k: v for k, v in a.items() if k not in ('fin_tokens', 'open_tokens', 'natural_stop')})
fin = a.get('fin_tokens') or []
if fin: print('finished tokens: n', len(fin), 'median', st.median(fin), 'max', max(fin))
print('open tokens', a.get('open_tokens'))
recs = r.get('records') or []
done = [x for x in recs if x.get('t_end')]
nat = [x for x in done if x.get('finish') == 'stop']
print('natural stops', len(nat), 'correct among them', sum(1 for x in nat if x.get('correct')), 'truncated', sum(1 for x in done if x.get('finish') == 'length'))
for x in done: print('  ', x['task'], x['test'], x.get('finish'), x.get('n'), x.get('nr'), 'C' if x.get('correct') else ('A' if x.get('answered') else '-'), round((x['t_end'] - x['t_start']) / 60, 1), 'min')
