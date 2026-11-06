"""Appendix B: every development run kept in analysis/s37/data/runs (4 Nov 2026).
Writes paper/tables/tB_dev_runs.md. Score = mean of per-game final_score (local scoring on the run's games, not the
leaderboard). Minutes = end_time - start_time of the run file."""
import gzip, json
from datetime import datetime
from pathlib import Path
from common import HERE, ROOT, CTL, SPEC2, INVALID, B12_FAMILY, dev_run_files

# Which analysis uses which run (the map asked for in review). Codes are explained in the table's legend.
FB25 = list(B12_FAMILY) + ['b28_tc_fb25']                 # the 17 FB25 runs with token logs (lever4b_two_numbers.py)
SAME = CTL + [r for r in SPEC2 if r not in INVALID]       # 6 valid same-agent runs
def uses(run):
    u = []
    if run in CTL: u.append('C')
    if run in SAME: u.append('S')
    if run in B12_FAMILY and run not in INVALID: u.append('F')
    if run in FB25: u.append('T')
    if run in FB25 and run not in INVALID: u.append('B')
    return ' '.join(u)
def group(run):
    if run in CTL: return 'control'
    if run in SPEC2: return 'serving-only change (spec2)'
    if run == 'b28_tc_fb25': return 'B12 variant (ThinkingCap, KILL)'
    if run in B12_FAMILY: return 'B12 variant'
    if run == 'f1_forklab': return 'fork replay: 32 saved states x 4 arms x 2'
    return ''
LATER = [('v5_fb50', 'control at 2x play time (V5)')]    # runs after the 4 Nov analysis freeze; T1 rows join when scored
for extra, lab in (('t1_fb25_nospec', 'T1: speculative decoding off (V10)'), ('t1r_fb25_nospec', 'T1 rerun, same notebook (V10)'),
                   ('t1c_fb25_ctl', 'control, same day as T1 (V10)'), ('v11_l1guard', 'level-1 games keep their slot (V11)')):
    if (HERE / 'data' / 'runs' / f'{extra}.benchmark.json.gz').exists(): LATER.append((extra, lab))

def row(p):
    d = json.load(gzip.open(p, 'rt'))
    g = d['game_runs']
    t0, t1 = (datetime.fromisoformat(d[k]) for k in ('start_time', 'end_time'))
    played = sum(1 for x in g if x.get('actions_per_level') and sum(x['actions_per_level']) > 0 or (x.get('final_generated_tokens') or 0) > 0)
    return dict(run=p.name.split('.')[0], date=t0.strftime('%Y-%m-%d'), games=len(g), played=played,
                unstarted=[x['game_id'].split('-')[0] for x in g if not (x.get('actions_per_level') and sum(x['actions_per_level']) > 0)
                           and not (x.get('final_generated_tokens') or 0) > 0],
                levels=sum(x['levels_completed'] or 0 for x in g), of=sum(x['number_of_levels'] for x in g),
                score=sum(x['final_score'] for x in g) / len(g), mins=(t1 - t0).total_seconds() / 60)
rows = sorted((row(p) for p in dev_run_files()), key=lambda r: (r['date'], r['run']))
later = [dict(row(HERE / 'data' / 'runs' / f'{r}.benchmark.json.gz'), label=lab) for r, lab in LATER]
scale = 100 if max(r['score'] for r in rows) <= 1.0 else 1
out = ['| # | Run file | Date | Games | Levels cleared | Score | Run minutes | Group | Used in |',
       '|---|---|---|---|---|---|---|---|---|']
def line(i, r, grp, use):
    games = f"{r['played']} of {r['games']}" if r['played'] < r['games'] else f"{r['games']}"
    if r['unstarted']: grp = (grp + '; ' if grp else '') + 'partial: ' + ', '.join(r['unstarted']) + ' never started'
    return (f"| {i} | {r['run']} | {r['date']} | {games} | {r['levels']} of {r['of']} | {r['score'] * scale:.1f} | "
            f"{r['mins']:.0f} | {grp} | {use} |")
for i, r in enumerate(rows, 1):
    out.append(line(i, r, group(r['run']), uses(r['run'])))
n_games = sum(r['games'] for r in rows)
out.append(f'| | **Later runs** (after the 4 Nov analysis freeze; not in the counts above) | | | | | | | |')
for i, r in enumerate(later, len(rows) + 1):
    out.append(line(i, r, r['label'], ''))
out.append(f"\n{len(rows)} runs, {n_games:,} game runs. Score is local (the run's own games), not a leaderboard score. "
           f"Games shows games played of games scheduled when they differ. Used in: C = crack-model fit and the V5 baseline "
           f"({len(CTL)} controls); S = same-agent noise: Fig. 3, the level SD and the A/A spread of b ({len(SAME)} runs); "
           f"F = family variance and efficiency cost, Sections 3.3 and 4.3 ({sum(1 for r in B12_FAMILY if r not in INVALID)} runs); "
           f"T = the level-attempt fit of b, Section 3.5 (every played attempt of {len(FB25)} runs); "
           f"B = run-level comparisons of b, Section 4.5 ({sum(1 for r in FB25 if r not in INVALID)} runs).")
(ROOT / 'paper' / 'tables' / 'tB_dev_runs.md').write_text('\n'.join(out) + '\n')
print(len(rows), 'runs', n_games, 'game runs; later', len(later), '; score scale x', scale)
print('\n'.join(out[-8:]))
