"""V10 follow-up (exploratory; written 5 Nov 2026 ~13:30 UTC, after the sealed readout of T1, its rerun and T1c).
Not part of the sealed test: by the sealed validity rule both T1 runs are invalid, so P-V10-1 is UNTESTED.
For each run: generated tokens (solver_note tokens=), levels, games completed, games never started (tokens = 0), and
when games started (wallclock_seconds of each game's first logged step, seconds since the run began). For the T1 runs
it also reads the sealed P-V10-1 table (sealed/v10_supply_prediction.txt, read-only) and interpolates the band at the
measured supply ratio, exactly as v10_readout.py would for a valid run.
Run: python3 analysis/s37/v10_followup.py > analysis/s37/out/v10_followup.out"""
import gzip, json, re, statistics
import numpy as np
from common import HERE, CTL, SPEC2_VALID
SEALED = HERE / 'sealed' / 'v10_supply_prediction.txt'
TAB = [tuple(float(x) for x in m.groups()) for m in
       re.finditer(r'^\s+(\d\.\d\d)\s+([\d.]+)\s+([\d.]+) to\s+([\d.]+)$', SEALED.read_text(), re.M)]
S_CTL = float(re.search(r'/ ([\d.]+)M, the mean of the four FB25 control runs', SEALED.read_text()).group(1)) * 1e6
BUDGET_S = 121 * 60
def band(r):
    rs = [t[0] for t in TAB]
    return tuple(float(np.interp(r, rs, [t[i] for t in TAB])) for i in (1, 2, 3))
def stats(name):
    V = json.load(gzip.open(HERE / 'data' / 'runs' / f'{name}.benchmark.json.gz', 'rt'))['game_runs']
    tok = {g['game_id'][:4]: int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1)) for g in V}
    start = {g['game_id'][:4]: g['history'][0]['wallclock_seconds'] for g in V if g.get('history')}
    lev = {g['game_id'][:4]: g['levels_completed'] for g in V}
    won = sum(g['levels_completed'] >= g['number_of_levels'] for g in V)
    return dict(tokens=sum(tok.values()), levels=sum(lev.values()), won=won, never=[k for k, t in tok.items() if t == 0],
                start=start, lev=lev)
print(f'sealed table rows {len(TAB)}; control mean tokens {S_CTL / 1e6:.4f}M; per-game budget {BUDGET_S} s')
for name in ('t1_fb25_nospec', 't1r_fb25_nospec', 't1c_fb25_ctl') + tuple(CTL):
    s = stats(name); r = s['tokens'] / S_CTL
    late = sorted(s['start'].items(), key=lambda kv: kv[1])
    n_late = sum(1 for _, t in late if t > BUDGET_S / 2)
    print(f'{name}: tokens {s["tokens"] / 1e6:.3f}M (r {r:.3f}), levels {s["levels"]}, completed {s["won"]}, '
          f'never started {len(s["never"])} ({", ".join(s["never"]) or "-"}); started after half the budget {n_late}; '
          f'last start {late[-1][0]} at {late[-1][1] / 60:.0f} min')
    if name.startswith('t1_') or name.startswith('t1r_'):
        pt, lo, hi = band(r)
        print(f'  exploratory: sealed band at r {r:.3f}: {pt:.1f} ({lo:.1f} to {hi:.1f}); observed {s["levels"]} -> '
              f'{"inside" if lo <= s["levels"] <= hi else "outside"} (not a test: the run is invalid)')
last = [max(stats(n)['start'].values()) / 60 for n in CTL + ['t1c_fb25_ctl']]
print(f'last game start in the {len(last)} control runs (4 controls and T1c): {min(last):.0f} to {max(last):.0f} min '
      f'of the {BUDGET_S // 60}-minute budget, which counts from the start of the run for every game')
ctl = [stats(n)['levels'] for n in CTL]; t1c = stats('t1c_fb25_ctl')['levels']
same = [stats(n)['levels'] for n in CTL + SPEC2_VALID]
print(f'controls {sorted(ctl)} mean {statistics.mean(ctl):.2f} SD {statistics.stdev(ctl):.2f}; with T1c ({t1c}): '
      f'mean {statistics.mean(ctl + [t1c]):.2f} SD {statistics.stdev(ctl + [t1c]):.2f}')
print(f'same-agent FB25 runs {sorted(same)} mean {statistics.mean(same):.2f} SD {statistics.stdev(same):.2f}; with T1c: '
      f'n {len(same) + 1} mean {statistics.mean(same + [t1c]):.2f} SD {statistics.stdev(same + [t1c]):.2f}')
