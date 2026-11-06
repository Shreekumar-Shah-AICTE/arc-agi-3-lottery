"""V11 readout by the sealed rule (analysis/s37/sealed/v11_l1guard_prediction.txt; post-seal note v11_post_seal_note.txt).
Written 5 Nov 2026 ~14:55 UTC, while V11 was running and before any V11 output existed.
Run: python3 analysis/s37/v11_readout.py [KERNEL_LOG] > analysis/s37/out/v11_readout.out
Data: data/runs/v11_l1guard.benchmark.json.gz (benchmark.json from kernel shreekumarshah/arc3-v11-l1guard-b).
P-V11-1: N0 = games that started (non-empty history) and ended with levels_completed == 0. PASS if N0 <= 1, else FAIL."""
import json, re, statistics, sys
from pathlib import Path
from common import HERE
from v5_budget_prediction import load

RUN = 'v11_l1guard'
REF = {'b12_fb25': (1, 106), 'b28_ctl_fb25': (4, 95), 'b32_ctl_r2': (2, 103), 'b32_ctl_r3': (3, 107),
       'b32_spec2_r2': (2, 97), 'b32_spec2_r3': (3, 107), 't1c_fb25_ctl': (2, 112)}   # (N0, total levels), sealed reference
SIX = ['sk48', 'bp35', 'dc22', 'g50t', 'sc25', 'cd82']
BUDGET = 121

def tokens(g):
    return int(re.search(r'tokens=(\d+)', g.get('solver_note') or 'tokens=0').group(1))

G = load(RUN)
started = [g for g in G if g.get('history')]
never = [g['game_id'][:4] for g in G if not g.get('history')]
zero = [g for g in started if g['levels_completed'] == 0]
n0 = len(zero)
total = sum(g['levels_completed'] or 0 for g in G)
print(f'V11 readout ({RUN}): {len(G)} games in benchmark.json, {len(started)} started, never started: {never or "none"}')

# validity lines from the kernel log
log = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if log and log.exists():
    raw = log.read_text()
    try:
        text = ''.join(x.get('data', '') for x in json.loads(raw))
    except ValueError:
        text = raw
    i_d, i_g = text.find('#OURS_FORM ok version=d_prime'), text.find('#V11_L1GUARD ok')
    hits = re.findall(r'#V11_L1GUARD hits=(\d+)', text)
    fresh = re.findall(r'#OURS_FRESH replaced=(\d+)', text)
    print(f'log: "#OURS_FORM ok version=d_prime" {"found" if i_d >= 0 else "MISSING"}; "#V11_L1GUARD ok" '
          f'{"found" if i_g >= 0 else "MISSING"}{" (after it)" if 0 <= i_d < i_g else ""}; guard hits {hits[-1] if hits else "n/a"}; '
          f'fresh games queued by D\' {fresh[-1] if fresh else "n/a"}')
    valid = 0 <= i_d < i_g and len(G) == 25
else:
    print('log: not given, validity lines not checked')
    valid = len(G) == 25
print(f'validity: {"VALID" if valid else "INVALID"}{" (PARTIAL: some games never started)" if never else ""}')

print(f'\nP-V11-1 (sealed): N0 = {n0} started games ended with 0 levels ({", ".join(g["game_id"][:4] for g in zero) or "none"})')
print(f'  rule: PASS if N0 <= 1, FAIL if N0 >= 2  ->  {"PASS" if n0 <= 1 else "FAIL"}'
      + (f'  (scored among the {len(started)} started games)' if never else ''))
if never:
    print(f'  N0 counting the {len(never)} never-started games as zeros: {n0 + len(never)}')
print(f'  reference N0 per run: {", ".join(str(v[0]) for v in REF.values())} (mean {statistics.mean(v[0] for v in REF.values()):.2f})')

print('\nsecondary (reported, not scored)')
print(f'  total levels: {total} of {sum(g["number_of_levels"] for g in G)} (reference {", ".join(str(v[1]) for v in REF.values())}; '
      f'mean {statistics.mean(v[1] for v in REF.values()):.1f}, SD {statistics.stdev(v[1] for v in REF.values()):.1f}; '
      f'z {(total - statistics.mean(v[1] for v in REF.values())) / statistics.stdev(v[1] for v in REF.values()):+.2f})')
st = sorted(((g['history'][0]['wallclock_seconds'] / 60, g['game_id'][:4]) for g in started), reverse=True)
print(f'  last game to start: {st[0][1]} at minute {st[0][0]:.1f} (reference 69 to 108); next: '
      + ', '.join(f'{n} {m:.1f}' for m, n in st[1:4]))
print(f'  games that never started: {len(never)} (reference 0)')
by = {g['game_id'][:4]: g for g in G}
for n in SIX:
    g = by.get(n)
    if g is None:
        print(f'  {n}: not in run'); continue
    h = g.get('history') or []
    print(f'  {n}: {g["levels_completed"]} of {g["number_of_levels"]} levels, tokens {tokens(g) / 1e3:.0f}k, '
          f'start {h[0]["wallclock_seconds"] / 60 if h else float("nan"):.1f}, last step {h[-1]["wallclock_seconds"] / 60 if h else float("nan"):.1f} min')
print('  zero-level games: tokens on level 1, last logged step, shelved (last step more than 30 min before minute 121)')
for g in zero:
    h = g['history']
    last = h[-1]['wallclock_seconds'] / 60
    print(f'    {g["game_id"][:4]}: tokens {tokens(g) / 1e3:.0f}k, start {h[0]["wallclock_seconds"] / 60:.1f}, last {last:.1f} min, '
          f'actions {len(h)}, shelved {"yes" if last < BUDGET - 30 else "no"}')
if not zero:
    print('    none')
print(f'  games completed (all levels): {sum(1 for g in G if g["levels_completed"] == g["number_of_levels"])}; '
      f'total tokens {sum(tokens(g) for g in G) / 1e6:.3f}M (FB25 control mean 4.5585M)')
