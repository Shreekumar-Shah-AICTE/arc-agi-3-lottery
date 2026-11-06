"""Ledger audit for the paper (4 Nov 2026): which experiment rows end negative, and which positive.
Input: state/EXPERIMENTS.csv (157 rows). Negative = verdict starts with KILL, KILLED, NOISE, FAIL, PARK or INVALID.
Positive-looking = verdict starts with PASS or contains WEAK/FINDING. Writes out/ledger_negatives.csv; prints counts."""
import csv, collections, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
all_rows = list(csv.DictReader(open(ROOT / 'state' / 'EXPERIMENTS.csv', encoding='utf-8')))
# 5 Nov: the paper's own model tests (V5 and the V10 runs, track 'ARC-AGI-3 dev (paper)') are reported in Section 4.4
# and listed apart here, so an invalid test run is not counted as a negative result of the agent work.
PAPER_TESTS = 'ARC-AGI-3 dev (paper)'
rows = [r for r in all_rows if r['track'] != PAPER_TESTS]
def first(v):
    m = re.match(r'[A-Za-z_\-]+', (v or '').strip()); return m.group(0).upper() if m else ''
NEG = ('KILL', 'KILLED', 'NOISE', 'FAIL', 'PARK', 'INVALID')
neg = [r for r in rows if first(r['verdict']) in NEG]
pos = [r for r in rows if first(r['verdict']).startswith('PASS') or 'WEAK' in first(r['verdict']) or first(r['verdict']) == 'FINDING']
print(f'{len(rows)} ledger rows; negative {len(neg)}:', dict(collections.Counter(first(r["verdict"]) for r in neg)))
print(f'ledger total {len(all_rows)} rows; the paper\'s own test runs ({PAPER_TESTS}, Section 4.4), not counted above: '
      + ', '.join(f"{r['id']} {first(r['verdict'])}" for r in all_rows if r['track'] == PAPER_TESTS))
print('negative by track:', dict(collections.Counter(r['track'] for r in neg)))
with open(ROOT / 'analysis' / 's37' / 'out' / 'ledger_negatives.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id', 'date', 'track', 'verdict_word', 'hypothesis', 'verdict'])
    for r in neg: w.writerow([r['id'], r['date'], r['track'], first(r['verdict']), (r['hypothesis'] or '')[:160], (r['verdict'] or '')[:200]])
print(f'positive-looking rows: {len(pos)}')
for r in pos: print('  ', r['id'], r['date'], r['track'], '|', (r['hypothesis'] or '')[:90], '|', (r['verdict'] or '')[:110])
