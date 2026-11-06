"""Table 1 (4 Nov 2026): all our ARC-AGI-3 submissions with public score and place on the final public board
(other teams scoring >= the draw, our own row removed). Private scores are withheld by the hosts until verification.
Writes paper/tables/t1_arc3_submissions.md. "What it was" uses the plain descriptions in PLAIN (added 4 Nov ~13:45 UTC);
the descriptions as submitted to Kaggle stay in the source CSV."""
import csv
from common import HERE, ROOT, load_lb
PLAIN = {   # ref -> plain description (internal session and decision codes removed)
    '56712371': 'B0: fork of a public notebook (scottlegrand, 5.19); pipeline check',
    '56745504': "B1: fork of dfranzen's Milestone 2 notebook",
    '56769866': 'B1 resubmitted (noise sample 2)',
    '56790419': 'X6: B1 + ledger of solved levels',
    '56832595': "B12: X6 ledger v2 + D' slot priority",
    '56858951': 'B12 resubmitted (noise sample 2)',
    '56890682': 'B15: B12 on a vLLM runtime',
    '56919974': 'B12 resubmitted (noise sample 3)',
    '56936216': 'LEN: B12 + dead-trace stub and cap notice',
    '57003717': 'A1: fork of a different public notebook (arena p50)',
    '57052442': 'B12 resubmitted (noise sample 4)',
    '57082387': 'HT2: B12 + stall detector + fresh-eyes analyst calls (second build)',
    '57117599': 'HT: B12 + stall detector + fresh-eyes analyst calls (first build)',
    '57153031': 'B12 resubmitted (noise sample 5)',
    '57760408': 'B28: B12 + 40-minute start-up timeout',
    '57815956': 'B30: B12 + 50-minute start-up timeout + end cap',
    '57904835': 'B30 resubmitted',
}
rows = list(csv.DictReader(open(HERE / 'data' / 'lb' / 'arc-agi-3_our_submissions_20261104T0951.csv', encoding='utf-8')))
sc = load_lb('arc-agi-3'); best = max(float(r['publicScore']) for r in rows)
others = list(sc); others.remove(best)
out = ['| # | Date (UTC) | Ref | What it was | Public score | Place on final board |', '|---|---|---|---|---|---|']
for i, r in enumerate(sorted(rows, key=lambda r: r['date']), 1):
    x = float(r['publicScore']); place = 1 + sum(1 for s in others if s >= x - 1e-9)
    out.append(f"| {i} | {r['date'][:16]} | {r['ref']} | {PLAIN.get(r['ref'], r['description'][:90]).replace('|', '/')} | {x:.2f} | {place:,} |")
out.append(f'\nFinal public board: {len(sc):,} teams (fetched 4 Nov 2026 09:51 UTC). Private scores: not yet published. '
           'Ref is the Kaggle submission ID.')
(ROOT / 'paper' / 'tables' / 't1_arc3_submissions.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(out[:3])); print('rows', len(rows))
