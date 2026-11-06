"""Table 2 (5 Nov 2026): all our ARC-AGI-2 submissions with public score and place on the final public board
(other teams scoring >= the draw, our own row removed), as Table 1 does for ARC-AGI-3. Private scores are withheld by
the hosts until verification. Writes paper/tables/t2_arc2_submissions.md. "What it was" uses plain descriptions
(from state/EXPERIMENTS.csv and D52-D53); the descriptions as submitted to Kaggle stay in the source CSV."""
import csv
from common import HERE, ROOT, load_lb
PLAIN = {   # ref -> plain description (internal session and decision codes removed)
    '56712373': "A0: fork of a public notebook (mikelou1's perfpatch, public score 33.89)",
    '56745390': 'A1: A0 + dynamic token cap and a fixed hash seed',
    '56769864': 'A2: A1 + a second-pass seed ensemble on idle GPU time',
    '56800905': 'E2.1: our model-first pipeline (27B model, 48k token cap, 2 attempts)',
    '56833698': 'A0 resubmitted (noise sample 2)',
    '56858952': 'A0 resubmitted (noise sample 3)',
    '57016836': 'PP: the public minimal-perfpatch notebook, unchanged',
    '57082458': 'PP resubmitted (noise sample 2)',
    '57117601': 'PP resubmitted (noise sample 3)',
    '57153034': 'PP resubmitted (noise sample 4)',
}
rows = list(csv.DictReader(open(HERE / 'data' / 'lb' / 'arc-agi-2_our_submissions_20261104T1743.csv', encoding='utf-8')))
sc = load_lb('arc-agi-2'); best = max(float(r['publicScore']) for r in rows)
others = list(sc); others.remove(best)
out = ['| # | Date (UTC) | Ref | What it was | Public score | Place on final board |', '|---|---|---|---|---|---|']
for i, r in enumerate(sorted(rows, key=lambda r: r['date']), 1):
    x = float(r['publicScore']); place = 1 + sum(1 for s in others if s >= x - 1e-9)
    out.append(f"| {i} | {r['date'][:16]} | {r['ref']} | {PLAIN.get(r['ref'], r['description'][:90]).replace('|', '/')} | {x:.2f} | {place:,} |")
out.append(f'\nFinal public board: {len(sc):,} teams (fetched 4 Nov 2026 11:16 UTC). Private scores: not yet published. '
           'Ref is the Kaggle submission ID. The seven A0 and PP draws are the "one public notebook lineage" of Section 4.8.')
(ROOT / 'paper' / 'tables' / 't2_arc2_submissions.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print('\n'.join(out)); print('rows', len(rows))
