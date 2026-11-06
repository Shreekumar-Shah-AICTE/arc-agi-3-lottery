"""Appendix A (4 Nov 2026): the 32 negative rows of the experiment ledger (from out/ledger_negatives.csv).
Writes paper/tables/tA_negative_results.md."""
import csv
from common import HERE, ROOT
rows = list(csv.DictReader(open(HERE / 'out' / 'ledger_negatives.csv', encoding='utf-8')))
out = ['| # | Ledger id | Date | Track | Hypothesis (short) | Verdict |', '|---|---|---|---|---|---|']
for i, r in enumerate(sorted(rows, key=lambda r: r['date']), 1):
    h = r['hypothesis'].replace('|', '/')[:100]; v = r['verdict'].replace('|', '/')[:120]
    out.append(f"| {i} | {r['id']} | {r['date']} | {r['track']} | {h} | {v} |")
out.append('\nRow 30 also played only 24 of 25 games (lf52 never admitted, D84). Like rows 31 and 32 it is left out of '
           'run-level comparisons; its played attempts stay in the level-attempt fit (Appendix B, "partial").')
(ROOT / 'paper' / 'tables' / 'tA_negative_results.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print(len(rows), 'rows')
