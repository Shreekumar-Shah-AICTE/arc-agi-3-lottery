"""Appendix A (4 Nov 2026): the 32 negative rows of the experiment ledger (from out/ledger_negatives.csv).
Writes paper/tables/tA_negative_results.md."""
import csv
from common import HERE, ROOT
from negatives_kind import kind
from collections import Counter
rows = list(csv.DictReader(open(HERE / 'out' / 'ledger_negatives.csv', encoding='utf-8')))
out = ['| # | Ledger id | Date | Track | Kind | Hypothesis (short) | Verdict |', '|---|---|---|---|---|---|---|']
for i, r in enumerate(sorted(rows, key=lambda r: r['date']), 1):
    h = r['hypothesis'].replace('|', '/')[:100]; v = r['verdict'].replace('|', '/')[:120]
    out.append(f"| {i} | {r['id']} | {r['date']} | {r['track']} | {kind(r['id'])} | {h} | {v} |")
out.append('\nRow 30 also played only 24 of 25 games (lf52 never admitted, D84). Like rows 31 and 32 it is left out of '
           'run-level comparisons; its played attempts stay in the level-attempt fit (Appendix B, "partial").')
C = Counter(kind(r['id']) for r in rows)
out.append(f"\nKind (added 7 Nov 2026): GPU change = a distinct change to the ARC-AGI-3 agent or its serving, run on the GPU ({C['GPU change']} rows); "
           f"ARC-AGI-2 ({C['ARC-AGI-2']}); CPU only = a simulation or log analysis without a new GPU run ({C['CPU only']}); rerun = identical code, "
           f"a noise sample ({C['rerun']}); readout = a second row for a change listed above ({C['readout']}).")
(HERE / 'out' / 'ledger_kinds.out').write_text(' '.join(f"{k.replace(' ', '_')}={n}" for k, n in sorted(C.items())) + f" total={len(rows)}\n")
(ROOT / 'paper' / 'tables' / 'tA_negative_results.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
print(len(rows), 'rows')
