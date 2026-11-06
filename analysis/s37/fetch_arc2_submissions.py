"""Our ARC-AGI-2 submissions with public and private scores (read-only Kaggle API call, run 4 Nov 2026).
Writes data/lb/arc-agi-2_our_submissions_<UTC>.csv (ref, date, description, status, publicScore, privateScore).
The registry (paper/build_results.py) reads the three fork draws from this file.
Needs KAGGLE_USERNAME and KAGGLE_API_TOKEN in the environment. Read-only: no submissions."""
import csv, datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
COMP = 'arc-prize-2026-arc-agi-2'
OUT = Path(__file__).resolve().parent / 'data' / 'lb'
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M')
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions(COMP, page_size=100) or []
with open(OUT / f'arc-agi-2_our_submissions_{STAMP}.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['ref', 'date', 'description', 'status', 'publicScore', 'privateScore'])
    for s in subs:
        w.writerow([s.ref, s.date, (s.description or '').replace('\n', ' ')[:120], s.status, s.public_score, s.private_score])
print('our submissions:', len(subs), '| with private score:', sum(1 for s in subs if s.private_score not in (None, '')))
print('stamp', STAMP)
