"""Fetch the final ARC-AGI-3 boards after the 2026-11-02 23:59 UTC close (run 4 Nov 2026).
Writes, under data/lb/:
  arc-agi-3_public_final_<UTC>.csv.gz   Kaggle leaderboard download (Rank, TeamId, ..., Score, SubmissionCount)
  arc-agi-3_api_board_<mode>_<UTC>.csv.gz  GetLeaderboard pages (TeamId, TeamName, SubmissionDate, Score),
                                          mode = default and override_public
  arc-agi-3_our_submissions_<UTC>.csv   our submissions with public and private scores
Which API board is private is decided by comparing it with the public download (printed).
Needs KAGGLE_USERNAME and KAGGLE_API_TOKEN in the environment. Read-only: no submissions."""
import csv, gzip, io, sys, zipfile, datetime, tempfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest
COMP = 'arc-prize-2026-arc-agi-3'
OUT = Path(__file__).resolve().parent / 'data' / 'lb'
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M')
api = KaggleApi(); api.authenticate()

def save_gz(path, header, rows):
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

# 1. public download (zip with one or more CSVs)
with tempfile.TemporaryDirectory() as td:
    api.competition_leaderboard_download(COMP, td, quiet=True)
    z = zipfile.ZipFile(Path(td) / f'{COMP}.zip')
    print('download zip members:', z.namelist())
    for name in z.namelist():
        data = z.read(name).decode('utf-8-sig')
        kind = 'private' if 'private' in name.lower() else 'public'
        dest = OUT / f'arc-agi-3_{kind}_final_{STAMP}.csv.gz'
        with gzip.open(dest, 'wt', encoding='utf-8', newline='') as fh: fh.write(data)
        print('saved', dest.name, 'rows', data.count('\n') - 1)
pub = {}
with gzip.open(OUT / f'arc-agi-3_public_final_{STAMP}.csv.gz', 'rt', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        if r.get('Score'): pub[r['TeamId']] = float(r['Score'])

# 2. GetLeaderboard, default and override_public
for mode in ('default', 'override_public'):
    rows, token = [], None
    try:
        with api.build_kaggle_client() as kg:
            while True:
                rq = ApiGetLeaderboardRequest(); rq.competition_name = COMP; rq.page_size = 1000
                if mode == 'override_public': rq.override_public = True
                if token: rq.page_token = token
                rs = kg.competitions.competition_api_client.get_leaderboard(rq)
                rows += [(s.team_id, s.team_name, s.submission_date, s.score) for s in rs.submissions]
                token = rs.next_page_token
                if not token or not rs.submissions: break
    except Exception as e:
        print(mode, 'failed:', type(e).__name__, str(e)[:200]); continue
    save_gz(OUT / f'arc-agi-3_api_board_{mode}_{STAMP}.csv.gz', ['TeamId', 'TeamName', 'SubmissionDate', 'Score'], rows)
    same = sum(1 for t, _, _, s in rows if str(t) in pub and s not in (None, '') and abs(float(s) - pub[str(t)]) < 1e-9)
    print(f'{mode}: {len(rows)} teams; equal to public download for {same}/{sum(1 for t,*_ in rows if str(t) in pub)} matched teams')

# 3. our submissions
subs, page = [], None
with open(OUT / f'arc-agi-3_our_submissions_{STAMP}.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['ref', 'date', 'description', 'status', 'publicScore', 'privateScore'])
    for s in api.competition_submissions(COMP, page_size=100) or []:
        w.writerow([s.ref, s.date, (s.description or '').replace('\n', ' ')[:120], s.status, s.public_score, s.private_score])
        subs.append(s)
print('our submissions:', len(subs), '| with private score:', sum(1 for s in subs if s.private_score not in (None, '')))
print('stamp', STAMP)
