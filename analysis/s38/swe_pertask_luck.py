"""Is run-to-run noise on SWE-Bench-Verified per-task luck, as on ARC-AGI-3 (Section 4.3)? Exploratory, 7 Nov 2026, written
after the paper's tests. Data: resolved ids per run from Bjarnason et al. (2026), fetched by fetch_swe_runs.py (CC BY 4.0).
For each configuration (model x scaffold x temperature) with m runs over N = 500 tasks: observed variance of the run
resolution rate, and the variance that independent per-task luck predicts, sum_i p_i(1 - p_i) m/(m - 1) / N^2 with
p_i = c_i/m. The share predicted/observed is the share of run-level variance that per-task luck accounts for, as in 4.3. Null band:
parametric bootstrap with the same p_i (seed 0, 4000 draws). Output: out/swe_pertask_luck.json and .out."""
import json, numpy as np
from pathlib import Path
H = Path(__file__).parent
runs = json.load(open(H / 'data/swe_runs/resolved_ids.json'))['runs']
cfg = {}
for k, v in runs.items(): cfg.setdefault(k.split('/')[0], []).append(v)
rng = np.random.default_rng(0); out = {'configs': {}}; num = den = 0.0; nulls = []
for c in sorted(cfg):
    rs = cfg[c]; m = len(rs); N = rs[0]['total']; assert all(r['total'] == N for r in rs)
    ids = sorted({i for r in rs for i in r['resolved_ids']})
    X = np.array([[i in set(r['resolved_ids']) for i in ids] for r in rs], float)   # m x (tasks ever solved)
    tot = X.sum(1); p = X.mean(0)
    obs = tot.var(ddof=1); pred = (p * (1 - p)).sum() * m / (m - 1)
    sim = rng.random((4000, m, len(p))) < p
    st = sim.sum(2); sp = sim.mean(1)
    pv = (sp * (1 - sp)).sum(1) * m / (m - 1); ov = np.maximum(st.var(1, ddof=1), 1e-9); rat_null = pv / ov
    nulls.append((pv, ov))
    out['configs'][c] = dict(runs=m, tasks=N, mean_pct=round(100 * tot.mean() / N, 2), sd_obs_pp=round(100 * np.sqrt(obs) / N, 2),
        sd_pred_pp=round(100 * np.sqrt(pred) / N, 2), share=round(pred / obs, 2),
        never=int(N - len(ids)), always=int((p == 1).sum()), sometimes=int(((p > 0) & (p < 1)).sum()),
        null90=[round(float(np.quantile(rat_null, q)), 2) for q in (0.05, 0.95)])
    num += pred; den += obs
pooled_null = sum(pv for pv, _ in nulls) / sum(ov for _, ov in nulls)
C = out['configs'].values()
out['pooled'] = dict(configs=len(cfg), runs=sum(d['runs'] for d in C), share=round(num / den, 2),
    null90=[round(float(np.quantile(pooled_null, q)), 2) for q in (0.05, 0.95)],
    share_min=min(d['share'] for d in C), share_max=max(d['share'] for d in C),
    sd_obs_min=min(d['sd_obs_pp'] for d in C), sd_obs_max=max(d['sd_obs_pp'] for d in C),
    sometimes_min=min(d['sometimes'] for d in C), sometimes_max=max(d['sometimes'] for d in C),
    never_min=min(d['never'] for d in C), never_max=max(d['never'] for d in C))
(H / 'out').mkdir(exist_ok=True)
(H / 'out/swe_pertask_luck.json').write_text(json.dumps(out, indent=1))
lines = [f"{c}: {d['runs']} runs, mean {d['mean_pct']}%, SD {d['sd_obs_pp']} pp observed vs {d['sd_pred_pp']} predicted, per-task share {d['share']} (null 90% {d['null90']}), never {d['never']}, sometimes {d['sometimes']}, always {d['always']}" for c, d in out['configs'].items()]
lines.append(f"pooled over {out['pooled']['configs']} configurations ({out['pooled']['runs']} runs): per-task share {out['pooled']['share']} (null 90% {out['pooled']['null90']}); per configuration {out['pooled']['share_min']} to {out['pooled']['share_max']}")
(H / 'out/swe_pertask_luck.out').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
