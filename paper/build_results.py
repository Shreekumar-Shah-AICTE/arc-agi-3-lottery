"""Number registry for the paper (started 4 Nov 2026). Every number in paper/draft/*.md should be here, extracted
from its source file by a regex or computed from data, so a changed source breaks the build instead of the text.
Run: python3 paper/build_results.py  -> paper/results.json ; then python3 paper/check_numbers.py (unmatched numbers)."""
import json, re, sys, statistics
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / 'analysis' / 's37'))
from common import B12_LB, load_lb
O = 'analysis/s37/out/'
REPORT = 'paper/sources/arcagi3_report_excerpts.txt'   # verbatim excerpts of the ARC-AGI-3 Technical Report (URL inside)
NB = 'kaggle/b12_fb25/b12_fb25.ipynb'
E = {}
def add(key, path, pattern, group=1):
    p = Path(path) if path.startswith('/') else R / path
    m = re.search(pattern, p.read_text(encoding='utf-8', errors='replace'), re.S)
    if not m: raise SystemExit(f'MISSING {key}: {path} /{pattern}/')
    E[key] = {'value': m.group(group).rstrip('.'), 'source': path}
def put(key, value, source): E[key] = {'value': str(value), 'source': source}
# crack model fit (4 FB25 controls)
for k, g in (('crack_q', 1), ('crack_median_k', 2), ('crack_s', 3)):
    add(k, O + 'lever3_stuck.out', r'fit q ([\d.]+) median ([\d.]+)k s ([\d.]+)', g)
for k, g in (('fit_attempts', 1), ('fit_clears', 2), ('fit_censored', 3)):
    add(k, 'analysis/s37/v5_budget_prediction.py', r'the\s+(\d+) level attempts of the 4 FB25 control runs \((\d+) clears, (\d+) censored', g)
add('halffit_pred_mean', O + 'v5_budget_prediction.out', r'mean ([\d.]+) vs actual mean ([\d.]+)', 1)
add('halffit_actual_mean', O + 'v5_budget_prediction.out', r'mean ([\d.]+) vs actual mean ([\d.]+)', 2)
# level effects and noise models
add('icc', O + 'lever3c_level_difficulty.out', r'\(ICC\): ([\d.]+)')
add('within_sd', O + 'lever3c_level_difficulty.out', r'within-cell \(run-to-run\) SD ([\d.]+)')
add('between_sd', O + 'lever3c_level_difficulty.out', r'between-cell SD ([\d.]+)')
for h in ('H0', 'H1', 'H2'):
    add(f'{h}_sd', O + 'lever3d_noise_model.out', h + r':[^\n]*?run-to-run SD within a world: mean\s+([\d.]+)')
add('obs_mean_fb25', 'analysis/s37/lever3d_noise_model.py', r'six same-agent FB25 runs\): mean ([\d.]+), SD ([\d.]+)', 1)
add('obs_sd_fb25', 'analysis/s37/lever3d_noise_model.py', r'six same-agent FB25 runs\): mean ([\d.]+), SD ([\d.]+)', 2)
# two numbers and run counts
add('aft_s', O + 'lever4b_two_numbers.out', r'residual s ([\d.]+)')
add('b_aa_sd', O + 'lever4b_two_numbers.out', r'A/A spread of b[^\n]*SD ([\d.]+)')
add('runs_raw_levels', O + 'lever4b_two_numbers.out', r'runs per arm, raw level totals: ([\d.]+)')
add('runs_b', O + 'lever4b_two_numbers.out', r'crack-cost b on shared levels: ([\d.]+)')
add('corr_two_numbers', O + 'lever4b_two_numbers.out', r'corr\(predicted, observed level difference vs ctl\) = ([\d.]+)')
add('runs_ratio', O + 'lever4b_two_numbers.out', r'ratio \(raw levels / b\): ([\d.]+)x')
add('runs_ratio_lo', O + 'lever4b_two_numbers.out', r'leave-one-out ratio range: ([\d.]+)x to ([\d.]+)x', 1)
add('runs_ratio_hi', O + 'lever4b_two_numbers.out', r'leave-one-out ratio range: ([\d.]+)x to ([\d.]+)x', 2)
add('same_valid_runs', O + 'lever4b_two_numbers.out', r'A/A spread of b over the (\d+) valid')
add('runs_raw_levels7', O + 'lever4b_two_numbers.out', r'review-copy cohort[^\n]*-> ([\d.]+) and ([\d.]+) runs per arm', 1)
add('runs_b7', O + 'lever4b_two_numbers.out', r'review-copy cohort[^\n]*-> ([\d.]+) and ([\d.]+) runs per arm', 2)
add('lv_sd_same', O + 'lever4b_two_numbers.out', r'levels SD over the same \d+ runs: ([\d.]+)')
add('var_groups', O + 'lever4b_two_numbers.out', r'; (\d+) variants, Bonferroni')
add('bonf_z', O + 'lever4b_two_numbers.out', r'Bonferroni \|z\| >= ([\d.]+)')
add('ht_b', O + 'lever4b_two_numbers.out', r'ht \(harness\)\s+runs 2\s+mean b (-[\d.]+)\s+z -([\d.]+)', 1)
add('ht_z', O + 'lever4b_two_numbers.out', r'ht \(harness\)\s+runs 2\s+mean b (-[\d.]+)\s+z -([\d.]+)', 2)
add('tc_z', O + 'lever4b_two_numbers.out', r'ThinkingCap \(model\)\s+runs 1\s+mean b -[\d.]+\s+z -([\d.]+)')
add('lb_runs_3pt', O + 'lever4_runcount.out', r'\+3 points: (\d+) submissions')
add('lb_runs_5pt', O + 'lever4_runcount.out', r'\+5 points: (\d+) submissions')
add('one_run_resolve', O + 'lever4_runcount.out', r'= ([\d.]+) points \(LB\)')
add('best_of_100', O + 'lever4_runcount.out', r'n 100: \+([\d.]+) points')
# data inventory
add('dev_runs', O + 'crack_summary.txt', r'from (\d+) runs')
add('game_runs', O + 'crack_summary.txt', r'(\d+) \(run, game\) rows')
add('ledger_rows', O + 'ledger_negatives.out', r'(\d+) ledger rows')
add('ledger_neg', O + 'ledger_negatives.out', r'negative (\d+):')
add('ledger_rows_total', O + 'ledger_negatives.out', r'ledger total (\d+) rows')
# ARC-AGI-2 plateau
add('arc2_teams', O + 'arc2_plateau_lottery.out', r'board: (\d+) teams')
add('arc2_plateau_teams', O + 'arc2_plateau_lottery.out', r'plateau \[27, 35\): (\d+) teams')
add('arc2_draw_mean', O + 'arc2_plateau_lottery.out', r'draws mean ([\d.]+), SD ([\d.]+)', 1)
add('arc2_draw_sd', O + 'arc2_plateau_lottery.out', r'draws mean ([\d.]+), SD ([\d.]+)', 2)
# scoring and compute facts
for i, nm in enumerate(('med', 'lo', 'hi')):   # 14 valid B12-family runs (eff_cost.py)
    add(f'eff_cost_{nm}', O + 'eff_cost.out', r'B12 family \(\d+ runs\): loss median ([\d.]+)%, range ([\d.]+)% to ([\d.]+)%', i + 1)
add('eff_cost_runs', O + 'eff_cost.out', r'B12 family \((\d+) runs\)')
import re as _re
_off = {m.group(1): float(m.group(2)) for m in _re.finditer(r'^(\S+)\s+games\s+\d+\s+official\s+([\d.]+)', (R / O / 'eff_cost.out').read_text(), _re.M)}
put('ctl_dev_score_mean', f"{sum(_off[r] for r in ('b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3')) / 4:.1f}", O + 'eff_cost.out (mean official score of the 4 controls)')
add('stream_min_per_game', NB, r'FB25 \(D47\): ([\d.]+) stream-min per game')
add('true_sub_minutes', NB, r'max_runtime_s_per_game = (532)\*60')
add('fb25_minutes', NB, r'max_runtime_s_per_game = (121)\*60')
add('hidden_games', NB, r'game count should be (110)')
add('rhae_cap', REPORT, r'cap the maximum score for a level at (1\.15)x')
add('random_win_odds', REPORT, r'more often than 1 in ([\d,]+) times')
add('streams', NB, r'the real run \((10) x 532 / 110\)')
add('public_games', 'analysis/s37/v5_budget_prediction.py', r'(25) public games')
import csv as _csv
_subs = list(_csv.DictReader(open(R / 'analysis/s37/data/lb/arc-agi-3_our_submissions_20261104T0951.csv', encoding='utf-8')))
put('arc3_submissions', len(_subs), 'analysis/s37/data/lb/arc-agi-3_our_submissions_20261104T0951.csv')
put('forum_reroll_score', '38.85', 'external: Kaggle discussion 755320 ("identical code into 38.85 over ~100 submissions"), read 2 Nov 2026')
put('forum_rerolls', '100', 'external: Kaggle discussion 755320, read 2 Nov 2026')
put('o3_semi', '75.7', 'external: ARC Prize blog, F. Chollet, 20 Dec 2024, o3 on ARC-AGI-1 semi-private within the $10k limit, read 6 Nov 2026')
put('o3_high', '87.5', 'external: ARC Prize blog, 20 Dec 2024, high-compute o3 configuration, read 6 Nov 2026')
put('o3_ratio', '172', 'external: ARC Prize blog, 20 Dec 2024, high-compute configuration used 172x the compute, read 6 Nov 2026')
put('bj_traj', '60,000', 'external: Bjarnason et al. 2026, arXiv 2602.07150 abstract, read 6 Nov 2026')
put('bj_lo', '2.2', 'external: Bjarnason et al. 2026, arXiv 2602.07150 abstract (pass@1 varies by 2.2 to 6.0 points)')
put('bj_hi', '6.0', 'external: Bjarnason et al. 2026, arXiv 2602.07150 abstract')
put('wd_share', '54', 'external: Wiedmann et al. 2026, arXiv 2610.01618 abstract (about 54% of outcome variance)')
put('hardy_benchmarks', '22', 'external: Hardy et al. 2026, arXiv 2610.00651 abstract (22 benchmarks)')
put('astra_std', '62.7', 'external: ARC Prize blog (arcprize.org/blog/astra), GPT-6 Astra (max), Standard harness, semi-private, read 6 Nov 2026')
put('astra_pa', '99.9', 'external: ARC Prize blog (arcprize.org/blog/astra), best Provider Adapter score, semi-private, read 6 Nov 2026')
put('oai_before', '13.3', 'external: OpenAI, 29 Jul 2026, GPT-5.6 Sol with the official harness, ARC-AGI-3 public set, read 6 Nov 2026')
put('oai_after', '38.3', 'external: OpenAI, 29 Jul 2026, with retained reasoning and compaction, read 6 Nov 2026')
_AA = 'analysis/s37/out/aa_calibration.out'   # aa_calibration.py (6 Nov, exploratory A/A check of the b test)
add('aa_within_max', _AA, r'other than b32_spec2_r3: largest \|z\| ([\d.]+)')
add('aa_sd5', _AA, r'five-run A/A SD ([\d.]+)')
add('aa_z_out', _AA, r'b32_spec2_r3\s+b -0\.116.*?z -([\d.]+)')
add('aa_cross', _AA, r'past \|z\| 2\.64 at the five-run SD: (\d+) of')
add('b_spec2_r3', O + 'lever4b_two_numbers.out', r'b32_spec2_r3\s+\d+\s+[\d.]+\s+(-[\d.]+)')
add('b_spec2_a', O + 'lever4b_two_numbers.out', r'b31_spec2_fb25\s+\d+\s+[\d.]+\s+(-[\d.]+)')
_AB = 'analysis/s37/out/aa_bootstrap_b.out'   # aa_bootstrap_b.py (6 Nov, level bootstrap of b, seeded)
add('ab_refits', _AB, r'level bootstrap, (\d+) refits')
add('ab_se_len', _AB, r'len\s+runs 1\s+mean b -0\.086\s+bootstrap SE ([\d.]+)')
add('ab_se_tc', _AB, r'ThinkingCap\s+runs 1\s+mean b -0\.150\s+bootstrap SE ([\d.]+)')
add('ab_z_tc', _AB, r'ThinkingCap\s+runs 1\s+mean b -0\.150\s+bootstrap SE [\d.]+\s+z -([\d.]+)')
add('ab_z_ht', _AB, r'ht\s+runs 2\s+mean b -0\.095\s+bootstrap SE [\d.]+\s+z -([\d.]+)')
add('ab_z_spec2r3', _AB, r'b32_spec2_r3 against its own bootstrap SE: z -([\d.]+)')
add('ab_past', _AB, r'past \|z\| 2\.64 with bootstrap SEs: (\d+) of')
add('best_of_n', O + 'lever4_runcount.out', r'n (100): \+')
# --- Results 4.3, 4.6, 4.7, 4.8 (added 4 Nov ~12:15 UTC)
add('fb25_lo', O + 'lever3_noise.out', r'b28_ctl_fb25\s+ctl\s+levels\s+(\d+)')
add('fb25_hi', O + 'lever3_noise.out', r'b32_ctl_r3\s+ctl\s+levels\s+(\d+)')
V2O = O + 'v2_variance_family.out'   # v2_variance_family.py (14 valid B12-family runs)
for i, nm in enumerate(('runs', 'obs_sd', 'pred_sd', 'share')):
    add(f'v2_{nm}', V2O, r'(\d+) runs x 25 games: mean [\d.]+, observed SD ([\d.]+), predicted SD ([\d.]+), share (\d+)%', i + 1)
for i, nm in enumerate(('ci_level', 'ci_lo', 'ci_hi')):
    add(f'v2_{nm}', V2O, r'bootstrap (\d+)% interval for the share: (\d+)% to (\d+)%', i + 1)
add('v2_paircorr', V2O, r'across runs: ([+-][\d.]+)')
add('v2_games_change', V2O, r'varies: (\d+) of 25')
add('v2_max_swing', V2O, r'max swing (\d+) levels')
add('tok_lo', O + 'lever3_noise.out', r'b32_spec2_r3\s+spec2 levels\s+\d+ won\s+\d+ tokens ([\d.]+)M')
add('tok_hi', O + 'lever3_noise.out', r'b12_fb25\s+ctl\s+levels\s+\d+ won\s+\d+ tokens ([\d.]+)M')
_tt = [float(x) for x in re.findall(r'(?m)^  b\S+\s+(?:ctl|spec2)\s+levels\s+\d+ won\s+\d+ tokens ([\d.]+)M', (R / O / 'lever3_noise.out').read_text())]
assert len(_tt) == 6; put('tok_cv', f'{100 * statistics.stdev(_tt) / statistics.mean(_tt):.1f}', 'computed from out/lever3_noise.out (6 runs)')
add('corr_lv_tok', O + 'lever3_noise.out', r'corr\(levels, total tokens\) over runs: (-[\d.]+)')
add('ratio_q', O + 'lever3_noise.out', r'mean 0\.25 \|[^\n]*\| ([\d.]+) \[')
add('ratio_full', O + 'lever3_noise.out', r'mean 1\.00 \|[^\n]*\| ([\d.]+) \[')
add('perm_p_half', O + 'lever3_noise.out', r'f 0\.50: var\(total\) observed[^\n]*= ([\d.]+)')
add('H2_lo', O + 'lever3d_noise_model.out', r'H2:[^\n]*10-90%\s+([\d.]+)-\s*([\d.]+)', 1)
add('H2_hi', O + 'lever3d_noise_model.out', r'H2:[^\n]*10-90%\s+([\d.]+)-\s*([\d.]+)', 2)
add('open_per_run', O + 'lever3_stuck.out', r'per run ([\d.]+)\)')
add('open_total', O + 'lever3_stuck.out', r'open attempts at 121 min: (\d+)')
add('open_wall_pct', O + 'lever3_stuck.out', r'expected walls [\d.]+ \((\d+)%\)')
add('lb_sigma', O + 'lever4_runcount.out', r'sigma ([\d.]+)\)')
for n in (2, 5, 10, 20, 50):
    add(f'best_of_{n}', O + 'lever4_runcount.out', r'n\s+%d: \+([\d.]+) points' % n)
add('lb_t3', O + 'lever4_runcount.out', r'LB \+3 points: formula 15, t-test (\d+)')
add('lb_t5', O + 'lever4_runcount.out', r'LB \+5 points: formula 6, t-test (\d+)')
for k, g in (('sd_lb_lo', 1), ('sd_lb_hi', 2), ('n3_lo', 3), ('n3_hi', 4)):
    add(k, O + 'lever4_runcount.out', r'LB SD 2.87 from 5 runs: ([\d.]+) to ([\d.]+) points; runs per arm for \+3.0: (\d+) to (\d+)', g)
add('eff10_levels', O + 'lever4b_two_numbers.out', r'10% efficiency change: \+([\d.]+) levels')
add('eff10_b', O + 'lever4b_two_numbers.out', r'vs ([\d.]+) on the b scale')
add('variant_runs', O + 'lever4b_two_numbers.out', r'variant runs \((\d+)\)')
add('hazard_gain0', O + 'lever5_scheduler_sim.out', r'SD 0\.0[^\n]*\n(?:[^\n]*\n){3}\s+hazard vs fair: \+?(-?[\d.]+)%')
add('hazard_gain8', O + 'lever5_scheduler_sim.out', r'SD 0\.8[^\n]*\n(?:[^\n]*\n){3}\s+hazard vs fair: \+?(-?[\d.]+)%')
add('oracle_gain0', O + 'lever5_scheduler_sim.out', r'SD 0\.0[^\n]*\n(?:[^\n]*\n){4}\s+oracle vs fair: \+([\d.]+)%')
add('oracle_gain8', O + 'lever5_scheduler_sim.out', r'SD 0\.8[^\n]*\n(?:[^\n]*\n){4}\s+oracle vs fair: \+([\d.]+)%')
add('het_sd', 'analysis/s37/lever5_scheduler_sim.py', r'exp\(N\(0, (0\.8)\^2\)\)')
add('wall_like_H', O + 'lever5b_wall_detector.out', r'H = (\d+)000', 1)
for c in (5, 10, 20, 40):
    add(f'auc_{c}k', O + 'lever5b_wall_detector.out', r'C = %dk tokens into the attempt[^\n]*\n\s+leave-one-game-out logistic AUC ([\d.]+)' % c)
add('arc2_plateau_pct', O + 'arc2_plateau_lottery.out', r'teams \((\d+)% of the board\)')
add('arc2_pred_1', O + 'arc2_plateau_lottery.out', r'1-\s+1\s+\d+\s+[\d.]+ \([\d.]+\)\s+([\d.]+)')
add('arc2_obs_1', O + 'arc2_plateau_lottery.out', r'1-\s+1\s+\d+\s+([\d.]+)')
add('arc2_n1_teams', O + 'arc2_plateau_lottery.out', r'1-\s+1\s+(\d+)')
add('arc2_pred_top', O + 'arc2_plateau_lottery.out', r'64- inf\s+\d+\s+[\d.]+ \([\d.]+\)\s+([\d.]+) \(n = (\d+)\)\s+([\d.]+)', 1)
add('arc2_top_n', O + 'arc2_plateau_lottery.out', r'64- inf\s+\d+\s+[\d.]+ \([\d.]+\)\s+([\d.]+) \(n = (\d+)\)\s+([\d.]+)', 2)
add('arc2_pred_top_sd1', O + 'arc2_plateau_lottery.out', r'64- inf\s+\d+\s+[\d.]+ \([\d.]+\)\s+([\d.]+) \(n = (\d+)\)\s+([\d.]+)', 3)
add('arc2_obs_top', O + 'arc2_plateau_lottery.out', r'64- inf\s+\d+\s+([\d.]+)')
add('arc2_top_teams', O + 'arc2_plateau_lottery.out', r'64- inf\s+(\d+)')
add('arc2_spearman', O + 'arc2_plateau_lottery.out', r'inside the plateau: ([\d.]+)')
add('arc2_rise_obs', O + 'arc2_plateau_lottery.out', r'rise from 1 to (\d+) entries: observed ([\d.]+) points; best of n at SD [\d.]+ gives ([\d.]+) \((\d+)%\), at the byte-identical SD [\d.]+ ([\d.]+) \((\d+)%\)', 2)
add('arc2_rise_81', O + 'arc2_plateau_lottery.out', r'rise from 1 to (\d+) entries: observed ([\d.]+) points; best of n at SD [\d.]+ gives ([\d.]+) \((\d+)%\), at the byte-identical SD [\d.]+ ([\d.]+) \((\d+)%\)', 3)
add('arc2_share_81', O + 'arc2_plateau_lottery.out', r'rise from 1 to (\d+) entries: observed ([\d.]+) points; best of n at SD [\d.]+ gives ([\d.]+) \((\d+)%\), at the byte-identical SD [\d.]+ ([\d.]+) \((\d+)%\)', 4)
add('arc2_rise_34', O + 'arc2_plateau_lottery.out', r'rise from 1 to (\d+) entries: observed ([\d.]+) points; best of n at SD [\d.]+ gives ([\d.]+) \((\d+)%\), at the byte-identical SD [\d.]+ ([\d.]+) \((\d+)%\)', 5)
add('arc2_share_34', O + 'arc2_plateau_lottery.out', r'rise from 1 to (\d+) entries: observed ([\d.]+) points; best of n at SD [\d.]+ gives ([\d.]+) \((\d+)%\), at the byte-identical SD [\d.]+ ([\d.]+) \((\d+)%\)', 6)
_AP = O + 'arc2_plateau_lottery.out'   # 5 Nov correction: A0's three draws are unchanged resubmissions too
_RU = r'unchanged resubmissions: A0 \(3 draws\) mean ([\d.]+) SD ([\d.]+); PP \(4 draws\) mean ([\d.]+) SD ([\d.]+); pooled within-notebook SD ([\d.]+); notebook means differ by ([\d.]+)'
for i, nm in enumerate(('a0_mean', 'a0_sd', 'pp_mean', 'pp_sd', 'pool_sd', 'nb_gap'), 1): add(f'arc2_{nm}', _AP, _RU, i)
_RF = r"run-luck floor, rise from 1 to \d+ entries at the pooled SD [\d.]+: ([\d.]+) \((\d+)%\); at A0's SD [\d.]+: ([\d.]+) \((\d+)%\)"
for i, nm in enumerate(('rise_pool', 'share_pool', 'rise_86', 'share_86'), 1): add(f'arc2_{nm}', _AP, _RF, i)
_V9S = 'analysis/s37/sealed/v9_arc2_private_lb_prediction.txt'   # lists our 7 scored ARC-AGI-2 draws
_m = re.search(r'(\d) byte-identical resubmits\s+\(([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+); SD [\d.]+\) and (\d) forks', (R / _V9S).read_text())
_ident = [float(x) for x in _m.groups()[1:5]]; put('arc2_ident_sd', f'{statistics.stdev(_ident):.2f}', 'computed from ' + _V9S)
put('arc2_ident_n', _m.group(1), _V9S); put('arc2_fork_n', _m.group(6), _V9S)
_SUB2 = 'analysis/s37/data/lb/arc-agi-2_our_submissions_20261104T1743.csv'   # fetch_arc2_submissions.py (read-only)
_s2 = {r['ref']: r for r in __import__('csv').DictReader(open(R / _SUB2, encoding='utf-8'))}
_fork = [float(_s2[ref]['publicScore']) for ref in ('56712373', '56833698', '56858952')]   # the three A0-fork draws
put('arc2_fork_sd', f'{statistics.stdev(_fork):.2f}', 'computed from ' + _SUB2 + ' (submissions 56712373, 56833698, 56858952)')
add('ext_runs_n', O + 'lever2_external.out', r'(romantamrazov)') if False else put('ext_runs_n', 6, O + 'lever2_external.out (6 external rows)')
for k, pat in (('duck_med_lo', r'keithtyser[^\n]*med\s+([\d.]+)k'), ('duck_med_hi', r'jeroencottaar[^\n]*med\s+([\d.]+)k'),
               ('duck_q_lo', r'chiakazirim[^\n]*q ([\d.]+)'), ('duck_q_hi', r'jeroencottaar[^\n]*q ([\d.]+)'),
               ('slow_med', r'foysalemonshanto[^\n]*med\s+([\d.]+)k'), ('fast_med', r'romantamrazov[^\n]*med\s+([\d.]+)k')):
    add(k, O + 'lever2_external.out', pat)
_ext = re.findall(r'^(?:chiakazirim|foysalemonshanto|jeroencottaar|keithtyser|romantamrazov|wuliao0)[^\n]*\|\s+([\d.]+) vs (\d+)\s*$', (R / O / 'lever2_external.out').read_text(), re.M)
assert len(_ext) == 6, _ext
put('ext_max_err', f'{max(abs(float(a) - float(b)) for a, b in _ext):.1f}', 'computed from lever2_external.out')
for i, k in enumerate(('p4_0', 'p4_1', 'p4_2', 'p4_3')):
    add(k, O + 'lever2b_franzen_4pass.out', r'f 1\.00 \([^)]*\): pass totals \[(\d+), (\d+), (\d+), (\d+)\]', i + 1)
add('p4_var', O + 'lever2b_franzen_4pass.out', r'f 1\.00 [^\n]*var\(total\)\s+([\d.]+)')
add('p4_null', O + 'lever2b_franzen_4pass.out', r'f 1\.00 [^\n]*null median\s+([\d.]+)')
add('p4_p', O + 'lever2b_franzen_4pass.out', r'f 1\.00 [^\n]*P\(null <= obs\) = ([\d.]+)')
add('plateau_lo', O + 'arc2_plateau_lottery.out', r'plateau \[(\d+), (\d+)\)', 1)
add('plateau_hi', O + 'arc2_plateau_lottery.out', r'plateau \[(\d+), (\d+)\)', 2)
for n in (2, 5, 10, 20, 50, 100):
    add(f'best_n_{n}', O + 'lever4_runcount.out', r'n\s+(%d): \+' % n)
add('power', O + 'lever4_runcount.out', r'power ([\d.]+), two-sided ([\d.]+)', 1)
add('alpha', O + 'lever4_runcount.out', r'power ([\d.]+), two-sided ([\d.]+)', 2)
add('h2_range_lo_pct', O + 'lever3d_noise_model.out', r'H2:[^\n]*(10)-(90)%', 1)
add('h2_range_hi_pct', O + 'lever3d_noise_model.out', r'H2:[^\n]*(10)-(90)%', 2)
for c in (5, 10, 20, 40):
    add(f'checkpoint_{c}k', O + 'lever5b_wall_detector.out', r'C = (%d)k tokens' % c)
# --- Results 4.1, 4.2, 4.5 and Approach 3.3 additions (4 Nov ~12:25 UTC)
_subs_sorted = sorted(_subs, key=lambda r: r['date'])
_b12_date = next(r['date'] for r in _subs_sorted if r['ref'] == '56832595')
_later = [r for r in _subs_sorted if r['date'] > _b12_date and 'B12 resubmit' not in r['description'] and not r['description'].startswith('A1 ')]
put('later_variants_n', len(_later), 'our submissions csv'); _lv = [float(r['publicScore']) for r in _later]
put('later_min', f'{min(_lv):.2f}', 'our submissions csv'); put('later_max', f'{max(_lv):.2f}', 'our submissions csv')
put('later_mean', f'{statistics.mean(_lv):.2f}', 'our submissions csv')
put('b12_mean5', f'{statistics.mean(B12_LB):.2f}', 'computed from B12_LB')
put('a1_score', next(r['publicScore'] for r in _subs if r['description'].startswith('A1 ')), 'our submissions csv')
TRO = O + 'v5_truncation.out'   # v5_truncation.py (budget retrodiction, exploratory)
_d25 = [int(x) for x in re.search(r'D25 actual \[(\d+), (\d+)\]', (R / TRO).read_text()).groups()]
_tr = [int(x) for x in re.search(r'FB25 cut at the D25 per-game token count \[(\d+), (\d+), (\d+), (\d+)\]', (R / TRO).read_text()).groups()]
put('d25_a', _d25[0], TRO); put('d25_b', _d25[1], TRO)
for i in range(4): put(f'trunc_{i}', _tr[i], TRO)
put('trunc_mean', f'{statistics.mean(_tr):.1f}', 'computed from ' + TRO)
put('trunc_share', f'{100 * statistics.mean(_tr) / statistics.mean(_d25):.0f}', 'computed from ' + TRO + ' (mean cut-off FB25 / mean D25)')
add('km62_obs', O + 'fig_survival_fit.out', r'KM at (\d+)k: ([\d.]+), model ([\d.]+)', 2)
add('km62_mod', O + 'fig_survival_fit.out', r'KM at (\d+)k: ([\d.]+), model ([\d.]+)', 3)
add('km62_at', O + 'fig_survival_fit.out', r'KM at (\d+)k: ([\d.]+), model ([\d.]+)', 1)
_x7 = json.load(open(R / 'analysis/s7/x7_mech.json'))   # x7_mech.py on the X6 v2 and X7 serving logs (3 Oct)
_hit = {k: v['prefill_cached_tok'] / (v['prefill_cached_tok'] + v['prefill_new_tok']) for k, v in _x7.items()}
put('prefix_before', f"{100 * _hit['X6v2']:.0f}", 'computed from analysis/s7/x7_mech.json (X6 v2)')
put('prefix_after', f"{100 * _hit['X7']:.0f}", 'computed from analysis/s7/x7_mech.json (X7)')
put('turns_cut', f"{100 * (1 - _x7['X7']['requests'] / _x7['X6v2']['requests']):.0f}", 'computed from analysis/s7/x7_mech.json (agent requests)')
put('digest_calls', '541', 'X2 run (b3) request logs, counted 1 Oct 2026: scene() helper called 0 times in 541 unique tool calls; '
    'the request logs were not kept, so this number has no file in the repository')
_tb = {m.group(1): (int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(7))) for m in re.finditer(
    r'^(b\w+)\s+(\d+)\s+([\d.]+)\s+([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s*$', (R / O / 'lever4b_two_numbers.out').read_text(), re.M)}
assert len(_tb) == 17, len(_tb)
import sys as _sys; _sys.path.insert(0, str(R / 'analysis' / 's37')); from common import INVALID as _INVALID  # partial runs (not all 25 games played)
_tbv = {k: v for k, v in _tb.items() if k not in _INVALID}
put('fb25_logged_runs', len(_tbv), O + 'lever4b_two_numbers.out (valid runs only)')
put('b_min', f"{min(v[2] for v in _tbv.values()):.3f}", O + 'lever4b_two_numbers.out (valid runs)'); put('b_max', f"{max(v[2] for v in _tbv.values()):+.3f}", O + 'lever4b_two_numbers.out (valid runs)')
_ctl = ['b12_fb25', 'b28_ctl_fb25', 'b32_ctl_r2', 'b32_ctl_r3']; _sp2 = ['b32_spec2_r2', 'b32_spec2_r3']   # valid spec2 runs
_d = statistics.mean(_tb[k][2] for k in _sp2) - statistics.mean(_tb[k][2] for k in _ctl)
put('spec2_mean_b', f'{statistics.mean(_tb[k][2] for k in _sp2):.3f}', 'computed from lever4b table (2 valid spec2 runs)'); put('spec2_runs', len(_sp2), 'lever4b table')
put('spec2_z', f'{abs(_d) / (float(E["b_aa_sd"]["value"]) * (1 / len(_sp2) + 1 / 4) ** 0.5):.1f}', 'computed: |mean diff| / (A/A SD x sqrt(1/2 + 1/4))')
put('tc_supply', f"{_tb['b28_tc_fb25'][1]:.2f}", 'lever4b table'); put('tc_b', f"{_tb['b28_tc_fb25'][2]:.3f}", 'lever4b table')
put('tc_resid', f"{abs(_tb['b28_tc_fb25'][3]):.1f}", 'lever4b table')
put('ctl_supply', f"{statistics.mean(_tb[k][1] for k in _ctl):.1f}", 'lever4b table (mean of 4 controls)')
put('b15c_supply', f"{_tb['b15c_fb25'][1]:.2f}", 'lever4b table'); put('b15c_resid', f"{abs(_tb['b15c_fb25'][3]):.1f}", 'lever4b table')
put('b15c_supply_gain_pct', f"{100 * (_tb['b15c_fb25'][1] / statistics.mean(_tb[k][1] for k in _ctl) - 1):.0f}", 'computed from lever4b table')
for i, nm in enumerate(('l1', 'later')):   # discovery_split.py, 15 holdout games, 38 development runs (f1_forklab left out)
    add(f'eff_ratio_{nm}', O + 'discovery_split.out', r'hold: [^\n]*?median actions/human on cleared level 1 ([\d.]+) \(n=\d+\) vs later levels ([\d.]+)', i + 1)
for i, nm in enumerate(('n', 'med', 'q1', 'q3', 'cap', 'slow')):   # eff_cost.py, 14 valid family runs, all 25 games
    add(f'eff_ah_{nm}', O + 'eff_cost.out', r'cleared levels, \d+ family runs, all 25 games: n (\d+); actions / human baseline median ([\d.]+) '
        r'\(quartiles ([\d.]+) to ([\d.]+)\); at the 1.15 cap (\d+)%; slower than the baseline (\d+)%', i + 1)
add('holdout_games', 'analysis/s37/v3_hazard.py', r'the (15) original holdout games')
# leaderboard draws of B12 (byte-identical) and their ranks on the final public board
sc = load_lb('arc-agi-3')
for i, x in enumerate(B12_LB): put(f'b12_draw{i + 1}', f'{x:.2f}', 'analysis/s37/common.py B12_LB')
put('b12_sd', f'{statistics.stdev(B12_LB):.2f}', 'computed from B12_LB')
put('final_board_teams', f'{len(sc):,}', 'analysis/s37/data/lb/arc-agi-3_public_final_*.csv.gz')
_others = list(sc); _others.remove(max(B12_LB))   # drop our own team's row (best draw); ties count against the draw
for i, x in enumerate(B12_LB): put(f'b12_rank{i + 1}', f'{1 + sum(1 for s in _others if s >= x - 1e-9):,}', 'final board, our row removed')
add('b12_dev_score', O + 'crack_summary.txt', r'b12_fb25\s+games 25\s+mean score\s+([\d.]+)')
put('place_span', f"{int(E['b12_rank5']['value'].replace(',', '')) - int(E['b12_rank1']['value'].replace(',', '')):,}", 'computed: place of worst minus best identical draw')
# --- 4.4 sealed predictions (read-only sources)
SE = 'analysis/s37/sealed/arc3_private_lb_prediction.txt'; V5S = 'analysis/s37/sealed/v5_budget_prediction.txt'
add('h_sigma_approx', SE, r'run noise with SD sigma of about\s+([\d.]+)')
add('crowd_n', SE, r'Fitted to the ([\d,]+) teams with public\s+scores in \[(\d+), (\d+)\)', 1)
add('crowd_lo', SE, r'Fitted to the ([\d,]+) teams with public\s+scores in \[(\d+), (\d+)\)', 2)
add('crowd_hi', SE, r'Fitted to the ([\d,]+) teams with public\s+scores in \[(\d+), (\d+)\)', 3)
add('s_sigma', SE, r'Alternative S \(low noise\): sigma = ([\d.]+) with a real skill gain of ([\d.]+) points', 1)
add('s_gain', SE, r'Alternative S \(low noise\): sigma = ([\d.]+) with a real skill gain of ([\d.]+) points', 2)
for k, pat in (('P1', r'P1 [^\n]*H ([\d.]+) \(([\d.]+) to ([\d.]+)\)\. S ([\d.]+)\. H fails if above ([\d.]+)'),):
    for i, nm in enumerate(('h', 'lo', 'hi', 's', 'fail')): add(f'{k}_{nm}', SE, pat, i + 1)
for i, nm in enumerate(('h', 'lo', 'hi', 's')): add(f'P2_{nm}', SE, r'P2 [^\n]*H (\d+)% \((\d+)% to (\d+)%\)\. S (\d+)%', i + 1)
add('P2_fail', SE, r'H fails if above (\d+)%\.')
add('P2_chance', SE, r'Pure chance is (\d+)%')
add('P3_pub', SE, r'The same gap is\s+([\d.]+) points')
for i, nm in enumerate(('h', 'lo', 'hi', 's', 'fail')): add(f'P3_{nm}', SE, r'H ([\d.]+) \(([\d.]+) to ([\d.]+)\)\. S ([\d.]+)\. H fails if above ([\d.]+)\.\nP4', i + 1)
add('P3_ngroup', SE, r'teams in C with n >= (\d+)')
for i, nm in enumerate(('h', 'lo', 'hi', 's', 'fail')): add(f'P4_{nm}', SE, r'H ([\d.]+) points \(([\d.]+) to ([\d.]+)\)\. S ([\d.]+)\. H fails if below ([\d.]+)', i + 1)
for i, nm in enumerate(('h', 'lo', 'hi', 'fail')): add(f'P5_{nm}', SE, r'H ([\d.]+) \(([\d.]+) to ([\d.]+)\)\. H fails if below ([\d.]+)\. Noise', i + 1)
add('P5_cut', SE, r'final public score >= (\d+)')
for i, nm in enumerate(('n', 'lo', 'hi')): add(f'P6a_{nm}', SE, r'private scores of our (\d+) B12-family runs: ([\d.]+) to ([\d.]+)', i + 1)
for i, nm in enumerate(('pt', 'lo', 'hi')): add(f'P6b_{nm}', SE, r'private - D = (\d+) \(([\d.]+) to ([\d.]+)\)', i + 1)
add('seal_block', 'analysis/s37/sealed/ots_upgrade_check.txt', r'block (\d+)')
add('rule_min_in', SE, r'at least (\d+) of them fall inside')
add('rule_cross', SE, r'H fails if (\d+) or more of P1 to P4')
add('v5_minutes', 'analysis/s37/v5_budget_prediction.py', r'max_runtime_s_per_game = (242)\*60')
add('v5_pred', V5S, r'P-V5-1  Total levels completed in the 2x run: (\d+)\.')
add('v5_lo', V5S, r'90% prediction interval (\d+)-(\d+)', 1)
add('v5_hi', V5S, r'90% prediction interval (\d+)-(\d+)', 2)
add('v5_stuck', V5S, r'<= (\d+)\s+crack model fails toward "stuck"')
add('v5_oot', V5S, r'>= (\d+)\s+crack model fails toward "out of time"')
for i, nm in enumerate(('pt', 'lo', 'hi')): add(f'v5_games_{nm}', V5S, r'P-V5-2  Games completed: (\d+) \(90%: (\d+)-(\d+)\)', i + 1)
add('fb25_games_lo', V5S, r'FB25: (\d+)-(\d+)\.', 1); add('fb25_games_hi', V5S, r'FB25: (\d+)-(\d+)\.', 2)
add('fb25_mean_sealed', V5S, r'mean ([\d.]+), SD [\d.]+\)\. Games completed')
# V5 outcome (readout 4 Nov ~14:37 UTC by v5_readout.py, written before the data; follow-up v5_followup.py written after)
V5R = 'analysis/s37/out/v5_readout.out'; V5F = 'analysis/s37/out/v5_followup.out'
add('v5_obs', V5R, r'P-V5-1 total levels (\d+)')
add('v5_games_obs', V5R, r'P-V5-2 games completed (\d+)')
add('v5_l121', V5R, r'P-V5-3 levels by minute 121 (\d+)')
for i, nm in enumerate(('pt', 'lo', 'hi')): add(f'v5_l121_{nm}', V5R, r'\(sealed about (\d+), (\d+)-(\d+)\)', i + 1)
add('v5_rho_n', V5R, r'over (\d+) open games: ([-\d.]+)', 1); add('v5_rho', V5R, r'over (\d+) open games: ([-\d.]+)', 2)
for i, nm in enumerate(('med', 'lo', 'hi')): add(f'v5_rho_{nm}', V5R, r'sealed median ([-\d.]+), 90% ([-\d.]+) to ([-\d.]+)\)', i + 1)
add('v5_tokens_m', V5R, r'total tokens ([\d.]+)M')
add('v5_alt_a', V5R, r'A saturation ~(\d+)'); add('v5_alt_b', V5R, r'B log-linear ~(\d+)')
add('v5_alt_c_lo', V5R, r'C linear (\d+)-(\d+)', 1); add('v5_alt_c_hi', V5R, r'C linear (\d+)-(\d+)', 2)
add('v5_loglogistic', V5R, r'log-logistic variant ([\d.]+)')
add('v5_gain_obs', V5F, r'gain observed ([\d.]+) vs predicted ([\d.]+)', 1)
add('v5_gain_pred', V5F, r'gain observed ([\d.]+) vs predicted ([\d.]+)', 2)
add('v5_stuck_n', V5F, r'(\d+) games stuck early')
for i, nm in enumerate(('1x', 'pred', 'obs', 'excess')):
    add(f'v5_stuck_{nm}', V5F, r'1\.5: [^)]*\): 1x ([\d.]+), predicted ([\d.]+), V5 (\d+); excess over prediction ([\d.]+)', i + 1)
add('v5_rest_excess', V5F, r'other \d+ games: [^\n]*excess over prediction ([\d.]+)')
add('v5_block', 'analysis/s37/out/v5_ots_check.txt', r'block (\d+)')
add('v5_stuck_cut', V5F, r'FB25 mean <= ([\d.]+):')
add('v5_seal_lead_h', 'analysis/s37/out/v5_ots_check.txt', r'by about ([\d.]+) hours')
VT = 'analysis/s37/out/v5_seal_timing.txt'  # GitHub push event, benchmark.json start/end, block time (all UTC)
for nm, lab in (('commit', 'commit'), ('push', 'push'), ('start', 'run start'), ('note', 'post-seal note'), ('anchor', 'anchor'), ('end', 'run end')):
    add(f'v5_t_{nm}', VT, lab + r' (\d{1,2}:\d{2})')
add('v5_push_lead_min', VT, r'push to run start: (\d+) minutes')
add('v5_start_anchor_min', VT, r'run start to Bitcoin anchor: (\d+) minutes')
add('v5_anchor_lead_h', VT, r'Bitcoin anchor to run end: ([\d.]+) hours')
VO = 'analysis/s37/out/v9_v10_ots_check.txt'
add('v9_block', VO, r'v9_arc2[^\n]*\n[^\n]*\n\s*ots: [^\n]*block (\d+)')
add('v9_anchor_t', VO, r'block 974554 hash \w+ time 2026-11-04T(\d\d:\d\d)')
put('v5_excess_total', int(E['v5_obs']['value']) - int(E['v5_pred']['value']), 'v5_obs minus v5_pred (sealed point)')
# post-seal note (committed f2a04ca 10:48 UTC, before any V5 output) and follow-up 2 (after the readout, exploratory)
PSN = 'analysis/s37/sealed/v5_post_seal_note.txt'; V5F2 = 'analysis/s37/out/v5_followup2.out'
add('v5_dyn_lo', PSN, r'H2 ([\d.]+) levels', 1); add('v5_dyn_hi', PSN, r'H0 ([\d.]+), H1', 1)
add('v5_rule_lo', PSN, r'a total of (\d+)-(\d+) favours the dynamic simulator', 1)
add('v5_rule_hi', PSN, r'a total of (\d+)-(\d+) favours the dynamic simulator', 2)
add('v5_fade_min', V5F2, r'V5 by minute ([\d.]+) (\d+), by minute 121', 1)
add('v5_stuck_by_fade', V5F2, r'games stuck at 1x [^\n]*V5 by minute [\d.]+ (\d+), by minute 121')
add('v5_stuck_ctl_lo', V5F2, r'games stuck at 1x [^\n]*controls at their end \(121 min\) \[(\d+), (\d+), (\d+), (\d+)\]', 4)
add('v5_stuck_ctl_hi', V5F2, r'games stuck at 1x [^\n]*controls at their end \(121 min\) \[(\d+), (\d+), (\d+), (\d+)\]', 1)
add('v5_half2_pred', V5F2, r'all 25 games: gain after minute 121 predicted ([\d.]+) given the tokens generated, observed (\d+)', 1)
add('v5_half2_obs', V5F2, r'all 25 games: gain after minute 121 predicted ([\d.]+) given the tokens generated, observed (\d+)', 2)
add('v5_tail_k', V5F2, r'V5 attempts past (\d+)k tokens: (\d+); clears (\d+) vs expected ([\d.]+); Poisson P\(>= \d+\) = ([\d.]+)', 1)
add('v5_tail_n', V5F2, r'V5 attempts past (\d+)k tokens: (\d+); clears (\d+) vs expected ([\d.]+); Poisson P\(>= \d+\) = ([\d.]+)', 2)
add('v5_tail_obs', V5F2, r'V5 attempts past (\d+)k tokens: (\d+); clears (\d+) vs expected ([\d.]+); Poisson P\(>= \d+\) = ([\d.]+)', 3)
add('v5_tail_exp', V5F2, r'V5 attempts past (\d+)k tokens: (\d+); clears (\d+) vs expected ([\d.]+); Poisson P\(>= \d+\) = ([\d.]+)', 4)
add('v5_tail_p', V5F2, r'V5 attempts past (\d+)k tokens: (\d+); clears (\d+) vs expected ([\d.]+); Poisson P\(>= \d+\) = ([\d.]+)', 5)
add('v5_endgame_pct', V5F2, r'fades over the last (\d+)% of the per-game budget')
add('v5_endgame_1x', V5F2, r'from minute ([\d.]+) at 121 min and from minute ([\d.]+) at 242 min', 1)
add('v5_endgame_2x', V5F2, r'from minute ([\d.]+) at 121 min and from minute ([\d.]+) at 242 min', 2)
_st = re.findall(r'  (b\S+): games complete at minute [\d.]+ \d+, at minute 121 (\d+); open attempts \d+, mean age ([\d.]+)k tokens, expected walls [\d.]+; sealed method from this state \(second-half tokens 1\.35x first-half\): \+([\d.]+) levels', (R / V5F2).read_text())
assert len(_st) == 4, _st
put('v5_ctl_done121_lo', min(int(x[1]) for x in _st), V5F2); put('v5_ctl_done121_hi', max(int(x[1]) for x in _st), V5F2)
put('v5_ctl_age_lo', min(float(x[2]) for x in _st), V5F2); put('v5_ctl_age_hi', max(float(x[2]) for x in _st), V5F2)
put('v5_sealed_from_ctl_lo', min(float(x[3]) for x in _st), V5F2); put('v5_sealed_from_ctl_hi', max(float(x[3]) for x in _st), V5F2)
add('v5_done121', V5F2, r'V5 at minute 121: games complete at minute [\d.]+ \d+, at minute 121 (\d+); open attempts \d+, mean age ([\d.]+)k', 1)
add('v5_age121', V5F2, r'V5 at minute 121: games complete at minute [\d.]+ \d+, at minute 121 (\d+); open attempts \d+, mean age ([\d.]+)k', 2)
add('v5_sealed_from_v5', V5F2, r'V5 at minute 121: [^\n]*sealed method from this state \(second-half tokens 1\.35x first-half\): \+([\d.]+) levels')
_w = re.search(r'stuck games: controls levels at [\d.]+ -> 121 (\d+)->(\d+), (\d+)->(\d+), (\d+)->(\d+), (\d+)->(\d+); tokens in the window ([\d.]+)M, ([\d.]+)M, ([\d.]+)M, ([\d.]+)M \| V5 (\d+)->(\d+), tokens ([\d.]+)M', (R / V5F2).read_text())
_g = [int(_w.group(2 * k + 2)) - int(_w.group(2 * k + 1)) for k in range(4)]; _t = [float(_w.group(9 + k)) for k in range(4)]
put('v5_stuck_win_gain_lo', min(_g), V5F2); put('v5_stuck_win_gain_hi', max(_g), V5F2)
put('v5_stuck_win_tok_lo', min(_t), V5F2); put('v5_stuck_win_tok_hi', max(_t), V5F2)
put('v5_stuck_win_v5_from', _w.group(13), V5F2); put('v5_stuck_win_v5_to', _w.group(14), V5F2); put('v5_stuck_win_v5_tok', _w.group(15), V5F2)
# scheduler simulation with and without the end-of-run rule (exploratory, 5 Nov after the operator's OK; before V11's readout)
SCH = O + 'sched_sealedcalc_check.out'
for k, lab in (('sch_1x_on', r'1x on'), ('sch_2x_on', r'2x on'), ('sch_sealed_on', r'sealed from 1x on')):
    add(k, SCH, r'\n  ' + lab + r' +([\d.]+) levels')
add('sch_worlds', SCH, r'scheduler simulator: (\d+) worlds x (\d+) runs', 1); add('sch_runs', SCH, r'scheduler simulator: (\d+) worlds x (\d+) runs', 2)
add('sch_gain_on', SCH, r'true gain, rule on \(2x on - 1x on\): \+([\d.]+)')
add('sch_gain_off', SCH, r'true gain, rule off \(2x off - 1x off\): \+([\d.]+)')
add('sch_sealed_gain_on', SCH, r'sealed gain from 1x on states: \+([\d.]+)')
add('sch_short_on', SCH, r'shortfall from 1x on states \(sealed - true 2x on\): -([\d.]+)')
add('sch_short_off', SCH, r'shortfall from 1x off states \(sealed - true 2x off\): -([\d.]+)')
add('sch_short_diff', SCH, r'minus shortfall from 1x off states: -([\d.]+) \(SE ([\d.]+)\)', 1)
add('sch_short_diff_se', SCH, r'minus shortfall from 1x off states: -([\d.]+) \(SE ([\d.]+)\)', 2)
add('sch_v5state_short', SCH, r'from the 2x state at 121: sealed further levels - true further levels: -([\d.]+)')
add('sch_short_diff_d40', O + 'sched_sealedcalc_check_drain40k.out', r'minus shortfall from 1x off states: -([\d.]+)')
add('sch_short_diff_d80', O + 'sched_sealedcalc_check_drain80k.out', r'minus shortfall from 1x off states: -([\d.]+)')
add('sch_drain_k', 'analysis/s37/sched_endgame_sim.py', r'base (\d+)k, the size of')
put('sch_v5_excess', round(float(E['v5_obs']['value']) - float(E['v5_pred']['value']) - float(E['sch_short_on']['value']), 1),
    'v5_obs - v5_pred - sch_short_on (V5 above the simulator-corrected expectation)')
add('h_sigma_alt', SE, r'S and H at sigma ([\d.]+) fit about equally well')
add('h_sigma_meas', SE, r'H at sigma ([\d.]+) fits worse')
# distance to 85% / 100% (distance_85.py, exploratory model extrapolation; Discussion)
D85 = O + 'distance_85.out'
add('d85_cap_inf', D85, r'\n0\.06\s+[\d.]+ \([\d.]+\)\s+[\d.]+ \([\d.]+\)\s+[\d.]+ \([\d.]+\)\s+[\d.]+ \([\d.]+\)\s+([\d.]+) \(')
add('d85_q_needed', D85, r'unlimited budget: 85% needs q <= ([\d.]+)')
add('d85_nowall_k', D85, r'no walls: 85% needs a budget multiplier of about ([\d.]+)x')
# caps by game set (exact k=inf form; added 6 Nov 2026 after outside review round 2, D102)
for nm, lab in (('pool4', r'pooled, four controls'), ('pool8', r'pooled, eight runs'), ('ho4', r'held-out, four controls'), ('ho8', r'held-out, eight runs')):
    for i, part in enumerate(('all25', 'ho15'), 1):
        add(f'd85_cap_{nm}_{part}', D85, lab + r'\s+q [\d.]+: ([\d.]+) on the 25 public level counts, ([\d.]+) on the 15 held-out games', i)
for nm, qq in (('mix4', r'0\.110'), ('mix8', r'0\.070')):
    add(f'd85_cap_{nm}', D85, r'25-game mix, dev q 0\.000 and held-out q ' + qq + r' \([a-z ]+\): ([\d.]+)')
add('d85_q_needed_ho', D85, r'held-out games: 85% needs q <= ([\d.]+)')
# a first look inside the walls from the action logs (wall_autopsy.py, exploratory, D103)
WA = O + 'wall_autopsy.out'
add('wa_dc22_h', WA, r'dc22 L5\s+(\d+)\s')
add('wa_dc22_actmax', WA, r'dc22 L5: \d+ runs, (\d+) actions')
add('wa_dc22_tpa', WA, r'dc22 L5: \d+ runs, \d+ actions, \d+k tokens, ([\d,]+) per action')
add('wa_dc22_need', WA, r'dc22 L5: .*?, ([\d]+)k tokens needed')
add('wa_g50t_act', WA, r'g50t L5: \d+ runs, (\d+) actions')
add('wa_g50t_ah', WA, r'g50t L5: .*? per action, ([\d.]+) of h')
add('wa_ah_cleared', WA, r'median a/h ([\d.]+), median tokens per action')
add('wa_tpa_cleared', WA, r'median tokens per action ([\d,]+)\n')
add('wa_tpa_walls', WA, r'summary: walls median tokens per action ([\d,]+)')
add('wa_many', WA, r'still did not clear: (\d+) of')
add('wa_short', WA, r'walls where no run took even the median cleared share of h \(a/h [\d.]+\): (\d+) of')
add('wa_auc', WA, r'P\(wall ratio > cleared ratio\) = ([\d.]+)')
add('wa_h_rel_walls', WA, r'walls median ([\d.]+) \(range')
add('wa_h_rel_cleared', WA, r'\(level 2 or later, n=\d+\): median ([\d.]+)')
# walls, shelving and start order (walls.py, exploratory, 5 Nov 2026 after V5, T1, T1 rerun and T1c; Section 4.3, Discussion)
W = O + 'walls.out'
for i, nm in enumerate(('att', 'open', 'q', 'med', 's', 'cap'), 1):
    add(f'walls_refit_{nm}', W, r'fit 7 same-agent runs \+ V5: attempts (\d+) \(open (\d+)\); q ([\d.]+) median ([\d.]+)k s ([\d.]+); unlimited cap, independent walls: (\d+) of', i)
add('walls_v5_obs', W, r'V5 observed: (\d+) levels at 242 min')
for i, nm in enumerate(('never', 'runs'), 1):
    add(f'walls_{nm}', W, r'wall census \(pooled fit q [\d.]+\): (\d+) levels reached but never cleared in any of (\d+) runs', i)
for i, nm in enumerate(('likely', 'fixed_cap'), 1):
    add(f'walls_{nm}', W, r'levels with P\(wall\) >= 0\.5: (\d+); cap if those are fixed walls: (\d+) of', i)
add('walls_mixed', W, r'levels cleared in some runs and open in others: (\d+)')
add('walls_nlev_total', W, r'cap if those are fixed walls: \d+ of (\d+)')
for i, nm in enumerate(('runs', 'max', 'total', 'v5', 'p'), 1):
    add(f'walls_dc22_{nm}', W, r'dc22 L5\s+(\d+)\s+(\d+)k\s+(\d+)k\s+(\d+)k\s+([\d.]+)', i)
add('walls_share_sealed', W, r'four controls at 121 min, 4 controls \(sealed fit\): (\d+)% of')
for i, nm in enumerate(('share_refit', 'open_ctl'), 1):
    add(f'walls_{nm}', W, r'four controls at 121 min, 7 same-agent runs \+ V5: (\d+)% of (\d+) open attempts', i)
for i, nm in enumerate(('k1', 'k1_lv', 'k2', 'k2_lv', 'inf', 'inf_lv', 'nowall_k'), 1):
    add(f'walls_refit_{nm}', W, r'7 same-agent runs \+ V5 \(q [\d.]+, median [\d.]+k, s [\d.]+\): k=1 ([\d.]+) \(([\d.]+) levels\); k=2 ([\d.]+) \(([\d.]+)\); k=inf ([\d.]+) \(([\d.]+)\); no walls: 85% at about ([\d.]+)x', i)
for i, nm in enumerate(('var_tot', 'var_zero', 'zero_pct', 'zero_lo', 'zero_hi'), 1):
    add(f'walls_{nm}', W, r'summed per-game variance over the 7 runs: ([\d.]+); part due to 0 levels vs some: ([\d.]+) \((\d+)%; bootstrap over runs 90% interval (\d+)% to (\d+)%\)', i)
for i, nm in enumerate(('shelved_n', 'gameruns'), 1):
    add(f'walls_{nm}', W, r'\n  (\d+) of (\d+) game runs; at level 1', i)
for i, nm in enumerate(('zero_n', 'zero_shelved', 'zero_by20'), 1):
    add(f'walls_{nm}', W, r'0-level game runs in total: (\d+), of which shelved: (\d+); shelved by minute 20 \(last logged step at or before minute 20\): (\d+)', i)
for i, nm in enumerate(('shelf_med', 'shelf_lo', 'shelf_hi'), 1):
    add(f'walls_{nm}', W, r'tokens on level 1 when shelved: median (\d+)k \(range (\d+)k-(\d+)k\)', i)
for i, nm in enumerate(('v5l1_n', 'v5l1_lo', 'v5l1_hi'), 1):
    add(f'walls_{nm}', W, r'all (\d+) cleared level 1 in V5: True; tokens on level 1 from (\d+)k to (\d+)k', i)
for i, nm in enumerate(('late_n',), 1):
    add(f'walls_{nm}', W, r'games starting after minute 30 on average: (\d+) \(', i)
for i, nm in enumerate(('late_share', 'early_share'), 1):
    add(f'walls_{nm}', W, r'their mean share of levels cleared ([\d.]+) vs ([\d.]+) for the rest', i)
for i, nm in enumerate(('start', 'start_lo', 'start_hi', 'levels', 'nlev'), 1):
    add(f'walls_lf52_{nm}', W, r'lf52: start\s+([\d.]+) min \(\s*(\d+)-\s*(\d+)\), levels\s+([\d.]+) of (\d+)', i)
for i, nm in enumerate(('n', 'slope', 'p'), 1):
    add(f'walls_start_{nm}', W, r'within-game effect of a later start \(7 runs, (\d+) game runs\): ([-+\d.]+) levels per 10 minutes later; permutation p = ([\d.]+)', i)
add('walls_start_sd', W, r'within-game SD of the start minute: ([\d.]+) min')
# 4.1: B12 against the base notebook B1 on the leaderboard (b1_vs_b12.py, exploratory, 5 Nov 2026)
BV = O + 'b1_vs_b12.out'; T1T = 'paper/tables/t1_arc3_submissions.md'
add('b1_lb_1', T1T, r"B1: fork of dfranzen's Milestone 2 notebook \| ([\d.]+)")
add('b1_lb_2', T1T, r'B1 resubmitted \(noise sample 2\) \| ([\d.]+)')
add('b1_lb_mean', BV, r'B1 mean ([\d.]+) \(n=2\)')
add('b1_b12_diff', BV, r'B12 - B1 = ([\d.]+) points')
for i, nm in enumerate(('sd', 'se', 'z', 'p', 'n'), 1):
    add(f'b1_b12_{nm}', BV, r'B12 SD ([\d.]+): SE ([\d.]+), z ([\d.]+), two-sided p ([\d.]+); submissions per arm to detect a true [\d.]+ with power 0\.8: (\d+)', i)
BA = O + 'b1_vs_b12_aft.out'
add('b1aft_games', BA, r'AFT fit on the (\d+) dev games of B1')
for i, nm in enumerate(('b', 'sd', 'x'), 1):
    add(f'b1aft_{nm}', BA, r'B12 mean b vs B1: \+?([-\d.]+) \(SD over the 7 B12 runs ([\d.]+)\); tokens per crack x([\d.]+)', i)
for i, nm in enumerate(('lo', 'hi'), 1):
    add(f'b1aft_{nm}', BA, r'B12 - B1 in b, 90% interval \+?([-\d.]+) to \+?([-\d.]+)', i)
DH = O + 'dev_holdout_fit.out'
for pre, lab in (('dh4', 'four controls'), ('dh8', r'eight runs \(7 at 121 min \+ V5\)')):
    for part, plab in (('dev', r'dev \(10 games\)'), ('ho', r'holdout \(15 games\)')):
        for i, nm in enumerate(('att', 'q', 'med', 's', 'lv', 'nl', 'pct'), 1):
            add(f'{pre}_{part}_{nm}', DH, lab + ', ' + plab + r': attempts (\d+); q ([\d.]+) median ([\d.]+)k s ([\d.]+); levels per run ([\d.]+) of (\d+) \((\d+)%\)', i)
# Appendix C: the slot rule D' and harness settings, from the B12 notebook (embedded ours_form_priority.py and setup_env)
for nm in ('level_pow', 'eff_const', 'eff_pow', 'pace_pow', 'pace_ref', 'c_action', 'action_scale', 'c_token', 'token_scale',
           'lin_floor', 'rel_pow', 'b1', 'b2', 'b3', 'fade'):
    add(f'dp_{nm}', NB, r'D_PRIME = \{[^}]*?\\"' + nm + r'\\": ([\d_.]+)')
    E[f'dp_{nm}']['value'] = E[f'dp_{nm}']['value'].replace('_', '')
for i, nm in enumerate(('m_hi', 'm_lo'), 1): add(f'dp_{nm}', NB, r'min\((4)\.0, max\((0\.25), \(params\[\\"pace_ref\\"\]', i)
for i, nm in enumerate(('rel_hi', 'rel_lo'), 1): add(f'dp_{nm}', NB, r'min\((2)\.0, max\((0\.5), rel / params', i)
for i, nm in enumerate(('n_hi', 'n_lo'), 1): add(f'dp_{nm}', NB, r'n_clamped = min\((\d+), max\((\d+), n\)\)', i)
add('dp_norm_num', NB, r'norm = (\d+) / \(n_clamped')
_c = re.search(r"'LOCAL_ANALYZER_CONTEXT_WINDOW': \((\d+)\+(\d+)\)\*1024", (R / NB).read_text())
put('agent_ctx_k', int(_c.group(1)) + int(_c.group(2)), NB)
add('agent_max_out_k', NB, r"'LOCAL_ANALYZER_MAX_OUTPUT': (\d+)\*1024")
add('agent_drain_k', NB, r"'ARC3_CONTEXT_DRAIN_TOKENS': (\d+)\*1024")
add('agent_hist_turns', NB, r"'ARC3_HISTORY_ASSISTANT_TURNS': (\d+)")
add('agent_temp', NB, r"'LOCAL_ANALYZER_TEMPERATURE': '([\d.]+)'")
add('agent_streams', NB, r"'ARC3_MAX_ACTIVE_STREAMS': (\d+)")
for i, nm in enumerate(('n', 'actions', 'tokens', 'ratio'), 1):
    add(f'dp_shelf_{nm}', W, r"median shelved 0-level state \((\d+) game runs: (\d+) actions, (\d+)k tokens on level 1\): A\*M\*C = ([\d.]+) x norm", i)
for gm in ('su15', 'lf52'):
    for i, nm in enumerate(('start', 'levels', 'nlev', 'mean1x'), 1):
        add(f'walls_v5_{gm}_{nm}', W, gm + r': V5 start\s+([\d.]+), levels (\d+) of (\d+) \(121-min mean ([\d.]+)\)', i)
# --- 4.4 V9: sealed ARC-AGI-2 private-board prediction (read-only source, sealed 4 Nov ~16:53 UTC)
V9S = 'analysis/s37/sealed/v9_arc2_private_lb_prediction.txt'
for i, nm in enumerate(('sigma', 'mu', 'tau', 'f', 'beta')):
    add(f'v9_prim_{nm}', V9S, r'PRIMARY: sigma = ([\d.]+) \(our pooled same-code SD\), model HB \(mu ([\d.]+), tau ([\d.]+), f ([\d.]+), beta ([\d.]+)\)', i + 1)
add('v9_alt_all7', V9S, r'NAMED ALTERNATIVES: sigma = ([\d.]+) \(SD of all our 7 draws\)')
add('v9_alt_fit', V9S, r'sigma = ([\d.]+) \(best public-board fit\)')
add('v9_alt_s', V9S, r'S, low noise\s+\(sigma = ([\d.]+), beta')
add('v9_ll_hi_sigma', V9S, r'-[\d.]+ at 1\.0, -[\d.]+ at ([\d.]+)\)')
for i, nm in enumerate(('lo', 'hi', 'n')): add(f'v9_c_{nm}', V9S, r'C = teams with a final public score in \[(\d+), (\d+)\) \(([\d,]+) teams\)', i + 1)
_P = {'P1': r'P1 Spearman\(public, private\) within C: ([\d.]+) \[([\d.]+), ([\d.]+)\]; ([\d.]+) / ([\d.]+) / ([\d.]+)\.',
      'P2': r'P2 Share [^\n]*: (\d+)% \[(\d+)%, (\d+)%\]; (\d+)% / (\d+)% / (\d+)%\.',
      'P3': r'on the public board\): ([\d.]+) \[([\d.]+), ([\d.]+)\]; ([\d.]+) / ([\d.]+) / ([\d.]+)\.',
      'P4': r'over all of C: ([\d.]+) \[([\d.]+), ([\d.]+)\];\s+([\d.]+) / ([\d.]+) / ([\d.]+)\.'}
for k, pat in _P.items():
    for i, nm in enumerate(('prim', 'lo', 'hi', 'a081', 'a100', 's')): add(f'v9_{k}_{nm}', V9S, pat, i + 1)
add('v9_P2_chance', V9S, r'Chance is (\d+)%')
add('v9_P3_pub', V9S, r'the same gap is ([\d.]+)\s+on the public board')
for i, nm in enumerate(('ngrp', 'n1')): add(f'v9_P3_{nm}', V9S, r'\((\d+) vs (\d+) teams;', i + 1)
for i, nm in enumerate(('n', 'cut', 'prim', 'fail')):
    add(f'v9_P5_{nm}', V9S, r'P5 Spearman\(public, private\) among the (\d+) teams with a public score >= (\d+): about ([\d.]+)\. Fails if below ([\d.]+)\.', i + 1)
add('v9_rule_min_in', V9S, r'Primary passes if at least (\d+) of P1 to P4')
add('v9_rule_ratio', V9S, r'at most ([\d.]+) times the smallest')
# --- 4.4 V10: token-supply test (sealed 4 Nov ~16:41 UTC, read-only; readouts by v10_readout.py written before the data;
#     v10_followup.py is exploratory, written 5 Nov after the readouts)
V10S = 'analysis/s37/sealed/v10_supply_prediction.txt'; V10F = O + 'v10_followup.out'
add('v10_drift_lo', V10S, r'T1c totals ([\d.]+) to ([\d.]+) levels', 1); add('v10_drift_hi', V10S, r'T1c totals ([\d.]+) to ([\d.]+) levels', 2)
add('v10_tok_tol_pct', V10S, r'its tokens are within (\d+)% of')
for nm, path in (('t1', O + 'v10_readout.out'), ('t1r', O + 'v10_readout_rerun.out')):
    for i, k in enumerate(('games', 'tokens_m', 'levels', 'won')):
        add(f'v10_{nm}_{k}', path, r'T1 \S+: games (\d+), tokens ([\d.]+)M, levels (\d+), completed (\d+)', i + 1)
    add(f'v10_{nm}_r', path, r'measured supply ratio r = ([\d.]+)')
for i, k in enumerate(('tokens_m', 'tok_ratio', 'levels', 'won')):
    add(f'v10_t1c_{k}', O + 'v10_readout.out', r'T1c \S+: games \d+, tokens ([\d.]+)M \(([\d.]+) of control mean\), levels (\d+), completed (\d+)', i + 1)
for nm, run in (('t1', 't1_fb25_nospec'), ('t1r', 't1r_fb25_nospec')):
    add(f'v10_{nm}_never', V10F, r'\n' + run + r': [^\n]*never started (\d+)')
    for i, k in enumerate(('pt', 'lo', 'hi')):
        add(f'v10_{nm}_band_{k}', V10F, r'\n' + run + r': [^\n]*\n  exploratory: sealed band at r [\d.]+: ([\d.]+) \(([\d.]+) to ([\d.]+)\)', i + 1)
add('v10_last_start_lo', V10F, r'control runs \(4 controls and T1c\): (\d+) to (\d+) min', 1)
add('v10_last_start_hi', V10F, r'control runs \(4 controls and T1c\): (\d+) to (\d+) min', 2)
add('v10_same_sd6', V10F, r'same-agent FB25 runs \[[^\]]*\] mean [\d.]+ SD ([\d.]+); with T1c: n (\d+) mean ([\d.]+) SD ([\d.]+)', 1)
add('v10_same_n7', V10F, r'same-agent FB25 runs \[[^\]]*\] mean [\d.]+ SD ([\d.]+); with T1c: n (\d+) mean ([\d.]+) SD ([\d.]+)', 2)
add('v10_same_sd7', V10F, r'same-agent FB25 runs \[[^\]]*\] mean [\d.]+ SD ([\d.]+); with T1c: n (\d+) mean ([\d.]+) SD ([\d.]+)', 4)
add('v10_block', VO, r'v10_supply[^\n]*\n[^\n]*\n\s*ots: [^\n]*blocks (\d+) and (\d+)', 2)
add('v10_t_commit', VO, r'v10_supply[^\n]*\n\s*committed \w+ 2026-11-04T(\d\d:\d\d)')
# --- 4.4 V11: first-level guard (sealed 5 Nov 14:32 UTC; scored by v11_readout.py, written before the data)
V11S = 'analysis/s37/sealed/v11_l1guard_prediction.txt'; V11R = O + 'v11_readout.out'; V11T = O + 'v11_ots_check.txt'
add('v11_ref_n0_mean', V11S, r'Game runs with 0 levels per run: [\d, ]+ \(mean ([\d.]+), (\d+) in total\)', 1)
add('v11_ref_n0_total', V11S, r'Game runs with 0 levels per run: [\d, ]+ \(mean ([\d.]+), (\d+) in total\)', 2)
add('v11_sk48_ref', V11S, r'sk48 is one of them in (\d) of 7 runs')
add('v11_guard_tok', V11S, r'fewer than ([\d,]+) generated tokens on the')
add('v11_started', V11R, r'25 games in benchmark\.json, (\d+) started')
add('v11_never', V11R, r'games that never started: (\d+) \(reference 0\)')
add('v11_n0', V11R, r'N0 = (\d+) started games ended with 0 levels')
add('v11_n0_never', V11R, r'N0 counting the \d+ never-started games as zeros: (\d+)')
add('v11_total', V11R, r'total levels: (\d+) of (\d+)', 1)
add('v11_ref_total_mean', V11R, r'mean ([\d.]+), SD ([\d.]+); z', 1); add('v11_ref_total_sd', V11R, r'mean ([\d.]+), SD ([\d.]+); z', 2)
add('v11_last_start', V11R, r'last game to start: \w+ at minute ([\d.]+)')
add('v11_hits', V11R, r'guard hits (\d+)')
add('v11_zero_tok', V11R, r'tn36: tokens (\d+)k')
add('v11_won', V11R, r'games completed \(all levels\): (\d+)')
add('v11_tokens_m', V11R, r'total tokens ([\d.]+)M \(FB25 control mean ([\d.]+)M\)', 1)
_v11 = re.search(r'total tokens ([\d.]+)M \(FB25 control mean ([\d.]+)M\)', (R / V11R).read_text())
put('v11_tok_ratio', round(float(_v11.group(1)) / float(_v11.group(2)), 2), V11R + ' (V11 tokens / FB25 control mean)')
add('v11_t_commit', V11T, r'committed \w+ at 2026-11-05T(\d\d:\d\d)')
add('v11_t_push', V11T, r'kernel v1 pushed (\d\d:\d\d)')
add('v11_block', V11T, r'BitcoinBlockHeaderAttestation\((\d+)\)')
add('v11_block_t', V11T, r'block \d+ time 2026-11-05T(\d\d:\d\d)')
SE2 = O + 'sched_endgame_sim.out'
add('sch_guard_zero', SE2, r'121 on \+ L1 guard  levels mean [^\n]*zero-level games ([\d.]+); [^\n]*never started ([\d.]+)', 1)
add('sch_guard_never', SE2, r'121 on \+ L1 guard  levels mean [^\n]*zero-level games ([\d.]+); [^\n]*never started ([\d.]+)', 2)
add('sch_on_zero', SE2, r'\n  121 on             levels mean [^\n]*zero-level games ([\d.]+); [^\n]*never started ([\d.]+)', 1)
add('sch_on_never', SE2, r'\n  121 on             levels mean [^\n]*zero-level games ([\d.]+); [^\n]*never started ([\d.]+)', 2)
SGS = O + 'sched_guard_supply.out'   # exploratory, written after the V11 readout
for k, sc, lab in (('sch_g_full_two', '1.00', 'guard   '), ('sch_ng_low_two', '0.86', 'no guard'), ('sch_g_low_two', '0.86', 'guard   ')):
    add(k, SGS, r'supply x' + sc.replace('.', r'\.') + ', ' + lab + r': never started [\d.]+ per run \(2 or more in (\d+)% of runs\)')
add('sch_g_low_levels', SGS, r'supply x0\.86, guard   : [^\n]*levels ([\d.]+)')
# 7 Nov 2026 (exploratory): per-task luck on SWE-Bench-Verified, from the public runs of Bjarnason et al. (analysis/s38).
_SW = 'analysis/s38/out/swe_pertask_luck.json'; _sw = json.load(open(R / _SW))['pooled']
for _k in ('configs', 'runs', 'share', 'share_min', 'share_max', 'sd_obs_min', 'sd_obs_max', 'sometimes_min', 'sometimes_max', 'never_min', 'never_max'):
    put(f'swe_{_k}', _sw[_k], _SW)
put('swe_null90_lo', _sw['null90'][0], _SW); put('swe_null90_hi', _sw['null90'][1], _SW); put('swe_tasks', 500, _SW); put('swe_runs_per_config', 10, _SW); put('swe_null_draws', 4000, 'analysis/s38/swe_pertask_luck.py (parametric bootstrap draws)')
json.dump(E, open(R / 'paper' / 'results.json', 'w'), indent=1)
print(f'{len(E)} numbers ->', R / 'paper' / 'results.json')
