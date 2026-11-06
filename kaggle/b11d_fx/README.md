# b11_fx (submission build) and b11d_fx (dev mechanism run): frontier mode, round-4 milestone 1
Status update 21:20 UTC: G0 FAILED (11/58 = 19.0% < 25%); per the rule below, neither kernel is pushed. See state/DECISIONS.md D31.
Status 2026-10-03: BUILT, NOT PUSHED. Neither kernel may be pushed, run on GPU, or submitted without the operator's explicit go.
Built with: `python3 kaggle/make_bline.py <dir> <slug> lab/bline_x6_frontier.patch <FLAGS...>` (see kaggle/make_bline.py).

## What changes vs X6 v2 (kaggle/b4_x6, sub 56790419, LB 29.29)
- The patch cell carries lab/bline_x6_frontier.patch = lab/bline_x6.patch + lab/frontier_x6.patch. The frontier part adds
  inference/framework/frontier_mode.py and inference/framework/fx_explorer.py, plus a `_frontier_step` hook in solver.py step_env that
  runs only when ARC3_FRONTIER_MODE=1. With the flag off the build behaves as X6 v2.
- b11_fx flags: ARC3_FRONTIER_MODE=1, ARC3_FRONTIER_STALL_TOKENS=60000, ARC3_FRONTIER_STALL_TURNS=10, ARC3_FRONTIER_MIN_LEVEL=1,
  ARC3_FRONTIER_BUDGET=40000 (actions), ARC3_FRONTIER_TMAX_S=1200, ARC3_FRONTIER_MAX_CONCURRENT=8 (explorers at once),
  ARC3_FRONTIER_MAX_ATTEMPTS=1 (per level), ARC3_FRONTIER_MAX_STATES=30000, ARC3_FRONTIER_LOG=/kaggle/working/frontier_events.jsonl.
  X6 v2 flags unchanged (ARC3_SCENE_DIGEST=0, ARC3_LEVEL_LEDGER=1). Diff vs b4_x6 = patch cell + these flags (checked).
- b11d_fx: same patch; dev settings: the 15 holdout public games (bp35 cd82 cn04 dc22 g50t ka59 lf52 ls20 m0r0 s5i5 sk48 sp80 su15
  tn36 wa30), 25 min per game, ARC3_FRONTIER_STALL_TOKENS=40000, STALL_TURNS=6, BUDGET=40000, TMAX_S=300, MAX_CONCURRENT=4.
  Threshold from our logs (analysis/s7/stall_cost.py, 9 runs, 415 level attempts; VERIFIED): after 20k tokens on a level the LLM
  still clears 64.6% of levels (median level score 115), after 40k 46.5% (105.6), after 60k 25.0% (49.6, n=32). A 20k dev rule would
  hand the explorer levels the LLM clears at the cap; 40k is the lowest threshold that still yields about 16 events per holdout run
  (64 attempts reached 40k in 4 holdout runs). b11d_fx tests the mechanism; b11_fx keeps 60k for the 9-h run.

## Mechanism (per game thread)
1. Stall rule: generated tokens on the current level >= STALL_TOKENS, LLM turns on it >= STALL_TURNS, attempts on this level <
   MAX_ATTEMPTS, remaining game time >= MIN_REMAINING_S.
2. The thread releases its priority-gate slot (another game gets the GPU), waits for an explorer slot, sends level RESET and runs FX
   (graph search over frame digests; click targets from connected components; BFS over the frontier; unmask fallback when the masked
   graph is exhausted) for up to BUDGET actions or TMAX_S seconds.
3. Cleared: the level is completed in the scored env; the LLM is told the shortest known solving path and continues on level k+1.
   Not cleared: RESET plus replay of the LLM's own actions on this level, then a frame-equality check; the LLM continues where it was.
   If the replayed frame differs, the event is logged as 'reset' and the LLM continues from the level start with a note.
4. Every event is appended to frontier_events.jsonl (game, level, tokens, turns, actions used, seconds, outcome, solution length).

## Pre-registered gates (written 2026-10-03 21:00 UTC, before any GPU data)
- G0 (CPU, local engine, 58 frontier level starts from logged X6/X8 runs, FX at 40k actions): P(solve) >= 25% (the critic's
  "< 10/40" kill line). Result (21:10 UTC 3 Oct): FAIL, 11/58 = 19.0% at 40k (flat to 60k). Per this rule, G1 is not run.
- G1 (b11d_fx, one run, about 0.7 GPU-h like the 38-min X6 holdout run). Kill the build if any of:
  (a) a traceback from frontier_mode or fx_explorer, or a game thread that never returns;
  (b) zero 'cleared' events;
  (c) 'reset' outcomes > 20% of events (restore broken);
  (d) holdout levels < 25 (X6 v2 holdout 28; controls 22 and 24).
  Recorded, not gated: events fired, P(clear | fired), seconds and actions per event, explorer actions/s, P(LLM clears k+1 | opened),
  mean game score vs X6 v2 holdout 13.26 (a blind clear of a level the LLM would have cleared later lowers the score, not the count).
  Caveat: dev envs are in-process (no REST), so explorer speed in b11d_fx overstates gateway speed (local REST proxy: 80.6 actions/s
  per client, lab/frontier/gw_bench.py).
- G2 (ship b11_fx as the default ARC-AGI-3 build, operator go required): G1 passes AND cleared events >= 3 in the G1 run. The LB is
  recorded but cannot judge the arm: the expected gain (+1 to +2.5 points, INFERENCE) is below one-sample LB noise (SD 1.6-4). LB floor
  for keeping it: >= 26.3 (X6 v2 29.29 minus 3).
- Charter K2 (+15% levels on dev plus holdout, paired, >= 2 runs) is the bar for any claim that frontier mode moves us toward the 36
  cluster. Milestone 1 is not expected to meet it (about 3-5 extra levels on about 74 per dev+holdout pair, +4-7%, INFERENCE).
  K4: <= 10 GPU-h in total before the go/kill decision.
- EXIT: drop frontier mode for this competition if G1 fails twice after one bug-fix iteration, or if two paired dev+holdout runs show
  fewer levels than X6 v2 on both sets.

## How to run (after the operator's go)
- Push dev: `kaggle kernels push -p kaggle/b11d_fx` (KAGGLE_API_TOKEN inline). Read: `kaggle kernels output shreekumarshah/b11d-fx -p
  /data/kaggle_ro/out/b11d` then count levels with analysis/s7/run_report.py and read /kaggle/working/frontier_events.jsonl.
- Local tests (CPU): `bash lab/frontier/build_tree.sh` then
  `cd lab/frontier && PYTHONPATH=/data/.pylib:tree/src/tufa-arc-agi-framework/src:tree/src/ARC3-Inference python3 -m pytest -q tests/`
