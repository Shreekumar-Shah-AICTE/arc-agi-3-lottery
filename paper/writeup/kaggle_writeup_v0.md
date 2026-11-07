<!-- Kaggle writeup v0, 4 Nov 2026 ~12:40 UTC. Stand-alone summary of the PDF (target 900-1,200 words; cap 1,500).
     Every number is copied from paper/draft and checked by paper/check_numbers.py. Placeholders: [LINK: ...] ([V5] filled 4 Nov ~14:50 UTC). -->

# The ARC-AGI-3 Lottery: a testable model of LLM-agent scores under Kaggle compute

## The puzzle

We submitted the same notebook, byte for byte, five times to the ARC-AGI-3 Kaggle leaderboard. It scored 33.37, 29.69,
28.24, 26.76 and 26.21 (SD 2.87), which placed it 197th, 1,096th, 1,503rd, 1,836th and 1,920th of 5,073 teams. Identical
code spans 1,723 places. The notebook (B12) is dfranzen's public Milestone 2 agent on Tufa Labs' Duck harness, plus a
ledger of solved-level rules and a slot-priority rule from shiiin9's notebook, running Qwen3.8-Flash-Next on one RTX Pro 6000 GPU. Of the 157 agent experiments in our ledger, 32 were negative results, and
nothing we tried after this notebook beat its own run-to-run noise. Below is a testable model of why, and of what can beat the noise.

## The model in one paragraph

An agent plays a game level by level. On each level it spends generated tokens until it "cracks" the level: the moment
it finds what the level wants. With probability q the level is a wall that will not crack within any budget we can
afford; otherwise the wait for the crack is lognormal in tokens. Fitted on four control runs, the model has three numbers: q = 0.06, a
median wait of 22.1k tokens and a spread s = 1.05. These pool two kinds of games: the 10 we developed on show no walls, and the 15 we held out (used only for two early design checks) carry all of them (q 0.11), so the held-out fit is the better guide to the hidden games. The noise is luck near the budget edge: whether the hard late levels crack just before the clock runs out, and whether a game's first level cracks before the harness's slot
rule shelves it (about half of the per-game noise; in simulation the rule moves noise rather than adding it). The model reproduces the run-to-run SD of levels (5.6 against 5.3 observed on six runs; 6.0 with a seventh) only when each level's difficulty persists from run to run, with or without walls.

## Result 1: run-to-run luck is big enough to reorder similar agents

The luck is per game. Across 13 valid runs of one agent family, 22 of the 25 public games change their level count, by up to 7 levels,
and per-game scores are uncorrelated within a run. Throughput is not the cause: total tokens per run vary little
(a CV of about 2.5%).

- One leaderboard draw per arm resolves only differences of 11.4 points or more.
- A team's best of 100 identical entries sits about 7 points above its mean, so in a crowded band rank depends on entries and luck, not only on code.
- Sorted after the fact, our 32 negative results (17 of them distinct GPU-tested changes) either cut the tokens available to the levels that count, raised the
  crack cost, or did not change cracking at all. On the 14 valid runs with token logs (controls and variants), no variant passes a multiple-comparison
  test for crack cost against the control (a level bootstrap agrees; on the five true controls alone, two do, Section 4.5).
- B12 needs as many tokens per crack as dfranzen's agent without the two additions (a factor of 1.01; exploratory).

## Result 2: predictions sealed before their outcomes were known

Each prediction and its decision rule were hashed (SHA-256), time-stamped in Bitcoin (OpenTimestamps) and pushed to our private working repo on GitHub before the outcome was known; the private-leaderboard hashes were also posted on Kaggle (discussion 755123). Scorecard of the five tests: one pass, one inconclusive, one unscored, two pending.

- Doubling the play time per game (pushed to our private repo ten minutes before the run started; its public time stamp came 24 minutes into the four-hour run): 127 levels on the 25 public games (90% interval 110 to 145), against 95 to 107 levels
  in four runs at the normal budget. The doubled run cleared 141: inside the interval, so by the sealed rule the model passes, but 14 above
  the point prediction, and a log-linear alternative stated in advance (about 140) fits about 2.4 times better, which the sealed file had said one run could not resolve.
- Cutting the token supply (speculative decoding off, same clock): a sealed table of level totals against the measured
  supply. The run and its one allowed rerun were both invalid by the sealed rule (games that waited for a turn never started), so the test is unscored.
- Keeping games on level 1 from being shelved by the harness's slot rule: at most 1 of the games that start
  should end with 0 levels (seven normal runs had 1 to 4, so with no effect a pass has about a 1 in 7 chance). By the letter of the rule it passed, with 1 of the 23 that started; all six games that had zeros before cleared level 1. But 2 games never
  started (counting them as zeros gives 3), so we count it as inconclusive.
- The private leaderboards: six predictions for ARC-AGI-3 and five for ARC-AGI-2 (sealed after its close), scored when the hosts publish them.

## Result 3: a similar pattern on ARC-AGI-2

Seven of our scored draws of one public notebook lineage on ARC-AGI-2 have mean 29.66 and SD 0.81. On the final public
board (3,627 teams), 2,133 teams (59%) sit on a plateau between 27 and 35. If these teams mostly resubmit near-identical
public notebooks, each team's score is roughly the best of its n draws. Without fitting anything to the board, run luck of unchanged code (the pooled SD of our resubmissions, 0.61) explains about half (52%) of the rise of the plateau's mean from 29.41 for single-entry teams to 32.28 at 64 or more entries; the SD of all seven draws, which pool two notebooks, gives 70%. The rest is code differences between notebooks or effort. The sealed ARC-AGI-2 prediction tests this on the private board. On SWE-Bench-Verified, the 120 public runs of Bjarnason et al. are consistent with independent per-task luck (exploratory): it accounts for 86% of their run-to-run variance, where pure per-task luck gives 81% to 127%. This tests independence across tasks, not the crack model.

## What to do

- **Count runs before believing a gain.** Rerun the unchanged control to measure the SD, compare crack cost on shared levels
  instead of level totals (about 5 times fewer runs on our data, at least 2.8), and report the SD and the number of runs;
  our CPU notebook ([LINK: run-count notebook]) does the arithmetic on a team's own run logs. Leaderboards could show the number of entries next to
  each best score.
- **Buy tokens for the games that need them.** In the model a change raises the score only if it raises cracks per
  GPU-second (faster serving, fewer wasted tokens, fewer games stalled at the end), and one run shows it only if the gain
  beats the noise floor.
- **Attack walls.** In the action logs, on 6 of the 9 likely walls the agent took as many moves as it usually needs and still missed the goal; on the other 3 it cannot make enough moves, at about twice the usual tokens per move (1,597 against 756, median over the 9). A scheduler that knew
  which games were walled would gain 13% to 17% in simulation; one that only chases the highest crack hazard gains
  nothing, and our attempt to predict walls from the logs scored close to chance.
- **Limits.** One agent family, one model and one GPU type, and most estimates rest on four to seven runs (Section 6).

Why this matters for the 85% goal: a field that learns from leaderboard differences learns slowly if many of them are luck; a model of the luck tells a team when a result is worth acting on. If walls are permanent, they, not time, set the distance to 85%: at the fitted wall rate (6%; 4% refitted with the doubled run), even unlimited play caps this agent at about 56% (69%) on the 15 held-out games, the better guide, and at about 73% (81%) on all 25 public games; 85% needs a wall rate of 3.1% or less (a model extrapolation, not a measurement). A lognormal wait without walls, though, fits almost as well and predicts the doubled run better (146 levels); a run at four times the play time would tell them apart (141 levels with walls, 172 without). The fitted wall share also keeps falling with longer play, and a bootstrap over games puts a 90% interval of 48% to 91% on the held-out cap. With walls, tokens buy only the
cracks near the budget edge, and the rest is skill.

## Links and disclosure

- Paper (PDF: full methods, all submissions and runs, the 32 negative results): [LINK: PDF]
- Notebook: [LINK: notebook arc3-b12-x6dprime]. Code, data and sealed files: [LINK: repo]. Sealed post: Kaggle discussion 755123.
- This work was carried out with an AI agent under the author's direction; the paper's Section 3.7 describes how.
