<!-- Appendix E, 7 Nov 2026: the full text of Sections 4.6 to 4.8 and of three exploratory paragraphs of Section 4.3 as of 6 Nov, plus the SWE-Bench-Verified check added on 7 Nov
     (moved verbatim from paper/draft/04_results_noise_runs_universality.md). Numbers are registered in
     build_results.py and checked by check_numbers.py. -->
**Fewer walls than the four-run fit says (Section 4.3).** Refitted on all eight runs of this agent (the seven valid 121-minute runs
and V5; 1,003 attempts), the wall share falls from 6% to 4% of levels (median wait 24.2k tokens, s 1.10). The refit has
seen V5, so this is a consistency check, not a test: it expects 101.2 levels at 121 minutes and 140.2 at twice the
budget (observed: 102.75 on average and 141). It calls 28% of the controls' open attempts walls. Fourteen levels were
reached but never cleared in any of the eight runs, and the refit makes 9 of them walls with probability 0.5 or more.
The clearest is level 5 of dc22: reached in 6 runs and left open after 880k tokens in total (218k in V5 alone). If
those nine are fixed walls, unlimited play caps this agent at 154 of the 183 public levels. Another 40 levels were
cleared in some runs and not in others.

**Inside the walls (Section 4.3).** This check is exploratory, written after outside reviewers asked what the walls are,
and it uses the action logs only, because the agent's own notes are not in our data. For each of the 9 likely walls we
compared the actions the agent took on the level with the human baseline h that the scorer uses. On the held-out levels
it clears, the agent needs a median of 0.72 h actions at 756 generated tokens per action; on the walls it spends 1,597
tokens per action (median over the 9), about twice as long per move. The walls are of two kinds. In 6 of the 9, some
run took at least 0.72 h actions and still found no way through: g50t level 5, for example, got 244 actions, 2.54 times
the human baseline. These look like failures to find the goal or the mechanic, not a shortage of moves. In the other 3,
no run came close to the actions a person needs: dc22 level 5 asks people for 324 actions, the agent took at most 102,
and at its 1,950 tokens per action there, reaching 0.72 h would take about 456k tokens, roughly twice the 218k it got
in the doubled run. Walls are not simply the longest levels: relative to the earlier levels of their game, their human
baselines are only somewhat larger than those of the levels the agent clears (median ratio 1.98 against 1.31; AUC 0.56).
Telling a wrong goal from a missing mechanic needs the agent's reasoning, which a rerun that keeps its notes, or a
person playing the level, can supply.

**Half of the per-game noise is the first level (Section 4.3).** Levels are played in order, so a game that never clears level 1
scores nothing. Over the seven 121-minute runs, the split between 0 levels and at least one accounts for 49% of the summed per-game variance
(bootstrap over runs, 90% interval 39% to 57%), almost all of it in four games (sc25, cd82, dc22 and g50t). These zeros
are not walls. Of the 17 game runs that ended with 0 levels, 14 were shelved by the harness's slot rule. At most 10 games
play at once; a game hands back its slot whenever its context is trimmed and then queues behind games with a higher
priority, and its own priority falls with the tokens and actions spent on its current level (Appendix C). The shelved
games had spent 56k to 132k tokens on level 1 (median 77k) and did not play again; 7 of them stopped by minute 20. In V5
all 6 of these games cleared level 1, after 40k to 350k tokens. A game's first crack therefore races the slot rule's
patience as well as the clock, and that patience can run out early in the run. A sealed one-run test that protects
games on level 1 from shelving passed (V11, Section 4.4): 1 of the 23 games that started ended with 0 levels, but 2
games never started.

**How many runs it takes to see a real change (Section 4.6).** On the leaderboard one submission has an SD of 2.87 points. To detect a true difference with power 0.8 (two-sided test at
0.05) takes 15 submissions per arm for a gain of 3 points and 6 for a gain of 5; one run per arm resolves only 11.4 points.
These counts use the normal approximation; because the SD is estimated from the runs themselves, a t-test needs one more
run per arm (16 and 7, simulated). The SD is itself uncertain: from five submissions its 90% interval is 1.86 to 6.81
points, so the 15 could be anywhere from 7 to 81.
Resubmission alone buys points: the best of n identical submissions beats their mean by 1.62 (n = 2), 3.34 (5), 4.42 (10),
5.36 (20), 6.45 (50) and 7.20 points (100).

In development runs, the two numbers of Section 3.5 help. A 10% change in crack efficiency moves a run's level total by
4.4 levels and its crack cost b by 0.095. On the 6 valid same-agent runs, b has an A/A SD of 0.051 and the level total an
SD of 5.3, so a b comparison needs 4.5 runs per arm against 22.6 for raw level totals, 5.1 times fewer. The factor rests
on six runs and is fragile: leaving out one run at a time gives 2.8 to 65.1 times, so we claim only that b needs several
times fewer runs, at least about 2.8. On the 8 valid variant runs, the two numbers predict the level difference from the
control with r = 0.68. A CPU notebook (paper/notebook/how_many_runs.ipynb in our repository) computes these run counts,
fits b for two arms from run logs (with our B1 and B12 runs as the example) and repeats the fit-half, predict-half check
of Section 4.2 on a user's own runs.

**Methods the model rules out (Section 4.7).** We simulated schedulers under the fitted model before spending GPU time on them. Giving the 10 streams to the games with
the highest current crack hazard gains +0.0% over equal sharing when games are alike, and -3.6% when their difficulty
differs (log-median SD 0.8). An oracle that knows which games are walled and skips them gains +16.9% and +13.4%. So a
scheduler is worth building only with a wall detector. We then tried to predict, from the run logs (actions, clicks,
tokens per call, level index, earlier levels' costs), which open attempts would not crack within the next 30k tokens,
checking at 5k, 10k, 20k and 40k tokens into an attempt. Leave-one-game-out AUCs were 0.544, 0.491, 0.533 and 0.582,
close to chance, so we did not build it.

**The same lottery elsewhere (Section 4.8).** ARC-AGI-2. Seven of our scored draws of one public notebook lineage have mean 29.66 and SD 0.81 (Table 2). They come from
two notebooks, each resubmitted unchanged: A0 three times (mean 29.03, SD 0.86) and PP four times (mean 30.14, SD 0.34). On the final public
ARC-AGI-2 board (3,627 teams), 2,133 teams (59%) sit on a plateau between 27 and 35. If these teams mostly resubmit
near-identical public notebooks, each team's score is roughly the best of its n draws. Without fitting anything to the board, the best of n
draws from our seven predicts a plateau mean of 29.65 for single-entry teams rising to 31.67 at 91 entries; observed
means rise from 29.41 (282 teams with one entry) to 32.28 (64 teams with 64 or more) (Fig. 6). From 1 to 91 entries that
is 2.01 of the observed 2.87 points (70%). Run luck alone, at the pooled SD of our unchanged resubmissions (0.61), gives 1.50 (52%);
at PP's SD alone (0.34) it would be 0.84 (29%), and at A0's (0.86) 2.15 (75%). Inside the plateau, score and number of
entries have a rank correlation of 0.55. The observed slope is steeper than an SD of 0.81 predicts and matches an SD of
1.0 at the top. The two notebooks' means differ by 1.11 points, so the plateau's spread mixes run luck with small code
differences, and teams with more entries may also improve their code between entries. Run luck is about half of the effect,
with wide bounds from only seven draws. Section 4.4 gives a sealed prediction of how this plateau reorders on the private board.

Other teams' harnesses. We fitted the crack model separately to six public runs by other teams. Four forks of the Duck
harness show our median crack wait (20.2k to 28.1k tokens) but more walls (q from 0.22 to 0.37). Two runs show no walls at
all (q = 0.00), one with a much slower median (65.1k) and one, on 10 games only, with a faster one (11.4k). In every case
the per-run fit reproduces the run's level total within 1.3 levels.

One caveat on independence. dfranzen's public notebook plays four passes over the 25 games, not all at the same time. Their
level totals (102, 106, 104 and 105) vary far less than independent games would make them (variance 2.9 against a null
median of 40.9, p = 0.020). Some harnesses therefore couple passes, so our independent-game noise is an upper bound for
them. Fewer level-1 zeros do not explain it: every pass started all 25 games and ended 1, 1, 2 and 3 of them with no
level (sk48 in all four; analysis/s37/franzen_4pass_zeros.py), against 1 to 4 (mean 2.43) in our seven reference runs.
We have not found the mechanism.

**SWE-Bench-Verified (Section 4.8; exploratory, added 7 Nov 2026).** analysis/s38/fetch_swe_runs.py reads the per-run
harness reports of Bjarnason et al. from their public dataset
(huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts, CC BY 4.0) and keeps the ids of the resolved tasks: 12
configurations of 10 runs each over 500 tasks. analysis/s38/swe_pertask_luck.py takes each task's success rate p across
the m runs of a configuration and predicts the variance of a run's number of resolved tasks as the sum over tasks of
p(1 - p), times m/(m - 1) for the small sample. Pooled over the 12 configurations, the predicted variance is 86% of the
observed one; redrawing runs under pure per-task luck with the same p (a parametric bootstrap, 4,000 draws, seed 0)
gives a 90% band of 81% to 127% for this share. Per configuration the share is 51% to 200%, and the observed run SD is
0.75 to 1.76 points of resolution rate. In each configuration 100 to 331 tasks are never solved and 144 to 309 are
solved in some runs only. The check uses final outcomes only; the trajectories in the same dataset also record tokens,
which a crack-model fit in token time would need.
