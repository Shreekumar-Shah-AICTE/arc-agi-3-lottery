<!-- Draft v0, 4 Nov 2026 ~12:20 UTC; 6 Nov ~19:15 UTC reordered to open with the contribution (hosts' guide) and cut from 307 words. Abstract. [V5] filled 4 Nov ~14:50 UTC from out/v5_readout.out. -->

## Abstract
We present a testable model of what sets an LLM agent's ARC-AGI-3 score under fixed Kaggle compute, and of which changes
can beat its run-to-run noise. Byte-identical code submitted five times placed between 197th and
1,920th of 5,073 teams. In the model, each level waits for a "crack" (the moment the agent finds what the level wants),
measured in generated tokens: with probability q the level is a wall within the budget, otherwise the wait is lognormal.
Fitted on four control runs, the model reproduces the run-to-run noise (SD 5.6 against 5.3 levels) only if each level's difficulty persists across runs; a version without walls does as well. The walls sit in the 15 games we did not develop on, where, if walls are permanent, even unlimited
play would reach only about 56% to 69% of the score (73% to 81% over all 25 public games). A lognormal wait without walls fits almost as well and predicts the doubled run better, and the fitted wall share keeps falling with longer play, so the data do not yet separate walls from a long tail. Either way, a change beats the noise
only if it adds cracks per GPU-second. Five predictions were sealed in advance. Doubling the play time was predicted to clear 127 levels (90% interval 110 to 145) and cleared
141: inside the interval but 14 above the point prediction, about half of the gap traced later to the harness's endgame.
A guard that keeps games on their first level from being shelved was inconclusive (a pass by the letter in one partial run), a token-supply cut went unscored, and the private ARC-AGI-3 and ARC-AGI-2 boards are pending. On ARC-AGI-2, run luck of unchanged code accounts for about half of the rise of a 2,133-team plateau's mean with the number of entries; on 120 public SWE-Bench-Verified runs, the run noise is consistent with independent per-task luck. We give run counts and a two-number comparison that needed about 5 times fewer
runs on our data.
