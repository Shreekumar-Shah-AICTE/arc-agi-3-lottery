<!-- Draft v0, 4 Nov 2026 ~11:55 UTC. Section 1. [V5] filled 4 Nov ~14:50 UTC.
     Ranks from the final public ARC-AGI-3 board (fig_noise_and_hazard.py; 5,073 teams). -->

## 1. Introduction

ARC-AGI asks whether a system can learn new skills efficiently, from little experience, the way people do [Chollet19].
ARC-AGI-3 makes the test interactive. An agent plays games it has never seen, with no instructions, and is scored on
the levels it completes and on how many actions it needs compared with people [ARC3]. The Paper Prize asks how much a
paper raises the chance of anyone reaching 85% on ARC-AGI [ARCPaper26]; on ARC-AGI-3 the grand prize needs 100%
[ARC3Kaggle]. Progress toward either line is read off leaderboards, so whatever moves a leaderboard decides what the field learns.

Our question started with a resubmission. We submitted the same notebook, byte for byte, five times to the ARC-AGI-3
Kaggle leaderboard. It scored 33.37, 29.69, 28.24, 26.76 and 26.21 (SD 2.87). On the final public board of 5,073 teams (the 2 Nov close as fetched on 4 Nov),
those five draws place 197th, 1,096th, 1,503rd, 1,836th and 1,920th (Fig. 1; ties counted against us). Over the following weeks we ran dozens of
experiments on the agent. Thirty-two ended as killed, noise, failed, parked or invalid (Appendix A), and nothing we tried
after this notebook beat its run-to-run noise (with one leaderboard draw per arm, a gain needs about 11.4 points to be
detected reliably; Section 4.6).

This paper asks what sets an LLM agent's ARC-AGI-3 score under fixed compute, and what follows for progress and for
evaluation. Our answer is a small model. Each level of each game waits for a "crack", the moment the agent finds what
the level wants, and we measure the wait in generated tokens. With probability q the level is a wall that the agent will
not crack within the budget; otherwise the wait is lognormal. Three numbers fitted on four control runs describe the
agent, though they pool two kinds of games: the 10 we developed on show no walls, and the 15 we held out carry all
of them (Section 4.3). Level difficulty that persists from run to run explains why identical runs differ by about 5 levels rather than
12. The token budget then turns the model into a score. What is new is the pairing of a mechanism with sealed
tests: a model of where an LLM agent's leaderboard noise comes from, checked by predictions time-stamped before their
data existed, two of them for private leaderboards that are not yet published.

Contributions:
1. A mechanism (Section 3): the crack model and its link to the official score, with checks against held-out budgets,
   against other teams' runs and against rival waiting-time models without walls, which the data do not yet rule out (Section 4.3).
2. The cause of the noise (Section 4.3): luck in which near-the-edge levels a run clears before its budget ends, on top of
   level difficulty and walls that do not change between runs. About half of the per-game noise is whether a game clears
   its first level before the harness's slot rule shelves it; in simulation that rule moves noise onto first levels
   rather than adding it (the crack waits alone carry 90% of the run-to-run variance).
3. Sealed predictions (Section 4.4): for the private ARC-AGI-3 and ARC-AGI-2 leaderboards (pending until the hosts
   publish them); for doubling the play time (141 levels against 127 predicted: inside the 90% interval but near its top;
   in a simulation made after the result, about half of the excess comes from an endgame rule in the harness that shaped
   the control runs the sealed calculation continued); for a cut in token supply (unscored, because both slower runs left
   games unplayed); and for a guard that keeps games on level 1 from being shelved (inconclusive: by the letter it passed in one partial run, with 1
   zero-level game among the 23 that started, against 1 to 4 in seven reference runs, but 2 games never started).
4. Tools for anyone comparing agents (Sections 4.5-4.6): two numbers per agent that needed about 5 times fewer runs than
   raw level totals on our data (at least 2.8 times), run counts for leaderboard comparisons, and our 32 negative results
   sorted by what each change did to the run.
5. A similar pattern on ARC-AGI-2 (Section 4.8): 2,133 teams share a plateau whose mean score rises with their number of
   entries; run luck of unchanged code alone accounts for about half of the rise (52%; 70% with the SD of all seven of our draws, which pools two notebooks).
   On the 120 public SWE-Bench-Verified runs of Bjarnason et al., per-task luck likewise accounts for the run-to-run noise
   (86% pooled; exploratory).

Why this matters for the 85% goal: if most differences between similar agents are luck, a team that tunes against the
leaderboard learns noise, and a field that reads leaderboards learns less than it thinks. The model says which kind of
change can beat the noise (more cracks per GPU-second, fewer walls) and how many runs it takes to know. The ARC Prize
2024 report describes the paper awards as meant "to reward novel concepts regardless of how their solutions scored"
[ARC24]. Our own score is modest (best draw 33.37); our aim is to explain why scores like ours move the way they do.
