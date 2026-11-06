<!-- Draft v0, 4 Nov 2026 ~11:55 UTC. Section 3 of the paper. Numbers to be checked against results.json.
     Sources: v5_budget_prediction.py (model and fit), lever3c/3d (level effects, H0-H2), lever4b (two numbers),
     [ARC3] eq. 1-3 (scoring), state/intel/2026-10-07_osint4.md F5 (efficiency cost). -->

## 3. Approach

### 3.1 Setting and data
ARC-AGI-3 on Kaggle hides 110 games. Half of them set the public leaderboard and half the private one. A submission
runs on one RTX Pro 6000 GPU within a 9-hour limit. Our agent starts from dfranzen's public Milestone 2 notebook, which is
built on Tufa Labs' Duck harness (Jeroen Cottaar). We added a ledger of solved levels: the rule the model wrote for each
solved level stays in the prompt for the whole game, even after older history is dropped (X6). In a simple form it resembles the skill library of Voyager [Wang23] and
the stored self-reflections of Reflexion [Shinn23]. We also took the
slot-priority formula from shiiin9's public notebook (Form D1), which decides which games get the 10 generation streams
and favours games that have reached later levels (D'). Together they make the B12 notebook. The language model is
Qwen3.8-Flash-Next (4-bit weights) with a multi-token-prediction draft model, served by SGLang with 10 parallel streams.

We use four kinds of data. (1) Our Kaggle submissions: 17 on ARC-AGI-3 and 10 on ARC-AGI-2 [ARC2], with their leaderboard
scores (Tables 1 and 2). (2) 39 development runs of our agents on the public games (1,022 scheduled game runs, counting games the scheduler never started; Appendix B, whose last five rows are this paper's own test runs from Section 4.4), each with a per-action log of
generated tokens. One of them (f1_forklab) replays 32 saved game states in 256 short forks instead of playing whole
games. In three of these runs (and in three of the five test runs) the stream scheduler never admitted one or two of the 25 games; we call the other runs valid, and
every run-level number uses valid runs only. The standard control is FB25: all 25 games at once, up to 121 minutes per game, 10 streams, one pass.
That gives each game 48.4 stream-minutes, the same as a real submission (10 streams x 532 minutes / 110 games).
(3) Six public runs by other teams, whose logs we parsed the same way. (4) The final public leaderboards. We report the
development runs as measurements of noise and of the model, never as performance claims; performance claims use the
Kaggle leaderboard only.

### 3.2 The crack model
An agent plays a game level by level. On each level it spends generated tokens (thinking and acting) until it finds
what the level wants and clears it. We call the clear a crack and measure the wait T in generated tokens, not in actions
or seconds, because tokens are what the budget buys.

The model has three numbers. With probability q a level is a wall: the agent will not crack it within any budget we can
give it. Otherwise the wait is lognormal, log T ~ Normal(mu, s^2). A run ends when the budget ends, so the last attempt
in each game is right-censored at the tokens it had spent. This is a mixture cure model [Farewell82] in token time. We
fit it by maximum likelihood to the 480 level attempts of the four FB25 control runs (411 clears, 69 open at the end):
q = 0.06, median wait exp(mu) = 22.1k tokens, s = 1.05. These pooled numbers average two populations (Section 4.3): the 10
games we developed on show no walls (q 0.000, median 16.9k), while the 15 held-out games carry all of them (q 0.110,
median 35.7k; q 0.070 on all eight runs). The held-out fit is the better guide to the hidden games, so Section 5 gives
the caps for both.

The model turns a budget into levels. A game with token budget B clears level after level while the summed waits stay
below B. Games share the server, so a game that finishes early hands its share of the streams to the games still
playing. Our simulator shares the tokens of finished games equally among the games still playing, an approximation
of the real scheduler; Appendix D adds a simulation of the harness's slot rule itself.

### 3.3 From levels to score
ARC-AGI-3 scores a completed level by action efficiency against a human baseline h: min(1.15, (h/a)^2) for a actions.
A game's score is the smaller of two weighted averages with weights w_l = l: the share of levels completed and the mean
level score. The total is the mean over games [ARC3, eq. 1-3]. On the 13 valid runs of our main agent family the efficiency term
costs 0.9% to 8.2% of the score compared with completion alone (median 5.0%). The reason is that on the levels it clears the
agent usually needs fewer actions than the human baseline. Here a counts every action logged on the level, resets
included, and h is the human baseline the scorer uses for that level. Over the 1,353 levels these runs cleared on all 25
games, a/h has a median of 0.65 (quartiles 0.44 to 0.93); 75% of the cleared levels sit at the 1.15 cap and 19% took
more actions than the baseline. The completion term therefore decides most game scores: the score is set mostly by how many levels each game clears before its
budget runs out. Later levels weigh more, so the levels that a run reaches only with luck are also the ones that count
most.

### 3.4 Where the noise comes from
The same three numbers drawn fresh for every level and run (model H0) give a run-to-run SD of about 12 levels, far above
the 5.3 we observe across six same-agent FB25 runs. The data say why. Waits are mostly a property of the level: across
runs of one agent, the between-level SD of log wait is 0.80 and the within-level (run luck) SD is 0.49, so 73% of the
variance belongs to the level. If level difficulty is fixed and walls are drawn fresh each run (H1), the SD
is still too high. If both are fixed, so that a wall stays a wall in every run (H2), the simulated SD is 5.6 against
5.3 observed. Each run therefore holds the same loaded tickets: the hard levels and the walls do not move, and the luck
is in how many near-the-edge levels a run happens to clear before its budget ends.

### 3.5 Two numbers per agent
To compare two agents we fit, on the same levels, log T(run r, level l) = a_l + b_r + e with e ~ Normal(0, s^2) and the
open attempts censored (a lognormal accelerated-failure-time model with level effects, fitted by EM; s = 0.55). Each run
then has a crack cost b_r, the log multiplier of tokens needed per level relative to the control, and a token supply
S_r, the tokens it generated. Their combination m_r = (S_r / S_ctl) exp(-b_r) is an effective budget, which we convert to
levels with the control's own budget curve. Because a_l absorbs the level's difficulty, b is far less noisy than a
level count: its per-run standard error is 0.06-0.07.

### 3.6 Sealed predictions
Each prediction was written as a text file with its decision rule, hashed with SHA-256, time-stamped with OpenTimestamps
(anchored in Bitcoin, so anyone can check the time with the proof file), and committed before its outcome was known; the
hashes of both private-leaderboard predictions were also posted on Kaggle (discussion 755123: the ARC-AGI-3 hash in the
post, the ARC-AGI-2 hash in a comment). The scorers (score_sealed.py, v5_readout.py, v10_readout.py, v11_readout.py, score_v9.py) were written before the outcomes. Sealed files
are never edited; any later thought goes into a separate, dated note.

### 3.7 How this work was done

This work was carried out with an AI agent, starting on 30 September 2026. The agent wrote the analysis code,
ran the experiments and drafted the text. The author chose the research questions, approved every GPU run and
leaderboard submission, made the final leaderboard selections, published the notebook and posted the sealed hash. The
repository keeps the written record of the work: a decision log, a ledger of 162 experiment rows (157 agent experiments plus this paper's five test runs), the sealed
predictions with their hashes, and a registry that ties each number in this paper to the script output it comes from.
