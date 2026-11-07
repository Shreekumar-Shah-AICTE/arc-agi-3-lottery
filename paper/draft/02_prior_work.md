<!-- Draft v0, 4 Nov 2026 ~11:50 UTC; 6 Nov ~18:50 UTC added Dwork15, Kwa25, ARCo3, Bjarnason26, Wiedmann26, Hardy26, ARCAstra26 and OpenAI26 (sources checked on the web the same evening). Section 2 of the paper. Citation keys as in paper/REFERENCES.md.
     Notes for the check pass: (a) Roelofs19 and Bouthillier21 claims re-checked 4 Nov ~12:27 UTC against the official abstract pages
     (papers.nips.cc paper 9117; arxiv.org/abs/2103.03098): both match; Roelofs's finding (little evidence of substantial
     overfitting) is now stated. The earlier flagged web result is not relied on;
     (b) every number must match results.json.
     7 Nov 2026: tightened from about 1,580 to about 1,100 words; every citation kept; small paragraphs merged. -->

## 2. Prior work

**Noise in machine-learning evaluation.** Run-to-run variation is often as large as the differences that papers report.
In deep reinforcement learning the random seed alone can change a conclusion [Henderson18], which led to power
calculations for the number of seeds [Colas18] and to interval estimates for the few-run regime [Agarwal21]. Benchmark
variance has known sources, such as data sampling, initialization and hyperparameter choice [Bouthillier21]. For
language models, seed variance has been measured across common benchmarks [Madaan24] and tied to a benchmark's
signal-to-noise ratio [Heineman25]. Authors are asked to report the expected best score as a function of the number of
trials [Dodge19], and to use clustered standard errors and question-level paired comparisons [Miller24]. For agents,
single-run pass@1 on 60,000 SWE-Bench-Verified trajectories shifts by 2.2 to 6.0 points with the run chosen
[Bjarnason26]; rerunning the same configuration explains about 54% of the outcome variance on scientific-agent tasks
[Wiedmann26]; and leaderboard variance splits into signal and noise across 22 agent benchmarks [Hardy26].

We share this advice: report the spread and the number of runs. We differ in three ways. First, we give a mechanism, not
only a measurement: a waiting-time model with three fitted numbers predicts how large the run-to-run noise is and how it
changes with the budget. Second, we test it with predictions sealed before their data existed. Third, the setting is a
hidden-test competition. A team sees one noisy draw per day and cannot inspect the test, so the noise is part of the
score, not a reporting choice.

**Leaderboards and resubmission.** A public leaderboard reports a team's best entry, so it computes Dodge et al.'s
quantity with n equal to the team's number of entries. Across over one hundred Kaggle competitions, public and final
rankings show little evidence of substantial overfitting from test-set reuse [Roelofs19]. A leaderboard can limit what
repeated submission learns [BlumHardt15], and a holdout set can be reused across many adaptive analyses without
overfitting to it [Dwork15]. On Chatbot Arena, providers who test many private variants and publish only the best one
gain rank through that selection [Singh25]. Our case differs in kind. The noise sits in the run, not in the split, so
selecting the best public entry selects luck that will not carry over to the private half. On ARC-AGI-2, the plateau of
2,133 teams rises with the number of entries much as the best of n draws of one notebook would (Section 4.8).

**Repeated sampling and test-time compute.** Coverage grows steadily with the number of samples [Brown24], and the power
law of that growth traces to a heavy-tailed distribution of per-problem success rates [Schaeffer25]. The crack model
works the same way in token time, with fixed level difficulty in place of problem difficulty. A fixed test-time budget
can be spent well or badly [Snell24], resampling against an imperfect verifier has limits [Stroebl24], and pass^k
exposes the gap between an agent's best and its reliable behaviour [Yao24]. These works treat sampling as part of the
method. On Kaggle, two kinds of resampling happen outside it. Inside a run, more generated tokens act like extra samples
(Section 4.4 doubles the play time); across runs, the leaderboard resamples whole submissions. The crack model makes the
first explicit, and the best-of-n view the second.

**Waiting-time models.** The closest ancestor of the crack model is the run-time distribution of a Las Vegas algorithm,
by which stochastic search can be evaluated [HoosStutzle98]. Randomized backtracking search has heavy-tailed solve
times that random restarts remove [Gomes00], and a restart schedule exists within a logarithmic factor of the best one
when the distribution is unknown [Luby93]; we did not test whether restarting a long attempt on a level helps an LLM
agent. Running several randomized solvers side by side, or splitting one processor among them, is an algorithm portfolio
[Huberman97] [GomesSelman01], and SATzilla picks a solver per instance from predicted log run times, keeping timed-out runs
as censored lower bounds [Xu08]. Our scheduler simulations (Section 4.7) ask the portfolio question for games that share
one token budget; the Gittins index [Gittins79] is the classical rule for splitting effort among independent projects, and
we did not test it. Our wall share is the cured fraction of a mixture cure model, in which some units never have the event during
follow-up [BerksonGage52] [Farewell82]; we apply it in generated-token time, with censoring at the end of each run.
Telling a cured unit from a late event is a known difficulty of these models [MallerZhou96]. For us it is the question
of whether a wall is a very slow crack (Section 4.3).

**Comparing agents with fewer runs.** Agents must be compared at equal cost [KapoorAgents24]. We describe each agent by
two numbers: how many tokens it can generate in the budget (supply) and how many it needs per level (crack cost).
Comparing crack cost on the same levels is a blocked design, in which level effects cancel as question effects do in
paired comparisons of evals [Miller24], much as item response models let tinyBenchmarks evaluate language models on far
fewer items [Polo24]: most of the variance in crack times belongs to the level, not to the run (Section 3.4). The estimator itself is standard: a lognormal accelerated-failure-time regression with
level effects [KalbfleischPrentice80], fitted by EM over the censored attempts [Dempster77]. It needed
several times fewer runs than raw level counts (about 5 times on our data, at least 2.8; Section 4.6). METR summarizes an
agent by the task length, in human working time, that it completes half the time [Kwa25]. Our crack cost is the
agent-side counterpart, and its curve also shows where more budget stops buying (the walls).

**Preregistration.** Predictions time-stamped before the data exist separate tests from post-hoc stories [Nosek18].
Preregistration is not new to machine learning (a NeurIPS 2020 workshop was devoted to it [PreregML20]). We sealed ours
with SHA-256 hashes, OpenTimestamps proofs anchored in Bitcoin, and a public Kaggle post; what we add is its use on a
live competition's hidden leaderboard.

**ARC.** Chollet defines intelligence as skill-acquisition efficiency [Chollet19]. ARC-AGI-3 makes the test interactive
and scores the actions an agent needs against a human baseline, level by level [ARC3]. The hosts qualified every game so
that random play cannot beat its non-tutorial levels: random sweeps of one million steps, and a probability of a random
win below 1 in 10,000 per level [ARC3, Section 3.5.1]. The luck we measure is therefore not random play but the question
of when a capable agent's search finds a level's key. Several ARC methods search over samples: Greenblatt sampled about
8,000 programs per task and gained about 3 points of accuracy per doubling of samples [Greenblatt24], and
product-of-experts scoring [Franzen25] and self-improving evolutionary program search [SOAR] also rank sampled
candidates. Others change the learner, with induction and transduction models trained on synthetic tasks [Li24] or
test-time training [Akyurek24]. Compute alone also moves ARC scores: on ARC-AGI-1, OpenAI's o3 preview scored 75.7% of
the semi-private tasks within the leaderboard's compute limit and 87.5% with 172 times the compute [ARCo3]. On ARC-AGI-3,
frontier systems outside Kaggle's limits score far higher: GPT-6 Astra reached 62.7% of the semi-private set with the
hosts' standard harness and 99.9% with a harness that keeps the model's reasoning state between requests [ARCAstra26].
The hosts report a 'binary-like pattern': once a frontier model understands a game's mechanics, it plays within the range
of human action efficiency [ARCAstra26]. That matches the crack model's premise that a level is a wait for one insight,
followed by play. Where those papers improve the search or the learner, we ask what a fixed compute budget does to the
score of any such search, and what that implies for reading leaderboards. Competitors already knew that resubmitting
helps: a forum post describes a team turning identical code into 38.85 over about 100 submissions (Kaggle discussion
755320). We put a number on it: for our notebook, the best of 100 identical submissions is expected to score 7.2 points
above a single one (Section 4.6).
