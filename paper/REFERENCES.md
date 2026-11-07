# References for the paper (verified 4 Nov 2026 by web lookup unless marked TO VERIFY)

Format: short key, full citation, where verified. Used in "Prior work" (outline v2, section 2).

## Variance and reporting in ML evaluation
- [Bouthillier21] X. Bouthillier et al. "Accounting for Variance in Machine Learning Benchmarks." MLSys 2021. arXiv 2103.03098 (abstract checked 4 Nov 2026: variance from data sampling,
  initialization and hyperparameter choice).
- [Henderson18] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, D. Meger. "Deep Reinforcement Learning That Matters."
  AAAI 2018 (vol. 32). arXiv 1709.06560. (ojs.aaai.org/index.php/aaai/article/view/11694)
- [Colas18] C. Colas, O. Sigaud, P.-Y. Oudeyer. "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments." arXiv 1806.08295.
- [Agarwal21] R. Agarwal et al. "Deep Reinforcement Learning at the Edge of the Statistical Precipice." NeurIPS 2021. arXiv 2108.13264.
- [Dodge19] J. Dodge et al. "Show Your Work: Improved Reporting of Experimental Results." EMNLP-IJCNLP 2019, pp. 2185-2194
  (aclanthology.org/D19-1224). Expected best validation score as a function of the number of trials.
- [Madaan24] L. Madaan et al. "Quantifying Variance in Evaluation Benchmarks." arXiv 2406.10229.
- [Heineman25] D. Heineman, V. Hofmann, I. Magnusson, Y. Gu, N. A. Smith, H. Hajishirzi, K. Lo, J. Dodge. "Signal and Noise: A Framework
  for Reducing Uncertainty in Language Model Evaluation." arXiv 2508.13144 (Aug 2025).
- [Miller24] E. Miller. "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations." arXiv 2411.00640
  (Nov 2024; checked 5 Nov 2026: recommends clustered standard errors and question-level paired differences).
- [Bjarnason26] B. H. Bjarnason, A. Silva, M. Monperrus. "On Randomness in Agentic Evals." arXiv 2602.07150 (2026; abstract checked
  6 Nov 2026: 60,000 SWE-Bench-Verified trajectories; single-run pass@1 varies by 2.2 to 6.0 points with the run chosen).
- [Wiedmann26] L. Wiedmann, L. Girrbach, C. Schmid, Z. Akata. "Agents Are Systems, Not Models: Rethinking Agentic Evaluation."
  arXiv 2610.01618 (2026; abstract checked 6 Nov 2026: about 54% of outcome variance from repeating the same configuration).
- [Hardy26] M. Hardy, R. Azam, A. Reuel, M. Kochenderfer, S. Koyejo. "Agent Evaluation Reliability: More Tasks Won't (Always)
  Fix An Agent Leaderboard." arXiv 2610.00651 (2026; abstract checked 6 Nov 2026: Bayesian variance decomposition, 22 agent benchmarks).
- [Polo24] F. Maia Polo, L. Weber, L. Choshen, Y. Sun, G. Xu, M. Yurochkin. "tinyBenchmarks: evaluating LLMs with fewer examples."
  arXiv 2402.14992 (ICML 2024).
- [Kwa25] T. Kwa, B. West, J. Becker et al. "Measuring AI Ability to Complete Long Software Tasks." arXiv 2503.14499 (2025;
  NeurIPS 2025; checked 6 Nov 2026: the 50% time horizon, in human working time, of tasks an agent completes).

## Leaderboards, resubmission and selection
- [Roelofs19] R. Roelofs et al. "A Meta-Analysis of Overfitting in Machine Learning." NeurIPS 2019. Over 100 Kaggle competitions, public vs final ranking: little evidence of substantial overfitting
  (abstract checked 4 Nov 2026, papers.nips.cc paper 9117).
- [BlumHardt15] A. Blum, M. Hardt. "The Ladder: A Reliable Leaderboard for Machine Learning Competitions." ICML 2015, PMLR 37.
- [Dwork15] C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, A. Roth. "The reusable holdout: Preserving validity in adaptive
  data analysis." Science 349(6248), 2015. doi 10.1126/science.aaa9375 (checked 6 Nov 2026; STOC 2015 version arXiv 1411.2664).
- [Singh25] S. Singh, Y. Nan, A. Wang, D. D'Souza, S. Kapoor, A. Üstün, S. Koyejo, Y. Deng, S. Longpre, N. A. Smith et al.
  "The Leaderboard Illusion." arXiv 2504.20879 (2025; checked 5 Nov 2026: undisclosed private testing of many variants on
  Chatbot Arena and selective disclosure of the best scores bias the leaderboard).

## Repeated sampling and test-time compute
- [Brown24] B. Brown et al. "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling." arXiv 2407.21787.
- [Snell24] C. Snell, J. Lee, K. Xu, A. Kumar. "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model
  Parameters." arXiv 2408.03314.
- [Stroebl24] B. Stroebl, S. Kapoor, A. Narayanan. "The Limits of Inference Scaling Through Resampling." arXiv 2411.17501 (2024;
  authors corrected 5 Nov 2026 from the arXiv page).
- [Schaeffer25] R. Schaeffer, J. Kazdan, J. Hughes, J. Juravsky, S. Price, A. Lynch, E. Jones, R. Kirk, A. Mirhoseini, S. Koyejo.
  "How Do Large Language Monkeys Get Their Power (Laws)?" ICML 2025 (PMLR 267). arXiv 2502.17578 (checked 5 Nov 2026:
  aggregate power laws from a heavy-tailed distribution of per-problem success probabilities).
- [Yao24] S. Yao et al. "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." arXiv 2406.12045 (pass^k).
- [KapoorAgents24] S. Kapoor et al. "AI Agents That Matter." arXiv 2407.01502.

## Run-time distributions of randomized search (added 5 Nov 2026 after the manual reviews)
- [HoosStutzle98] H. H. Hoos, T. Stützle. "Evaluating Las Vegas Algorithms - Pitfalls and Remedies." UAI 1998. arXiv 1301.7383
  (abstract checked 5 Nov 2026: evaluation by empirical run-time distributions).
- [Gomes00] C. P. Gomes, B. Selman, N. Crato, H. Kautz. "Heavy-Tailed Phenomena in Satisfiability and Constraint Satisfaction
  Problems." Journal of Automated Reasoning 24:67-100, 2000. doi 10.1023/A:1006314320276 (random restarts eliminate heavy tails).
- [Luby93] M. Luby, A. Sinclair, D. Zuckerman. "Optimal speedup of Las Vegas algorithms." Information Processing Letters
  47(4):173-180, 1993. doi 10.1016/0020-0190(93)90029-9.
- [Huberman97] B. A. Huberman, R. M. Lukose, T. Hogg. "An Economics Approach to Hard Computational Problems." Science
  275(5296):51-54, 1997. doi 10.1126/science.275.5296.51 (checked 7 Nov 2026: algorithm portfolios built from notions of risk).
- [GomesSelman01] C. P. Gomes, B. Selman. "Algorithm portfolios." Artificial Intelligence 126(1-2):43-62, 2001.
  doi 10.1016/S0004-3702(00)00081-3 (checked 7 Nov 2026).
- [Xu08] L. Xu, F. Hutter, H. H. Hoos, K. Leyton-Brown. "SATzilla: Portfolio-based Algorithm Selection for SAT." Journal of
  Artificial Intelligence Research 32:565-606, 2008. doi 10.1613/jair.2490 (checked 7 Nov 2026: per-instance log-runtime
  models, with timed-out runs kept as censored lower bounds by the method of Schmee and Hahn).
- [Gittins79] J. C. Gittins. "Bandit Processes and Dynamic Allocation Indices." Journal of the Royal Statistical Society,
  Series B 41(2):148-177, 1979 (with discussion). doi 10.1111/j.2517-6161.1979.tb01068.x (checked 7 Nov 2026).

## Survival analysis and preregistration
- [Farewell82] V. T. Farewell. "The Use of Mixture Models for the Analysis of Survival Data with Long-Term Survivors."
  Biometrics 38(4):1041-1046, 1982. doi 10.2307/2529885. (Our "wall" share = the cured / never-event fraction.)
- [BerksonGage52] J. Berkson, R. P. Gage. "Survival curve for cancer patients following treatment." Journal of the American Statistical Association 47(259): 501-515, 1952. doi:10.1080/01621459.1952.10501187
- [MallerZhou96] R. Maller, X. Zhou. Survival Analysis with Long-Term Survivors. Wiley, Chichester, 1996 (271 pp.).
- [KalbfleischPrentice80] J. D. Kalbfleisch, R. L. Prentice. The Statistical Analysis of Failure Time Data. Wiley, New York,
  1980 (2nd ed. 2002, doi 10.1002/9781118032985, checked 7 Nov 2026). Accelerated-failure-time models.
- [Dempster77] A. P. Dempster, N. M. Laird, D. B. Rubin. "Maximum Likelihood from Incomplete Data via the EM Algorithm."
  Journal of the Royal Statistical Society, Series B 39(1):1-38, 1977 (with discussion). doi 10.1111/j.2517-6161.1977.tb01600.x
  (checked 7 Nov 2026).
- [PreregML20] L. Bertinetto, J. F. Henriques, S. Albanie, M. Paganini, G. Varol (eds.). NeurIPS 2020 Workshop on
  Pre-registration in Machine Learning. PMLR 148, 2021 (proceedings.mlr.press/v148, checked 5 Nov 2026).
- [Nosek18] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor. "The preregistration revolution." PNAS 115(11):2600-2606, 2018.
  doi 10.1073/pnas.1708274114.

## ARC (texts in /data/paperdata/panel, see urls.txt)
- [Chollet19] F. Chollet. "On the Measure of Intelligence." arXiv 1911.01547.
- [ARC24] F. Chollet, M. Knoop, G. Kamradt, B. Landers. "ARC Prize 2024: Technical Report." arXiv 2412.04604.
- [ARC2] F. Chollet, M. Knoop, G. Kamradt, B. Landers, H. Pinkard. "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems."
  arXiv 2505.11831.
- [ARC3] ARC Prize Foundation. "ARC-AGI-3 Technical Report" (arcprize.org/media/ARC_AGI_3_Technical_Report.pdf). Section 3.5.1:
  games qualified so random play cannot beat non-tutorial levels (1M-step random sweeps; P_win < 1/10,000 per level);
  scoring compares AI to an upper-median human baseline per level, then aggregates per game.
- [ARC3Kaggle] Kaggle. "ARC Prize 2026 - ARC-AGI-3", Prizes (kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/prizes),
  read 4 Nov 2026: the Grand Prize is unlocked if a team achieves a score of 100% on the competition leaderboard.
- [ARCPaper26] ARC Prize Foundation. "ARC Prize 2026 Paper Prize" (arcprize.org/competitions/2026/paper), read 4 Nov 2026:
  Progress asks "how much does the paper increase the overall chance of anyone achieving 85% on ARC-AGI?"
- [Franzen25] D. Franzen, J. Disselhoff, D. Hartmann. "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective." arXiv 2505.07859.
- [SOAR] J. Pourcel, C. Colas, P.-Y. Oudeyer. "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on
  ARC-AGI" (SOAR). openreview z4IG090qt2.
- [ARCo3] F. Chollet. "OpenAI o3 Breakthrough High Score on ARC-AGI-Pub." ARC Prize blog, 20 Dec 2024 (arcprize.org/blog/oai-o3-pub-breakthrough;
  read 6 Nov 2026: 75.7% on the semi-private set within the $10k compute limit, 87.5% at 172 times the compute).
- [ARCAstra26] ARC Prize Foundation. "OpenAI's GPT-6 Astra on ARC-AGI-3." ARC Prize blog, 2026 (arcprize.org/blog/astra; read
  6 Nov 2026: 62.7% on the semi-private set with the Standard harness, 99.9% with the Provider Adapter harness; once frontier AI
  understands the mechanics it generally executes within the range of human efficiency, a binary-like pattern).
- [OpenAI26] OpenAI. "How enabling two settings tripled our scores on the ARC-AGI-3 benchmark." 29 July 2026
  (openai.com/index/how-two-settings-tripled-our-arc-agi-3-scores; read 6 Nov 2026: GPT-5.6 Sol 13.3% to 38.3% on the public
  set with retained reasoning and compaction, 6x fewer output tokens).
- [Greenblatt24] R. Greenblatt. "Getting 50% (SoTA) on ARC-AGI with GPT-4o." Redwood Research blog, 17 June 2024
  (blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt; read 5 Nov 2026: about 8,000 sampled programs per task,
  "an additional 3% correct per doubling of k").
- [Li24] W.-D. Li, K. Hu, C. Larsen, Y. Wu, S. Alford, C. Woo, S. M. Dunn, H. Tang, M. Naim, D. Nguyen, W.-L. Zheng, Z. Tavares,
  Y. Pu, K. Ellis. "Combining Induction and Transduction for Abstract Reasoning." arXiv 2411.02272 (2024).
- [Akyurek24] E. Akyürek, M. Damani, A. Zweiger, L. Qiu, H. Guo, J. Pari, Y. Kim, J. Andreas. "The Surprising Effectiveness of
  Test-Time Training for Few-Shot Learning." arXiv 2411.07279 (2024; title as of v2).

## Agents and memory
- [Wang23] G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, A. Anandkumar. "Voyager: An Open-Ended Embodied Agent
  with Large Language Models." arXiv 2305.16291 (2023; a skill library of programs).
- [Shinn23] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, S. Yao. "Reflexion: Language Agents with Verbal Reinforcement
  Learning." NeurIPS 2023.
- [Zhao24] A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, G. Huang. "ExpeL: LLM Agents Are Experiential Learners." AAAI 2024,
  38(17):19632-19642. doi 10.1609/aaai.v38i17.29936; arXiv 2308.10144 (checked 7 Nov 2026).
- [Wang24] Z. Z. Wang, J. Mao, D. Fried, G. Neubig. "Agent Workflow Memory." arXiv 2409.07429 (2024; checked 7 Nov 2026).
