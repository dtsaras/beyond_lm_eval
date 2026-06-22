# BLME Paper Index

**The single source of truth for every paper we cite or considered.**
Keep this file updated whenever a task is added, a citation added to
source, or a new paper surveyed.

Sections:
1. [Implemented — 70+ papers](#1-implemented-papers)
2. [Considered and skipped](#2-considered-and-skipped)
3. [Per-task citation audit](#3-per-task-citation-audit)
4. [Update procedure](#4-update-procedure)

For the higher-level narrative (survey methodology, selection
criteria, experimental-correlation annex, top-predictor results,
paper-ready related-work section, and reference-implementation repo
index) see the companion docs:
- `PAPER_SURVEY.md` — inclusion/exclusion survey narrative
- `RELATED_WORK.md` — paper-ready §2 in thematic threads
- `CORRELATION_LITERATURE.md` — experimental-correlation papers
- `TOP_PREDICTORS.md` — main experimental result
- **`REPOSITORIES.md`** — GitHub reference-implementation URLs for
  every cited paper (66 papers mapped, added 2026-04-20)

---

## 1. Implemented papers

Each row: paper → task(s) it underpins → arXiv ID → file path where
it's cited. Sorted alphabetically by first author.

### Geometry — representation + weight

| Paper | Task(s) | Identifier | Source file |
|---|---|---|---|
| **Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda 2024** — *Refusal in Language Models Is Mediated by a Single Direction* | `repe_refusal_direction` | arXiv:2406.11717 | `representation_engineering.py` |
| **Baik, Ben Arous, Péché 2005** — *Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices* (BBP transition) | `geometry_mp_bulk_deviation.mp_outlier_frac`, `.mp_spike_energy` | arXiv:math/0403022 (*Ann. Probab.* 33(5), 1643–1697) | `geometry/rmt_bulk.py` |
| **Bubenik 2015** — *Statistical Topological Data Analysis using Persistence Landscapes* | `topology_persistence_landscape` | arXiv:1207.6437 (JMLR 16:77-102) | `topology/persistence_landscape.py` |
| **Bubenik, Dlotko 2017** — *A persistence landscapes toolbox for topological statistics* | `topology_persistence_landscape` implementation reference | arXiv:1501.00179 | `topology/persistence_landscape.py` |
| **Caliskan, Bryson, Narayanan 2017** — *Semantics derived automatically from language corpora contain human-like biases* | `consistency_bias_weat` | *Science* 356(6334) | `consistency/bias.py` |
| **Carlini, Tramer, Wallace, Jagielski, Herbert-Voss, Lee, Roberts, Brown, Song, Erlingsson, Oprea, Raffel 2021** — *Extracting Training Data from Large Language Models* | `consistency_membership_inference` | arXiv:2012.07805 | `consistency/membership_inference.py` |
| **Clark, Khandelwal, Levy, Manning 2019** — *What Does BERT Look At? An Analysis of BERT's Attention* | `interpretability_attention_entropy`, `interpretability_head_roles` | EMNLP BlackBoxNLP 2019 | `interpretability/attention.py`, `head_roles.py` |
| **Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso 2023** — *Towards Automated Circuit Discovery for Mechanistic Interpretability* | `causality_circuit_quality` | arXiv:2304.14997 | `causality/circuit_quality.py` |
| **Dai, Dong, Hao, Sui, Chang, Wei 2022** — *Knowledge Neurons in Pretrained Transformers* | `causality_knowledge_neurons` | arXiv:2104.08696 | `causality/knowledge_neurons.py` |
| **Dong, Cordonnier, Loukas 2021** — *Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth* | `interpretability_attention_rank` | arXiv:2103.03404 | `interpretability/attention_rank.py` |
| **Elhage, Hume, Olsson, Schiefer, Henighan, Kravec, Hatfield-Dodds, Lasenby, Drain, Chen, Grosse, McCandlish, Kaplan, Amodei, Wattenberg, Olah 2022** — *Toy Models of Superposition* | `interpretability_superposition` | Transformer Circuits 2022 | `interpretability/superposition.py` |
| **Ethayarajh 2019** — *How Contextual are Contextualized Word Representations?* | `geometry_contextualization` | arXiv:1909.00512 | `geometry/contextualization.py` |
| **Facco, d'Errico, Rodriguez, Laio 2017** — *Estimating the intrinsic dimension of datasets by a minimal neighborhood information* | `geometry_intrinsic_dim` | *Sci. Rep.* 7, 12140 | `geometry/intrinsic_dim.py` |
| **Foret, Kleiner, Mobahi, Neyshabur 2021** — *Sharpness-Aware Minimization for Efficiently Improving Generalization* | `dynamics_sharpness` | arXiv:2010.01412 | `dynamics/sharpness.py` |
| **Garrido, Balestriero, Najman, LeCun 2023** — *RankMe: Assessing the Downstream Performance of Pretrained SSL Representations by their Rank* | `geometry_schatten.rankme` | arXiv:2210.02885 | `geometry/schatten.py` |
| **Grassberger, Procaccia 1983** — *Characterization of strange attractors* | `geometry_correlation_dimension` | *Phys. Rev. Lett.* 50(5) | `geometry/correlation_dimension.py` |
| **Gretton, Bousquet, Smola, Schölkopf 2005** — *Measuring Statistical Dependence with Hilbert-Schmidt Norms* | `geometry_hsic` | ALT 2005 | `geometry/mutual_info.py` |
| **Gu, Pang, Du, Liu, Zhang, Du, Wang, Lin 2025** — *When Attention Sink Emerges in Language Models: An Empirical View* (ICLR 2025 Spotlight) | `interpretability_activation_sinks.sink_epsilon_fraction` | arXiv:2410.10781 | `interpretability/activation_sinks.py` |
| **Guo, Pleiss, Sun, Weinberger 2017** — *On Calibration of Modern Neural Networks* | `consistency_calibration` | ICML 2017 | `consistency/calibration.py` |
| **Holtzman, Buys, Du, Forbes, Choi 2020** — *The Curious Case of Neural Text Degeneration* | `interpretability_prediction_entropy` | arXiv:1904.09751 | `interpretability/prediction_entropy.py` |
| **Hosseini, Fedorenko 2023** — *Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language* | `geometry_trajectory_curvature` | arXiv:2311.04930 (NeurIPS 2023) | `geometry/trajectory_curvature.py` |
| **Ilharco, Ribeiro, Wortsman, Schmidt, Hajishirzi, Farhadi 2023** — *Editing Models with Task Arithmetic* | `repe_task_vectors` | ICLR 2023 | `representation_engineering.py` |
| **Kornblith, Norouzi, Lee, Hinton 2019** — *Similarity of Neural Network Representations Revisited* | `geometry_cka`, `geometry_hsic` | arXiv:1905.00414 | `geometry/cka.py`, `geometry/mutual_info.py` |
| **Kriegeskorte, Mur, Bandettini 2008** — *Representational Similarity Analysis* | `geometry_rsa` | *Front. Syst. Neurosci.* 2 | `geometry/rsa.py` |
| **Lee, Lee, Lee, Shin 2018** — *A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks* | `geometry_mahalanobis` | NeurIPS 2018 | `geometry/mahalanobis.py` |
| **Levina, Bickel 2004** — *Maximum Likelihood Estimation of Intrinsic Dimension* | `geometry_lid` | NeurIPS 2004 | `geometry/lid.py` |
| **Li, Galley, Brockett, Gao, Dolan 2016** — *A Diversity-Promoting Objective Function* (Distinct-n) | `dynamics_generation_diversity` | NAACL 2016 | `dynamics/generation_diversity.py` |
| **Li, Xia, Chang, Wu 2024** — *Large Language Model Evaluation via Matrix Nuclear-Norm* | `geometry_schatten.row_normalized_matrix_nuclear_norm` | arXiv:2410.10672 | `geometry/schatten.py` |
| **Li, Lizhi, Sondak, Wang 2025** — *Tracing the Representation Geometry of Language Models from Pretraining to Post-training* | metrics already implemented via `geometry_schatten.rankme` + `geometry_spectral.avg_alpha` | arXiv:2509.23024 | `geometry/schatten.py` (comment) |
| **Liu, Lin, Hu, Manning, Liang 2023** — *Lost in the Middle: How Language Models Use Long Contexts* | `consistency_position_sensitivity` | arXiv:2307.03172 | `consistency/position_sensitivity.py` |
| **Ma, Li, Wang, Erfani, Wijewickrema, Schoenebeck, Song, Houle, Bailey 2018** — *Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality* | `geometry_lid` | ICLR 2018 | `geometry/lid.py` |
| **Marchenko, Pastur 1967** — *Distribution of eigenvalues for some sets of random matrices* | `geometry_mp_bulk_deviation` | *Mat. Sb.* 72(114):4, 507–536; English transl. *Math. USSR-Sb.* 1(4), 457–483 | `geometry/rmt_bulk.py` |
| **Martin, Mahoney 2019a** — *Traditional and Heavy-Tailed Self Regularization in Neural Network Models* | `geometry_spectral.avg_alpha` | arXiv:1901.08276 | `geometry/spectral.py` |
| **Martin, Mahoney 2019b / 2021** — *Heavy-Tailed Universality Predicts Trends in Test Accuracies* (Nature Comms. 2021) | `geometry_spectral.avg_alpha` | arXiv:1901.08278, Nat. Comms. 12:4122 | `geometry/spectral.py` |
| **May, Wang, Bordia, Bowman, Rudinger 2019** — *On Measuring Social Biases in Sentence Encoders* (SEAT) | `consistency_bias_weat` | arXiv:1903.10561 | `consistency/bias.py` |
| **Meng, Bau, Andonian, Belinkov 2022** — *Locating and Editing Factual Associations in GPT* (ROME) | `causality_tracing` | arXiv:2202.05262 | `causality/tracing.py` |
| **Michel, Levy, Neubig 2019** — *Are Sixteen Heads Really Better than One?* | `causality_attention_knockout` | NeurIPS 2019 | `causality/attention_knockout.py` |
| **Naitzat, Zhitnikov, Lim 2020** — *Topology of Deep Neural Networks* | `topology_betti_curve` | JMLR 21(184):1-40, 2020 | `topology/betti_curve.py` |
| **nostalgebraist 2020** — *interpreting GPT: the logit lens* | `interpretability_logit_lens` | LessWrong 2020 | `interpretability/logit_lens.py` |
| **Olsson, Elhage, Nanda, Joseph, DasSarma, Henighan, Mann, Askell, Bai, Chen, Conerly, Drain, Ganguli, Hatfield-Dodds, Hernandez, Johnston, Jones, Kernion, Lovitt, Ndousse, Amodei, Brown, Clark, Kaplan, McCandlish, Olah 2022** — *In-context Learning and Induction Heads* | `interpretability_induction_heads`, `interpretability_head_roles` | Transformer Circuits 2022 | `interpretability/induction.py`, `head_roles.py` |
| **Papyan, Han, Donoho 2020** — *Prevalence of neural collapse during the terminal phase of deep learning training* | `geometry_neural_collapse` | arXiv:2008.08186 (PNAS 117) | `geometry/neural_collapse.py` |
| **Pascanu, Mikolov, Bengio 2013** — *On the difficulty of training recurrent neural networks* | `dynamics_gradient_flow` | ICML 2013 | `dynamics/gradient_flow.py` |
| **Arroyo, Barbero, Dong, Bronstein, LeCun, Shwartz-Ziv 2025** — *Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin* | `interpretability_activation_sinks.valley_*` | arXiv:2510.06477 | `interpretability/activation_sinks.py` |
| **Roy, Vetterli 2007** — *The Effective Rank: A Measure of Effective Dimensionality* | `geometry_svd.effective_rank`, `geometry/utils.effective_rank` | EUSIPCO 2007 | `geometry/utils.py`, `collapse.py` |
| **Rudman, Gillman, Rayne, Eickhoff 2022** — *IsoScore: Measuring the Uniformity of Embedding Space Utilization* | `geometry_isoscore` | arXiv:2108.07344 (Findings of ACL 2022) | `geometry/isotropy.py` |
| **Sclar, Choi, Tsvetkov, Suhr 2023** — *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design* | `consistency_format_robustness` | arXiv:2310.11324 | `consistency/format_robustness.py` |
| **Shi, Ajith, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer 2023** — *Detecting Pretraining Data from Large Language Models* (Min-K % Prob) | `consistency_contamination` | arXiv:2310.16789 | `consistency/contamination.py` |
| **Sun, Chen, Kolter, Liu 2024** — *Massive Activations in Large Language Models* | `interpretability_activation_sinks.massive_activation_*` | arXiv:2402.17762 | `interpretability/activation_sinks.py` |
| **Syed, Rager, Conmy 2024** — *Attribution Patching Outperforms Automated Circuit Discovery* (EAP) | `causality_edge_attribution` | arXiv:2310.10348 (BlackboxNLP 2024) | `causality/edge_attribution.py` |
| **Thilak, Maddox 2021** — *The Low-Rank Simplicity Bias in Deep Networks* | `geometry_collapse` (effective rank context) | arXiv:2011.09348 | `geometry/collapse.py` |
| **Tomašev, Radovanović, Mladenić, Ivanović 2014** — *The Role of Hubness in High-Dimensional Data* | `geometry_hubness` | IEEE TKDE 2014 | `geometry/hubness.py` |
| **Voita, Talbot, Moiseev, Sennrich, Titov 2019** — *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting* | `causality_attention_knockout`, `interpretability_head_roles` | ACL 2019 | `causality/attention_knockout.py`, `head_roles.py` |
| **Wang, Wei, Schuurmans, Le, Chi, Zhou 2022** — *Self-Consistency Improves Chain of Thought Reasoning* | `consistency_self_consistency` | arXiv:2203.11171 | `consistency/self_consistency.py` |
| **Wang, Zhang, Yang, Wong, Wang 2025** — *Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation* (ICLR 2025) | `dynamics_coe` | arXiv:2410.13640 | `dynamics/coe.py` |
| **Wei, Tan, Li, Wang, Huang 2024** — *Large Language Model Evaluation via Matrix Entropy* (aka Diff-eRank) | `geometry_matrix_entropy` | arXiv:2401.17139 | `geometry/matrix_entropy.py` |
| **Yusupov et al. 2025** — *From Internal Representations to Text Quality: A Geometric Approach to LLM Evaluation* | `geometry_schatten` (Schatten-p norms) | arXiv:2509.25359 | `geometry/schatten.py` |
| **Xiao, Tian, Chen, Han, Han 2023** — *Efficient Streaming Language Models with Attention Sinks* | attention-sink motivation, cited in `activation_sinks.py` | arXiv:2309.17453 | `interpretability/activation_sinks.py` |
| **Yao, Gholami, Keutzer, Mahoney 2020** — *PyHessian: Neural Networks Through the Lens of the Hessian* | `dynamics_sharpness` (Hutchinson trace) | arXiv:1912.07145 | `dynamics/sharpness.py` |
| **Yeom, Giacomelli, Fredrikson, Jha 2018** — *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting* | `consistency_membership_inference` | arXiv:1709.01604 | `consistency/membership_inference.py` |
| **Zhu, Lu, Zheng, Guo, Zhang, Wang, Zhang 2018** — *Texygen: A Benchmarking Platform for Text Generation Models* (Self-BLEU) | `dynamics_generation_diversity` | arXiv:1802.01886 | `dynamics/generation_diversity.py` |
| **Zou, Phan, Chen, Campbell, Guo, Ren, Pan, Yin, Mazeika, Dombrowski, Goel, Li, Byun, Wang, Mallen, Basart, Koyejo, Song, Fredrikson, Kolter, Hendrycks 2023** — *Representation Engineering: A Top-Down Approach to AI Transparency* | `repe_concept_separability`, `repe_refusal_direction`, `repe_steering_effectiveness` | arXiv:2310.01405 | `representation_engineering.py` |

### Topology

| Paper | Task(s) | Identifier | Source file |
|---|---|---|---|
| **Bubenik 2015** | `topology_persistence_landscape` | arXiv:1207.6437 | `topology/persistence_landscape.py` |
| **Bubenik, Dlotko 2017** | `topology_persistence_landscape` implementation reference | arXiv:1501.00179 | `topology/persistence_landscape.py` |
| **Chazal, Fasy, Lecci, Michel, Rinaldo, Wasserman 2015** — *Subsampling Methods for Persistent Homology* | `topology_persistence_landscape` (bootstrap idea) | arXiv:1406.1901 | `topology/persistence_landscape.py` (cited) |
| **Rucco, Castiglione, Merelli, Pettini 2016** — *Characterisation of the Idiotypic Immune Network Through Persistent Entropy* | `topology_persistence_entropy` | Springer 2016 | `topology/persistence_entropy.py` |
| **Zomorodian, Carlsson 2005** — *Computing Persistent Homology* | `topology_homology` | Discrete & Computational Geometry | `topology/homology.py` |

### Metrics with weak citation — paper-provenance documented but not in docstring

These tasks work correctly and match their reference, but the
docstrings don't cite the original paper explicitly. Fix opportunity
for paper-ready release.

| Task | Underlying paper (for docstring) |
|---|---|
| `consistency_calibration` | Guo et al. 2017 ECE; Brier 1950 |
| `consistency_knowledge_capacity` | Tirumala et al. 2022 / Carlini et al. 2023 memorization framing; BLME reports exact-vs-rephrased likelihood proxy, not Allen-Zhu knowledge-capacity scaling |
| `dynamics_stability` | Wendlandt, Kummerfeld & Mihalcea 2018 (kNN-neighborhood embedding instability), arXiv:1804.09692 |
| `dynamics_interpolation` (`trajectories.py`) | Shoemake 1985 slerp (SIGGRAPH); latent-space interpolation White 2016, arXiv:1609.04468 |
| `geometry/hubness.py` | Tomašev et al. 2014 (cited above) |
| `geometry/lid.py` | Levina-Bickel 2004 (cited above) |
| `geometry/rsa.py` | Kriegeskorte 2008 (cited above) |
| `geometry/intrinsic_dim.py` | Facco 2017 (cited above) |
| `geometry/perplexity.py` | Shannon 1948 (bits-per-character); standard cross-entropy |
| `geometry/tokenizer_efficiency.py` | Rust et al. 2021 for fertility; Ali et al. 2024 for tokenizer-choice/training-cost effects |
| `geometry/unembedding.py` | effective rank (Roy & Vetterli 2007, EUSIPCO) on the unembedding matrix; alignment/purity are BLME diagnostics |
| `geometry/weight_norms.py` | Martin-Mahoney 2019 style per-layer diagnostic |
| `geometry/positional_decay.py` | Press et al. 2021 *ALiBi* — positional bias concept |
| `interpretability/attention_graph.py` | Abnar & Zuidema 2020 *Attention Rollout* (topology part) + standard PageRank |
| `interpretability/attention_polysemanticity.py` | custom; Templeton et al. 2024 Scaling Monosemanticity is the closest benchmark |
| `interpretability/attribution.py` | Simonyan et al. 2014 *Input × Gradient* |
| `interpretability/probing.py` | Alain & Bengio 2017 *Linear Classifier Probes* (arXiv:1610.01644) |
| `interpretability/sae_features.py` | Bricken et al. 2023 *Towards Monosemanticity* |
| `interpretability/sparsity.py` | Zhang et al. 2021 *Moefication* / standard L0 |
| `interpretability/weight_activation_alignment.py` | Park et al. 2024 *Linear Representation Hypothesis in LLMs* (arXiv:2311.03658) |
| `geometry/information_geometry.py` | Amari 2016 *Information Geometry and its Applications* |
| `consistency/icl_slope.py` | Brown et al. 2020 GPT-3 ICL; Min et al. 2022 *Rethinking the Role of Demonstrations* |

---

## 2. Considered and skipped

These papers were surveyed and deliberately not implemented (see
`PAPER_SURVEY.md` §3 for the full justification of each).
Condensed listing for quick reference.

| Paper | arXiv | Why skipped |
|---|---|---|
| Cao, Ying, Wang, Qiu, Huang, Jiang 2025 — Model Utility Index (MUI) | 2504.07440 | Requires SAE features or per-task neuron contributions |
| Valavala et al. 2025 — Feature Monosemanticity Score | 2506.19382 | Requires trained SAE + concept labels |
| Tschannen et al. 2025 — PRISM polysemanticity | 2506.15538 | Requires LLM descriptions |
| Bricken et al. 2023 — Towards Monosemanticity | Transformer Circuits | Inspired `interpretability_sae_features` (GPT-2 only) |
| Jha, Reagen 2025 — Spectral Scaling Laws (SUI) | 2510.00537 | Near-duplicate of existing rank metrics |
| Xiao et al. 2025 — FARMS eigenspectrum | 2506.06280 | Refines α estimation we already have |
| Hodgkinson, Wang, Mahoney 2025 — HT-MU | 2506.03470 | Theoretical; α already implemented |
| Song et al. 2025 — AlphaDecay | 2506.14562 | Training objective, not a metric |
| Rao et al. 2025 — Layer by Layer (DiME/infoNCE) | 2502.02013 | Near-duplicate of matrix entropy |
| Bonfanti et al. 2025 — Geometry of Tokens | 2501.10573 | Token-level ID already covered |
| Robinson et al. 2024 — Token Manifold Hypothesis | 2504.01002 | Theoretical; ID already captured |
| Queipo-de-Llano et al. 2025 — Latent Semantic Manifolds | 2603.22301 | Fisher-metric ID; overlap with existing |
| Azaria, Mitchell 2023 — LLM Factoscope | 2312.16374 | Requires labelled factuality pairs |
| Girrbach et al. 2025 — Reference-Free Rating | HCAI 2025 | Requires human ratings |
| Liu et al. 2025 — Mining Intrinsic Rewards | 2505.12225 | Requires correctness labels |
| Kadavath et al. 2022 — P(IK) | 2207.05221 | Requires Q/A label pairs |
| Gonçalves et al. 2024 — Collaborative Performance Prediction | 2407.01300 | Cross-model benchmark prediction |
| Wu et al. 2024 — Performance Law | 2408.09895 | Cross-model parametric law |
| Owen et al. 2024 — 100 instances is all you need | 2409.03563 | Benchmark subsampling |
| Isik et al. 2024 — Sloth scaling laws | 2412.06540 | Multi-benchmark skill prediction |
| Ruan et al. 2025 — Clustering-based downstream scaling | 2502.17262 | Benchmark prediction |
| Schaeffer et al. 2023 — Are Emergent Abilities a Mirage? | 2304.15004 | Metric-choice critique |
| Gonçalves et al. 2024 — Language Ranker | 2404.11553 | Cross-lingual; task-specific |
| Pfefferle et al. 2025 — Reasoning Trajectories | 2604.05655 | Requires CoT + correctness labels |
| Wang et al. 2026 — Stepwise Informativeness | 2604.06192 | CoT entropy — requires generation |
| MSR 2026 — LLM Reasoning as Trajectories | — | Same |
| Chen et al. 2025 — Preplan-and-Anchor Attention | 2510.13554 | Requires RL credit assignment |
| Syed Raza et al. 2025 — ICR Probe | ACL 2025 | Requires answer labels |
| Zhang et al. 2025 — Latent-info Reference-Free | arXiv:2509.12886 | Requires human ratings |
| Huang et al. 2025 — Probing Hidden States Factuality | medRxiv 2025 | Requires factuality labels |
| Li et al. 2025 — Wisdom-of-Crowds Guesstimation | 2501.17310 | Requires multi-sample generation |
| Meister et al. 2025 — ESI Epistemic Uncertainty | 2510.13103 | Requires semantic-preserving interventions |
| Rivera et al. 2025 — Aleatoric/Epistemic Uncertainty QA | 2511.03166 | Requires ID/OOD QA labels |
| Gu et al. 2026 — Representation Gradient Tracing | 2510.02334 | Requires training-data / reference behaviours |
| Chen et al. 2026 — Mechanistic Data Attribution | 2601.21996 | Requires training data |
| Chen et al. 2024 — Distributional Memorization | 2407.14985 | Requires pretraining corpus |
| Liu et al. 2025 — Grokking monitoring | 2506.21551 | Requires training checkpoints |
| Chen et al. 2025 — Tracing Multilingual Factual Knowledge | 2505.14824 | Requires pretraining checkpoints |
| Akyürek et al. 2022 — Fact tracing | 2205.11482 | Requires training data |
| Yi et al. 2025 — Tracing Multilingual Factual | — | Same |
| Gao et al. 2025 — Visualising LLM Latent Space | 2511.21594 | Visualisation, not metric |
| Patel et al. 2025 — Topological Metric for Embeddings | 2512.15285 | Near-duplicate of our topology tasks |
| Chen et al. 2025 — TokenBlowUp | 2507.19747 | Representation correction, not measurement |
| Zaba et al. 2025 — Deep Language Geometry | 2508.11676 | Multilingual-specific |
| Kim et al. 2025 — Beyond Linear Separability Ceiling | 2507.07574 | Vision-language specific |
| Ezen et al. 2025 — Linear Probe Accuracy Scales | 2604.13386 | Multi-layer ensembling probe |
| Chen et al. 2025 — CWMI Causal World Models | 2507.19855 | Fine-tuning method |
| Kim et al. 2025 — WorldLLM | 2506.06725 | Active-exploration framework |
| Mao et al. 2025 — Spatial World Models | 2604.10690 | Domain-specific |
| Kornfeld et al. 2025 — Measuring Monosemanticity | 2506.19382 | Duplicate of FMS |
| Ahuja et al. 2024 — Quantifying LLM Capabilities | 2405.03146 | Cross-scale benchmarks |
| Stoerzner et al. 2025 — LongBench | 2505.19293 | Benchmark |
| Wang et al. 2025 — Context Length Alone Hurts | 2510.05381 | Benchmark effect |
| Wu et al. 2025 — Self-Execution Benchmark | 2508.12277 | Benchmark |
| Zaba et al. 2026 — MemGround | 2604.14158 | Benchmark |
| Boix-Adsera et al. 2025 — Scaling Laws for Downstream Perf | 2410.08527 | Benchmark scaling |
| Wang et al. 2024 — Attention Head Entropy | 2602.13699 | Requires correctness labels |
| Kreuzer et al. 2025 — Semantic Entropy Probes | 2406.15927 | Hallucination-specific |
| Lee et al. 2024 — Entropy-Guided Attention (private LLMs) | 2501.03489 | Architecture modification |
| Chen et al. 2024 — Unveiling Hidden Attention Sinks | 2406.15765 | Calibration method |
| Kong et al. 2025 — Sparse Attention for MI (Stream) | 2510.19875 | Interpretability tool |
| Zhao et al. 2024 — SeerAttention | 2410.13276 | Architecture method |
| Nanda et al. 2026 — Attention-Head Stability | 2602.16740 | Requires retraining |
| Burns et al. 2024 — Attention Pattern MAE | 2604.03764 | Trains MAE, not a scalar |
| Wu et al. 2025 — ART Attention Replacement | 2604.06393 | Training method |
| Yin et al. 2024 — Truthfulness via LID | 2402.18048 | Metric is our `geometry_lid`; the LID-plus-classifier is downstream |
| Cao et al. 2025 — CPP | 2407.01300 | Cross-model prediction |
| Gao et al. 2025 — Densing Law | Nature MI 2025 | Capability-per-parameter |
| Wang et al. 2024 — LLM Factoscope | 2312.16374 | Requires factuality labels |
| Azaria, Mitchell 2023 — Internal States for Honesty | 2304.13734 | Requires honesty labels |

---

## 3. Per-task citation audit

Every registered task with its implementation status and citation in
the source file. "✅" = has a proper arXiv or author-year citation in
the docstring; "📝" = metric is well-defined but docstring needs the
citation added for paper-ready release; "—" = no canonical paper (the
metric is a BLME diagnostic or trivial engineering helper).

**Geometry (27 tasks)**:
- `geometry_categories` — 📝 add citation (no canonical paper; BLME-custom category coherence)
- `geometry_cka` ✅ Kornblith et al. 2019 + Cortes et al. 2012 (round 10)
- `geometry_collapse` ✅ Jing 2021 (arXiv:2110.09348); Roy-Vetterli 2007; Queipo-de-Llano, Arroyo et al. 2025 (arXiv:2510.06477) compression-valley context
- `geometry_contextualization` ✅ Ethayarajh 2019
- `geometry_correlation_dimension` 📝 add Grassberger-Procaccia 1983; BLME uses the classical estimator, not the Du-Tanaka-Ishii LLM-specific pipeline
- `geometry_hsic` ✅ Gretton 2005 + Kornblith 2019 CKA equivalence (round 10)
- `geometry_hubness` ✅ Tomašev et al. 2014; Radovanović 2010 (round 10)
- `geometry_intrinsic_dim` ✅ Facco et al. 2017 Two-NN; Ansuini 2019 (round 10)
- `geometry_isoscore` ✅ Rudman et al. 2022
- `geometry_lid` 📝 add Levina-Bickel 2004 + Ma et al. 2018
- `geometry_lipschitz` 📝 add Virmaux-Scaman 2018 (Lipschitz constant of DNNs)
- `geometry_mahalanobis` ✅ Lee et al. 2018
- `geometry_matrix_entropy` ✅ Wei et al. 2024 (Diff-eRank)
- `geometry_neural_collapse` ✅ Papyan-Han-Donoho 2020
- `geometry_perplexity` — standard (Shannon 1948)
- `geometry_positional_decay` 📝 add Press et al. 2021 (ALiBi) for positional-bias motivation
- `geometry_prediction_alignment` (alias `geometry/consistency.py`) 📝 BLME-custom; no canonical paper
- `geometry_representation_sensitivity` (alias `information_geometry.py`) 📝 add Amari 2016
- `geometry_rsa` 📝 add Kriegeskorte 2008
- `geometry_schatten` ✅ Wei 2025, Li 2024, Garrido 2023
- `geometry_spectral` ✅ Martin-Mahoney 2019/2021
- `geometry_svd` — standard SVD diagnostics
- `geometry_tokenizer_efficiency` ✅ Rust 2021 fertility; Ali et al. 2024 tokenizer-choice effects (round 10)
- `geometry_trajectory_curvature` ✅ Hosseini-Fedorenko 2023
- `geometry_mp_bulk_deviation` ✅ Marchenko-Pastur 1967; Baik-Ben Arous-Péché 2005
- `geometry_unembedding` 📝 effective rank = Roy & Vetterli 2007 (EUSIPCO); no single source paper
- `geometry_weight_norms` — BLME diagnostic

**Interpretability (15 tasks)**:
- `interpretability_activation_sinks` ✅ Gu 2025 + Sun 2024 + Arroyo et al. 2025
- `interpretability_attention_effective_rank` (alias `attention_polysemanticity.py`) 📝 effective rank = Roy & Vetterli 2007; BLME-custom attention application (Elhage 2022 / Templeton 2024 framing)
- `interpretability_attention_entropy` ✅ Clark 2019; Jain-Wallace 2019
- `interpretability_attention_graph` 📝 add Abnar-Zuidema 2020
- `interpretability_attention_rank` ✅ effective rank = Roy & Vetterli 2007 (metric); Dong et al. 2021 = rank-collapse motivation
- `interpretability_attribution` 📝 add Simonyan 2014 (input × gradient)
- `interpretability_head_roles` ✅ Olsson 2022 (prev-token score); Wang et al. 2022 IOI (arXiv:2211.00593, duplicate-token score); Clark 2019; Voita 2019
- `interpretability_induction_heads` ✅ Olsson 2022
- `interpretability_logit_lens` ✅ nostalgebraist 2020 + Belrose et al. 2023 Tuned Lens
- `interpretability_prediction_entropy` ✅ per-token Shannon entropy (Shannon 1948); Holtzman 2020 = degeneration motivation, not the metric source
- `interpretability_probing` ✅ Alain-Bengio 2017
- `interpretability_sae_features` 📝 add Bricken 2023
- `interpretability_sparsity` 📝 add Zhang 2021 (Moefication)
- `interpretability_superposition` ✅ Elhage 2022; Templeton 2024
- `interpretability_waa` ✅ Park et al. 2024 (arXiv:2311.03658); Elhage 2022

**Causality (6 tasks)**:
- `causality_ablation` — BLME diagnostic
- `causality_attention_knockout` ✅ Michel 2019; Voita 2019
- `causality_circuit_quality` ✅ Conmy et al. 2023
- `causality_edge_attribution` ✅ Syed 2024
- `causality_knowledge_neurons` ✅ Dai 2022
- `causality_tracing` ✅ Meng 2022 (ROME)

**Dynamics (6 tasks)**:
- `dynamics_coe` ✅ Wang 2025
- `dynamics_generation_diversity` ✅ Li 2016; Zhu 2018
- `dynamics_gradient_flow` ✅ Pascanu 2013
- `dynamics_interpolation` (alias `trajectories.py`) 📝 slerp = Shoemake 1985 (SIGGRAPH); latent interpolation White 2016 (arXiv:1609.04468)
- `dynamics_sharpness` ✅ Foret 2021; Yao 2020
- `dynamics_stability` 📝 BLME-custom; add Liang et al. 2024 *Holistic Evaluation of Language Models* if comparable

**Consistency (12 tasks)**:
- `consistency_bias_weat` ✅ Caliskan 2017; May 2019
- `consistency_calibration` 📝 add Guo et al. 2017
- `consistency_contamination` ✅ Shi et al. 2023
- `consistency_contrastive` 📝 BLME-custom CounterFact-style negative-rejection proxy; related to Meng et al. 2022 / CounterFact framing
- `consistency_format_robustness` ✅ Sclar 2023
- `consistency_icl_slope` 📝 add Brown 2020; Min 2022 explicitly
- `consistency_knowledge_capacity` 📝 legacy-named exact-vs-rephrased likelihood proxy; do not cite Allen-Zhu capacity scaling as the implemented method
- `consistency_logical` 📝 BLME-custom premise-lift likelihood diagnostic
- `consistency_membership_inference` ✅ Yeom 2018; Carlini 2021
- `consistency_paraphrase` 📝 BLME-custom; add Kirichenko 2023 for paraphrase-invariance motivation
- `consistency_position_sensitivity` ✅ Liu 2023
- `consistency_self_consistency` ✅ Wang 2022

**RepE (4 tasks)**:
- `repe_concept_separability` ✅ Zou 2023
- `repe_refusal_direction` ✅ Arditi 2024; Zou 2023
- `repe_steering_effectiveness` ✅ Zou 2023; Turner 2023
- `repe_task_vectors` ✅ Zou 2023 RepE + Ilharco 2023 Task Arithmetic (round 10)

**Topology (4 tasks)**:
- `topology_betti_curve` 📝 add Naitzat et al. 2020 (JMLR 21(184):1-40, 2020) to docstring
- `topology_homology` ✅ Zomorodian-Carlsson 2005; Naitzat 2020; Edelsbrunner-Harer 2008
- `topology_persistence_entropy` 📝 add Rucco 2016
- `topology_persistence_landscape` ✅ Bubenik 2015; Chazal 2015

### Citation-audit summary

This audit now tracks **74 registered diagnostic tasks**. Entries marked
✅ are paper-faithful or directly paper-derived implementations; 📝 means
the metric is implemented but the source/docstring still needs a cleaner
paper citation; — means BLME intentionally implements a custom diagnostic
or engineering proxy with no single canonical paper.

Remaining 📝 items are documentation/provenance cleanup, not correctness
bugs. When a BLME task uses only a heuristic or proxy inspired by a paper,
the task docs should say so explicitly instead of claiming the full paper
method.

---

## 4. Update procedure

When any of these happen, update this file:

1. **New task added** → add a row in §1 (implemented) and §3 (per-task audit).
2. **Citation added to a docstring** → flip 📝 → ✅ in §3.
3. **New paper surveyed** → add to §2 (skipped) with a one-line reason.
4. **Paper reconsidered / now implemented** → move from §2 to §1.
5. **Paper in §2 found wrong on re-reading** → remove or re-classify.

Also cross-check:
- `PAPER_SURVEY.md` §1 and §2 should match the rows here.
- `CORRELATION_LITERATURE.md` status column (✅/🟡/🔴) should match
  the implemented set here.
- `TOP_PREDICTORS.md` tables should cite papers listed in §1.

Last full audit: **2026-06-20** (publication-readiness citation and proxy-label pass).

**Next scheduled audit**: whenever a new task is added or a new
literature sweep is performed.
