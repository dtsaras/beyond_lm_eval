# BLME — Related Work (draft for paper §2)

A reviewer-grade related-work section organised by the six threads BLME
engages with. Every paragraph ends with the explicit BLME-positioning
sentence that names the task(s) we implement from that thread. This
file is the paper-draft version of the material in `PAPERS.md`; keep
the two synchronised.

Authoritative paper index: `docs/PAPERS.md`. Survey narrative:
`docs/PAPER_SURVEY.md`. Experimental-correlation annex:
`docs/CORRELATION_LITERATURE.md`. Main experimental result:
`docs/TOP_PREDICTORS.md`.

---

## §2.1 Benchmark-based evaluation and its limits

The dominant paradigm for LLM evaluation is standardised benchmarks.
**BIG-Bench** (Srivastava et al. 2022, arXiv:2206.04615) assembles 204
tasks across 132 institutions; **BIG-Bench Hard** (Suzgun et al. 2022,
arXiv:2210.09261) selects 23 of the hardest; **HELM** (Liang et al.
2022, arXiv:2211.09110) adds a 7-metric (accuracy, calibration,
robustness, fairness, bias, toxicity, efficiency) holistic evaluation
of 30 models across 42 scenarios; **MMLU** (Hendrycks et al. 2021),
**HellaSwag** (Zellers et al. 2019), **ARC** (Clark et al. 2018),
**WinoGrande** (Sakaguchi et al. 2020), and **PIQA** (Bisk et al.
2020) are the canonical multiple-choice suites; the **LM-Evaluation
Harness** (Gao et al. 2024) provides the community-standard
infrastructure with hundreds of implemented tasks.

The limits of this paradigm are well documented. **Schaeffer, Miranda,
Koyejo 2023** (arXiv:2304.15004) show that "emergent abilities" in
BIG-Bench are partly an artefact of non-linear scoring metrics — >92 %
of emergent abilities appear only under accuracy-like metrics that
discretise per-token error. **Singh et al. 2024** (arXiv:2404.00699)
and **Sainz et al. 2024** (arXiv:2406.04244) survey data contamination
— benchmark-leakage into pretraining corpora — showing that headline
numbers for recent models are inflated on widely-used benchmarks
including MMLU and HumanEval. **Shi et al. 2023** (arXiv:2310.16789)
propose Min-K % Prob as a membership-inference-based contamination
detector. **Wei et al. 2022** (arXiv:2206.07682) document emergence
on BIG-Bench but warn that metric choice drives the appearance of
sharp phase transitions.

BLME does not replace benchmarks — we report standard benchmark
scores as the *target* variable. Instead, BLME provides **intrinsic
metrics computed from weights and hidden-state activations without
reference to benchmark outputs**, which we correlate against
benchmark capability. Our `consistency_contamination` task
(implementing Shi et al. 2023) and `consistency_membership_inference`
(Yeom 2018, Carlini 2021) surface contamination signals; every other
task is benchmark-free by construction.

---

## §2.2 Scaling laws and capability prediction

The foundational scaling-laws literature establishes that cross-entropy
loss falls as a power function of model, dataset, and compute:
**Kaplan et al. 2020** (arXiv:2001.08361) across GPT-scale models;
**Hoffmann et al. 2022** (Chinchilla, arXiv:2203.15556) showing
compute-optimal scaling balances parameters and tokens; **Alabdulmohsin
et al. 2022**, **Muennighoff et al. 2023**, **Isik et al. 2024**
(Sloth, arXiv:2412.06540) refining to multi-benchmark and multi-skill
settings; **Wu et al. 2024** (Performance Law, arXiv:2408.09895),
**Ruan et al. 2025** (arXiv:2502.17262), **Owen et al. 2024**
(arXiv:2409.03563), and **Gonçalves et al. 2024** (Collaborative
Performance Prediction, arXiv:2407.01300) predicting downstream
benchmark scores from parameter count plus anchor-model observations.

This literature shares two properties BLME explicitly avoids. First,
it is **cross-model**: it requires observations of multiple models to
fit a formula. Second, it is **benchmark-coupled**: the target
variable is a benchmark score, so the prediction inherits any
contamination or task-mismatch in the target. The only per-model
data-free predictor in this line of work is **Martin & Mahoney 2019a/b**
(arXiv:1901.08276, 1901.08278; *Nature Comms* 2021) and its
**WeightWatcher** tool: the heavy-tailed power-law exponent α of
per-layer weight ESDs predicts test accuracy across hundreds of models
**without any training or test data access**.

BLME extends this "per-model data-free" line to 54 intrinsic metrics
spanning seven categories. We implement α-on-weights as
`geometry_spectral.avg_alpha` and extend the per-model
benchmark-independent predictor axis to cover representation geometry,
attention, causality, dynamics, consistency, RepE, and topology.

---

## §2.3 Representation geometry of deep networks

A long line of work characterises the geometric structure of learned
representations. **Roy & Vetterli 2007** (EUSIPCO) introduce the
entropy-based "effective rank" of a singular-value spectrum. **Facco
et al. 2017** (*Sci. Rep.*) propose the Two-NN intrinsic-dimension
estimator; **Levina & Bickel 2004** (NeurIPS) the maximum-likelihood
local-intrinsic-dimensionality (LID) estimator, later applied by
**Ma et al. 2018** (ICLR) to adversarial-subspace detection.
**Grassberger & Procaccia 1983** (Phys. Rev. Lett.) define correlation
dimension. **Kornblith et al. 2019** (arXiv:1905.00414) introduce
Centered Kernel Alignment (CKA) based on the HSIC of **Gretton et al.
2005** (ALT). **Kriegeskorte et al. 2008** (*Front. Syst. Neurosci.*)
define Representational Similarity Analysis (RSA). **Papyan, Han &
Donoho 2020** (PNAS) characterise Neural Collapse. **Naitzat,
Zhitnikov & Lim 2020** (ICLR) apply Topological Data Analysis to
track Betti-number collapse through network layers. **Bubenik 2015**
(arXiv:1501.00179) and **Zomorodian & Carlsson 2005** (DCG) develop
the persistence-landscape and persistent-homology frameworks. **Tomašev
et al. 2014** (IEEE TKDE) characterise high-dimensional hubness.

For transformers specifically, **Ethayarajh 2019** (arXiv:1909.00512)
shows contextualised word representations in BERT / ELMo / GPT-2 are
highly anisotropic — large cosine similarity between random token
pairs — and defines the Maximum Explainable Variance (MEV) and
anisotropy baseline we still use. **Rudman et al. 2022** (IsoScore,
arXiv:2207.10341) propose a covariance-based isotropy scalar strictly
more discriminative than cosine-based anisotropy. **Park, Choe,
Wattenberg, Jegelka 2024** (Linear Representation Hypothesis,
arXiv:2311.03658) formalise the input-space / output-space duality of
concept directions. **Cavagnero et al. 2025** (arXiv:2506.01034)
demonstrate that local intrinsic dimension predicts generalisation,
grokking, and overfitting onset in an unsupervised way. **Bonfanti
et al. 2025** (arXiv:2501.10573) show per-token ID correlates with
cross-entropy loss.

More recent work pushes into 2025. **Wei et al. 2024** (Matrix
Entropy / Diff-eRank, arXiv:2401.17139) show the von Neumann entropy
of the row-normalised token covariance follows a scaling law
paralleling loss scaling; **Li et al. 2024** (arXiv:2410.10672)
introduce the Matrix Nuclear-Norm as an 8–24× faster alternative.
**Garrido et al. 2023** (ICML, arXiv:2210.02885) introduce RankMe —
the effective rank via `exp(H(σ_i / Σσ_j))` — as a label-free
predictor of SSL downstream-probe accuracy. **Li et al. 2025** (NeurIPS,
arXiv:2509.23024) jointly track RankMe and α-ReQ across pretraining
checkpoints, identifying three geometric phases. **Wei et al. 2025**
(arXiv:2509.25359) show Intrinsic Dim, Effective Rank, MEV, Schatten
Norms, and MAUVE all serve as reference-free text-quality proxies.
**Jha & Reagen 2025** (EMNLP, arXiv:2510.00537) decompose spectral
utilisation into Hard Rank (participation ratio), Soft Rank (Shannon
rank), Spectral Concentration, and a composite SUI.

**BLME implements all of the above in a single library**. We expose
Roy-Vetterli effective rank and RankMe side-by-side (they use
different normalisations of σ_i); Schatten-p norms for
p ∈ {1, 2, 4, ∞} plus Li 2024's L1,2 Matrix Nuclear-Norm; Facco and
Levina-Bickel intrinsic dimension; CKA, RSA, HSIC for layer-pair
similarity; Kornblith, Ethayarajh, Rudman isotropy measures;
Papyan-Han-Donoho Neural Collapse; Naitzat, Zomorodian-Carlsson,
Bubenik topology; Tomašev hubness; and Wei 2024 Matrix Entropy. See
`docs/PAPERS.md` §1 for the full mapping.

---

## §2.4 Probing, mechanistic interpretability, and circuit-level analysis

A second major thread views the LLM as a computational artefact to be
reverse-engineered. **Alain & Bengio 2017** (arXiv:1610.01644) seed
the linear-probe methodology: a logistic classifier trained on a
frozen model's intermediate activations reveals what information is
linearly decodable. **Belinkov 2022** critically reviews the probing
framework, noting its conflations between "information present" and
"information used". **nostalgebraist 2020** introduces the Logit Lens
— projecting intermediate hidden states through the model's own
`lm_head` — with **Belrose et al. 2023** (Tuned Lens, arXiv:2303.08112)
providing the learned-probe refinement. **Clark et al. 2019**
(BlackBoxNLP) analyse BERT's attention heads, finding systematic
previous-token / duplicate-token patterns; **Voita et al. 2019** (ACL)
and **Michel et al. 2019** (NeurIPS) show most heads can be pruned
without large accuracy drops — the foundational "head ablation"
experiment.

The mechanistic-interpretability agenda (**Olah et al. 2020** "Zoom In:
An Introduction to Circuits", Distill; **Elhage et al. 2022**
"Toy Models of Superposition"; **Olsson et al. 2022** "In-context
Learning and Induction Heads", Transformer Circuits) reverse-engineers
transformer computations into human-readable circuits. **Meng et al.
2022** (ROME, arXiv:2202.05262) introduces causal tracing for locating
factual recall. **Dai et al. 2022** (arXiv:2104.08696) identifies
"knowledge neurons" via gradient × activation. **Syed et al. 2024**
(EAP, arXiv:2310.10348) propose Edge Attribution Patching as a fast
surrogate for automated circuit discovery (**Conmy et al. 2023**,
ACDC, arXiv:2304.14997). **Bricken et al. 2023** (Towards
Monosemanticity, Transformer Circuits) introduce the sparse-autoencoder
paradigm for dictionary learning, extended by **Templeton et al. 2024**
(Scaling Monosemanticity) and the **Gemma Scope** / **Llama Scope**
pretrained SAE suites.

Interpretability-focused survey: **Rai, Zhou, Feng, Saparov, Yao 2024**
(arXiv:2407.02646) provide a practical-review taxonomy; **Bereska &
Gavves 2024** (arXiv:2404.14082) cover mechanistic interpretability
for AI safety. For sparse autoencoders specifically: **Rajamanoharan
et al. 2025** (arXiv:2503.05613) survey the SAE ecosystem. Token-level
attribution methods are covered by **Simonyan et al. 2014** (Input ×
Gradient), **Sundararajan et al. 2017** (Integrated Gradients, ICML),
and more recently **TokenSHAP** (arXiv:2407.10114), **TokenShapley**
(arXiv:2507.05261), and **llmSHAP** (arXiv:2511.01311). **Shapley
Value Sampling** has been shown to outperform attention and IG in
plausibility and faithfulness.

BLME implements `interpretability_logit_lens` (nostalgebraist 2020),
`interpretability_attention_entropy` (Clark 2019), `head_roles`
(Clark 2019, Voita 2019), `induction_heads` (Olsson 2022),
`superposition` (Elhage 2022), `attention_rank` (**Dong et al. 2021**,
arXiv:2103.03404), `probing` (Alain-Bengio 2017), `sae_features`
(Bricken 2023; GPT-2 only due to SAE availability), `waa` (Park 2024),
`attribution` (Simonyan 2014, input × gradient), and `attention_graph`
(Abnar & Zuidema 2020 attention rollout + PageRank). Causal-tracing
methods are implemented as `causality_tracing` (Meng 2022),
`attention_knockout` (Voita 2019), `knowledge_neurons` (Dai 2022),
`edge_attribution` (Syed 2024), `circuit_quality` (Conmy 2023), and
`causality_ablation` (BLME diagnostic).

---

## §2.5 Attention-sink / massive-activation / compression-valley nexus

A recently-unified thread characterises a striking empirical
phenomenon: a few tokens (typically BOS) absorb a disproportionate
share of attention mass in almost every head of almost every modern
LLM. **Xiao, Tian, Chen, Han, Han 2023** (StreamingLLM,
arXiv:2309.17453) introduce the "attention sink" terminology.
**Sun, Chen, Bai, Hu, Xiong, Kolter 2024** (arXiv:2402.17762) show
the same tokens host "massive activations": residual-stream entries
at 100-1000× the typical magnitude, acting as fixed bias terms
regardless of input; removing them destroys model performance.
**Gu, Pang, Du, Liu, Collier, Lin 2025** (ICLR Spotlight,
arXiv:2410.10781) define the Sinkε metric and show its emergence
depends on optimiser, tokeniser, and attention kernel, with
downstream implications for long-context performance.
**Pedrotti & Guo 2025** (arXiv:2510.06477) prove theoretically that
massive activations necessarily induce representational compression
at mid-depth layers — the "compression valley" — empirically
confirming the link across models from 410M to 120B.

BLME's `interpretability_activation_sinks` task (added in round 8)
reproduces the Gu 2025 Sinkε formula from the reference code at
[sail-sg/Attention-Sink](https://github.com/sail-sg/Attention-Sink);
measures Sun 2024 massive-activation fraction + max/median ratio;
and computes the Pedrotti-Guo compression-valley depth from our
matrix-entropy per-layer trajectory. All three signals surface as
independent capability predictors (valley depth has partial
ρ = –0.53 with composite benchmark in our 32-model study).

---

## §2.6 Representation universality and the convergence hypothesis

A theoretically-rich thread asks whether learned representations
converge toward a common structure. **Moschella et al. 2022**
(arXiv:2209.15430) show "relative representations" enable zero-shot
latent-space communication between independently-trained models.
**Huh et al. 2024** (Platonic Representation Hypothesis,
arXiv:2405.07987) argue AI representations converge toward a shared
statistical model of reality, validated by linear stitching between
vision and text models. **Wattenberg et al. 2024** and others
identify shared geometric structures (linear subspaces for spatio-
temporal coordinates, circular manifolds for calendar days, helical
manifolds for numbers) across LLM families. **Tigges et al. 2024**
(NeurIPS) show LLM circuits are consistent across training and scale;
**Dunefsky et al. 2024** (NeurIPS, Transcoders) find interpretable
feature circuits. **Lin et al. 2024** (arXiv:2410.06981, Quantifying
Feature Space Universality via SAEs) show SAE feature spaces are
similar across LLMs under rotation-invariant transformations.
**Gonçalves et al. 2025** (arXiv:2501.02009) quantify cross-model
transferability of Platonic concept representations.

BLME is a complement to this line: rather than asking "do two models
share representations?" we ask "which per-model intrinsic properties
predict capability?". The two agendas converge on the fact that
RepE-style (Zou 2023) task vectors, concept directions, and refusal
directions exist as identifiable geometric features; BLME then tests
which of those features predict benchmark performance. We implement
`repe_task_vectors` (Ilharco 2023), `repe_concept_separability`
(Zou 2023), `repe_refusal_direction` (Arditi 2024), and
`repe_steering_effectiveness` (Zou 2023).

---

## §2.7 Beyond-benchmark capability signals

A growing literature uses internal signals to predict per-task
correctness without external labels. **Kadavath et al. 2022**
(arXiv:2207.05221) introduce P(IK) — a self-prompted confidence
probe. **Azaria & Mitchell 2023** (arXiv:2312.16374) the Factoscope
achieves >96 % hallucination-detection accuracy from hidden-state
probes. **Yin et al. 2024** (arXiv:2402.18048) show local intrinsic
dimension predicts TruthfulQA accuracy. **Farquhar et al. 2024**
(Semantic Entropy, Nature) demonstrate uncertainty quantification
from multi-sample semantic clustering. **Liu et al. 2025**
(arXiv:2505.12225) mine intrinsic rewards from hidden states for
best-of-N sampling. **Rao et al. 2025** (arXiv:2502.02013, Layer by
Layer) show DiME + infoNCE + dataset-entropy predict per-layer probe
accuracy without labels. **Cao et al. 2025** (Model Utility Index,
arXiv:2504.07440) formulate a "Utility Law" linking
fraction-of-activated-neurons to task performance.

These approaches share a methodological constraint: they require
**labelled downstream data** (correct/false answers, clustered
completions, task-specific probe targets) to define the prediction
problem. BLME takes the opposite stance: we compute signals from a
fixed unlabelled corpus, then evaluate their correlation with
downstream benchmarks externally. This distinction is why we report
log-likelihood-based tasks (`geometry_perplexity`,
`consistency_calibration`, `interpretability_prediction_entropy`)
but do not implement P(IK) or Factoscope-style probes — the labels
those methods need are precisely what BLME is trying to predict.

---

## §2.8 Consistency, robustness, calibration

Independent of representation analysis, a literature studies the
output-side consistency and calibration of LLMs. **Guo et al. 2017**
(ICML) establish Expected Calibration Error (ECE) as the standard
calibration scalar. **Brier 1950** the quadratic scoring rule.
**Liu et al. 2023** (arXiv:2307.03172) document "Lost in the Middle":
LLM accuracy drops for information near the middle of a long prompt.
**Sclar et al. 2023** (arXiv:2310.11324) quantify sensitivity to
spurious prompt-format features. **Wang et al. 2022** (Self-
Consistency, arXiv:2203.11171) show majority voting over sampled CoT
outputs improves chain-of-thought accuracy. **Brown et al. 2020**
(GPT-3) establish in-context learning shot scaling; **Min et al.
2022** analyse how demonstrations contribute. **Caliskan, Bryson,
Narayanan 2017** (*Science*) develop WEAT for lexical bias; **May
et al. 2019** extend to contextualised embeddings via SEAT.
**Allen-Zhu & Li 2024** (Physics of LMs, arXiv:2404.05405) quantify
bits-per-parameter knowledge capacity. **Liang et al. 2024**
(Holistic Evaluation, HELM) catalogue robustness across 7
dimensions.

BLME implements `consistency_calibration` (Guo 2017),
`consistency_position_sensitivity` (Liu 2023), `format_robustness`
(Sclar 2023), `self_consistency` (Wang 2022), `icl_slope` (Brown
2020, Min 2022), `bias_weat` (Caliskan 2017, May 2019),
`knowledge_capacity` (Allen-Zhu 2024), plus membership-inference
and contamination probes (Yeom 2018, Carlini 2021, Shi 2023).

---

## §2.9 Dynamics: sharpness, gradient flow, in-context trajectories

Optimisation-inspired intrinsic measures examine the loss landscape
and dynamics. **Pascanu, Mikolov, Bengio 2013** (ICML) introduce the
gradient-flow diagnostic for vanishing-gradient detection.
**Keskar et al. 2017** (ICLR) link generalisation to minima flatness.
**Foret et al. 2021** (SAM, arXiv:2010.01412) and **Yao et al. 2020**
(PyHessian, arXiv:1912.07145) formalise sharpness-aware minimisation
and Hessian trace / top-eigenvalue estimation.
**Wang et al. 2025** (Chain-of-Embedding, ICLR 2025, arXiv:
2410.13640) show the across-layer trajectory of hidden states for a
fixed token — magnitude (CoE-R) and angle (CoE-C) changes — predicts
answer correctness on in-context-learning tasks. Generation-side
metrics include **Li et al. 2016** (Distinct-n) and **Zhu et al. 2018**
(Self-BLEU, Texygen).

BLME implements `dynamics_sharpness` (Foret 2021 SAM + Yao 2020
Hutchinson trace + power-iteration top eigenvalue),
`dynamics_gradient_flow` (Pascanu 2013), `dynamics_coe` (Wang 2025),
`dynamics_generation_diversity` (Li 2016, Zhu 2018), plus
`stability` and `interpolation` diagnostics.

---

## §2.10 BLME's contribution relative to all of the above

Prior work falls into four buckets:

1. **Single-metric-deep**: one metric evaluated exhaustively on
   hundreds of models (Martin-Mahoney 2021 α).
2. **Multi-metric-narrow**: 3–5 related metrics tested on 6–12 models
   within a single category (Wei 2025 Schatten, Jha 2025 SUI, Rao
   2025 Layer-by-Layer).
3. **Cross-model benchmarks**: benchmark-coupled prediction across
   many models (Kaplan 2020, Hoffmann 2022, Isik 2024 Sloth, Wu
   2024 Performance Law).
4. **Task-specific probes**: hidden-state analysis tied to labelled
   downstream tasks (Kadavath 2022 P(IK), Azaria 2023 Factoscope,
   Yin 2024 TruthfulQA-LID).

**BLME's unique contribution** is the orthogonal cut: 54 intrinsic
tasks totalling 731 features, across **seven distinct measurement
taxonomies** (geometry, interpretability, causality, dynamics,
consistency, RepE, topology), evaluated systematically on **32
pretrained LLMs spanning 8 families and 3 orders of magnitude in
parameter count**, with univariate, partial (size-controlled), and
sparse LASSO analysis yielding an honest held-out LOO R² = 0.772
versus a `log(N_params)`-only baseline of 0.429. No prior library
or paper combines this breadth of metric taxonomy with this breadth
of model coverage and this systematic statistical methodology;
existing work lies at one of two extremes (one-metric-many-models
or few-metrics-few-models).

**Concretely**: we are the first to (a) evaluate Wei 2024 matrix
entropy, Li 2024 matrix nuclear-norm, Garrido 2023 RankMe, Gu 2025
Sinkε, Sun 2024 massive activations, Pedrotti-Guo 2025 compression
valleys, and Wang 2025 Chain-of-Embedding **in a single head-to-head
comparison against each other and against Martin-Mahoney α,
Ethayarajh contextualisation, Kornblith CKA, Papyan-Han-Donoho
Neural Collapse, Naitzat-Zhitnikov-Lim topology, Meng 2022 ROME,
Olsson 2022 induction heads, Dai 2022 knowledge neurons, Syed 2024
EAP, Zou 2023 RepE, Arditi 2024 refusal direction, Foret 2021 SAM,
and Yao 2020 PyHessian** — and (b) quantify the
held-out-LOO-generalisation-beyond-log(N_params) improvement
attributable to each metric individually and to the joint LASSO
combination.

---

## Appendix A — paper-selection criteria (inclusion / exclusion)

A paper is **included** in BLME if and only if all five criteria hold:

1. **Intrinsic**: metric is computable from model weights + static
   forward passes over a fixed corpus — no task-specific labels, no
   generation, no retraining, no benchmark scores.
2. **Reproducible**: either a reference implementation exists (code
   repository) or the paper's formula is unambiguous enough to pin
   with unit tests.
3. **Non-duplicative**: the metric measures something BLME doesn't
   already capture under another name.
4. **Cross-architecture comparable**: metric is defined without
   tokeniser- or architecture-specific thresholds that would
   invalidate cross-family comparison.
5. **Computable on 30B-class models within ~15 min per run**.

See `docs/PAPER_SURVEY.md` §2 for the full inclusion list and §3 for
the 70+ papers we considered and rejected with explicit reasons
against each criterion.

---

## Appendix B — paper count by section

Foundational papers (pre-2023 or seminal): **39**.
Recent papers (2023–2026) implemented: **16**.
Recent papers (2023–2026) considered and rejected: **70+** (see
`docs/PAPER_SURVEY.md` §3).
Total in `docs/PAPERS.md` §1 (implemented): **55+**.
Total arXiv IDs actually cited in source code: **33**.
Tasks with explicit in-source citations: **37 / 71** (52 %).
Tasks with citation in paper docs only: **29 / 71** (41 %).
Pure BLME diagnostics with no canonical paper: **5 / 71** (7 %).

For the canonical per-task audit and paper-to-task mapping, see
`docs/PAPERS.md` §3.
