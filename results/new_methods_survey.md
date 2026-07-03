# BLME New-Methods Gap Survey (2023–2026)

**Goal.** Find diagnostic / representation / interpretability methods that BLME's
74-task portfolio does **not** include but arguably should, to make the portfolio
comprehensive. Each candidate is screened against BLME's selection criteria
(`PAPER_SURVEY.md` §4): *intrinsic* (weights + static forward pass, no task
labels, no generation, no retraining), *reproducible* (official/standard code or
unambiguous formula), *non-duplicative*, *cross-architecture comparable*,
*computable on 30B-class models*. It must yield a scalar/small-vector per model.

**Method.** ~20 targeted web searches across the requested gap axes (uncertainty/
hallucination geometry, universality/CKA-alternatives, weight-space spectral,
long-context/positional, quantization/robustness geometry, emergent/grokking,
SAE-free monosemanticity, label-light factuality, reasoning-trace geometry,
modern intrinsic-dimension/curvature). Every arXiv id and repo URL below was
verified to exist via search; arxiv.org/github.com direct-fetch was network-
blocked in this environment, so titles/authors/ids are confirmed from search
result snippets (multiple corroborating hits), not from fetching the PDF.

Status legend (parity-verifiability): **closed-form** = single scalar from a
formula over a matrix/SVD/eigendecomposition (easy parity test); **pipeline** =
multi-step but deterministic; **generation/label** = needs sampling or labels
(fails BLME mold).

---

## Ranked candidate table

Rank = portfolio-value × parity-verifiability × feasibility. "MUST-ADD" = clear
gap + official code + closed-form + fits the mold.

| # | Method | Paper (arXiv / venue) | Official repo (conf.) | BLME family | Form | Fits mold? | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Layer-linearity / Procrustes similarity** | Razzhigaev et al. 2024, *Your Transformer is Secretly Linear*, arXiv:2405.12250, ACL 2024 | [AIRI-Institute/LLM-Microscope](https://github.com/AIRI-Institute/LLM-Microscope) (HIGH; pip `llm-microscope`, exposes `procrustes_similarity`) | geometry/dynamics | closed-form (Procrustes residual per layer pair) | yes | **MUST-ADD** |
| 2 | **PHD intrinsic dimension** (persistent-homology dim of per-sentence token cloud) | Tulchinskii et al. 2023, *Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts*, arXiv:2306.04723, NeurIPS 2023 | [ArGintum/GPTID](https://github.com/ArGintum/GPTID) (HIGH) | geometry ∩ topology | pipeline (PH dim estimator, deterministic) | yes | **MUST-ADD** |
| 3 | **Vendi Score** (effective # of dissimilar elements = exp Shannon-entropy of kernel-matrix eigenvalues) | Friedman & Dieng 2023, *The Vendi Score*, arXiv:2210.02410, TMLR 2023 | [vertaix/Vendi-Score](https://github.com/vertaix/Vendi-Score) (HIGH; pip `vendi-score`) | geometry (rank/diversity) | closed-form | yes | **MUST-ADD** |
| 4 | **Zigzag persistence descriptors** (cross-layer topological feature lifespans) | Aksenov/Barannikov et al. 2025, *Persistent Topological Features in LLMs*, arXiv:2410.11042, ICML 2025 | [RitAreaSciencePark/ZigZagLLMs](https://github.com/RitAreaSciencePark/ZigZagLLMs) (HIGH) | topology | pipeline (zigzag PH, deterministic) | yes | strong add |
| 5 | **PH-Dim of representation point cloud** (Birdal-Lim TDA ID estimator) | Birdal, Lou, Guibas, Simsekli 2021, arXiv:2111.13171, NeurIPS 2021 | [tolgabirdal/PHDimGeneralization](https://github.com/tolgabirdal/PHDimGeneralization) (HIGH) | geometry ∩ topology | pipeline (Ripser-based, CPU) | yes | add (alt to #2) |
| 6 | **Metric-space Magnitude** (multi-scale effective size of latent metric space) | Limbeck, Andreeva, Sarkar, Rieck 2024, arXiv:2311.16054, NeurIPS 2024 | [aidos-lab/magnipy](https://github.com/aidos-lab/magnipy) (HIGH) | geometry/topology | closed-form (per scale) | yes | add |
| 7 | **Activation kurtosis** (4th-moment outlier/quantizability signature) | Akhondzadeh et al. 2025, *KurTail*, arXiv:2503.01483, EMNLP 2025 Findings | no clean official repo (method = scipy `kurtosis`) | geometry/robustness | closed-form | yes | add (robustness gap) |
| 8 | **CKNNA** (Centered Kernel Nearest-Neighbor Alignment; local-structure CKA variant) | Huh et al. 2024, *Platonic Representation Hypothesis*, arXiv:2405.07987, ICML 2024 | [minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep) (HIGH) | geometry (similarity) | closed-form | yes (needs 2nd ref space) | conditional add |
| 9 | **EigenScore** (log-det / mean-log-eigenvalue of covariance of N sampled responses) | Chen et al. 2024, *INSIDE*, arXiv:2402.03744, ICLR 2024 | [D2I-ai/eigenscore](https://github.com/D2I-ai/eigenscore) (HIGH) | uncertainty geometry | generation pipeline | **no** (needs sampling) | reject (same class as semantic entropy) |
| 10 | **Kernel Language Entropy (KLE)** | Nikitin et al. 2024, arXiv:2405.20003, NeurIPS 2024 | [AlexanderVNikitin/kernel-language-entropy](https://github.com/AlexanderVNikitin/kernel-language-entropy) (HIGH) | uncertainty | generation + similarity model | **no** | reject (same as semantic entropy) |
| 11 | **Attention-map spectral features** (Markov-chain spectral gap of attention) | Caspari et al. 2025, arXiv:2502.17598 | repo not confirmed | interpretability (attention) | closed-form | partially (needs labels for their use) | weak / watch |
| 12 | **Lookback-Lens ratio** (context-vs-generated attention mass) | Chuang et al. 2024, EMNLP 2024, arXiv:2407.07071 | [voidism/Lookback-Lens](https://github.com/voidism/Lookback-Lens) (HIGH) | interpretability (attention) | closed-form (but RAG-contextual) | partially | watch (overlaps attention_graph) |
| 13 | **VNE** (von Neumann entropy of autocorrelation matrix) | Kim et al. 2023, arXiv:2304.01434, CVPR 2023 | [jaeill/CVPR23-VNE](https://github.com/jaeill/CVPR23-VNE) (HIGH) | geometry | closed-form | yes | **already covered** by `geometry_matrix_entropy` |
| 14 | **Confidence-regulation / entropy neurons** | Stolfo, Wu, Gurnee et al. 2024, arXiv:2406.16254, NeurIPS 2024 | code via authors (MEDIUM) | interpretability | pipeline (null-space + weight-norm) | borderline | watch (overlaps prediction_entropy) |
| 15 | **Injectivity / collision rate** (SipIt) | Nikolaou et al. 2025, arXiv:2510.15511 | repo emerging (LOW) | geometry (theory) | n/a (existence result) | no scalar capability signal | reject |

---

## Per-candidate detail (top tier)

### 1. Layer-linearity / Procrustes similarity — **MUST-ADD**
- **Paper.** Razzhigaev, Mikhalchuk, Goncharova, Gerasimenko, Oseledets, Kuznetsov.
  *Your Transformer is Secretly Linear.* arXiv:2405.12250, **ACL 2024**. (id + venue verified)
- **Repo.** [AIRI-Institute/LLM-Microscope](https://github.com/AIRI-Institute/LLM-Microscope) — **HIGH**.
  Pip package `llm-microscope` exposes `procrustes_similarity` plus anisotropy/ID
  helpers. This is the official code for the paper.
- **Measures.** How well each layer-to-layer hidden-state transform is approximated
  by a single orthogonal (Procrustes) map: linearity score ≈ 0.99 for decoders.
- **Family.** geometry / dynamics (layer-transition geometry).
- **Closed-form?** Yes. Per adjacent layer pair `(X, Y)` of mean-centered hidden
  states: best orthogonal `R` via SVD of `XᵀY`, score = explained-variance of
  `XR` vs `Y` (1 − normalized residual). Aggregate mean/min over depth → scalar.
- **5-line compute.** `X = h_l - mean; Y = h_{l+1} - mean; U,_,Vt = svd(Xᵀ Y);
  R = U Vt; resid = ||XR − Y||_F² / ||Y||_F²; score_l = 1 − resid; report
  mean_l score_l, min_l score_l.`
- **Non-redundant?** YES. BLME has CKA (similarity), trajectory_curvature
  (angle of step-vectors), and matrix entropy (rank). None measures *orthogonal-
  linear approximability* of the layer transition. The Procrustes residual is a
  distinct signal (it penalizes non-orthogonal/non-linear warps that CKA
  tolerates). Distinct from `geometry_cka` (which is invariant to orthogonal
  *and* isotropic-scale transforms by construction).
- **Feasibility.** CPU forward-pass only; no labels; closed-form SVD of d×d.
  30B-class fine (d up to ~8k SVD is cheap). **Best single add: clean gap,
  official code, closed-form, trivial parity test.**

### 2. PHD intrinsic dimension (GPTID) — **MUST-ADD**
- **Paper.** Tulchinskii, Kuznetsov, Kushnareva, Cherniavskii, Nikolenko, Burnaev,
  Barannikov, Piontkovskaya. *Intrinsic Dimension Estimation for Robust Detection
  of AI-Generated Texts.* arXiv:2306.04723, **NeurIPS 2023**. (verified)
- **Repo.** [ArGintum/GPTID](https://github.com/ArGintum/GPTID) — **HIGH** (official; has `example.ipynb`).
- **Measures.** Persistent-Homology Dimension (PHD) of the per-sentence static/
  contextual token point cloud — a *topological* intrinsic-dimension estimator.
  Human text ≈ 9–10, generated ≈ 8; a clean capability/fluency-linked signal.
- **Family.** geometry ∩ topology.
- **Closed-form?** Pipeline but deterministic: fit `E(n) ∝ n^{(d−1)/d}` to the
  total H₀ persistence of random subsamples of size n; slope → PHD.
- **5-line compute.** `for n in sizes: take random subsample; ph0 = sum H0
  lifespans (Ripser); record (log n, log ph0); slope α = lstsq; PHD = 1/(1−α).`
- **Non-redundant?** YES vs `geometry_intrinsic_dim` (Two-NN) and `geometry_lid`
  (Levina-Bickel MLE): PHD is a *global topological* ID estimator (combines local
  + global structure via persistence), known to disagree with MLE/Two-NN in the
  regimes BLME cares about. The skip-list rejected Cavagnero (= MLE, duplicate)
  and Robinson (qualitative) but **never** evaluated a PH-based ID estimator with
  official code. This fills a real "modern ID variant" gap.
- **Feasibility.** CPU (Ripser H₀ only — fast); no labels; deterministic.

### 3. Vendi Score — **MUST-ADD**
- **Paper.** Friedman & Dieng. *The Vendi Score: A Diversity Evaluation Metric
  for Machine Learning.* arXiv:2210.02410, **TMLR 2023**. (verified)
- **Repo.** [vertaix/Vendi-Score](https://github.com/vertaix/Vendi-Score) — **HIGH** (pip `vendi-score`).
- **Measures.** Effective number of dissimilar elements: `exp(−Σ λᵢ log λᵢ)`
  where `λᵢ` are eigenvalues of the normalized `n×n` similarity (Gram) matrix
  `K/n` (trace 1). A rank/diversity scalar with an *interpretable* unit.
- **Family.** geometry (rank/diversity).
- **Closed-form?** Yes — eigenvalues of the Gram matrix; label-free; no reference set.
- **5-line compute.** `K = kernel(H, H); K = K/n; w = eigvalsh(K); w = w[w>0];
  VS = exp(−Σ w log w).` (q=1 order; q=∞ and q=0 orders also defined.)
- **Non-redundant?** Mostly YES. BLME's `effective_rank` normalizes σ² of the
  *covariance*; RankMe normalizes raw σ of the design matrix. Vendi normalizes
  eigenvalues of a *kernel/Gram* matrix and uses Shannon (not Rényi-2) entropy in
  the exponent — it generalizes to arbitrary similarity kernels (cosine, RBF),
  which the covariance-based metrics cannot. With a *linear* kernel it is closely
  related to effective_rank; with a *nonlinear* kernel it is genuinely new signal.
  Recommend the cosine/RBF-kernel variant to avoid duplicating effective_rank.
- **Feasibility.** CPU; eigendecomp of n×n (n = #tokens/sentences, keep ≤ 2k).

### 4. Zigzag persistence descriptors (ZigZagLLMs) — strong add
- **Paper.** *Persistent Topological Features in Large Language Models.*
  arXiv:2410.11042, **ICML 2025** (poster). (verified; authors incl. Barannikov / RIT
  Area Science Park group)
- **Repo.** [RitAreaSciencePark/ZigZagLLMs](https://github.com/RitAreaSciencePark/ZigZagLLMs) — **HIGH**.
- **Measures.** Cross-layer persistence of p-dim holes via **zigzag** persistence
  with a kNN filtration — tracks the *full evolutionary path* of topological
  features across layers (4 distinct processing phases), not per-layer-then-aggregate.
- **Family.** topology.
- **Non-redundant?** YES vs `topology_betti_curve` (per-layer β₀/β₁ then aggregate)
  and `topology_homology` (single point cloud). Zigzag tracks *feature identity
  across layers* — a signal BLME's per-layer-aggregate topology demonstrably
  discards. This is the one topology paper in the skip-list neighborhood with
  official code and a genuinely new descriptor.
- **Caveat.** Heaviest of the top tier (zigzag PH across all layers); subsample
  tokens aggressively. Output = small vector of descriptor summaries.

### 5. PH-Dim of representations (PHDimGeneralization) — add (alternative to #2)
- **Paper.** Birdal, Lou, Guibas, Simsekli. *Intrinsic Dimension, Persistent
  Homology and Generalization in Neural Networks.* arXiv:2111.13171, **NeurIPS 2021**.
- **Repo.** [tolgabirdal/PHDimGeneralization](https://github.com/tolgabirdal/PHDimGeneralization) — **HIGH** (`calculate_ph_dim`).
- Same PH-dimension estimator family as #2 (GPTID is essentially the NLP
  application of this). Pick **one** of #2/#5; GPTID (#2) is the LLM-tuned variant
  with verified separation results, so prefer it. PHDimGeneralization is the
  canonical "ID-of-point-cloud via PH" reference for parity.

### 6. Metric-space Magnitude (magnipy) — add
- **Paper.** Limbeck, Andreeva, Sarkar, Rieck. *Metric Space Magnitude for
  Evaluating the Diversity of Latent Representations.* arXiv:2311.16054, **NeurIPS 2024**.
- **Repo.** [aidos-lab/magnipy](https://github.com/aidos-lab/magnipy) — **HIGH**.
- **Measures.** Magnitude function `|tX|` of the latent metric space across
  scales `t` — a multi-scale invariant capturing effective size, curvature,
  density, entropy simultaneously. Summaries (area under magnitude curve, magnitude
  dimension) → small vector.
- **Non-redundant?** YES — multi-scale; no other BLME metric is scale-resolved.
  Theoretically links to ID, entropy, and curvature in one object. Newer / less
  battle-tested than #1–3, hence rank 6.
- **Feasibility.** Solve `(ζ) w = 1` for the similarity matrix `ζ_ij = exp(−t d_ij)`
  at several t; CPU, n ≤ ~2k.

### 7. Activation kurtosis — add (fills robustness/quantization gap)
- **Paper.** Akhondzadeh, Bojchevski, Eleftheriou, Dazzi. *KurTail: Kurtosis-based
  LLM Quantization.* arXiv:2503.01483, **EMNLP 2025 Findings**. (verified)
- **Repo.** No clean official repo found; but the *measurement* is the standard
  4th standardized moment (`scipy.stats.kurtosis`) — trivially parity-able and
  paper-faithful as a formula. (Confidence on a dedicated repo: NONE/LOW.)
- **Measures.** Per-channel / per-layer excess kurtosis of residual-stream and
  FFN activations — directly indexes outlier prevalence and *quantizability*.
- **Family.** geometry / robustness.
- **Non-redundant?** YES. BLME's `interpretability_activation_sinks` (Sun 2024
  massive-activation *fraction* with a `>100× median` threshold) and `superposition`
  capture *magnitude* outliers, not the *distributional tail shape*. Kurtosis is a
  smooth, threshold-free 4th-moment that the literature (KurTail, AMXFP4) shows
  predicts quantization error. This is the cleanest entry for the **quantization/
  robustness-geometry gap** the brief explicitly flags as possibly missing.
- **Feasibility.** CPU; closed-form; no labels.

### 8. CKNNA (local-structure CKA) — conditional add
- **Paper.** Huh, Cheung, Wang, Isola. *The Platonic Representation Hypothesis.*
  arXiv:2405.07987, **ICML 2024**. CKNNA (Centered Kernel Nearest-Neighbor
  Alignment) is the mutual-kNN-masked CKA introduced there.
- **Repo.** [minyoungg/platonic-rep](https://github.com/minyoungg/platonic-rep) — **HIGH**.
- **Measures.** CKA restricted to pairs that are mutual k-NN in *both* spaces —
  local-neighborhood alignment; sensitive where global CKA is not.
- **Non-redundant?** Partially. BLME has linear CKA/HSIC/RSA. CKNNA is a *local*
  alignment that catches structure CKA misses. BUT like CKA it is a *two-space*
  metric — to be a single-model scalar it must compare adjacent layers (or to a
  fixed reference), exactly as `geometry_cka` already does. So it's a CKA-family
  refinement, not a new family. Platonic-rep was already in REPOSITORIES.md
  universality section but **not** as a BLME task. Add only if a layer-adjacent
  CKNNA gives signal beyond layer-adjacent CKA (likely, but verify empirically).

---

## Explicit coverage-gap audit (brief's checklist a–j)

- **(a) Uncertainty / hallucination geometry** — *mostly already-rejected-class.*
  EigenScore (#9), KLE (#10), semantic entropy (already rejected), Lookback-Lens
  (#12, RAG-contextual) all need **generation or labels** → fail the static-forward
  mold for the same reason Farquhar 2024 was rejected. The *single-pass* core of
  EigenScore (log-det of token-covariance) reduces to BLME's `matrix_entropy`.
  **No must-add here.** Attention-map spectral features (#11) is the only
  single-pass option and its repo is unconfirmed.
- **(b) Feature-universality / stitching / CKA-alternatives** — CKNNA (#8) is the
  realistic add; mutual-kNN alignment. Model-stitching proper needs a 2nd trained
  model → out of mold. Platonic/Relative-reps already logged as universality
  (not tasks). **Partial gap; #8 conditional.**
- **(c) Weight-space spectral / scaling-law** — well covered (`geometry_spectral`
  α, `geometry_mp_bulk_deviation`, `geometry_schatten` MNN/RankMe). FARMS, SUI,
  AlphaDecay, Jha-Reagen all correctly rejected as duplicates. **No gap.**
- **(d) Long-context / positional structure** — `geometry_positional_decay` covers
  attention-vs-distance. RoPE high/low-frequency split (semantic vs positional)
  is a *possible* refinement but no single clean reference-code scalar emerged;
  most positional work is architecture-modification, not measurement. **Minor gap,
  low-confidence; not ranked.**
- **(e) Quantization / robustness geometry** — **real gap.** Activation **kurtosis**
  (#7) is the clean fill. Jacobian/spectral-norm robustness overlaps existing
  `geometry_lipschitz` + `dynamics_sharpness`. **#7 add.**
- **(f) Emergent-ability / grokking** — progress measures (weight entropy, circuit
  complexity, info-theoretic) **require training checkpoints** → out of mold (BLME
  evaluates finished models; already rejected this class). HTSR-grokking = existing
  `geometry_spectral`. **No add.**
- **(g) SAE-free monosemanticity** — the field is still SAE-bound (FMS, PRISM,
  feature-decorrelation) — all need a trained SAE or LLM-generated descriptions.
  BLME's `interpretability_superposition` is the SAE-free proxy. **No clean add.**
- **(h) Label-light knowledge / factuality** — logit-lens-based factuality and
  entropy/confidence neurons (#14) overlap `interpretability_logit_lens` +
  `interpretability_prediction_entropy`. Truthfulness probes need labels. **No add.**
- **(i) Reasoning-trace geometry** — CoE is in. Everything newer (REMA, reasoning
  manifold, stepwise informativeness, FAI/WAAD) needs **CoT generation + correctness
  labels** → out of mold (already rejected). **No add.**
- **(j) Modern intrinsic-dimension / curvature** — **real gap.** PHD/PH-dim (#2,#5)
  is a topological ID estimator BLME lacks (it only has Two-NN + MLE-LID). Magnitude
  (#6) is a multi-scale curvature/size invariant. **#2 must-add, #6 add.**

---

## Honest redundancy calls (gaps that are NOT actually gaps)
- **VNE (#13)** = von Neumann entropy of representation autocorrelation = BLME's
  `geometry_matrix_entropy` (Wei 2024). The CVPR paper uses it as a *regularizer*;
  the *measurement* is identical. **Covered.**
- **EigenScore single-pass core** = log-det of covariance eigenvalues = a monotone
  function of `matrix_entropy`. **Covered** (only the multi-generation version is
  new, and that fails the mold).
- **α-ReQ / SUI / FARMS / AlphaDecay** = `geometry_spectral`. **Covered** (already in skip list).
- **Confidence/entropy neurons (#14)** = mechanism behind `prediction_entropy`;
  identifying the neurons adds a pipeline without a clearly new capability scalar.
- **Lookback-Lens (#12)** context-attention ratio overlaps `interpretability_
  attention_graph` (BOS-sink ratio / edge Gini) and is RAG-context-specific.

---

## Bottom line — recommended additions, in priority order
1. **Layer-linearity / Procrustes** (2405.12250, LLM-Microscope) — MUST-ADD.
2. **PHD intrinsic dim** (2306.04723, GPTID) — MUST-ADD.
3. **Vendi Score** (2210.02410, vertaix/Vendi-Score) — MUST-ADD (use nonlinear kernel).
4. **Zigzag persistence** (2410.11042, ZigZagLLMs) — strong add (topology family).
5. **Activation kurtosis** (2503.01483) — add (quantization/robustness gap).
6. **Metric-space Magnitude** (2311.16054, magnipy) — add (multi-scale).
7. **CKNNA** (2405.07987, platonic-rep) — conditional (verify it beats adjacent CKA).

All seven fit the single-model, label-light, static-forward mold and have
verified official/standard code for parity. Items 1–3 are closed-form with
trivial parity tests and fill distinct, real gaps (layer-linearity, topological
ID, kernel-diversity). Items 4–6 are deterministic pipelines with official repos.
The uncertainty/hallucination, grokking, SAE-monosemanticity, and reasoning-trace
"gaps" are genuinely out-of-mold or already covered and should stay rejected.
