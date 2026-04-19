# Experimental correlations between intrinsic metrics and LLM performance

Direct answer to "which papers experimentally correlate intrinsic
metrics with downstream performance". This file is a focused
annex to `PAPER_SURVEY.md` — it lists only papers that run the
actual experiment.

Grouped by metric family. For each: the claimed relationship, the
experimental scope, and whether BLME captures that signal.

---

## 1. Spectral / weight-matrix metrics

### Martin & Mahoney 2019–2021 — Heavy-Tailed Self-Regularization (HT-SR)
- arXiv:1901.08276 ("Traditional and Heavy-Tailed Self Regularization")
- arXiv:1901.08278 ("Heavy-Tailed Universality Predicts Trends in Test
  Accuracies")
- *Nature Communications* 2021, "Predicting trends in the quality of
  state-of-the-art neural networks without access to training or
  testing data".
- Code: [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher).
- **Claim**: power-law exponent α of the per-layer weight-ESD
  (Empirical Spectral Density) correlates with held-out test
  accuracy. α ∈ [2, 6] → good generalisation; α > 6 or α < 2 →
  poor. Weighted-α ("universal capacity control") correlates
  strongly with test accuracy across hundreds of pretrained
  vision/NLP models in the Nature Comms paper.
- **Experimental scope**: hundreds of CV + NLP pretrained models,
  data-free (no training-data access needed).
- **BLME status**: ✅ implemented as
  `geometry_spectral.avg_alpha` / `min_alpha` / `max_alpha`
  (round-3-audited for Conv1D coverage and Roy-Vetterli
  convention).

### Garrido et al. 2023 — RankMe (ICML)
- arXiv:2210.02885.
- **Claim**: `exp(H(p_i))` with `p_i = σ_i / Σσ_j` of the embedding
  matrix **predicts downstream probe accuracy** of SSL models
  without any labels.
- **Scope**: SimCLR, VICReg, DINO, MAE across ImageNet linear-probe
  benchmarks.
- **BLME status**: ✅ implemented in round-7 `geometry_schatten.rankme`.

### Li et al. 2025 — Tracing Representation Geometry (NeurIPS 2025)
- arXiv:2509.23024.
- **Claim**: the joint trajectory of RankMe (effective rank) and
  α-ReQ (eigenvalue decay rate) across pretraining defines three
  geometric phases; the transition into the "compression-seeking"
  phase **coincides with the improvement in downstream task
  performance**.
- **Scope**: OLMo 1B-7B, Pythia 160M-12B across pretraining
  checkpoints + downstream benchmarks.
- **BLME status**: ✅ the metrics (RankMe, α-ReQ) are in
  `geometry_schatten.rankme` and `geometry_spectral.avg_alpha`
  respectively; phase labelling is derivable post-hoc from our
  per-layer profiles. Code release pending.

### Jha & Reagen 2025 — Spectral Scaling Laws (EMNLP 2025)
- arXiv:2510.00537.
- **Claim**: asymmetric power-law between Soft Rank (Shannon rank of
  normalised singular values) and FFN width, with Hard Rank
  (participation ratio) growing only sublinearly. Defines a
  **Spectral Utilization Index** (SUI) composite of both ranks +
  spectral concentration.
- **Scope**: LLaMA, GPT-2, nGPT families; correlates with loss.
- **BLME status**: 🟡 Hard Rank = our `geometry_svd.participation_ratio`;
  Soft Rank = our `effective_rank` and `rankme`; Spectral
  Concentration derivable from existing per-layer singular-value
  lists. SUI is a composite the aggregator can compute.

### Wei et al. 2024 — Matrix Entropy / Diff-eRank
- arXiv:2401.17139.
- **Claim**: matrix entropy of the normalised token covariance
  **follows a scaling law that tracks the loss-scaling law** —
  decreases as a power of model size.
- **Scope**: Pythia family (14M-12B), LLaMA 7B-65B, plus multimodal
  LLMs.
- **BLME status**: ✅ implemented as `geometry_matrix_entropy`
  (round-3-audited to per-sentence + row-L2-normalise + /log d).

### Li et al. 2024 — Matrix Nuclear-Norm
- arXiv:2410.10672.
- **Claim**: MNN (L1,2-norm of normalised hidden-state matrix)
  **decreases with model size**, tracking the compression-efficiency
  scaling law, at 8-24× speedup over matrix entropy.
- **Scope**: Cerebras-GPT (111M-6.7B), LLaMA 7B-65B.
- **BLME status**: ✅ implemented in round-7 `geometry_schatten`.

### Wei et al. 2025 — From Internal Representations to Text Quality
- arXiv:2509.25359.
- **Claim**: Intrinsic Dim, Effective Rank, MEV, MAUVE, and
  Schatten norms are **reference-free text-quality proxies** that
  produce consistent rankings of text generators across 6 models
  from 0.5B to 8B.
- **Scope**: 0.5B-8B autoregressive + diffusion text generators.
- **BLME status**: ✅ all 5 metrics are in BLME — `geometry_intrinsic_dim`,
  `geometry_svd.effective_rank`, `geometry_contextualization.per_layer.*.mev`,
  and `geometry_schatten.schatten_{1,2,4,inf}_last`.

---

## 2. Attention / activation metrics

### Gu et al. ICLR 2025 — Attention Sink Emergence
- arXiv:2410.10781.
- **Claim**: Sinkε metric depends on optimiser, tokeniser, attention
  kernel, pretraining-data composition. **Models without sinks
  perform worse on long-context tasks** — shows a correlation between
  sink strength and retrieval performance.
- **Scope**: LLaMA 7B-70B, Mistral, Qwen, Pythia 70M-12B, plus
  models trained from scratch with controlled varations.
- **BLME status**: ✅ in round-8 `interpretability_activation_sinks.sink_epsilon_fraction`.

### Sun et al. 2024 — Massive Activations
- arXiv:2402.17762.
- **Claim**: a few hidden-state entries (<0.01% of positions) are
  100-1000× the typical magnitude; **present in every modern
  LLM**, **fixed across inputs**, and **removing them destroys
  model performance** — causal evidence that massive activations
  encode a critical bias. Per-layer fraction and max/median ratio
  correlate with quantisation difficulty and downstream capability.
- **Scope**: LLaMA-1/2/3, Mistral, Vicuna, GPT-J, Pythia.
- **BLME status**: ✅ in round-8
  `interpretability_activation_sinks.massive_activation_{fraction,max_ratio}`.

### Pedrotti & Guo 2025 — Compression Valleys
- arXiv:2510.06477.
- **Claim**: sinks and massive activations are mechanically linked
  (massive activations → attention sink) and **produce a provable
  mid-depth entropy valley** whose depth correlates with the
  onset of in-context learning. Empirical sweep 410M-120B.
- **BLME status**: ✅ in round-8
  `interpretability_activation_sinks.valley_{layer,layer_norm,depth}`.

---

## 3. Intrinsic-dimension metrics

### Cavagnero et al. 2025 — Less is More (NAACL 2025)
- arXiv:2506.01034.
- **Claim**: mean of local intrinsic dimensions across tokens
  **predicts generalisation improvement, grokking onset, overfitting
  onset, and fine-tuning exhaustion** — all unsupervised.
- **Scope**: OLMo-1B, Pythia 70M-1B across pretraining checkpoints
  and fine-tuning runs.
- **BLME status**: ✅ implemented as `geometry_lid` (Levina-Bickel MLE).

### Yin et al. 2024 — Truthfulness via LID
- arXiv:2402.18048.
- **Claim**: local intrinsic dim of the response hidden state
  **predicts TruthfulQA correctness** via a shallow probe.
- **Scope**: LLaMA, GPT-J on TruthfulQA.
- **BLME status**: ✅ LID itself in `geometry_lid`; the probe-for-
  truthfulness part is a downstream task we don't label.

### Bonfanti et al. 2025 — Geometry of Tokens
- arXiv:2501.10573.
- **Claim**: per-token intrinsic dim + neighbourhood overlap +
  cosine similarity at each layer **correlate with per-token
  cross-entropy loss**.
- **Scope**: LLaMA, Pythia at multiple scales.
- **BLME status**: ✅ intrinsic dim + contextualisation cosines
  already in BLME.

---

## 4. Information-theoretic / layer-quality metrics

### Rao et al. 2025 — Layer by Layer (ICLR 2025)
- arXiv:2502.02013.
- **Claim**: DiME (difference of matrix-based entropies), infoNCE,
  and dataset entropy of token hidden states **negatively correlate
  with downstream probe accuracy at each layer** — selecting the
  best layer via these unsupervised metrics recovers near-optimal
  probe performance.
- **Scope**: 10+ LLMs, 30+ probing tasks.
- **BLME status**: 🟡 dataset entropy is a variant of our
  `geometry_matrix_entropy`; the per-layer relationship-to-probe-
  accuracy is implicit in our `interpretability_probing` layer sweep.

### Cao et al. 2025 — Model Utility Law (MUI)
- arXiv:2504.07440.
- **Claim**: MUI (fraction of active neurons/features during
  inference) follows an **inverse-log relationship with task
  accuracy** ("Utility Law") — more capable models expend less
  effort per task.
- **Scope**: GPT-2 (SAE-equipped), LLaMA-2 (neuron-level), Qwen,
  Mistral across MMLU, GSM8K, BBH.
- **BLME status**: 🔴 skipped (requires SAE or per-task-labelled
  neuron contributions; not a pure intrinsic metric).

---

## 5. Topology / persistent-homology metrics

### Naitzat, Zhitnikov, Lim 2020 — Topology of Deep Neural Networks
- ICLR 2020.
- **Claim**: Betti numbers (β₀ = connected components, β₁ = loops)
  decrease monotonically with depth in well-generalising networks;
  the **rate of decrease correlates with generalisation**.
- **BLME status**: ✅ `topology_betti_curve` with round-6-audited
  normalised-depth decay rate.

---

## 6. Scaling-law / cross-model experimental studies

### Kaplan et al. 2020 — Scaling Laws for Neural LMs
- arXiv:2001.08361.
- **Claim**: loss follows power laws in model size, dataset size,
  compute. Every paper in this list cites it.
- **BLME status**: our `geometry_perplexity` (NLL, PPL, BPC) feeds
  into any scaling-law fit.

### Hoffmann et al. 2022 — Chinchilla
- arXiv:2203.15556.
- **Claim**: optimal compute allocation between parameters and
  tokens; same power-law mechanics.
- **BLME status**: same as above.

### Gonçalves et al. 2024 — Collaborative Performance Prediction
- arXiv:2407.01300.
- **Claim**: latent task+model embeddings **predict held-out model
  performance** on downstream benchmarks.
- **BLME status**: 🔴 requires cross-model collaborative filtering;
  not a per-model intrinsic metric.

### Wu et al. 2024 — Performance Law of LLMs
- arXiv:2408.09895.
- **Claim**: simple parametric formula predicts MMLU scores from
  0.5B to 1T+ using only 10 open-source anchor models.
- **BLME status**: 🔴 cross-model formula, not intrinsic.

### Owen et al. 2024 — 100 instances is all you need
- arXiv:2409.03563.
- **Claim**: accurate full-benchmark prediction from only 100
  labelled instances.
- **BLME status**: 🔴 requires benchmark labels.

### Isik et al. 2024 — Sloth scaling laws
- arXiv:2412.06540.
- **Claim**: benchmarks are low-dimensional projections of latent
  skills; fits scaling laws per skill axis.
- **BLME status**: 🔴 requires benchmark scores.

### Ruan et al. 2025 — Clustering-Based Downstream Scaling
- arXiv:2502.17262.
- **Claim**: clusters of related benchmarks have predictable
  scaling; full-set performance predicted at 1.55% error on 70B.
- **BLME status**: 🔴 cross-benchmark prediction, not per-model
  intrinsic.

---

## 7. Hidden-state probing for correctness prediction

### Azaria & Mitchell 2023 — LLM factoscope / internal-state probes
- arXiv:2312.16374.
- **Claim**: a shallow probe on inter-layer hidden-state changes
  **achieves >96% accuracy detecting factual vs non-factual
  generations**, across LLaMA, GPT-3.5.
- **BLME status**: 🟡 we expose `interpretability_probing` as a
  baseline linear probe on next-token-id labels; the factoscope
  requires labelled correct/false pairs — out of scope for a
  label-free diagnostic.

### Girrbach et al. 2025 — Reference-Free Rating via Latent Info
- HCAI Munich 2025.
- **Claim**: a reference-free probe on response-hidden-state latent
  information **predicts human quality ratings**.
- **BLME status**: 🔴 requires ratings for probe training.

### Liu et al. 2025 — Mining Intrinsic Rewards from Hidden States
- arXiv:2505.12225.
- **Claim**: linear probe on response hidden states **predicts
  reasoning correctness** (for best-of-N sampling).
- **BLME status**: 🔴 requires correct/incorrect labels.

### Kadavath et al. 2022 — Language Models (Mostly) Know What They Know
- arXiv:2207.05221.
- **Claim**: a self-prompted "P(IK)" probe **calibrates LLM
  confidence against correctness** — strong correlation between
  P(IK) and actual accuracy.
- **BLME status**: 🟡 requires question-answering + label data; we
  report calibration via `consistency_calibration.ece` on
  downstream tasks, not intrinsic.

---

## 8. Direct summary — experimental-correlation papers grouped by BLME status

**✅ All claimed correlations are measured by a BLME task already** (14 papers):
Martin-Mahoney 2019/2021, Garrido 2023 RankMe, Li 2025 Tracing
Geometry (metrics yes, phase-label no), Wei 2024 matrix entropy,
Li 2024 MNN, Wei 2025 Text-Quality Geometric, Gu 2025 Sinkε,
Sun 2024 massive activations, Pedrotti & Guo 2025 compression
valleys, Cavagnero 2025 Local ID, Yin 2024 truthfulness LID,
Bonfanti 2025 token geometry, Naitzat 2020 Betti, Kaplan 2020
+ Hoffmann 2022 scaling laws.

**🟡 Correlation is measurable by BLME tasks but requires downstream label** (3 papers):
Rao 2025 Layer-by-Layer, Kadavath 2022 P(IK), Azaria 2023
Factoscope.

**🔴 Requires cross-model benchmark access, SAE training, or task
labels — out of scope** (8 papers):
Cao 2025 MUI, Gonçalves 2024 CPP, Wu 2024 Performance Law, Owen
2024 "100 instances", Isik 2024 Sloth, Ruan 2025 Clustering,
Liu 2025 Intrinsic Rewards, Girrbach 2025 Reference-Free.

### Most impactful experimental-correlation finding for BLME

**Pedrotti & Guo 2025** (arXiv:2510.06477) unifies three metrics
(attention sinks, massive activations, compression valleys) and
**proves theoretically** that they're manifestations of a single
mechanism — and shows the mid-layer entropy-valley depth
correlates with in-context-learning strength across 410M-120B.
This is the single strongest recent theoretical+experimental
correlation result for a set of intrinsic metrics, and all three
are now captured in BLME round 8.

### Runner-up

**Martin & Mahoney** Nature Communications 2021 — the landmark
result that **α alone** (per-layer heavy-tailed exponent)
predicts held-out test accuracy across hundreds of models
**without any training or test data**. This is the theoretical
foundation for BLME's `geometry_spectral.avg_alpha`.

---

## 9. What's missing

A few correlation directions I did not find dedicated
experimental papers for (and would be publishable if we ran them):

- **Schatten-p_norm vs. composite-benchmark ρ** in our 32-model
  grid (we already reported ρ ≈ -0.75 for Schatten-1 — no prior
  paper has done this at this scale across tokeniser families).
- **Compression-valley depth vs. in-context learning slope** —
  Pedrotti & Guo argue for this but didn't sweep 30+ instruct
  models.
- **Sinkε vs. format-robustness** — has not been explicitly tied
  to a cross-prompt consistency metric in the literature.

These are opportunities for the BLME paper itself to contribute
novel findings.
