# Task Fix Proposals

This document assesses each problematic BLME task and proposes either a concrete fix to make it more intrinsic, or explains why it is fundamentally a behavioral/data-dependent test and how to properly frame it as a Tier 3 task.

---

## Tasks Originally Proposed for Removal

### 1. `consistency_logical` — Logical Consistency

**Current implementation:** Splits LAMBADA passages at the midpoint into "premise" and "conclusion." Computes P(premise) and P(conclusion) as geometric mean of token probabilities. Flags violations where P(premise) > P(conclusion) + margin (default 0.1).

**Core problem:** The midpoint split creates no logical relationship. Probability ordering is confounded by text length and token frequency. This measures text difficulty, not logical reasoning.

**Fixable?** YES, with significant redesign.

**Proposed fix:**
1. Replace LAMBADA with a curated dataset of genuine logical entailment pairs (e.g., from SNLI, RTE, or a custom set of `{premise, entailed_conclusion, contradicted_conclusion}`).
2. Instead of comparing P(premise) vs P(conclusion), compare P(conclusion | premise) vs P(contradicted_conclusion | premise). Condition on the premise by computing log-probability of only the conclusion tokens after the premise.
3. Remove the arbitrary margin parameter. Report the raw ratio P(entailed | premise) / P(contradicted | premise) as the consistency score.
4. Normalize by conclusion length (per-token mean log-prob, not whole-sequence).

**Tier 3 framing:** Even with fixes, this measures the model's logical behavior on specific entailment pairs. Frame as "Logical Entailment Sensitivity" — a behavioral probe that tests whether the model's probability assignments respect basic logical relationships. Valid for cross-model comparison only when using a fixed, standardized entailment dataset.

---

### 2. `causality_ablation` — Ablation Robustness

**Current implementation:** For each ablation percentage k%, randomly selects k% of hidden dimensions in middle layers and replaces them with the sequence-mean activation. Measures loss increase.

**Core problem:** (a) Sequence-mean is not dataset-mean — a meaningful mean ablation should use the expected activation over a large corpus, not the mean over the current 128-token sequence. (b) Random dimension selection doesn't target functionally meaningful units. (c) Loss degradation is dominated by input difficulty.

**Fixable?** YES.

**Proposed fix:**
1. **Dataset-mean ablation:** Pre-compute the mean activation per dimension per layer over the entire evaluation corpus (store in cache). Use this dataset-level mean for ablation instead of per-sequence mean.
2. **Structured ablation targets:** Instead of random dimensions, ablate by functional unit — zero out entire attention heads or MLP neurons. Report per-head and per-neuron importance.
3. **Normalize by baseline loss:** Report degradation as a ratio (ablated_loss / baseline_loss) rather than absolute difference, making it comparable across texts.
4. **Fixed evaluation set:** Always use the same corpus for ablation to ensure cross-model comparability.

**Tier 3 framing:** Even with dataset-mean ablation, the degradation profile depends on the corpus used. Frame as "Ablation Robustness Profile" — a controlled intervention that measures how gracefully a model degrades under structured perturbation. The shape of the degradation curve (convex vs. linear vs. concave) is more informative than absolute values.

---

### 3. `causality_circuit_quality` — Circuit Quality

**Current implementation:** Ranks layers by importance via mean-ablation (loss increase when ablating). Selects top-25% as "circuit." Measures faithfulness (exp(-KL) between circuit and full model) and minimality (1 - circuit_size/total).

**Core problem:** (a) Mean-ablation importance is biased toward layers that handle common features. (b) Top-k% selection is arbitrary — no adaptive threshold. (c) Minimality is trivially just 1-k/N. (d) Faithfulness via KL on final-token probabilities is too narrow. (e) All metrics are dataset-dependent.

**Fixable?** PARTIALLY. The core concept is sound but the implementation needs significant rework.

**Proposed fix:**
1. **Adaptive circuit selection:** Instead of fixed 25%, use an elbow/knee-point detection on the sorted importance scores. The circuit consists of all layers above the elbow.
2. **Better importance metric:** Use activation patching (restore clean activations into a corrupted run) instead of mean-ablation for layer importance. This is closer to true causal importance.
3. **Broader faithfulness:** Compute KL over all token positions (not just the final token). Or use Jensen-Shannon divergence (symmetric, bounded).
4. **Replace minimality:** Instead of 1-k/N, use compression_ratio = circuit_faithfulness / circuit_size. A small circuit with high faithfulness is genuinely minimal.
5. **Fixed dataset:** Use the standardized WikiText corpus.

**Tier 3 framing:** Circuit discovery is inherently data-dependent (different inputs activate different circuits). Frame as "Circuit Compressibility" — measures how much of the model's computation on a standard corpus can be explained by a small subset of layers. The circuit size at a fixed faithfulness threshold (e.g., 0.9) is more meaningful than the composite quality score.

---

### 4. `dynamics_interpolation` — Latent Space Convexity

**Current implementation:** For random pairs of sequences, interpolates their final-layer last-token hidden states (h_interp = (1-alpha)*h1 + alpha*h2). Measures entropy of the decoded distribution at each interpolation step. Reports convexity_gap = entropy(midpoint) - mean(entropy(endpoints)).

**Core problem:** (a) Only interpolates one hidden state (last layer, last token) — ignores the full sequence representation. (b) Entropy at the interpolation midpoint doesn't validate convexity in any formal sense. (c) Results vary dramatically by which pair of texts is selected. (d) Only 10 interpolation steps.

**Fixable?** PARTIALLY. The instability across pairs is fundamental.

**Proposed fix:**
1. **More pairs, more steps:** Use 100+ pairs and 50 interpolation steps to get statistically stable estimates.
2. **Mean-pooled interpolation:** Interpolate mean-pooled representations (not just last token) to capture more of the sequence semantics.
3. **Multi-layer interpolation:** Compute convexity gap at each layer separately. The layer-wise convexity profile is more informative than a single number.
4. **Normalized entropy:** Divide entropy by log(vocab_size) to make it comparable across models with different vocabulary sizes.
5. **Use fixed pairs:** Instead of random pairs, use a standardized set of text pairs with known semantic relationships (similar, unrelated, contradictory).

**Tier 3 framing:** Latent convexity is inherently pair-dependent. Frame as "Latent Space Smoothness" — measures whether linear interpolation in hidden space produces smooth (not chaotic) output distributions. The aggregate convexity gap over many pairs is reasonably stable; individual pair results are not.

---

## Tasks Originally Proposed for Reclassification

### 5. `consistency_calibration` — Expected Calibration Error

**Current implementation:** Computes next-token prediction accuracy and confidence (max softmax probability) per token. Bins by confidence and measures the gap between predicted confidence and actual accuracy (ECE).

**Core problem:** ECE is a (model, dataset) joint property. Same model has different ECE on different text.

**Fixable as intrinsic?** NO. Calibration is fundamentally a property of the model's predictions on specific data, not a property of the model alone.

**Tier 3 framing:** Use as a dependent variable (Y) rather than a predictor (X). When computed on a fixed standardized corpus, ECE becomes a stable behavioral characterization. Report as "Corpus-Conditioned Calibration." Fix the binning: use adaptive quantile-based bins instead of uniform bins to avoid empty-bin artifacts.

---

### 6. `consistency_contamination` — Min-k% Memorization Detection

**Current implementation:** Computes per-token log-probabilities. Takes the bottom k% and reports the ratio min_k_mean / overall_mean. Ratio close to 1.0 suggests memorization (even the least-likely tokens are relatively likely).

**Core problem:** (a) Ratio is confounded with text difficulty — easy text has uniformly high probabilities. (b) No reference baseline to define "contaminated." (c) The 2 hardcoded example texts are too generic.

**Fixable?** YES, with a reference model.

**Proposed fix:**
1. **Reference model comparison:** Compute the min-k% ratio on a reference model known to NOT have seen the evaluation text (e.g., a small model trained on a known corpus). The difference in ratios between the test model and reference model is the contamination signal.
2. **Use holdout text:** Instead of generic sentences, use text from a controlled source that some models may have seen (e.g., recent arXiv abstracts published after different model training cutoffs).
3. **Z-score normalization:** Instead of the raw ratio, report a z-score of the model's min-k% ratio against a distribution of ratios from reference texts.

**Tier 3 framing:** Contamination detection is inherently about specific (model, text) pairs. Frame as "Corpus Familiarity Score" — a controlled comparison against a reference distribution. Cannot be truly intrinsic but is useful for flagging data leakage.

---

### 7. `consistency_paraphrase` — Paraphrase Invariance

**Current implementation:** Computes mean-pooled final-layer representations for (text1, paraphrase, unrelated). Reports L2 and cosine distances, plus isometry ratio = paraphrase_dist / unrelated_dist.

**Core problem:** (a) Mean pooling is lossy — different sentence structures produce different pooled representations even when semantically equivalent. (b) Paraphrase quality varies. (c) "Unrelated" text selection is arbitrary.

**Fixable?** YES.

**Proposed fix:**
1. **Multi-layer analysis:** Compute distances at every layer, not just the final one. Report the layer-wise invariance profile.
2. **Better pooling:** Use last-token representation (causal LM convention) instead of mean pooling. Or use both and report the comparison.
3. **Standardized paraphrase set:** Curate a fixed set of 50+ high-quality paraphrase pairs with controlled syntactic variation. Include the paraphrase set as a bundled asset.
4. **Controlled unrelated selection:** For each paraphrase pair (A, A'), select an unrelated text B that matches A in length and vocabulary overlap, so the distance comparison is fair.
5. **Statistical significance:** Report bootstrap confidence intervals on the isometry ratio.

**Tier 3 framing:** Paraphrase invariance measures how well the model maps semantically equivalent inputs to similar representations. Data-dependent but informative when using a fixed, high-quality paraphrase set. Frame as "Semantic Representation Stability."

---

### 8. `consistency_contrastive` — Contrastive Consistency

**Current implementation:** For paired factual/exclusive statements, computes sequence probability (geometric mean of token probs). Reports rejection_ratio = P(exclusive) / P(factual).

**Core problem:** (a) Sequence probability conflates text length with model confidence. (b) Cannot distinguish genuine rejection from length effects. (c) Dataset-dependent.

**Fixable?** YES.

**Proposed fix:**
1. **Conditional probability:** Instead of P(full_statement), compute P(answer_tokens | prompt). Use a shared prompt prefix and measure only the diverging completion tokens.
2. **Length-matched pairs:** Ensure factual and counterfactual completions have the same token count (pad or truncate).
3. **Per-token comparison:** Report per-token log-prob differences rather than whole-sequence ratios.
4. **Standardized dataset:** Bundle a curated set of 100+ factual/counterfactual pairs with controlled format (e.g., from CounterFact or LAMA).

**Tier 3 framing:** Contrastive consistency is a behavioral probe testing whether the model's probability assignments distinguish facts from counterfacts. Valid for cross-model comparison only with a fixed dataset and length-controlled pairs. Frame as "Factual Discrimination Score."

---

### 9. `consistency_knowledge_capacity` — Knowledge Capacity

**Current implementation:** For (prompt, exact_completion, rephrased_completion) triples, computes mean log-probability of the completion tokens. Reports memorization_score = logprob(exact) - logprob(rephrased) and generalization_ratio = logprob(rephrased) / logprob(exact).

**Core problem:** (a) Tokenization mismatch — tokenizing the prompt separately may produce different token IDs than tokenizing the full text. (b) Off-by-one prone slicing. (c) Only 3 hardcoded facts. (d) Shorter exact completions have higher per-token probability by default.

**Fixable?** YES.

**Proposed fix:**
1. **Robust tokenization:** Tokenize the full text (prompt + completion) as one unit. Find the completion boundary by tokenizing the prompt and matching the prefix of the full-text token IDs.
2. **Length normalization:** Always report per-token mean log-probability for the completion. Ensure exact and rephrased completions are comparable in token count.
3. **Larger dataset:** Bundle 100+ knowledge triples spanning diverse domains (geography, science, history, pop culture). Use LAMA or a curated subset as the basis.
4. **Multiple rephrasings:** For each fact, include 3-5 rephrasings and report the mean and variance of the generalization ratio.

**Tier 3 framing:** Knowledge capacity measures whether the model encodes factual knowledge in a format-invariant way. Inherently dataset-dependent but stable with a large, diverse knowledge set. Frame as "Knowledge Generalization Score."

---

### 10. `interpretability_prediction_entropy` — Prediction Entropy

**Current implementation:** Computes Shannon entropy of the softmax output distribution at every token position. Reports mean, std, median, p90 entropy and top-1/top-5 probabilities.

**Core problem:** Entropy is a property of the (model, input) pair, not the model alone. The same model produces low entropy on predictable text and high entropy on ambiguous text. Default dataset is 50 copies of the same sentence.

**Fixable as intrinsic?** NO. Output entropy is fundamentally input-dependent.

**Tier 3 framing:** When measured on a fixed corpus, prediction entropy becomes a stable model characteristic — it measures the model's average uncertainty on a standard workload. Fix the default dataset (use WikiText corpus instead of 50 copies of one sentence). Frame as "Corpus-Conditioned Prediction Uncertainty." The distribution of per-token entropy (not just the mean) is informative — heavy tails indicate the model is highly uncertain on specific tokens.

---

### 11. `interpretability_probing` — Linear Probing

**Current implementation:** Trains SGDClassifier (logistic regression) on hidden states at each layer to predict the next token. Reports per-layer accuracy.

**Core problem:** Hewitt & Liang (2019, "Designing and Interpreting Probes with Control Tasks") showed that high probe accuracy does not prove the model mechanistically uses the probed feature. The classifier may learn the task independently in high-dimensional space.

**Fixable?** YES, by adding a control task.

**Proposed fix:**
1. **Add control task:** Train a second probe on random labels (shuffled next-token IDs). The "selectivity" = probe_accuracy - control_accuracy measures how much the representation actually encodes the feature beyond chance.
2. **MCC instead of accuracy:** Use Matthew's Correlation Coefficient instead of raw accuracy, since next-token prediction has extreme class imbalance (50K+ classes).
3. **MDL probe (Voita & Titov 2020):** Replace logistic regression with an MDL-based probe that measures the description length required to encode the labeling, penalizing memorization.
4. **Probe multiple features:** In addition to next-token prediction, probe for POS tags, dependency labels, or named entity tags (requires tagged data).
5. **Use proper evaluation dataset:** Replace 50 copies of one sentence with the standardized WikiText corpus.

**Tier 3 framing:** Probing measures linear accessibility of features, not causal encoding. With a control task and selectivity metric, it becomes a meaningful (though still data-dependent) measure of "how linearly readable is feature X in this model's representations." Frame as "Linear Feature Accessibility."

---

### 12. `interpretability_attribution` — Component Attribution

**Current implementation:** Computes layer deltas (h_{l+1} - h_l), projects onto embedding space (delta @ E^T), identifies top-5 most similar vocabulary embeddings, and measures their pairwise cosine similarity as a "coherence score."

**Core problem:** (a) Hidden-space deltas are not guaranteed to be in embedding space. (b) Top-5 selection is arbitrary. (c) Coherence (pairwise cosine sim) doesn't prove causal contribution. (d) Default is 1 sample.

**Fixable?** PARTIALLY.

**Proposed fix:**
1. **Apply layer norm before projection:** If the model has a final layer norm, apply it to the delta before projecting to embedding space. This makes the projection more meaningful.
2. **Adaptive top-k:** Instead of fixed top-5, use the tokens whose similarity exceeds a threshold (e.g., top 1% of vocabulary by similarity).
3. **Semantic clustering:** Instead of pairwise cosine similarity, measure whether the top-k tokens belong to the same WordNet synset or semantic category. This is a stronger test of coherence.
4. **More samples:** Use 50+ texts from the standard corpus.
5. **Null distribution:** Compare coherence against random deltas (Gaussian noise with matched norm) to establish a baseline.

**Tier 3 framing:** Attribution coherence is inherently text-dependent (different texts activate different update directions). Frame as "Layer Update Semantic Coherence" — an exploratory metric that assesses whether layer-wise representation changes correspond to interpretable vocabulary directions. Always report with a null-distribution baseline.

---

### 13. `interpretability_attention_polysemanticity` — Attention Polysemanticity

**Current implementation:** Hooks into attention output projection modules (c_proj, out_proj, dense). Computes SVD of the output tensor (seq_len x hidden_size) and reports Shannon entropy of normalized singular values.

**Core problem:** (a) SVD is computed over (seq_len x hidden_size), which measures positional diversity of outputs, not per-head feature superposition. (b) The output projection operates on concatenated multi-head output, not individual heads. (c) When seq_len < hidden_size, SVD is artificially truncated.

**Fixable?** YES, with a significant algorithmic change.

**Proposed fix:**
1. **Per-head analysis:** Instead of hooking the output projection, hook the individual attention head value matrices or the pre-projection per-head outputs. Compute SVD per head.
2. **Activation covariance:** Compute SVD of the covariance matrix of per-head activations across tokens (not the raw activation matrix). This gives the dimensionality of the feature space each head uses, regardless of sequence length.
3. **Batch over multiple inputs:** Aggregate the covariance over many inputs before SVD, giving a stable estimate of the head's feature dimensionality.
4. **Participation ratio:** Instead of Shannon entropy, use participation ratio = (sum sigma_i)^2 / sum(sigma_i^2) as a more interpretable measure of effective dimensionality.

**Tier 3 framing:** Even with per-head covariance SVD, the features activated depend on input text. Frame as "Attention Head Feature Dimensionality" — measures how many effective dimensions each head uses on a standard corpus. Low dimensionality suggests specialized heads; high dimensionality suggests polysemantic heads.

---

### 14. `dynamics_coe` — Chain of Embedding Drift

**Current implementation:** Starting from a prompt, greedily generates tokens and tracks the magnitude and angle change of the final-token hidden state at each step. Reports mean/std of magnitude and angle changes.

**Core problem:** (a) Only tracks last-token representation. (b) Greedy decoding is deterministic — always picks the most likely token. (c) No baseline for "normal" drift values. (d) Magnitude and angle are correlated.

**Fixable?** YES.

**Proposed fix:**
1. **Multi-position tracking:** Track hidden states at multiple token positions (not just the last) to capture how the full context representation evolves.
2. **Multiple decoding strategies:** Run with greedy, nucleus sampling (p=0.9), and temperature=1.0. Compare drift profiles. If drift is similar across strategies, the metric is more model-intrinsic.
3. **Standardized prompts:** Use a fixed set of 20+ prompts with controlled properties (factual, creative, ambiguous) from a bundled dataset.
4. **Relative drift:** Normalize drift by the hidden state norm at each step to make it comparable across models with different activation scales.
5. **Drift acceleration:** Report the second derivative of the trajectory (are drift changes smooth or jerky?) as an additional diagnostic.

**Tier 3 framing:** Generation drift inherently depends on the generation trajectory, which depends on the prompt. Frame as "Generation Trajectory Stability" — measures how erratically the model's internal state changes during autoregressive generation on a controlled set of prompts. The prompt set should be fixed and diverse.

---

### 15. `repe_steering_effectiveness` — Steering Effectiveness

**Current implementation:** Extracts task vectors from contrastive pairs, then injects them (alpha * task_vector) into each layer's hidden state for a neutral prompt. Measures KL divergence between steered and baseline output distributions.

**Core problem:** (a) KL measures output shift, not steering quality — large KL could mean steering toward garbage. (b) Alpha is hardcoded at 1.0. (c) Only modifies last-token hidden state. (d) Success threshold (0.01 KL) is arbitrary.

**Fixable?** PARTIALLY.

**Proposed fix:**
1. **Directional KL:** Instead of raw KL divergence, measure whether the steered distribution shifts probability mass *toward* the positive concept tokens (and away from negative). Report "directional steering score" = log P(positive_tokens | steered) - log P(positive_tokens | baseline).
2. **Alpha sweep:** Test multiple alpha values (0.5, 1.0, 2.0, 5.0) and report the maximum achievable directional steering score. This captures the model's steerability envelope.
3. **Standardized contrastive pairs:** Bundle a fixed set of 10+ concept axes (truth/falsehood, positive/negative sentiment, formal/informal) with 20+ pairs each.
4. **Multiple neutral prompts:** Use 10+ diverse neutral prompts and report mean/variance of effectiveness.

**Tier 3 framing:** Steering effectiveness is fundamentally about (model, concept, prompt) triples. Frame as "Representational Steerability" — measures how much the model's output distribution can be controlled by linear interventions in hidden space. Valid for cross-model comparison only with fixed concept axes and prompt sets.

---

## Summary

| Task | Fixable? | Key Fix | Post-Fix Tier |
|------|----------|---------|---------------|
| `consistency_logical` | YES | Real entailment pairs + conditional probability | Tier 3 |
| `causality_ablation` | YES | Dataset-mean ablation + structured targets | Tier 3 |
| `causality_circuit_quality` | PARTIALLY | Adaptive thresholds + activation patching | Tier 3 |
| `dynamics_interpolation` | PARTIALLY | More pairs + multi-layer analysis | Tier 3 |
| `consistency_calibration` | NO (use as Y) | Quantile-based binning | Y-variable |
| `consistency_contamination` | YES | Reference model comparison | Tier 3 |
| `consistency_paraphrase` | YES | Multi-layer + standardized pairs + last-token pooling | Tier 3 |
| `consistency_contrastive` | YES | Conditional probability + length-matched pairs | Tier 3 |
| `consistency_knowledge_capacity` | YES | Robust tokenization + larger dataset | Tier 3 |
| `interpretability_prediction_entropy` | NO (use as Y) | Fix default dataset | Y-variable |
| `interpretability_probing` | YES | Add control task + selectivity metric | Tier 3 |
| `interpretability_attribution` | PARTIALLY | Layer norm + null distribution + semantic clustering | Tier 3 |
| `interpretability_attention_polysemanticity` | YES | Per-head covariance SVD + participation ratio | Tier 3 |
| `dynamics_coe` | YES | Standardized prompts + relative drift + multi-strategy | Tier 3 |
| `repe_steering_effectiveness` | PARTIALLY | Directional steering + alpha sweep + fixed axes | Tier 3 |
