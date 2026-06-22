# Consistency Tasks

This module evaluates the mathematical reliability and internal logical coherence of the model's likelihood representations, independent of ground-truth accuracy.

**Current registry coverage (12 tasks)**: `consistency_bias_weat`,
`consistency_calibration`, `consistency_contamination`,
`consistency_contrastive`, `consistency_format_robustness`,
`consistency_icl_slope`, `consistency_knowledge_capacity`,
`consistency_logical`, `consistency_membership_inference`,
`consistency_paraphrase`, `consistency_position_sensitivity`, and
`consistency_self_consistency`.

**Paper-faithful vs. BLME proxy notes**: Several consistency tasks use small
bundled prompt sets or intrinsic likelihood comparisons so they can run without
benchmark labels. Report them as reliability/proxy diagnostics, not as factual
accuracy measurements. `consistency_membership_inference` is a loss-based MIA
proxy because BLME does not know the model's actual training set.

---

## 1. Paraphrase Consistency
* **What are we measuring**: Whether paraphrased sentences land near each other in the final representation space relative to an unrelated baseline.
* **How are we measuring**: Computing last-token representation distances for paraphrase pairs and comparing them to unrelated-pair distances.
* **Hypothesis**: A robust latent space should map paraphrases to nearby geometric regions before tokenization differences dominate.
* **Citation/Paper**: BLME representation-isometry diagnostic inspired by paraphrase-invariance literature; it is not the ParaRel prediction-consistency metric from Elazar et al. 2021.
* **File & Function**: `src/blme/tasks/consistency/paraphrase.py` -> `ParaphraseInvarianceTask`
* **Critical Info**: Primary output is `representation_distance_ratio_l2`; legacy `isometry_ratio_l2` is an alias.

## 2. Likelihood Calibration
* **What are we measuring**: How well next-token confidence is calibrated against teacher-forced next-token correctness.
* **How are we measuring**: Binning top-1 next-token confidence and comparing it to whether the predicted token matches the corpus token; also reporting Brier score and calibration slope/intercept.
* **Hypothesis**: Over-parameterized LLMs often suffer from severe overconfidence, assigning 99% probability even when hallucinating. A well-calibrated geometric space prevents the probabilities from skewing uncontrollably.
* **Citation/Paper**: `Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.` [ICML 2017, ArXiv: 1706.04599]
* **File & Function**: `src/blme/tasks/consistency/calibration.py` -> `CalibrationTask`
* **Critical Info**: This adapts Guo et al.'s classifier calibration to language modeling; it is not downstream QA correctness calibration.

## 3. Logical Contradiction / Consistency
* **What are we measuring**: Whether a premise increases the likelihood of an entailed conclusion.
* **How are we measuring**: Scoring `P(conclusion | premise)` and `P(conclusion)` with shared continuation scoring; a violation occurs when conditioning on the premise decreases conclusion likelihood.
* **Hypothesis**: If the model's likelihoods reflect the entailment relation, the premise should lift the conclusion likelihood.
* **Citation/Paper**: Similar properties studied generally in neuro-symbolic and LLM consistency literature.
* **File & Function**: `src/blme/tasks/consistency/logical.py` -> `LogicalConsistencyTask`
* **Critical Info**: Does not evaluate truth, only internal agreement. A model can be logically consistent while being entirely factually wrong.

## 4. Contrastive Evaluation
* **What are we measuring**: A CounterFact-style negative-rejection proxy: does the model prefer a factual target over a mutually exclusive alternative under the same prompt?
* **How are we measuring**: Scoring factual and exclusive continuations and reporting probability ratios and differences.
* **Hypothesis**: A continuous geometry should distinctly separate true from false in its highest-likelihood paths. Failing contrastive tests means false attractors are geographically too close to true attractors.
* **Citation/Paper**: BLME likelihood diagnostic using CounterFact-style triples; related data source and framing come from Meng et al. 2022 ROME.
* **File & Function**: `src/blme/tasks/consistency/contrastive.py` -> `ContrastiveConsistencyTask`
* **Critical Info**: `mean_rejection_ratio` is P(false)/P(true); lower is better.

## 5. Data Contamination Detection (Min-k% Probability)
* **What are we measuring**: Whether the model has memorized specific text from its training data.
* **How are we measuring**: Analyzing the distribution of per-token log probabilities using the Min-k% method. If the lowest-probability tokens in a passage are still unusually high, it is a signature of memorized (rather than generalized) text. The primary score is `min_k_score`, the mean bottom-k% token log probability.
* **Hypothesis**: A model that has memorized text assigns uniformly high probabilities across all tokens, including those that would normally be surprising. Generalized knowledge shows more variance in per-token probabilities.
* **Citation/Paper**: `Shi, W., et al. (2023). Detecting Pretraining Data from Large Language Models.` [ArXiv: 2310.16789]
* **File & Function**: `src/blme/tasks/consistency/contamination.py` -> `ContaminationDetectionTask`
* **Critical Info**: The k_pct parameter (default: 20%) controls how many of the lowest-probability tokens are examined. Higher (less negative) `min_k_score` can indicate memorization-like text. When labeled calibration data are supplied, thresholds are **in-sample by default** (`in_sample_threshold`); use `calibration_mode: held_out` for a held-out calibration split.

## 6. Knowledge Capacity (Memorization vs Generalization)
* **What are we measuring**: Whether a model prefers exact factual completions over semantically equivalent rephrasings.
* **How are we measuring**: Comparing the token-level log probability of exact factual completions versus semantically equivalent rephrasings. A model that assigns similar probability to both has generalized; one that strongly prefers the exact form has memorized it.
* **Hypothesis**: Generalized knowledge should be robust to surface-level rephrasing. A large gap between exact and rephrased probabilities indicates brittle memorization rather than deep understanding.
* **Citation/Paper**: Related to `Tirumala, K., et al. (2022). Memorization Without Overfitting.` [NeurIPS 2022, ArXiv: 2205.10770] and `Carlini, N., et al. (2023). Quantifying Memorization Across Neural Language Models.` [ICLR 2023, ArXiv: 2202.07646]
* **File & Function**: `src/blme/tasks/consistency/knowledge_capacity.py` -> `KnowledgeCapacityTask`
* **Critical Info**: Despite the legacy task name, this is a paraphrase-probability / memorization proxy, not Allen-Zhu-style knowledge-capacity scaling. Primary output is `paraphrase_probability_ratio`; `generalization_ratio` is a legacy alias.

## 7. Position Sensitivity
* **What are we measuring**: Whether likelihood of recalling a key fact depends on where the fact appears in a long context.
* **How are we measuring**: Inserting bundled "needle" facts at relative positions and scoring the recall continuation NLL.
* **Hypothesis**: Lost-in-the-middle behavior appears as worse recall likelihood for middle positions than beginning/end positions.
* **Citation/Paper**: `Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts.` [ArXiv: 2307.03172]
* **File & Function**: `src/blme/tasks/consistency/position_sensitivity.py` -> `PositionSensitivityTask`
* **Critical Info**: Uses bundled facts and likelihood scoring, not full QA accuracy.

## 8. Format Robustness
* **What are we measuring**: How sensitive answer likelihoods and top-1 next-token choices are to prompt surface format.
* **How are we measuring**: Rendering bundled QA pairs in multiple formats and measuring NLL spread plus top-1 agreement.
* **Hypothesis**: More robust models should assign similar likelihoods across superficial prompt formats.
* **Citation/Paper**: `Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2023). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.` [ArXiv: 2310.11324]
* **File & Function**: `src/blme/tasks/consistency/format_robustness.py` -> `FormatRobustnessTask`
* **Critical Info**: This is an intrinsic likelihood diagnostic; it does not run the full benchmark suites used in the paper.

## 9. Self-Consistency
* **What are we measuring**: Agreement among multiple sampled completions for the same prompt.
* **How are we measuring**: Sampling completions with temperature and reporting first-token agreement, uniqueness, and entropy.
* **Hypothesis**: Concentrated, stable samples indicate more consistent generation under stochastic decoding.
* **Citation/Paper**: `Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models.` [ArXiv: 2203.11171]
* **File & Function**: `src/blme/tasks/consistency/self_consistency.py` -> `SelfConsistencyTask`
* **Critical Info**: BLME measures agreement without answer labels; it is not the paper's CoT majority-vote accuracy method.

## 10. Bias WEAT / SEAT
* **What are we measuring**: Association strength between target and attribute word sets in contextualized embeddings.
* **How are we measuring**: Running WEAT-style effect sizes and permutation p-values over bundled word sets.
* **Hypothesis**: Larger absolute WEAT effect sizes indicate stronger representational association biases.
* **Citation/Paper**: `Caliskan, Bryson, & Narayanan (2017). Semantics derived automatically from language corpora contain human-like biases.` [Science] and `May et al. (2019). On Measuring Social Biases in Sentence Encoders.` [ArXiv: 1903.10561]
* **File & Function**: `src/blme/tasks/consistency/bias.py` -> `WEATBiasTask`
* **Critical Info**: Bias scores are sensitive to templates and word sets; compare within the same configuration.

## 11. Membership Inference
* **What are we measuring**: Whether common factual sentences receive lower loss than unlikely/synthetic sentences.
* **How are we measuring**: Using negative loss as a membership score and reporting AUROC, loss gap, and shuffled counterfactual gaps.
* **Hypothesis**: Large loss gaps are consistent with memorization-like behavior.
* **Citation/Paper**: `Yeom et al. (2018). Privacy Risk in Machine Learning.` [ArXiv: 1709.01604] and `Carlini et al. (2021). Extracting Training Data from Large Language Models.` [ArXiv: 2012.07805]
* **File & Function**: `src/blme/tasks/consistency/membership_inference.py` -> `MembershipInferenceTask`
* **Critical Info**: This is a proxy split, not a verified training-set membership attack.

## 12. ICL Slope
* **What are we measuring**: How NLL changes as the prompt includes more in-context demonstrations.
* **How are we measuring**: Scoring bundled tasks at shot counts such as 0, 1, 2, and 4, then fitting a slope.
* **Hypothesis**: More negative slopes and larger NLL gains indicate stronger in-context learning from demonstrations.
* **Citation/Paper**: `Brown et al. (2020). Language Models are Few-Shot Learners.` and `Min et al. (2022). Rethinking the Role of Demonstrations.`
* **File & Function**: `src/blme/tasks/consistency/icl_slope.py` -> `ICLSlopeTask`
* **Critical Info**: Uses a small intrinsic bundle; use task-specific datasets for paper-grade ICL claims.
