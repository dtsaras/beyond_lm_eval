# Interpretability Tasks

This module contains metrics that probe the internal properties, specialized circuits, and representational capacity of the model's layers and heads.

**Registry coverage (19 tasks)**: `interpretability_activation_kurtosis`,
`interpretability_activation_sinks`,
`interpretability_attention_effective_rank`, `interpretability_attention_entropy`,
`interpretability_attention_graph`, `interpretability_attention_rank`,
`interpretability_attention_rollout`, `interpretability_attribution`,
`interpretability_head_roles`, `interpretability_inactive_heads`,
`interpretability_induction_heads`, `interpretability_logit_lens`,
`interpretability_logit_lens_kl`, `interpretability_prediction_entropy`,
`interpretability_probing`, `interpretability_sae_features`,
`interpretability_sparsity`, `interpretability_superposition`, and
`interpretability_waa`.

**Paper-faithful vs. BLME proxy notes**: `interpretability_activation_sinks`
implements paper-derived Sinkε, massive-activation, and compression-valley
diagnostics. `interpretability_attention_graph` is a BLME graph-centrality
diagnostic inspired by attention-sink/attention-rollout literature; PageRank
centralization alone is not definitive proof of attention-sink mechanics.
`interpretability_attention_effective_rank` is the registered task in
`attention_polysemanticity.py`; it reports an effective-rank proxy for head
capacity and should not be described as a paper-faithful polysemanticity or
monosemanticity method. Only `interpretability_logit_lens_kl` and
`interpretability_inactive_heads` declare headline keys.

---

## 1. Attention Entropy
* **What are we measuring**: The focus (sharpness) or dispersion of the probability distribution across an attention head.
* **How are we measuring**: Computing the Shannon entropy of the attention weights matrix. 
* **Hypothesis**: Low entropy means the head is sharply focused on a specific token (e.g., induction head), while high entropy means it's broadly attending to the context (e.g., a "bag of words" head).
* **Citation/Paper**: `Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What Does BERT Look at? An Analysis of BERT's Attention.` [ArXiv: 1906.04341]
* **File & Function**: `src/blme/tasks/interpretability/attention.py` -> `AttentionEntropyTask`
* **Critical Info**: Attention entropy is heavily correlated with the depth of the layer; deeper layers generally exhibit lower, more specialized entropy. Defaults: `num_samples: 100`, `batch_size: 1`. No headline keys or size-invariances are declared.

## 2. Attention Graph Modularity (Attention Sinks)
* **What are we measuring**: The structural topology of the attention matrix treated as a directed graph.
* **How are we measuring**: Computing the PageRank Centrality on the attention matrix to find specific bottleneck "sink" tokens, and the Edge Gini coefficient to measure global graph sparsity.
* **Hypothesis**: Models offload computation to "Attention Sinks" (usually the BOS token or newline characters) to act as a structural anchor. Finding the central node reveals the computational topology.
* **Citation/Paper**: `Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks.` [ArXiv: 2309.17453]
* **File & Function**: `src/blme/tasks/interpretability/attention_graph.py` -> `AttentionGraphTopologyTask`
* **Critical Info**: High centralization at index 0 is a useful attention-sink-like signal, but it is a proxy; confirm with Sinkε or intervention-based analyses before making mechanistic claims. Defaults: `num_samples: 10`. No headline keys or size-invariances are declared.

## 3. Attention Effective Rank (BLME head-capacity proxy)
* **What are we measuring**: The effective rank of attention-head value/output representations as a proxy for how much capacity a head uses across samples.
* **How are we measuring**: By computing the Singular Value Entropy (Effective Rank) of the isolated value-projection outputs corresponding to specific heads.
* **Hypothesis**: Higher effective rank suggests a head is using a broader subspace; lower rank suggests a more constrained or collapsed head. This is not a direct concept-level polysemanticity measurement.
* **Citation/Paper**: BLME proxy inspired by effective-rank diagnostics and the superposition literature (`Elhage et al. 2022`; closest concept-level benchmark: `Templeton et al. 2024`, Scaling Monosemanticity).
* **File & Function**: `src/blme/tasks/interpretability/attention_polysemanticity.py` -> `AttentionEffectiveRankTask`
* **Critical Info**: Registered task name is `interpretability_attention_effective_rank`; there is no `interpretability_attention_polysemanticity` task. Defaults: `num_samples: 3`. No headline keys or size-invariances are declared.

## 4. Induction Heads
* **What are we measuring**: The presence and strength of specialized "Induction Heads" that complete in-context patterns (e.g., A B ... A -> predicts B).
* **How are we measuring**: Generating a sequence of repeated random tokens, then analyzing the attention weights to check if current tokens heavily attend to the token immediately following their previous occurrence.
* **Hypothesis**: Induction heads are the fundamental mechanism behind in-context learning and zero-shot capabilities in LLMs.
* **Citation/Paper**: `Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context Learning and Induction Heads.` [ArXiv: 2209.11895]
* **File & Function**: `src/blme/tasks/interpretability/induction.py` -> `InductionHeadTask`
* **Critical Info**: These heads typically form abruptly around the middle layers during pre-training in a "phase change". Defaults: `seq_len: 30`, `num_samples: 20`. No headline keys or size-invariances are declared.

## 5. Logit Lens
* **What are we measuring**: What the model "believes" the next token should be at each intermediate layer before the final decision.
* **How are we measuring**: Multiplying the hidden states of intermediate layers directly against the vocabulary unembedding matrix to decode their implicit trajectory.
* **Hypothesis**: The model constructs its final prediction iteratively. By decoding early layers, we can see exactly when factual knowledge is injected into the residual stream.
* **Citation/Paper**: `Nostalgebraist. (2020). Interpreting GPT: the logit lens.` [LessWrong Blog Post, No Academic Proceeding]
* **File & Function**: `src/blme/tasks/interpretability/logit_lens.py` -> `LogitLensTask`
* **Critical Info**: Plagued by scaling issues because early representations are not in the same linear space as the final unembedding layer vocabulary. Defaults: `num_samples: 100`. No headline keys or size-invariances are declared.

## 6. Token Attribution (Gradient-based)
* **What are we measuring**: How much each preceding input token contributed to the likelihood of generating a specific target token.
* **How are we measuring**: Computing the gradient of the predicted logit with respect to the input embeddings, taking the L2 norm (InputXGradient).
* **Hypothesis**: Not all context tokens are equal. Saliency mapping reveals which entities heavily bias the model's specific outputs.
* **Citation/Paper**: `Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps.` (General Saliency)
* **File & Function**: `src/blme/tasks/interpretability/attribution.py` -> `ComponentAttributionTask`
* **Critical Info**: Gradients can be noisy. This requires a backward pass, which is much slower and more memory-intensive than standard inference. Defaults: `num_samples: 50`. No headline keys or size-invariances are declared.

## 7. Prediction Entropy
* **What are we measuring**: The model's confidence or uncertainty in its next-token prediction over the vocabulary distribution.
* **How are we measuring**: The Shannon entropy of the Softmax output probabilities.
* **Hypothesis**: High prediction entropy implies the model is genuinely guessing or hallucinating over flat distributions, whereas low prediction entropy indicates hard memorization or high structural constraint.
* **Citation/Paper**: `Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration.` [ICLR 2020, ArXiv: 1904.09751]
* **File & Function**: `src/blme/tasks/interpretability/prediction_entropy.py` -> `PredictionEntropyTask`
* **Critical Info**: Correlates heavily with `perplexity`, but normalizes out the specific text likelihood, giving a pure measure of constraint width. Defaults: `num_samples: 100`. No headline keys or size-invariances are declared.

## 8. Activation Sparsity
* **What are we measuring**: The frequency of inactive (zeroed-out or highly negative) neurons in the MLP feed-forward blocks.
* **How are we measuring**: Computing the L0 pseudo-norm fraction (percentage of active neurons) and the Kurtosis (heavy-tailedness) of the post-GELU/SwiGLU activations.
* **Hypothesis**: LLMs demonstrate severe activation sparsity; only a tiny fraction of the network fires for a given token. This translates to efficient computation and specialized feature maps.
* **Citation/Paper**: Related to `Zhang et al. (2021). Moefication: Transformer Feed-forward Layers are Mixtures of Experts.` and later contextual-sparsity analyses. BLME reports activation-rate diagnostics; it does not implement Deja Vu's inference-time predictor.
* **File & Function**: `src/blme/tasks/interpretability/sparsity.py` -> `ActivationSparsityTask`
* **Critical Info**: Relu networks have hard sparsity (true 0s), whereas GELU models have soft sparsity (negative values near 0). The task supports thresholding for soft sparsity. Defaults: `num_samples: 5`. No headline keys or size-invariances are declared.

## 9. Linear Probing
* **What are we measuring**: How linearly accessible bundled labels are from hidden states.
* **How are we measuring**: Extracting hidden states and training a regularized Logistic Regression probe. Evaluated via cross-validated accuracy/AUC.
* **Hypothesis**: If a linear probe can retrieve a label with high accuracy, the model has made that label linearly decodable; this does not prove the model uses that feature causally.
* **Citation/Paper**: `Alain, G. & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes.` [ArXiv: 1610.01644]. See also `Belinkov (2022)` for limitations of probing.
* **File & Function**: `src/blme/tasks/interpretability/probing.py` -> `LinearProbingTask`
* **Critical Info**: The metric is essentially measuring the capacity of the *probe*, not just the model, so high regularisation is required to prevent the probe from learning the task entirely. Defaults: `num_samples: 50`, `max_tokens: 128`, `max_probe_samples: 5000`. No headline keys or size-invariances are declared.

## 10. Weight-Activation Alignment (WAA)
* **What are we measuring**: The mechanistic capacity utilization of the network.
* **How are we measuring**: Computing the Cosine Similarity between the empirical principal components of the actual inference activations (dynamic) and the principal singular vectors of the static weight matrices.
* **Hypothesis**: If weight and activation eigenvectors are aligned, the model is using its learned capacity cleanly. Misalignment means the static weights contain parameters irrelevant to dynamic generation, causing parameter waste.
* **Citation/Paper**: Related to `Park, K., Choe, Y. J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models.` [ICML 2024, arXiv:2311.03658]. BLME's WAA is a proxy diagnostic, not that paper's full linear-representation analysis.
* **File & Function**: `src/blme/tasks/interpretability/weight_activation_alignment.py` -> `WeightActivationAlignmentTask`
* **Critical Info**: Heavily reliant on computing local SVD on weight matrices, making it expensive for massive models (>70B params) without specific approximations. Defaults: `num_samples: 5`. No headline keys or size-invariances are declared.

## 11. SAE Feature Dimensionality (sae_features)
* **What are we measuring**: The structural sparsity and disentanglement of the representation using a Sparse Autoencoder (SAE).
* **How are we measuring**: Running the inputs through pre-trained SAE dictionaries (via `sae-lens`) to extract L0 norms and active feature counts.
* **Hypothesis**: Traditional hidden states are in superposition. SAEs force features to be sparse and disentangled. Analyzing the SAE features reveals the true atomic semantic variables the model operates on.
* **Citation/Paper**: `Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models.` [ICLR 2024, ArXiv: 2309.08600]
* **File & Function**: `src/blme/tasks/interpretability/sae_features.py` -> `SAEFeatureDimensionalityTask`
* **Critical Info**: Strictly requires the external `sae-lens` library to map to established SAE dictionaries for the specific tested model. Defaults: `num_samples: 5`, `sae_release: gpt2-small-res-jb`, `sae_id: blocks.8.hook_resid_pre`. No headline keys or size-invariances are declared.

## 12. Superposition Index (Neuron Polysemanticity)
* **What are we measuring**: The degree of superposition (polysemanticity) in model neurons — whether individual neurons encode multiple unrelated features.
* **How are we measuring**: Analyzing the bimodality coefficient of per-neuron activation distributions within MLP layers. A bimodal activation distribution suggests a neuron encodes multiple features (fires strongly for distinct, unrelated inputs). Also measures neuron utilization rate — the fraction of neurons with non-trivial activation variance.
* **Hypothesis**: In superposition, individual neurons compress multiple unrelated features into their activation range. High polysemanticity (bimodality coefficient > 0.555) indicates severe superposition, while low values indicate cleaner, monosemantic neurons.
* **Citation/Paper**: `Elhage, N., et al. (2022). Toy Models of Superposition.` [ArXiv: 2209.10652] and `Templeton, A., et al. (2024). Scaling Monosemanticity.` [Transformer Circuits Thread]
* **File & Function**: `src/blme/tasks/interpretability/superposition.py` -> `SuperpositionIndexTask`
* **Critical Info**: The bimodality coefficient (BC) uses skewness and kurtosis: BC = (skewness^2 + 1) / kurtosis. Values > 0.555 suggest bimodality. Higher mean_polysemanticity_index indicates more severe superposition across the model. Defaults: `num_samples: 5`, `max_neurons: 256`. No headline keys or size-invariances are declared.

## 13. Attention Rank Collapse
* **What are we measuring**: Effective rank of attention matrices across layers and heads.
* **How are we measuring**: Computing Roy-Vetterli effective rank on attention maps and summarizing collapse trends across depth.
* **Hypothesis**: Very low attention rank suggests attention maps are collapsing to a small set of directions or tokens.
* **Citation/Paper**: `Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.` [ICML 2021, ArXiv: 2103.03404]
* **File & Function**: `src/blme/tasks/interpretability/attention_rank.py` -> `AttentionRankCollapseTask`
* **Critical Info**: Requires eager attention weights on modern transformer backends. Defaults: `num_samples: 50`. No headline keys or size-invariances are declared.

## 14. Attention Head Roles
* **What are we measuring**: Simple structural roles such as previous-token and duplicate-token attention.
* **How are we measuring**: Measuring how much heads attend to immediately previous tokens and repeated-token positions.
* **Hypothesis**: A larger fraction of specialized heads can indicate more developed in-context and copy-like mechanisms.
* **Citation/Paper**: `Clark et al. (2019). What Does BERT Look At?` and `Voita et al. (2019). Analyzing Multi-Head Self-Attention.`
* **File & Function**: `src/blme/tasks/interpretability/head_roles.py` -> `HeadRolesTask`
* **Critical Info**: BLME reports simple role fractions; it does not implement a full OV/copying-score decomposition. Defaults: `seq_len: 30`, `num_samples: 20`. No headline keys or size-invariances are declared.

## 15. Activation Sinks, Massive Activations, and Compression Valley
* **What are we measuring**: Three related phenomena: attention sink concentration, massive residual activations, and entropy valleys across layers.
* **How are we measuring**: Computing Gu et al.'s Sinkε, BOS attention mass, residual outlier fractions/ratios, and the minimum of a per-layer matrix-entropy profile.
* **Hypothesis**: These metrics describe emergent bias-token and compression mechanisms that often appear in modern LLMs.
* **Citation/Paper**: `Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks.` [ArXiv: 2309.17453]; `Gu et al. (2025). When Attention Sink Emerges in Language Models.` [ICLR 2025, ArXiv: 2410.10781]; `Sun et al. (2024). Massive Activations in Large Language Models.` [ArXiv: 2402.17762]; `Arroyo et al. (2025). Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin.` [ArXiv: 2510.06477]
* **File & Function**: `src/blme/tasks/interpretability/activation_sinks.py` -> `ActivationSinksTask`
* **Critical Info**: Sinkε is formula-faithful; the compression-valley output is a BLME matrix-entropy proxy, not the full paper pipeline. Defaults: `num_samples: 10`, `max_length: 128`, `sink_epsilon: 0.3`, `massive_threshold_ratio: 100.0`. No headline keys or size-invariances are declared.

## 16. Logit-Lens KL and Decision Depth (interpretability_logit_lens_kl)
* **What are we measuring**: When, through depth, the next-token distribution settles: how far each layer's decoded distribution is from the output, how much it moves in total along the stack, and from which normalized depth the final top-1 token is locked in.
* **How are we measuring**: Every block output h_l (l = 0..L-1; the last one hooked before the final norm, so the norm is applied once) is decoded with the model's own final norm and unembedding, final logit softcap included: p_l = softmax(W_U norm(h_l)), so p_{L-1} is the model's output distribution. Per token position: KL(p_l || p_{L-1}), KL(p_l || p_{l+1}), the lens entropy, and the margin log p_l(y*) - max_{v != y*} log p_l(v) of the final top-1 y*. Positions are every token of the first `num_samples` (100) corpus passages, each truncated to 512 tokens (a short passage is not replaced by a later one), except the sinks (position 0 and massive-activation rows). Headline: `kl_churn` = sum_l KL(p_l || p_{l+1}) (total distributional movement; identity blocks add exactly 0); `kl_to_final_auc` = (1/L) sum_l KL(p_l || p_{L-1}) (Hu et al.'s AUC on normalized depth); `decision_depth` = mean (s + 1) / L, s the first layer from which y* stays the lens's top-1 (Hu et al.'s time to decision). Also `first_agreement_depth` (the first layer where y* is the top-1), `entropy_auc`, and the per-layer profiles `kl_to_final_profile`, `kl_next_profile`, `entropy_profile`, `top1_agreement_profile` on the depth grid `_meta_layer_depth` = (l + 1) / L. The released, depth-summed AUCs and the reverse direction KL(p_{L-1} || p_l), which the tuned lens trains on, are `_meta_`.
* **Hypothesis**: Models differ in how early, in normalized depth, they settle on their prediction and in how much the decode moves on the way there; Entropy-Lens reports that logit-lens entropy profiles are family-specific and invariant under depth rescaling, a lineage signature these profiles can test.
* **Citation/Paper**: `Hu, J., Lepori, M. A., & Franke, M. (2025). Signatures of human-like processing in Transformer forward passes.` [ArXiv: 2504.14107; code jennhu/model-human-processing@69e405c]; `Entropy-Lens: Uncovering Decision Strategies in LLMs.` [ArXiv: 2502.16570]; `Belrose, N., et al. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens.` [ArXiv: 2303.08112]
* **Paper summary**: Hu et al. decode every layer of GPT-2, Llama-2, Llama-3.1, Gemma-2, OLMo-2 and Falcon-3 models (124M-405B) with the logit lens (checked with the tuned lens) and turn the layer-wise trajectories of correct/incorrect answer pairs into process measures (AUC, time to decision, boosting, two-stage magnitude), which they compare with output-layer measures as predictors of human accuracy, response times and other processing measures (e.g. mouse tracking) on five tasks.
* **File & Function**: `src/blme/tasks/interpretability/logit_lens_kl.py` -> `LogitLensKLTask`
* **Critical Info**: Headline keys: `kl_churn`, `kl_to_final_auc`, `decision_depth`. `kl_churn` is declared invariant to identity blocks inserted after any block, at interior and trailing placements (`tests/tasks/invariance/test_interpretability_logit_lens_kl.py::test_churn_is_exact_under_identity_blocks`); a block before block 0 would expose the embedding output, which the block-output profile does not hold. `kl_to_final_auc` and both depths move under identity blocks by construction: appended blocks scale them by L0 / L (`test_auc_and_depths_move_under_identity_blocks`). Hu et al.'s measures need correct/incorrect answer pairs; on free text the final top-1 plays the correct answer and the strongest other token of each layer the incorrect one, and their AUC is applied to KL and entropy and divided by L. The release computes no KL: `kl_churn` and the KL direction (the layer's own distribution first) are BLME adaptations. Departures from the release: the final logit softcap is applied; the norm and unembedding run in float32 and the log-softmax in float64 at every size, the forward pass in the model's dtype; positions are pooled over passages and not over answer stimuli. This is a separate task from `interpretability_logit_lens`. Early-layer logit-lens decodes are poor for some model families, which raises the early KL terms without any change in when the model decides. `_meta_final_lens_max_abs_logit_diff` checks the last-layer decode against the model's own logits. Positions are decoded in chunks of `chunk_elements` / V, so L x T x V never sits in memory.

## 17. Inactive-Head Fraction (interpretability_inactive_heads)
* **What are we measuring**: The share of attention heads whose output is small next to the other heads of their layer, so that zeroing them barely changes the model.
* **How are we measuring**: The released Avg Head Output Norm (LN) score. For each of the first 1000 corpus passages (up to 512 tokens), a pre-hook on each attention layer's output projection reads every query head's slice of its input (attn_weights @ V); the head's L2 norm is averaged over all tokens and divided by the mean over the layer's heads. A head is inactive on a passage when this score is below tau = 0.5; the fraction is the per-layer share of inactive (passage, head) pairs averaged over the layers. `inactive_head_threshold_5pct` is the score below which 5% of all (passage, layer, head) scores fall, the release's own 5% threshold.
* **Hypothesis**: Sandoval-Segura et al. find that about 12% of the heads of Llama, OLMo-2 and Qwen2.5 models are inactive on average, that this output-norm score finds more such heads than scores built on the attention sink, and that models of one family can differ.
* **Citation/Paper**: `Sandoval-Segura, P., et al. (2026). Identifying and Evaluating Inactive Heads in Pretrained LLMs.` [ICLR 2026, ArXiv: 2504.03889]; code psandovalsegura/inactive-heads.
* **File & Function**: `src/blme/tasks/interpretability/inactive_heads.py` -> `InactiveHeadsTask`
* **Critical Info**: Headline keys: `inactive_head_fraction` and `inactive_head_threshold_5pct`. No key is declared size-invariant: inserted identity blocks move the fraction (`tests/tasks/invariance/test_interpretability_inactive_heads.py::test_identity_blocks_move_the_fraction`), because their heads write nothing yet the score is read before W_O and rates them as ordinary heads; zero-padding the width adds no heads and changes no score (`test_zero_padding_the_width_adds_no_heads`). Defaults: `num_samples: 1000`, `max_length: 512`, `tau: 0.5`; each passage is scored once. The release picks tau per model at the quantiles that capture 0-30% of heads on MMLU (5-shot) and checks them by zeroing; BLME uses one fixed tau and reports the fraction over a tau grid and the release's 0-30% thresholds in `_meta_`. Heads that share a GQA value head are scored separately, as in the release. On gated attention the score reads the gated output; linear-attention and other non-softmax layers are skipped and counted.

## 18. Activation Kurtosis (interpretability_activation_kurtosis)
* **What are we measuring**: How heavy-tailed the per-channel activation distributions of the residual stream are, as a signal of outlier features.
* **How are we measuring**: For each hidden dimension at each layer, the excess (Fisher) kurtosis E[(x - mu)^4] / sigma^4 - 3 of its values across tokens, with population moments. Reported: the mean and maximum over channels and layers, the fraction of (channel, layer) pairs above `kurtosis_threshold: 10`, the kurtosis of the pooled activation tensor, last-layer scalars and per-layer profiles.
* **Hypothesis**: A few channels with very large kurtosis indicate the massive-activation regime; such outlier channels are what makes low-bit quantization hard.
* **Citation/Paper**: `Akhondzadeh, Bojchevski, Eleftheriou, & Dazzi (2025). KurTail: Kurtosis-based LLM Quantization.` [Findings of EMNLP 2025, ArXiv: 2503.01483]; massive activations: `Sun et al. (2024). Massive Activations in Large Language Models.` [ArXiv: 2402.17762]
* **File & Function**: `src/blme/tasks/interpretability/activation_kurtosis.py` -> `ActivationKurtosisTask`
* **Critical Info**: A normal channel has excess kurtosis 0. The per-channel kernel follows `scipy.stats.kurtosis(A, axis=0, fisher=True, bias=True)` (`tests/tasks/parity/test_activation_kurtosis_parity.py`). Reads the shared hidden-state cache when available; otherwise it runs its own forward pass. Counts are `_meta_`. Defaults: `num_samples: 100`, `kurtosis_threshold: 10`, `use_cache: true`. No headline keys or size-invariances are declared.

## 19. Attention Rollout (interpretability_attention_rollout)
* **What are we measuring**: Cumulative token-to-token influence across layers, with the residual connection taken into account.
* **How are we measuring**: Each layer's attention is averaged over heads, augmented with the identity and re-normalised to stay row-stochastic; the rollout is the cumulative matrix product from the first layer to the last (Abnar & Zuidema 2020, Sec. 3). The final-layer rollout is summarised across samples: mean influence from the first token, from the last token, mean self-influence (the diagonal), the mean row-wise Gini coefficient and the mean normalised row entropy.
* **Hypothesis**: Concentrated rollout rows (high Gini, low entropy) mean each position draws on few input tokens; high self-influence means little mixing across positions.
* **Citation/Paper**: `Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers.` [ACL 2020, ArXiv: 2005.00928]
* **File & Function**: `src/blme/tasks/interpretability/attention_rollout.py` -> `AttentionRolloutTask`
* **Critical Info**: The rollout kernel is checked against the reference implementation's `compute_joint_attention(add_residual=True)` in `tests/tasks/parity/test_attention_rollout_parity.py`. Reads cached attention maps when available; otherwise it runs its own forward pass with `output_attentions=True`. Rollout treats head-averaged attention as the only mixing mechanism and ignores MLPs and value transformations. Defaults: `num_samples: 10`. No headline keys or size-invariances are declared.
