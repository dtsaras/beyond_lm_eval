# Interpretability Tasks

This module contains metrics that directly probe the internal properties, specialized circuits, and representational capacity of the model's layers and heads.

**Current registry coverage (15 tasks)**: `interpretability_activation_sinks`,
`interpretability_attention_effective_rank`, `interpretability_attention_entropy`,
`interpretability_attention_graph`, `interpretability_attention_rank`,
`interpretability_attribution`, `interpretability_head_roles`,
`interpretability_induction_heads`, `interpretability_logit_lens`,
`interpretability_prediction_entropy`, `interpretability_probing`,
`interpretability_sae_features`, `interpretability_sparsity`,
`interpretability_superposition`, and `interpretability_waa`.

**Paper-faithful vs. BLME proxy notes**: `interpretability_activation_sinks`
implements paper-derived Sinkε, massive-activation, and compression-valley
diagnostics. `interpretability_attention_graph` is a BLME graph-centrality
diagnostic inspired by attention-sink/attention-rollout literature; PageRank
centralization alone is not definitive proof of attention-sink mechanics.
`interpretability_attention_effective_rank` is the registered task in
`attention_polysemanticity.py`; it reports an effective-rank proxy for head
capacity and should not be described as a paper-faithful polysemanticity or
monosemanticity method.

---

## 1. Attention Entropy
* **What are we measuring**: The focus (sharpness) or dispersion of the probability distribution across an attention head.
* **How are we measuring**: Computing the Shannon entropy of the attention weights matrix. 
* **Hypothesis**: Low entropy means the head is sharply focused on a specific token (e.g., induction head), while high entropy means it's broadly attending to the context (e.g., a "bag of words" head).
* **Citation/Paper**: `Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What Does BERT Look at? An Analysis of BERT's Attention.` [ArXiv: 1906.04341]
* **File & Function**: `src/blme/tasks/interpretability/attention.py` -> `AttentionEntropyTask`
* **Critical Info**: Attention entropy is heavily correlated with the depth of the layer; deeper layers generally exhibit lower, more specialized entropy.

## 2. Attention Graph Modularity (Attention Sinks)
* **What are we measuring**: The structural topology of the attention matrix treated as a directed graph.
* **How are we measuring**: Computing the PageRank Centrality on the attention matrix to find specific bottleneck "sink" tokens, and the Edge Gini coefficient to measure global graph sparsity.
* **Hypothesis**: Models offload computation to "Attention Sinks" (usually the BOS token or newline characters) to act as a structural anchor. Finding the central node reveals the computational topology.
* **Citation/Paper**: `Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2023). Efficient Streaming Language Models with Attention Sinks.` [ArXiv: 2309.17453]
* **File & Function**: `src/blme/tasks/interpretability/attention_graph.py` -> `AttentionGraphTopologyTask`
* **Critical Info**: High centralization at index 0 is a useful attention-sink-like signal, but it is a proxy; confirm with Sinkε or intervention-based analyses before making mechanistic claims.

## 3. Attention Effective Rank (BLME head-capacity proxy)
* **What are we measuring**: The effective rank of attention-head value/output representations as a proxy for how much capacity a head uses across samples.
* **How are we measuring**: By computing the Singular Value Entropy (Effective Rank) of the isolated value-projection outputs corresponding to specific heads.
* **Hypothesis**: Higher effective rank suggests a head is using a broader subspace; lower rank suggests a more constrained or collapsed head. This is not a direct concept-level polysemanticity measurement.
* **Citation/Paper**: BLME proxy inspired by effective-rank diagnostics and the superposition literature (`Elhage et al. 2022`; closest concept-level benchmark: `Templeton et al. 2024`, Scaling Monosemanticity).
* **File & Function**: `src/blme/tasks/interpretability/attention_polysemanticity.py` -> `AttentionEffectiveRankTask`
* **Critical Info**: Registered task name is `interpretability_attention_effective_rank`; stale recipe/docs references to `interpretability_attention_polysemanticity` are invalid.

## 4. Induction Heads
* **What are we measuring**: The presence and strength of specialized "Induction Heads" that complete in-context patterns (e.g., A B ... A -> predicts B).
* **How are we measuring**: Generating a sequence of repeated random tokens, then analyzing the attention weights to check if current tokens heavily attend to the token immediately following their previous occurrence.
* **Hypothesis**: Induction heads are the fundamental mechanism behind in-context learning and zero-shot capabilities in LLMs.
* **Citation/Paper**: `Olsson, C., Elhage, N., Nanda, N., et al. (2022). In-context Learning and Induction Heads.` [ArXiv: 2209.11895]
* **File & Function**: `src/blme/tasks/interpretability/induction.py` -> `InductionHeadTask`
* **Critical Info**: These heads typically form abruptly around the middle layers during pre-training in a "phase change".

## 5. Logit Lens
* **What are we measuring**: What the model "believes" the next token should be at each intermediate layer before the final decision.
* **How are we measuring**: Multiplying the hidden states of intermediate layers directly against the vocabulary unembedding matrix to decode their implicit trajectory.
* **Hypothesis**: The model constructs its final prediction iteratively. By decoding early layers, we can see exactly when factual knowledge is injected into the residual stream.
* **Citation/Paper**: `Nostalgebraist. (2020). Interpreting GPT: the logit lens.` [LessWrong Blog Post, No Academic Proceeding]
* **File & Function**: `src/blme/tasks/interpretability/logit_lens.py` -> `LogitLensTask`
* **Critical Info**: Plagued by scaling issues because early representations are not in the same linear space as the final unembedding layer vocabulary.

## 6. Token Attribution (Gradient-based)
* **What are we measuring**: How much each preceding input token contributed to the likelihood of generating a specific target token.
* **How are we measuring**: Computing the gradient of the predicted logit with respect to the input embeddings, taking the L2 norm (InputXGradient).
* **Hypothesis**: Not all context tokens are equal. Saliency mapping reveals which entities heavily bias the model's specific outputs.
* **Citation/Paper**: `Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps.` (General Saliency)
* **File & Function**: `src/blme/tasks/interpretability/attribution.py` -> `ComponentAttributionTask`
* **Critical Info**: Gradients can be noisy. This requires a backward pass, which is much slower and more memory-intensive than standard inference.

## 7. Prediction Entropy
* **What are we measuring**: The model's confidence or uncertainty in its next-token prediction over the vocabulary distribution.
* **How are we measuring**: The Shannon entropy of the Softmax output probabilities.
* **Hypothesis**: High prediction entropy implies the model is genuinely guessing or hallucinating over flat distributions, whereas low prediction entropy indicates hard memorization or high structural constraint.
* **Citation/Paper**: `Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration.` [ICLR 2020, ArXiv: 1904.09751]
* **File & Function**: `src/blme/tasks/interpretability/prediction_entropy.py` -> `PredictionEntropyTask`
* **Critical Info**: Correlates heavily with `perplexity`, but normalizes out the specific text likelihood, giving a pure measure of constraint width.

## 8. Activation Sparsity
* **What are we measuring**: The frequency of inactive (zeroed-out or highly negative) neurons in the MLP feed-forward blocks.
* **How are we measuring**: Computing the L0 pseudo-norm fraction (percentage of active neurons) and the Kurtosis (heavy-tailedness) of the post-GELU/SwiGLU activations.
* **Hypothesis**: LLMs demonstrate severe activation sparsity; only a tiny fraction of the network fires for a given token. This translates to efficient computation and specialized feature maps.
* **Citation/Paper**: Related to `Zhang et al. (2021). Moefication: Transformer Feed-forward Layers are Mixtures of Experts.` and later contextual-sparsity analyses. BLME reports activation-rate diagnostics; it does not implement Deja Vu's inference-time predictor.
* **File & Function**: `src/blme/tasks/interpretability/sparsity.py` -> `ActivationSparsityTask`
* **Critical Info**: Relu networks have hard sparsity (true 0s), whereas GELU models have soft sparsity (negative values near 0). The task supports thresholding for soft sparsity.

## 9. Linear Probing
* **What are we measuring**: How linearly accessible bundled labels are from hidden states.
* **How are we measuring**: Extracting hidden states and training a regularized Logistic Regression probe. Evaluated via cross-validated accuracy/AUC.
* **Hypothesis**: If a linear probe can retrieve a label with high accuracy, the model has made that label linearly decodable; this does not prove the model uses that feature causally.
* **Citation/Paper**: `Alain, G. & Bengio, Y. (2017). Understanding intermediate layers using linear classifier probes.` [ArXiv: 1610.01644]. See also `Belinkov (2022)` for limitations of probing.
* **File & Function**: `src/blme/tasks/interpretability/probing.py` -> `LinearProbingTask`
* **Critical Info**: The metric is essentially measuring the capacity of the *probe*, not just the model, so high regularisation is required to prevent the probe from learning the task entirely.

## 10. Weight-Activation Alignment (WAA)
* **What are we measuring**: The mechanistic capacity utilization of the network.
* **How are we measuring**: Computing the Cosine Similarity between the empirical principal components of the actual inference activations (dynamic) and the principal singular vectors of the static weight matrices.
* **Hypothesis**: If weight and activation eigenvectors are aligned, the model is using its learned capacity cleanly. Misalignment means the static weights contain parameters irrelevant to dynamic generation, causing parameter waste.
* **Citation/Paper**: Related to `Park, K., Choe, Y. J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models.` [ICML 2024, arXiv:2311.03658]. BLME's WAA is a proxy diagnostic, not that paper's full linear-representation analysis.
* **File & Function**: `src/blme/tasks/interpretability/weight_activation_alignment.py` -> `WeightActivationAlignmentTask`
* **Critical Info**: Heavily reliant on computing local SVD on weight matrices, making it expensive for massive models (>70B params) without specific approximations.

## 11. SAE Feature Dimensionality (sae_features)
* **What are we measuring**: The structural sparsity and disentanglement of the representation using a Sparse Autoencoder (SAE).
* **How are we measuring**: Running the inputs through pre-trained SAE dictionaries (via `sae-lens`) to extract L0 norms and active feature counts.
* **Hypothesis**: Traditional hidden states are in superposition. SAEs force features to be sparse and disentangled. Analyzing the SAE features reveals the true atomic semantic variables the model operates on.
* **Citation/Paper**: `Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models.` [ICLR 2024, ArXiv: 2309.08600]
* **File & Function**: `src/blme/tasks/interpretability/sae_features.py` -> `SAEFeatureDimensionalityTask`
* **Critical Info**: Strictly requires the external `sae-lens` library to map to established SAE dictionaries for the specific tested model.

## 12. Superposition Index (Neuron Polysemanticity)
* **What are we measuring**: The degree of superposition (polysemanticity) in model neurons — whether individual neurons encode multiple unrelated features.
* **How are we measuring**: Analyzing the bimodality coefficient of per-neuron activation distributions within MLP layers. A bimodal activation distribution suggests a neuron encodes multiple features (fires strongly for distinct, unrelated inputs). Also measures neuron utilization rate — the fraction of neurons with non-trivial activation variance.
* **Hypothesis**: In superposition, individual neurons compress multiple unrelated features into their activation range. High polysemanticity (bimodality coefficient > 0.555) indicates severe superposition, while low values indicate cleaner, monosemantic neurons.
* **Citation/Paper**: `Elhage, N., et al. (2022). Toy Models of Superposition.` [ArXiv: 2209.10652] and `Templeton, A., et al. (2024). Scaling Monosemanticity.` [Transformer Circuits Thread]
* **File & Function**: `src/blme/tasks/interpretability/superposition.py` -> `SuperpositionIndexTask`
* **Critical Info**: The bimodality coefficient (BC) uses skewness and kurtosis: BC = (skewness^2 + 1) / kurtosis. Values > 0.555 suggest bimodality. Higher mean_polysemanticity_index indicates more severe superposition across the model.

## 13. Attention Rank Collapse
* **What are we measuring**: Effective rank of attention matrices across layers and heads.
* **How are we measuring**: Computing Roy-Vetterli effective rank on attention maps and summarizing collapse trends across depth.
* **Hypothesis**: Very low attention rank suggests attention maps are collapsing to a small set of directions or tokens.
* **Citation/Paper**: `Dong, Y., Cordonnier, J. B., & Loukas, A. (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth.` [ICML 2021, ArXiv: 2103.03404]
* **File & Function**: `src/blme/tasks/interpretability/attention_rank.py` -> `AttentionRankCollapseTask`
* **Critical Info**: Requires eager attention weights on modern transformer backends.

## 14. Attention Head Roles
* **What are we measuring**: Simple structural roles such as previous-token and duplicate-token attention.
* **How are we measuring**: Measuring how much heads attend to immediately previous tokens and repeated-token positions.
* **Hypothesis**: A larger fraction of specialized heads can indicate more developed in-context and copy-like mechanisms.
* **Citation/Paper**: `Clark et al. (2019). What Does BERT Look At?` and `Voita et al. (2019). Analyzing Multi-Head Self-Attention.`
* **File & Function**: `src/blme/tasks/interpretability/head_roles.py` -> `HeadRolesTask`
* **Critical Info**: BLME reports simple role fractions; it does not implement a full OV/copying-score decomposition.

## 15. Activation Sinks, Massive Activations, and Compression Valley
* **What are we measuring**: Three related phenomena: attention sink concentration, massive residual activations, and entropy valleys across layers.
* **How are we measuring**: Computing Gu et al.'s Sinkε, BOS attention mass, residual outlier fractions/ratios, and the minimum of a per-layer matrix-entropy profile.
* **Hypothesis**: These metrics describe emergent bias-token and compression mechanisms that often appear in modern LLMs.
* **Citation/Paper**: `Xiao et al. (2023). Efficient Streaming Language Models with Attention Sinks.` [ArXiv: 2309.17453]; `Gu et al. (2025). When Attention Sink Emerges in Language Models.` [ICLR 2025, ArXiv: 2410.10781]; `Sun et al. (2024). Massive Activations in Large Language Models.` [ArXiv: 2402.17762]; `Arroyo et al. (2025). Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin.` [ArXiv: 2510.06477]
* **File & Function**: `src/blme/tasks/interpretability/activation_sinks.py` -> `ActivationSinksTask`
* **Critical Info**: Sinkε is formula-faithful; the compression-valley output is a BLME matrix-entropy proxy, not the full paper pipeline.
