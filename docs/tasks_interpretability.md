# Interpretability Tasks

This module contains metrics that directly probe the internal properties, specialized circuits, and representational capacity of the model's layers and heads.

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
* **Critical Info**: High centralization at index 0 without context is a definitive signature of Attention Sink mechanics.

## 3. Attention Head Polysemanticity (Superposition)
* **What are we measuring**: The degree of concept superposition within individual attention heads.
* **How are we measuring**: By computing the Singular Value Entropy (Effective Rank) of the isolated value-projection outputs corresponding to specific heads.
* **Hypothesis**: A single attention head compresses multiple unrelated atomic concepts into its subspace. High SVD entropy signifies severe polysemanticity (superposition), while low entropy indicates a clean, monosemantic head.
* **Citation/Paper**: Derived from the superposition framework established in `Elhage, N., et al. (2022). Toy Models of Superposition` [ArXiv: 2209.10652].
* **File & Function**: `src/blme/tasks/interpretability/attention_polysemanticity.py` -> `AttentionHeadPolysemanticityTask`
* **Critical Info**: Because attention acts as a routing mechanism, highly polysemantic heads route multiple conflicting signals simultaneously.

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
* **Citation/Paper**: Typical uncertainty metric.
* **File & Function**: `src/blme/tasks/interpretability/prediction_entropy.py` -> `PredictionEntropyTask`
* **Critical Info**: Correlates heavily with `perplexity`, but normalizes out the specific text likelihood, giving a pure measure of constraint width.

## 8. Activation Sparsity
* **What are we measuring**: The frequency of inactive (zeroed-out or highly negative) neurons in the MLP feed-forward blocks.
* **How are we measuring**: Computing the L0 pseudo-norm fraction (percentage of active neurons) and the Kurtosis (heavy-tailedness) of the post-GELU/SwiGLU activations.
* **Hypothesis**: LLMs demonstrate severe activation sparsity; only a tiny fraction of the network fires for a given token. This translates to efficient computation and specialized feature maps.
* **Citation/Paper**: `Liu, Z., et al. (2023). Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time.` [ArXiv: 2310.17157]
* **File & Function**: `src/blme/tasks/interpretability/sparsity.py` -> `ActivationSparsityTask`
* **Critical Info**: Relu networks have hard sparsity (true 0s), whereas GELU models have soft sparsity (negative values near 0). The task supports thresholding for soft sparsity.

## 9. Linear Probing
* **What are we measuring**: How linearly accessible a specific high-level concept (e.g., Parts of Speech) is within the hidden states.
* **How are we measuring**: Extracting hidden states and training a simple supervised Logistic Regression classifier to separate the concepts. Evaluated via Cross-Entropy or Accuracy.
* **Hypothesis**: If a linear probe can retrieve the concept with high accuracy, the model has actively constructed a structural geometric boundary for that concept in its primary representation space.
* **Citation/Paper**: `Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances.` [ArXiv: 2102.12452]
* **File & Function**: `src/blme/tasks/interpretability/probing.py` -> `LinearProbingTask`
* **Critical Info**: The metric is essentially measuring the capacity of the *probe*, not just the model, so high regularisation is required to prevent the probe from learning the task entirely.

## 10. Weight-Activation Alignment (WAA)
* **What are we measuring**: The mechanistic capacity utilization of the network.
* **How are we measuring**: Computing the Cosine Similarity between the empirical principal components of the actual inference activations (dynamic) and the principal singular vectors of the static weight matrices.
* **Hypothesis**: If weight and activation eigenvectors are aligned, the model is using its learned capacity cleanly. Misalignment means the static weights contain parameters irrelevant to dynamic generation, causing parameter waste.
* **Citation/Paper**: Broadly derived from general intrinsic dimensionality and capacity literature in LLMs.
* **File & Function**: `src/blme/tasks/interpretability/weight_activation_alignment.py` -> `WeightActivationAlignmentTask`
* **Critical Info**: Heavily reliant on computing local SVD on weight matrices, making it expensive for massive models (>70B params) without specific approximations.

## 11. SAE Feature Dimensionality (sae_features)
* **What are we measuring**: The structural sparsity and disentanglement of the representation using a Sparse Autoencoder (SAE).
* **How are we measuring**: Running the inputs through pre-trained SAE dictionaries (via `sae-lens`) to extract L0 norms and active feature counts.
* **Hypothesis**: Traditional hidden states are in superposition. SAEs force features to be sparse and disentangled. Analyzing the SAE features reveals the true atomic semantic variables the model operates on.
* **Citation/Paper**: `Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models.` [ArXiv: 2309.08600]
* **File & Function**: `src/blme/tasks/interpretability/sae_features.py` -> `SAEFeatureDimensionalityTask`
* **Critical Info**: Strictly requires the external `sae-lens` library to map to established SAE dictionaries for the specific tested model.
