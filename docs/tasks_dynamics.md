# Dynamics Tasks

This module assesses the mathematical stability and topological progression of the model's representations over the course of the forward pass or autoregressive generation trajectory.

**Current registry coverage (6 tasks)**: `dynamics_coe`,
`dynamics_generation_diversity`, `dynamics_gradient_flow`,
`dynamics_interpolation`, `dynamics_sharpness`, and `dynamics_stability`.

**Paper-faithful vs. BLME proxy notes**: `dynamics_coe` implements Wang et al.'s
CoE magnitude/angle scores, but BLME reads the prompt's final-token hidden-state
chain by default instead of mean-pooling generated output tokens; treat it as a
prompt-side CoE variant. `dynamics_stability` is an empirical perturbation proxy,
not an exact Lyapunov-exponent calculation.

---

## 1. Embedding-Neighborhood Stability
* **What are we measuring**: How stable token embedding nearest-neighbor neighborhoods are under a reference model comparison or a seeded embedding perturbation.
* **How are we measuring**: Sampling vocabulary rows, computing cosine kNN sets before and after the reference/perturbation, then reporting Jaccard overlap.
* **Hypothesis**: Robust embedding spaces preserve local neighborhoods under small perturbations or related checkpoints. Low overlap indicates brittle local geometry.
* **Citation/Paper**: BLME diagnostic inspired by representation-stability and neighborhood-overlap analyses; it is not a Lyapunov-exponent or Jacobian calculation.
* **File & Function**: `src/blme/tasks/dynamics/stability.py` -> `NeighborhoodStabilityTask`
* **Critical Info**: Default mode is `embedding_noise`; set `reference_model_path` to compare against another checkpoint. Do not report this as an exact dynamical-systems Lyapunov exponent.

## 2. Center of Expansion (COE)
* **What are we measuring**: How a token representation changes from embedding space through successive transformer layers.
* **How are we measuring**: Computing adjacent-layer magnitude changes, adjacent-layer angle changes, and Wang et al.'s CoE-R / CoE-C scores over the hidden-state chain.
* **Hypothesis**: Correct and incorrect responses can induce different latent-space trajectories; CoE scores are intended as output-free self-evaluation signals.
* **Citation/Paper**: `Wang, Y., Zhang, P., Yang, B., Wong, D. F., & Wang, R. (2025). Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation.` [ICLR 2025, ArXiv: 2410.13640]
* **File & Function**: `src/blme/tasks/dynamics/coe.py` -> `ChainOfEmbeddingTask`
* **Critical Info**: BLME's default `token_position: last` evaluates the prompt's final token without generation. This is close to the paper's geometry but not identical to the paper's generated-output mean pooling.

## 3. Latent Interpolation (Convexity Proxy)
* **What are we measuring**: Whether the latent probability space is continuously convex between two valid representations.
* **How are we measuring**: Selecting two distinct hidden state vectors ($h_1$, $h_2$) and decoding the points linearly interpolated between them via the language modeling head. We check output entropy across the line.
* **Hypothesis**: A robust representation space should generally construct smooth gradients between concepts. If the exact middle point ($0.5 h_1 + 0.5 h_2$) collapses into extreme entropy (random noise), the semantic space is severely non-convex and structurally brittle.
* **Citation/Paper**: Standard latent-space interpolation / slerp diagnostic adapted to NLP representations; no single canonical LLM paper.
* **File & Function**: `src/blme/tasks/dynamics/trajectories.py` -> `LatentInterpolationTask`
* **Critical Info**: Relies explicitly on the model's Unembedding matrix structure; any failure here may be the fault of the linear head rather than the transformer block.

## 4. Loss-Landscape Sharpness
* **What are we measuring**: Local curvature of the model's loss around the current parameters.
* **How are we measuring**: Hutchinson trace estimates, power iteration, and SAM-style loss increase over a small parameter perturbation.
* **Hypothesis**: Sharper local curvature can indicate brittle optimization geometry or poorer generalization.
* **Citation/Paper**: `Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization.` [ICLR 2021] and `Yao et al. (2020). PyHessian.` [ArXiv: 1912.07145]
* **File & Function**: `src/blme/tasks/dynamics/sharpness.py` -> `LossSharpnessTask`
* **Critical Info**: Second-order derivatives may require eager attention instead of SDPA/Flash Attention.

## 5. Gradient Flow
* **What are we measuring**: Per-layer gradient norm propagation under next-token cross-entropy.
* **How are we measuring**: Capturing layer inputs with hooks and backpropagating the shifted language-modeling loss.
* **Hypothesis**: Vanishing or exploding gradients across depth indicate training or conditioning pathologies.
* **Citation/Paper**: `Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks.` [ICML 2013]
* **File & Function**: `src/blme/tasks/dynamics/gradient_flow.py` -> `GradientFlowTask`
* **Critical Info**: This is a diagnostic backward pass on the evaluated model, not a training run.

## 6. Generation Diversity
* **What are we measuring**: Diversity and repetition collapse in sampled continuations.
* **How are we measuring**: Distinct-n, Self-BLEU, entropy drift, token repetition, and repeated 4-gram rates over multiple sampled completions.
* **Hypothesis**: Degenerate generation often shows low distinct-n, high Self-BLEU, falling entropy, and repeated phrases.
* **Citation/Paper**: `Li et al. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models.` [NAACL 2016] and `Zhu et al. (2018). Texygen.` [ArXiv: 1802.01886]
* **File & Function**: `src/blme/tasks/dynamics/generation_diversity.py` -> `GenerationDiversityTask`
* **Critical Info**: Uses sampling, so set seeds externally when comparing exact values across runs.

