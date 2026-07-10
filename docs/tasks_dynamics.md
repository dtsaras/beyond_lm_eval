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

**Verification campaign (2026-07, Batch 3)**: every task below carries a
`Verification` field recording its evidence tier — official-code or
official-package parity where a reference exists, plus analytic anchors and
independent recomputation.

---

## 1. Embedding-Neighborhood Stability
* **What are we measuring**: How stable token embedding nearest-neighbor neighborhoods are under a reference model comparison or a seeded embedding perturbation.
* **How are we measuring**: Sampling vocabulary rows, computing cosine kNN sets before and after the reference/perturbation, then reporting Jaccard overlap.
* **Hypothesis**: Robust embedding spaces preserve local neighborhoods under small perturbations or related checkpoints. Low overlap indicates brittle local geometry.
* **Citation/Paper**: BLME diagnostic (proxy-only). Inspiration: `Wendlandt, L., Kummerfeld, J. K., & Mihalcea, R. (2018). Factors Influencing the Surprising Instability of Word Embeddings.` [NAACL-HLT 2018, ArXiv: 1804.09692] — BLME's Jaccard-with-configurable-k is a monotone transform (J = o/(2−o)) of that paper's 10-NN percent-overlap stability; it is not a Lyapunov-exponent or Jacobian calculation.
* **Paper summary**: Wendlandt et al. define embedding stability as the percent overlap of the ten nearest neighbors of a word across embedding spaces (different seeds/corpora/algorithms) and show word2vec/GloVe neighborhoods are surprisingly unstable, with frequency and part-of-speech among the drivers — and that downstream models inherit this instability. BLME transplants the neighborhood-overlap idea to LLM input embeddings, either across two checkpoints (`reference_model` mode) or under a seeded relative-scale Gaussian perturbation (`embedding_noise` mode).
* **File & Function**: `src/blme/tasks/dynamics/stability.py` -> `NeighborhoodStabilityTask`
* **Critical Info**: Default mode is `embedding_noise`; set `reference_model_path` to compare against another checkpoint. Do not report this as an exact dynamical-systems Lyapunov exponent.
* **Verification** (2026-07, campaign B3): three independent numeric routes agree exactly (sklearn NearestNeighbors, scipy cdist + exact-Fraction Jaccard, in-repo reference) — diff 0.0; the previously untested `reference_model` branch is now covered with its analytic anchor (self-reference → Jaccard exactly 1.0, std 0.0; `tests/tasks/test_dynamics.py`).

## 2. Chain of Embedding (CoE)
* **What are we measuring**: How a token representation changes from embedding space through successive transformer layers.
* **How are we measuring**: Computing adjacent-layer magnitude changes, adjacent-layer angle changes, and Wang et al.'s CoE-R / CoE-C scores over the hidden-state chain.
* **Hypothesis**: Correct and incorrect responses can induce different latent-space trajectories; CoE scores are intended as output-free self-evaluation signals.
* **Citation/Paper**: `Wang, Y., Zhang, P., Yang, B., Wong, D. F., & Wang, R. (2025). Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation.` [ICLR 2025, ArXiv: 2410.13640]
* **Paper summary**: Wang et al. treat the per-layer hidden states of a query (its "chain of embedding") as a latent reasoning trajectory and show its shape differs systematically between correct and incorrect answers, enabling self-evaluation with no output labels. Their scores combine per-step magnitude and angle changes normalized by the end-to-end change (Eq. 2–3), yielding CoE-R (real combination, Eq. 5) and CoE-C (complex-plane centroid magnitude). **Attribution caveat verified against the original**: the paper's printed Eq. 6/7 uses raw M·exp(iA), its Algorithm 2 normalizes only the magnitude, and the official code (`Alsace08/Chain-of-Embedding score.py`) normalizes both — BLME follows the official code.
* **File & Function**: `src/blme/tasks/dynamics/coe.py` -> `ChainOfEmbeddingTask`
* **Critical Info**: BLME's default `token_position: last` evaluates the prompt's final token without generation. This is close to the paper's geometry but not identical to the paper's generated-output mean pooling (flagged in the result dict).
* **Verification** (2026-07, campaign B3): `_coe_from_chain` vs verbatim-transcribed official `CoEScoreInfo` on random chains with Z_Mag ≠ 1: coe_c ≤ 1.9e-9, coe_r ≤ 7.2e-8 (float32 acos); a Z_Mag ≠ 1 regression case is now pinned in-test against the official formulas (the old fixture had Z_Mag exactly 1, hiding the Eq. 3 normalization).

## 3. Latent Interpolation (Convexity Proxy)
* **What are we measuring**: Whether the latent probability space is continuously convex between two valid representations.
* **How are we measuring**: Selecting two distinct hidden state vectors ($h_1$, $h_2$) and decoding the points linearly interpolated between them via the language modeling head. We check output entropy across the line, plus a slerp trajectory for comparison.
* **Hypothesis**: A robust representation space should generally construct smooth gradients between concepts. If the exact middle point ($0.5 h_1 + 0.5 h_2$) collapses into extreme entropy (random noise), the semantic space is severely non-convex and structurally brittle.
* **Citation/Paper**: BLME diagnostic (proxy-only); the slerp kernel follows `White, T. (2016). Sampling Generative Networks.` [ArXiv: 1609.04468] (which credits Shoemake 1985), via the official `dribnet/plat` implementation.
* **Paper summary**: White shows that linear interpolation between Gaussian latent points leaves the high-probability shell of the prior, and proposes spherical linear interpolation (slerp, Shoemake's great-arc formula) so interpolation paths stay on-manifold — the standard trick for latent-space traversals in generative models. BLME uses both: norm-corrected lerp and slerp between final hidden states, decoded through the LM head, with next-token entropy as the smoothness readout (the entropy/convexity summary is BLME's own diagnostic).
* **File & Function**: `src/blme/tasks/dynamics/trajectories.py` -> `LatentInterpolationTask`
* **Critical Info**: Relies explicitly on the model's Unembedding matrix structure; any failure here may be the fault of the linear head rather than the transformer block.
* **Verification** (2026-07, campaign B3): `_slerp` vs the VERBATIM official White/plat `interpolate.py` slerp: max diff 8.9e-16 over 140 random cases; end-to-end entropies match an independent float64 recomputation ≤ 8.1e-6 (documented epsilon); the <2-usable-rows infinite-loop bug is fixed (attempt cap, `c2911bf`) and the endpoint invariant (lerp ≡ slerp entropy at α = 0, 1) is now pinned in-test.

## 4. Loss-Landscape Sharpness
* **What are we measuring**: Local curvature of the model's loss around the current parameters.
* **How are we measuring**: Hutchinson trace estimates, power iteration for the top Hessian eigenvalue (signed Rayleigh quotient), and SAM-style loss increase over a small parameter perturbation, on a configurable parameter scope (default: last transformer block).
* **Hypothesis**: Sharper local curvature can indicate brittle optimization geometry or poorer generalization.
* **Citation/Paper**: `Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization.` [ICLR 2021, ArXiv: 2010.01412]; `Yao, Z., Gholami, A., Keutzer, K., & Mahoney, M. (2020). PyHessian: Neural Networks Through the Lens of the Hessian.` [IEEE Big Data 2020, ArXiv: 1912.07145]; Hutchinson trace: `Hutchinson (1990)`; `Bai, Fahey & Golub (1996)`.
* **Paper summary**: Foret et al. link the sharpness of the loss landscape to generalization and train by minimizing the worst-case loss in a ρ-ball; their Eq. 2 (p=2) perturbation ε̂ = ρ·∇L/‖∇L‖ is exactly BLME's SAM-sharpness probe L(θ+ε̂)−L(θ). Yao et al.'s PyHessian makes Hessian spectral analysis tractable for deep nets via Hessian-vector products: top eigenvalues by power iteration (signed Rayleigh quotient), trace by Hutchinson's stochastic estimator, and shows these track training phenomena (batch size, residual connections). BLME implements both probes on the last-block parameter subset by default so the cost stays bounded for large models.
* **File & Function**: `src/blme/tasks/dynamics/sharpness.py` -> `LossSharpnessTask`
* **Critical Info**: Second-order derivatives may require eager attention instead of SDPA/Flash Attention (the task detects this and errors cleanly). Hutchinson/power-iteration probe vectors are seeded (`seed` config, default 42) — B3 made the estimates reproducible run-to-run.
* **Verification** (2026-07, campaign B3): power iteration == exact `eigh` λ_max to <1e-9 (f64) and == pip PyHessian `eigenvalues(top_n=1)` to <1e-5; the earlier ‖Hv‖ form dropped the eigenvalue SIGN — fixed to the signed Rayleigh quotient (`2d01ae0`) and now pinned by a negative-dominant-Hessian anchor (diag(−5,3) → −5, not +5); SAM delta == analytic quadratic answer ρ‖θ‖+ρ²/2 exactly, with parameter restoration asserted; parity tests now import the real helpers (previously they transcribed the loop), and a real `evaluate()` end-to-end determinism test exists.

## 5. Gradient Flow
* **What are we measuring**: Per-layer gradient norm propagation under next-token cross-entropy.
* **How are we measuring**: Capturing layer inputs with hooks and backpropagating the shifted language-modeling loss; summarizing with entropy, a slope over normalized depth, and a vanishing ratio.
* **Hypothesis**: Vanishing or exploding gradients across depth indicate training or conditioning pathologies.
* **Citation/Paper**: `Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks.` [ICML 2013, ArXiv: 1211.5063]
* **Paper summary**: Pascanu et al. formalize vanishing/exploding gradients in recurrent nets: the error signal decomposes into products of step-to-step Jacobians (their Sec. 1.1, Eq. 3–5), whose norms shrink or blow up exponentially (Sec. 2), and they propose clipping and a regularizer. BLME's refined adaptation swaps the time axis for the DEPTH axis of a transformer: ‖dL_CE/dh_l‖_F per layer plays the role of the per-step gradient norm; the entropy/slope/vanishing-ratio summaries are BLME's own (no repo parity claimed — the paper's official code is RNN training code with no diagnostic kernel to port).
* **File & Function**: `src/blme/tasks/dynamics/gradient_flow.py` -> `GradientFlowTask`
* **Critical Info**: This is a diagnostic backward pass on the evaluated model, not a training run. The slope regresses log-norm on NORMALIZED depth (layer/(n_layers−1)) so it is comparable across depths.
* **Verification** (2026-07, campaign B3): per-layer norms vs autograd-free central finite differences (float64): max rel diff 1.1e-7; entropy == scipy.stats.entropy exactly; slope == scipy.stats.linregress to 3e-17; analytic anchors (ln 4 and −4) pinned. The module docstring described a legacy argmax-logit variant — corrected to the shipped CE-loss definition.

## 6. Generation Diversity
* **What are we measuring**: Diversity and repetition collapse in sampled continuations.
* **How are we measuring**: Distinct-n, Self-BLEU, entropy drift, token repetition, and repeated 4-gram rates over multiple sampled completions.
* **Hypothesis**: Degenerate generation often shows low distinct-n, high Self-BLEU, falling entropy, and repeated phrases.
* **Citation/Paper**: `Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models.` [NAACL 2016, ArXiv: 1510.03055]; `Zhu, Y., Lu, S., Zheng, L., Guo, J., Zhang, W., Wang, J., & Yu, Y. (2018). Texygen: A Benchmarking Platform for Text Generation Models.` [SIGIR 2018, ArXiv: 1802.01886]
* **Paper summary**: Li et al. observe that neural conversation models collapse onto bland high-frequency replies and propose a mutual-information objective; their distinct-1/2 metrics (distinct n-grams scaled by total generated tokens) became the standard diversity readout. Zhu et al.'s Texygen benchmarking platform contributes Self-BLEU — each generated text BLEU-scored against the other generations — where high Self-BLEU means mode collapse. BLME reports distinct-n per completion (scope flagged in the output as `per_completion_mean`, vs Li's corpus-level scope) and the exact Texygen Self-BLEU procedure (nltk smoothing method1, leave-one-out).
* **File & Function**: `src/blme/tasks/dynamics/generation_diversity.py` -> `GenerationDiversityTask`
* **Critical Info**: Sampling-based; pass `seed` in the task config for reproducible completions (the task seeds torch's RNG before `generate()` — transformers has no per-call generator kwarg, and passing one crashes generation, fixed in `7f9ba4c`).
* **Verification** (2026-07, campaign B3): `_self_bleu_single` vs verbatim-transcribed Texygen `calc_bleu`: worst diff 2.8e-17 (gram 3 and 4); `_distinct_n` vs the Li 2016 formula: 600 checks, diff 0.0; entropy quarters == scipy. The seeded path is regression-tested against a fake model that rejects unknown `generate()` kwargs (the old test enshrined the broken `generator=` API). Texygen author list corrected in PAPERS.md (last author is Yong Yu).
