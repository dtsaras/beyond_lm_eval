# Dynamics Tasks

This module assesses the mathematical stability and topological progression of the model's representations over the course of the forward pass or autoregressive generation trajectory.

---

## 1. Local Generation Stability (Lyapunov Exponents / Jacobians)
* **What are we measuring**: The sensitivity of the generation trajectory to minuscule perturbations in the initial condition.
* **How are we measuring**: Approximating the largest Lyapunov exponent or measuring the spectral norm of the input-output Jacobian over consecutive token generations.
* **Hypothesis**: Text generation is a discrete dynamical system. A system with a positive Lyapunov exponent is chaotic, meaning small prompt changes exponentially alter the output. A negative exponent implies structural stability and convergence to semantic attractors.
* **Citation/Paper**: Principles derived from deterministic chaos theory applied to RNNs. Specific implementation generalized for LLMs.
* **File & Function**: `src/blme/tasks/dynamics/stability.py` -> `NeighborhoodStabilityTask`
* **Critical Info**: Extremely complex to compute for deep networks. We often substitute exact Jacobians with empirical forward-pass noise injection due to computational scaling laws.

## 2. Center of Expansion (COE)
* **What are we measuring**: How the semantic "center of mass" of generated text drifts across the latent space over long sequences.
* **How are we measuring**: Extracting the sentence/token embeddings at regular intervals, computing their geometric centroid, and tracking its spatial trajectory.
* **Hypothesis**: Coherent, highly-focused text tightly orbits a specific semantic center in the representation space. Tangential rambling or hallucination causes the Center of Expansion to wildly fracture and drift into unrelated subspace regions.
* **Citation/Paper**: Internal BLME specialized metric.
* **File & Function**: `src/blme/tasks/dynamics/coe.py` -> `ChainOfEmbeddingTask`
* **Critical Info**: Relies heavily on the representation properties functioning identically at generation step $T=1$ versus step $T=500$, which breaks down if attention mechanisms (like RoPE) deteriorate over distance.

## 3. Latent Interpolation (Convexity)
* **What are we measuring**: Whether the latent probability space is continuously convex between two valid representations.
* **How are we measuring**: Selecting two distinct hidden state vectors ($h_1$, $h_2$) and decoding the points linearly interpolated between them via the language modeling head. We check output entropy across the line.
* **Hypothesis**: A robust representation space should generally construct smooth gradients between concepts. If the exact middle point ($0.5 h_1 + 0.5 h_2$) collapses into extreme entropy (random noise), the semantic space is severely non-convex and structurally brittle.
* **Citation/Paper**: Standard technique in Generative Adversarial Network (GAN) and latent space validation mapped to NLP representations.
* **File & Function**: `src/blme/tasks/dynamics/trajectories.py` -> `LatentInterpolationTask`
* **Critical Info**: Relies explicitly on the model's Unembedding matrix structure; any failure here may be the fault of the linear head rather than the transformer block.

