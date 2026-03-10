# Causality Tasks

This module contains metrics that utilize causal interventions (ablations, tracing, knockouts) to move beyond correlation and rigorously prove the mechanistic role of specific parameters or activations in an LLM.

---

## 1. Causal Tracing (ROME)
* **What are we measuring**: The precise location (layer and token position) where specific factual associations are injected into the residual stream.
* **How are we measuring**: By corrupting the input embedding to destroy factual recall, and then systematically restoring the clean hidden states at specific layers to see exactly which restoration recovers the original output probability.
* **Hypothesis**: Factual knowledge is highly localized in early-middle MLP modules, acting as key-value stores. Causal tracing maps this retrieval process.
* **Citation/Paper**: `Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT.` [NeurIPS 2022, ArXiv: 2202.05262]
* **File & Function**: `src/blme/tasks/causality/tracing.py` -> `CausalTracingTask`
* **Critical Info**: Computationally expensive as it requires $L \times T$ forward passes (where $L$ is layers and $T$ is tokens) for a single factual prompt to construct the causal heatmap.

## 2. Activation Ablation (Mean/Zero)
* **What are we measuring**: The absolute necessity of specific layers or heads for generating the original prediction.
* **How are we measuring**: Calculating the baseline prediction, physically zeroing out or mean-ablating the activation of a target component during the forward pass, and measuring the resulting drop in probability for the target token.
* **Hypothesis**: The model has redundant pathways. Components that, upon ablation, severely degrade performance are "critical path" components. Mean ablation is preferred over zero-ablation as it preserves the overall norm geometry.
* **Citation/Paper**: Standard mechanistic practice popularized in: `Wang, K., et al. (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.` [ArXiv: 2211.00593]
* **File & Function**: `src/blme/tasks/causality/ablation.py` -> `AblationRobustnessTask`
* **Critical Info**: Simply observing activation magnitude is insufficient for importance; a neuron may fire strongly but be ignored by downstream components. Interventional ablation confirms causality.

## 3. Attention Knockout
* **What are we measuring**: The reliance of the model on specific token-to-token attention edges.
* **How are we measuring**: Forcing specific cells in the attention matrix to zero (often the diagonal, or edges leading back to the subject token) and measuring the impact on the final output distribution. 
* **Hypothesis**: LLMs often route information via very sparse, specific attention bridges. Knocking out the structural attention edge prevents information routing entirely.
* **Citation/Paper**: General methodology related to attention head pruning and causal abstractions.
* **File & Function**: `src/blme/tasks/causality/attention_knockout.py` -> `AttentionKnockoutTask`
* **Critical Info**: Works via sophisticated PyTorch forward hooks that directly intercept and clone-modify the `attn_weights` tensor before it multiplies the Value matrix.

## 4. Circuit Quality (Faithfulness and Minimality)
* **What are we measuring**: Whether a small subset of model layers (a "circuit") can faithfully reproduce the full model's behavior.
* **How are we measuring**: Using mean ablation to rank each layer's causal importance, identifying the top-k% most important layers as the circuit, then ablating all non-circuit layers and measuring how closely the circuit's output distribution matches the full model's via KL divergence. The final score is the harmonic mean of faithfulness (circuit reproduces full model) and minimality (circuit uses few layers).
* **Hypothesis**: If a compact circuit faithfully reproduces model behavior, the model's computation is concentrated in a small subset of layers. Low circuit quality suggests distributed computation across many layers.
* **Citation/Paper**: `Chan, L., et al. (2022). Causal Scrubbing.` and `Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability.` [NeurIPS 2023, ArXiv: 2304.14997]
* **File & Function**: `src/blme/tasks/causality/circuit_quality.py` -> `CircuitQualityTask`
* **Critical Info**: Computationally expensive — requires multiple forward passes per layer for importance ranking, plus additional passes for faithfulness evaluation.
