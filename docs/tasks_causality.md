# Causality Tasks

This module contains metrics that use causal interventions or attribution proxies (ablations, tracing, knockouts, saliency, and first-order patching) to test whether specific parameters or activations influence an LLM's outputs. These diagnostics provide evidence about mechanism; they do not by themselves prove a complete causal circuit.

**Current registry coverage (6 tasks)**: `causality_ablation`,
`causality_attention_knockout`, `causality_circuit_quality`,
`causality_edge_attribution`, `causality_knowledge_neurons`, and
`causality_tracing`.

**Paper-faithful vs. BLME proxy notes**: `causality_tracing` follows the
ROME-style causal tracing setup. `causality_edge_attribution` is a simplified
per-layer first-order attribution-patching proxy using shuffled-token
corruption, not full edge-level EAP circuit recovery. `causality_circuit_quality`
uses layer-ranking and non-circuit ablation as a BLME circuit-quality heuristic,
not a full implementation of ACDC or causal scrubbing.

---

## 1. Causal Tracing (ROME)
* **What are we measuring**: The precise location (layer and token position) where specific factual associations are injected into the residual stream.
* **How are we measuring**: By corrupting the input embedding to destroy factual recall, and then systematically restoring the clean hidden states at specific layers to see exactly which restoration recovers the original output probability.
* **Hypothesis**: Factual knowledge is highly localized in early-middle MLP modules, acting as key-value stores. Causal tracing maps this retrieval process.
* **Citation/Paper**: `Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT.` [NeurIPS 2022, ArXiv: 2202.05262]
* **File & Function**: `src/blme/tasks/causality/tracing.py` -> `CausalTracingTask`
* **Critical Info**: Computationally expensive as it requires $L \times T$ forward passes (where $L$ is layers and $T$ is tokens) for a single factual prompt to construct the causal heatmap.

## 2. Activation Ablation (Mean/Zero)
* **What are we measuring**: The sensitivity of the original prediction to ablating residual-stream feature dimensions.
* **How are we measuring**: Computing baseline loss, then mean-ablating sampled fractions of hidden dimensions and measuring the resulting loss increase.
* **Hypothesis**: Models whose predictions depend on a small set of residual dimensions should degrade sharply under feature ablation; models with more distributed representations should degrade more gradually.
* **Citation/Paper**: BLME residual-feature robustness diagnostic inspired by standard mechanistic ablation practice. It is not the IOI circuit-ablation method from Wang et al. 2022.
* **File & Function**: `src/blme/tasks/causality/ablation.py` -> `AblationRobustnessTask`
* **Critical Info**: Simply observing activation magnitude is insufficient for importance; a neuron may fire strongly but be ignored by downstream components. Interventional ablation confirms causality.

## 3. Attention Knockout
* **What are we measuring**: The reliance of the model on specific attention heads.
* **How are we measuring**: Zeroing each head's contribution before the output projection and measuring the loss impact.
* **Hypothesis**: LLMs often route information through a small number of specialized heads. High impact concentration means only a few heads dominate the measured behavior.
* **Citation/Paper**: Related to attention head pruning and analysis: `Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One?` and `Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.`
* **File & Function**: `src/blme/tasks/causality/attention_knockout.py` -> `AttentionKnockoutTask`
* **Critical Info**: This is head knockout, not arbitrary attention-edge knockout. Use edge-attribution or task-specific interventions for edge-level circuit claims.

## 4. Circuit Quality (Faithfulness and Minimality)
* **What are we measuring**: Whether a small subset of model layers (a "circuit") can faithfully reproduce the full model's behavior.
* **How are we measuring**: Using mean ablation to rank each layer's causal importance, identifying the top-k% most important layers as the circuit, then ablating all non-circuit layers and measuring how closely the circuit's output distribution matches the full model's via KL divergence. The final score is the harmonic mean of faithfulness (circuit reproduces full model) and minimality (circuit uses few layers).
* **Hypothesis**: If a compact circuit faithfully reproduces model behavior, the model's computation is concentrated in a small subset of layers. Low circuit quality suggests distributed computation across many layers.
* **Citation/Paper**: `Chan, L., et al. (2022). Causal Scrubbing.` and `Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability.` [NeurIPS 2023, ArXiv: 2304.14997]
* **File & Function**: `src/blme/tasks/causality/circuit_quality.py` -> `CircuitQualityTask`
* **Critical Info**: Computationally expensive — requires multiple forward passes per layer for importance ranking, plus additional passes for faithfulness evaluation.

## 5. Edge Attribution Patching (simplified per-layer proxy)
* **What are we measuring**: How concentrated a first-order attribution score is across layers for a clean/corrupted prompt pair.
* **How are we measuring**: Computing `(h_clean - h_corrupted) · grad(target_logit | h_clean)` per layer, where the corrupted input is a reproducible token shuffle.
* **Hypothesis**: If the attribution mass is concentrated in a small set of layers, those layers are likely carrying most of the first-order contribution to the target prediction.
* **Citation/Paper**: `Syed, A., Rager, C., & Conmy, A. (2024). Attribution Patching Outperforms Automated Circuit Discovery.` [BlackboxNLP 2024, ArXiv: 2310.10348]
* **File & Function**: `src/blme/tasks/causality/edge_attribution.py` -> `EdgeAttributionTask`
* **Critical Info**: This is not the full paper method: BLME aggregates per layer and uses shuffled-token corruption, so report it as an attribution proxy rather than recovered circuit edges.

## 6. Knowledge Neurons
* **What are we measuring**: Which MLP intermediate activations have high saliency for factual target-token logits.
* **How are we measuring**: Capturing MLP down-projection inputs and backpropagating the target logit to score neuron-level attribution concentration.
* **Hypothesis**: Factual predictions often rely on sparse MLP directions; strong concentration suggests a few neurons/layers dominate the target logit.
* **Citation/Paper**: `Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers.` [ACL 2022, ArXiv: 2104.08696]
* **File & Function**: `src/blme/tasks/causality/knowledge_neurons.py` -> `KnowledgeNeuronsTask`
* **Critical Info**: This saliency diagnostic localizes candidate neurons; it does not implement Dai et al.'s full integrated-gradient thresholding or fact editing/validation protocol.
