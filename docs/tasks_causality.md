# Causality Tasks

This module contains metrics that use causal interventions or attribution proxies (ablations, tracing, knockouts, saliency, and first-order patching) to test whether specific parameters or activations influence an LLM's outputs. These diagnostics provide evidence about mechanism; they do not by themselves prove a complete causal circuit.

**Registry coverage (7 tasks)**: `causality_ablation`,
`causality_attention_knockout`, `causality_circuit_quality`,
`causality_edge_attribution`, `causality_knowledge_neurons`,
`causality_layer_skip`, and `causality_tracing`.

**Paper-faithful vs. BLME proxy notes**: `causality_tracing` follows the
ROME-style causal tracing setup. `causality_edge_attribution` is a simplified
per-layer first-order attribution-patching proxy using shuffled-token
corruption, not full edge-level EAP circuit recovery. `causality_circuit_quality`
uses layer-ranking and non-circuit ablation as a BLME circuit-quality heuristic,
not a full implementation of ACDC or causal scrubbing.

---

## 1. Causal Tracing (ROME)
* **What are we measuring**: The precise location (layer and token position) where specific factual associations are injected into the residual stream.
* **How are we measuring**: For each (prompt, subject, target) triple: run the clean prompt, then add Gaussian noise to the subject-token embeddings (`n_noise_samples: 10` draws; the noise scale is derived from the model's embeddings unless `noise_std` is set), then restore the clean hidden state at the subject positions of one layer at a time and record how much of the lost target probability is recovered.
* **Hypothesis**: Factual knowledge is highly localized in early-middle MLP modules, acting as key-value stores. Causal tracing maps this retrieval process.
* **Citation/Paper**: `Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT.` [NeurIPS 2022, ArXiv: 2202.05262]
* **File & Function**: `src/blme/tasks/causality/tracing.py` -> `CausalTracingTask`
* **Critical Info**: Reports the per-layer average indirect effect, `max_aie`, `max_causal_layer_idx` and `causal_entropy`. Defaults: `num_samples: 3`. No headline keys or size-invariances are declared. Cost grows with layers x noise draws per prompt. Without a dataset of triples the task falls back to a small bundled set of factual prompts.

## 2. Activation Ablation (Mean/Zero)
* **What are we measuring**: The sensitivity of the original prediction to ablating residual-stream feature dimensions.
* **How are we measuring**: Computing baseline cross-entropy, then mean-ablating a random fraction of residual-stream coordinates (`ablation_percentages: [0.01, 0.05, 0.1, 0.25]`) in a middle band of layers and measuring the loss increase; the area under the degradation curve summarises the profile.
* **Hypothesis**: Models whose predictions depend on a small set of residual dimensions should degrade sharply under feature ablation; models with more distributed representations should degrade more gradually.
* **Citation/Paper**: BLME residual-feature robustness diagnostic inspired by standard mechanistic ablation practice. It is not the IOI circuit-ablation method from Wang et al. 2022.
* **File & Function**: `src/blme/tasks/causality/ablation.py` -> `AblationRobustnessTask`
* **Critical Info**: This is a coarse residual-feature robustness probe, not localized MLP-neuron or attention-head ablation. Defaults: `num_samples: 5`. No headline keys or size-invariances are declared. Activation magnitude alone does not establish importance; the intervention measures the effect on the loss.

## 3. Attention Knockout
* **What are we measuring**: The reliance of the model on specific attention heads.
* **How are we measuring**: Zeroing each head's contribution before the output projection and measuring the loss impact.
* **Hypothesis**: LLMs often route information through a small number of specialized heads. High impact concentration means only a few heads dominate the measured behavior.
* **Citation/Paper**: Related to attention head pruning and analysis: `Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One?` and `Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.`
* **File & Function**: `src/blme/tasks/causality/attention_knockout.py` -> `AttentionKnockoutTask`
* **Critical Info**: This is head knockout, not arbitrary attention-edge knockout. Reports `max_knockout_impact`, `mean_knockout_impact`, `head_impact_gini_coefficient` and `per_head_impacts`. Defaults: `num_samples: 3`. No headline keys or size-invariances are declared. Use edge-attribution or task-specific interventions for edge-level circuit claims.

## 4. Circuit Quality (Faithfulness and Minimality)
* **What are we measuring**: Whether a small subset of model layers (a "circuit") can faithfully reproduce the full model's behavior.
* **How are we measuring**: Using mean ablation to rank each layer's causal importance, identifying the top-k% most important layers as the circuit, then ablating all non-circuit layers and measuring how closely the circuit's output distribution matches the full model's via KL divergence. The final score is the harmonic mean of faithfulness (circuit reproduces full model) and minimality (circuit uses few layers).
* **Hypothesis**: If a compact circuit faithfully reproduces model behavior, the model's computation is concentrated in a small subset of layers. Low circuit quality suggests distributed computation across many layers.
* **Citation/Paper**: `Chan, L., et al. (2022). Causal Scrubbing.` and `Conmy, A., et al. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability.` [NeurIPS 2023, ArXiv: 2304.14997]
* **File & Function**: `src/blme/tasks/causality/circuit_quality.py` -> `CircuitQualityTask`
* **Critical Info**: A layer-level heuristic (`diagnostic_method: layer_mean_ablation_circuit_proxy`), not ACDC or causal scrubbing. Defaults: `num_samples: 3`, `top_k_pct: 25`. No headline keys or size-invariances are declared. Requires several forward passes per layer for the importance ranking plus further passes for the faithfulness evaluation.

## 5. Layer attribution
* **What are we measuring**: How concentrated a first-order attribution score is across layers for a clean/corrupted prompt pair.
* **How are we measuring**: Computing `(h_clean - h_corrupted) · grad(target_logit | h_clean)` per layer, where the corrupted input is a reproducible token shuffle.
* **Hypothesis**: If the attribution mass is concentrated in a small set of layers, those layers are likely carrying most of the first-order contribution to the target prediction.
* **Citation/Paper**: `Syed, A., Rager, C., & Conmy, A. (2024). Attribution Patching Outperforms Automated Circuit Discovery.` [BlackboxNLP 2024, ArXiv: 2310.10348]
* **File & Function**: `src/blme/tasks/causality/edge_attribution.py` -> `EdgeAttributionTask`
* **Critical Info**: This is not the full paper method: BLME aggregates per layer and uses shuffled-token corruption, so report it as an attribution proxy (`diagnostic_method: residual_layer_gradient_patch_proxy`) and not as recovered circuit edges. Defaults: `num_samples: 10`. No headline keys or size-invariances are declared.

## 6. Neuron saliency
* **What are we measuring**: Which MLP intermediate activations have high saliency for factual target-token logits.
* **How are we measuring**: Capturing MLP down-projection inputs and backpropagating the target logit to score neuron-level attribution concentration.
* **Hypothesis**: Factual predictions often rely on sparse MLP directions; strong concentration suggests a few neurons/layers dominate the target logit.
* **Citation/Paper**: `Dai, D., Dong, L., Hao, Y., Sui, Z., Chang, B., & Wei, F. (2022). Knowledge Neurons in Pretrained Transformers.` [ACL 2022, ArXiv: 2104.08696]
* **File & Function**: `src/blme/tasks/causality/knowledge_neurons.py` -> `KnowledgeNeuronsTask`
* **Critical Info**: A gradient-times-activation saliency diagnostic (`diagnostic_method: ffn_gradient_activation_saliency`) that localizes candidate neurons; it does not implement Dai et al.'s integrated-gradient path integral, thresholding, or fact editing/validation protocol. Runs on a bundled fact set unless the dataset supplies `prompt`/`target` pairs. No headline keys or size-invariances are declared.

## 7. Layer-Skip Effect Profile
* **What are we measuring**: How much the model's later computation and its output depend on each block, and the depth at which that dependence sits (Csordas et al.'s depth score).
* **How are we measuring**: For each prompt, 4 split points t are drawn from range(8, T-4, 8). Block s is skipped (its output set to its input) on tokens 1..t-1, and on tokens >= t we record the largest relative change of every block's contribution ||c - c'|| / ||c|| and the largest L2 change of the output distribution ||p - p'||. Both are maximised over split points and the 10 prompts. A block's importance is its mean effect on the later blocks; the depth score is d = sum_s s * e_s / sum_s e_s (s = 1..L), reported for the future-computation and the output effects, and divided by L in the `*_ratio` keys.
* **Hypothesis**: Csordas et al. find that layers in the second half contribute little to later computation, so deeper models do not compose deeper computations; a low depth-score ratio means the model's effective depth is a small share of its nominal depth.
* **Citation/Paper**: `Csordas, R., Manning, C. D., & Potts, C. (2025). Do Language Models Use Their Depth Efficiently?` [NeurIPS 2025, ArXiv: 2505.13898]; code robertcsordas/llm_effective_depth.
* **File & Function**: `src/blme/tasks/causality/layer_skip.py` -> `LayerSkipEffectTask`
* **Critical Info**: Headline keys: `skip_depth_score_future_ratio`, `skip_depth_score_output_ratio`, `skip_output_effect_total`, `skip_output_effect_max`. `skip_output_effect_total` and `skip_output_effect_max` are declared invariant to identity blocks at interior, leading and trailing placements (`tests/tasks/invariance/test_causality_layer_skip.py::test_output_effects_under_identity_blocks`); the depth-score ratios are not, since an appended identity block lowers them by L0 / L. Costs 4 L forward passes per prompt (40 L at the default `num_samples: 10`). The release uses GSM8K problems with their solutions; BLME uses corpus passages up to `max_length: 512` tokens. The depth score follows the paper's 1-based layer index (the release's `effect_layer_score` is 0-based). The maxima over tokens and prompts grow with the prompt length and count, so compare models at the same `num_samples` and `max_length`.
