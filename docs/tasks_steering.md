# Steering Tasks

This module contains tasks aimed at characterizing structural intervention techniques to directly override the internal behavior of the language model geometry.

---

## 1. Steerability (Concept Vectors)
* **What are we measuring**: The capability of the model to fundamentally alter its generation style, behavior, or semantic domain when a specific static "Concept Vector" is injected into the mid-layer residual stream.
* **How are we measuring**: We test both typical baseline generation and conceptually-steered generation (by adding $+ c\vec{v}$ to activations). We then evaluate the resulting KL-divergence and logical shifts in the token probability distributions.
* **Hypothesis**: LLMs cluster high-level behaviors along linear subspace directions. Pushing the hidden states precisely along this axis "steers" the model without necessitating gradient-based finetuning.
* **Citation/Paper**: Popularized widely by Anthropic and related alignment researchers. Specifically evaluated in Representation Engineering frameworks. `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]
* **File & Function**: `src/blme/tasks/steering/concept_vectors.py` -> `ConceptSteerabilityTask`
* **Critical Info**: Steering relies completely on the linearity of the underlying features in the latent space. Non-linear features generally cannot be effectively steered using constant vector addition.

## 2. In-Weight Editing Efficacy 
* **What are we measuring**: (Legacy) Standard memory editing evaluation.
* **How are we measuring**: Routing backward-compatible metric parameters designed to evaluate explicit weight modification via ROME/MEMIT.
* **Hypothesis**: Editing factual associations requires altering the parametric memory matrix directly. Evaluated mainly for legacy compatibility.
* **Citation/Paper**: Legacy adaptation derived from standard ROME interventions [No specific conference paper].
* **File & Function**: `src/blme/tasks/steering/editing.py` -> `EditingTask`
* **Critical Info**: Mostly deprecated in favor of rigorous Causal and RepE task workflows.
