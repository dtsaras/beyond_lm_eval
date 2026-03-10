# Representation Engineering Tasks

This module contains metrics that analyze and manipulate high-level concepts using the top-down methodology of Representation Engineering (RepE).

---

## 1. Task Vector Creation and Application
* **What are we measuring**: The presence and effectiveness of linear "Task Vectors" that guide In-Context Learning (ICL).
* **How are we measuring**: Computing the average difference in hidden states between a prompt that includes in-context examples (the "learn" phase) and a zero-shot prompt. We then explicitly add this extracted "Task Vector" to a zero-shot prompt (the "apply" phase) to see if it replicates few-shot performance without the actual context.
* **Hypothesis**: In-Context Learning operates mechanistically by compressing the given training examples into a singular, linear task vector in the latent space.
* **Citation/Paper**: `Hendel, R., Geva, M., & Globerson, A. (2023). In-Context Learning Creates Task Vectors.` [EMNLP 2023 Findings, ArXiv: 2310.15916]
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `TaskVectorGeometryTask`
* **Critical Info**: Validates that prompt engineering is fundamentally just shifting the hidden geometric space by a single static vector.

## 2. Concept Separability (Linear Artificial Tomography)
* **What are we measuring**: How linearly separable high-level behavioral or cognitive concepts (e.g., truthfulness vs deception) are in the representation space.
* **How are we measuring**: Using Linear Artificial Tomography (LAT) / Principal Component Analysis (PCA) on a dataset of contrasting prompt behaviors. We determine the principal "Reading Vector" and measure the accuracy of separating the two concepts.
* **Hypothesis**: LLMs understand high-level concepts via simple linear directions rather than complex non-linear circuits.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `ConceptSeparabilityTask`
* **Critical Info**: Acts as the "Reading" phase of Representation Engineering, validating that the underlying structure supports Top-Down control interventions.

## 3. Steering Effectiveness
* **What are we measuring**: Whether representation steering (injecting task vectors into the residual stream) meaningfully alters the model's output distribution.
* **How are we measuring**: Extracting task vectors from contrastive text pairs (text_pos/text_neg), then injecting them at each layer during forward passes on neutral prompts. The output distribution shift is measured via KL divergence between the steered and unsteered outputs.
* **Hypothesis**: If representation engineering works, injecting a task vector at the right layer should cause a measurable shift in the output distribution. The steering success rate measures what fraction of layers produce a significant effect.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `SteeringEffectivenessTask`
* **Critical Info**: The `steering_alpha` parameter controls injection magnitude (default: 1.0). The `steering_threshold` parameter (default: 0.01 KL divergence) determines the minimum effect for a layer to count as "successful." Best steering layer identifies where intervention is most effective.
