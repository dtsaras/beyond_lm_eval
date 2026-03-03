# Representation Engineering Tasks

This module contains metrics that analyze and manipulate high-level concepts using the top-down methodology of Representation Engineering (RepE).

---

## 1. Task Vector Creation and Application
* **What are we measuring**: The presence and effectiveness of linear "Task Vectors" that guide In-Context Learning (ICL).
* **How are we measuring**: Computing the average difference in hidden states between a prompt that includes in-context examples (the "learn" phase) and a zero-shot prompt. We then explicitly add this extracted "Task Vector" to a zero-shot prompt (the "apply" phase) to see if it replicates few-shot performance without the actual context.
* **Hypothesis**: In-Context Learning operates mechanistically by compressing the given training examples into a singular, linear task vector in the latent space.
* **Citation/Paper**: `Hendel, R., Geva, M., & Globerson, A. (2023). In-Context Learning Creates Task Vectors.` [ArXiv: 2310.15916]
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `TaskVectorGeometryTask`
* **Critical Info**: Validates that prompt engineering is fundamentally just shifting the hidden geometric space by a single static vector.

## 2. Concept Separability (Linear Artificial Tomography)
* **What are we measuring**: How linearly separable high-level behavioral or cognitive concepts (e.g., truthfulness vs deception) are in the representation space.
* **How are we measuring**: Using Linear Artificial Tomography (LAT) / Principal Component Analysis (PCA) on a dataset of contrasting prompt behaviors. We determine the principal "Reading Vector" and measure the accuracy of separating the two concepts.
* **Hypothesis**: LLMs understand high-level concepts via simple linear directions rather than complex non-linear circuits.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `ConceptSeparabilityTask`
* **Critical Info**: Acts as the "Reading" phase of Representation Engineering, validating that the underlying structure supports Top-Down control interventions.
