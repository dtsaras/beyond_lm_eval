# Representation Engineering Tasks

This module contains metrics that analyze and manipulate high-level concepts using the top-down methodology of Representation Engineering (RepE).

**Current registry coverage (4 tasks)**: `repe_concept_separability`,
`repe_refusal_direction`, `repe_steering_effectiveness`, and
`repe_task_vectors`.

**Paper-faithful vs. BLME proxy notes**: The RepE tasks are paper-derived
diagnostics over hidden-state directions, but BLME reports measurement summaries
rather than reproducing each paper's full intervention/evaluation suite.

---

## 1. Task / Reading Vector Geometry
* **What are we measuring**: The geometry of contrastive activation-space directions ("reading vectors") across layers.
* **How are we measuring**: Computing mean positive-minus-negative hidden-state directions for bundled or user-provided contrastive pairs, then reporting per-layer vector norms and positive/negative mean cosine similarity.
* **Hypothesis**: If a concept or behavior is linearly represented, contrastive hidden-state means should separate along a stable direction.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]. Related weight-space task-vector framing: `Ilharco et al. (2023). Editing Models with Task Arithmetic.` [ICLR 2023, ArXiv: 2212.04089].
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `TaskVectorGeometryTask`
* **Critical Info**: BLME reports measurement summaries only; it does not prove that all prompt engineering reduces to a single static vector.

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

## 4. Refusal Direction
* **What are we measuring**: Whether harmful and harmless prompts separate along a linear hidden-state direction.
* **How are we measuring**: Computing the difference-of-means direction between bundled harmful and harmless prompts, then reporting direction norm, projection gap, and separability AUROC across layers.
* **Hypothesis**: Refusal behavior in aligned models is often mediated by a comparatively low-dimensional direction; stronger separation suggests a clearer refusal representation.
* **Citation/Paper**: `Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction.` [ArXiv: 2406.11717]. Related top-down framing: `Zou et al. (2023). Representation Engineering.`
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `RefusalDirectionTask`
* **Critical Info**: BLME measures separability and direction strength; it does not perform the paper's full refusal ablation/editing protocol.
