# Representation Engineering Tasks

This module contains metrics that analyze and manipulate high-level concepts using the top-down methodology of Representation Engineering (RepE).

**Current registry coverage (4 tasks)**: `repe_concept_separability`,
`repe_refusal_direction`, `repe_steering_effectiveness`, and
`repe_task_vectors`.

**Paper-faithful vs. BLME proxy notes**: The RepE tasks are paper-derived
diagnostics over hidden-state directions, but BLME reports measurement summaries
rather than reproducing each paper's full intervention/evaluation suite.

**Verification campaign (2026-07, Batch 2)**: every task below carries a
`Verification` field recording its evidence tier — official-code parity vs
`andyzoujm/representation-engineering` @ `5455d8a` and
`andyrdt/refusal_direction` @ `9d852fae`, plus independent numeric references.

---

## 1. Task / Reading Vector Geometry
* **What are we measuring**: The geometry of contrastive activation-space directions ("reading vectors") across layers.
* **How are we measuring**: Computing mean positive-minus-negative hidden-state directions at the last token for bundled or user-provided contrastive pairs, then reporting per-layer vector norms and positive/negative mean cosine similarity.
* **Hypothesis**: If a concept or behavior is linearly represented, contrastive hidden-state means should separate along a stable direction.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]. Related weight-space task-vector framing: `Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, & Farhadi (2023). Editing Models with Task Arithmetic.` [ICLR 2023, ArXiv: 2212.04089].
* **Paper summary**: Zou et al. propose Representation Engineering: instead of dissecting circuits bottom-up, read and control high-level concepts top-down from population-level hidden-state directions. Their LAT pipeline elicits a concept with contrastive stimuli and extracts a linear "reading vector" — §3.1.1 Step 3 lists the difference-of-cluster-means used here, taken at the last token of decoder models. They show honesty, emotion, and harmlessness concepts are linearly readable and controllable across open models. Ilharco et al. introduced the task-vector idea in *weight* space (fine-tuned minus pre-trained weights compose arithmetically); BLME measures the activation-space RepE analogue, not the weight-space one.
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `TaskVectorGeometryTask`
* **Critical Info**: BLME reports measurement summaries only; it does not prove that all prompt engineering reduces to a single static vector.
* **Verification** (2026-07, campaign B2): official-code parity — direction kernel == RepE `ClusterMeanRepReader.get_rep_directions` (difference of class means, no normalization) to 0.0 in float64 on the pinned fixture and ≤4.5e-7 rel end-to-end on distilgpt2 (float32 forward); dataset-fallback path pinned at ≤8.2e-6. Law-of-cosines analytic identity + determinism tests in `test_comprehensive_parity.py`.

## 2. Concept Separability (held-out linear probe)
* **What are we measuring**: How linearly separable high-level behavioral or cognitive concepts (e.g., truthfulness vs deception) are in the representation space.
* **How are we measuring**: A held-out logistic-regression separability probe: per layer, stratified k-fold CV of a logistic-regression classifier on mean-pooled hidden states, reporting cross-validated AUC and accuracy. This is a refined adaptation in the paper's linear-reading family (Zou et al. §5.1 also evaluate logistic-regression directions); it does **not** compute the PCA/LAT reading vector itself.
* **Hypothesis**: LLMs represent high-level concepts via simple linear directions rather than complex non-linear circuits; a prerequisite for top-down control is that a linear probe separates the concept.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]
* **Paper summary**: See §1. Relevant here: the paper's "reading" experiments show concepts become linearly decodable in mid-to-late layers, with detection accuracy from linear readers (PCA-based LAT and logistic-regression directions) far above chance; BLME's cross-validated probe quantifies exactly that linear accessibility per layer.
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `ConceptSeparabilityTask`
* **Critical Info**: Acts as the "Reading" phase of Representation Engineering, validating that the underlying structure supports Top-Down control interventions. Labels must be binary integers (0/1); NaN layers (fp16/bf16 deep-layer artifacts) are excluded via nan-aware summaries.
* **Verification** (2026-07, campaign B2): per-layer AUC/acc == independent `sklearn.cross_val_score` to 0.0 on a controlled model; planted 20σ concept → AUC exactly 1.0; verbatim official PCA-LAT kernel recovers the planted direction (|cos| = 0.9998) on the same activations, confirming the probe measures the same structure. B2 fixed: NaN-layer poisoning of `max_auc/mean_auc` (now nan-aware with all-NaN error) and the string-label crash (now a clean error dict).

## 3. Steering Effectiveness
* **What are we measuring**: Whether representation steering (injecting task vectors into the residual stream) meaningfully alters the model's output distribution.
* **How are we measuring**: Extracting task vectors from contrastive text pairs (text_pos/text_neg), then injecting them at the last token during forward passes on neutral prompts (up to 5 depth-sampled layers on deep models: 0, n/4, n/2, 3n/4, n−1). The output distribution shift is measured via KL(base‖steered) at the next-token distribution.
* **Hypothesis**: If representation engineering works, injecting a task vector at the right layer should cause a measurable shift in the output distribution. The steering success rate measures what fraction of *tested* layers produce a significant effect.
* **Citation/Paper**: `Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.` [ArXiv: 2310.01405]; `Turner, A., Thiergart, L., Leech, G., Udell, D., Vazquez, J., Mini, U., & MacDiarmid, M. (2023). Steering Language Models With Activation Engineering.` [ArXiv: 2308.10248]
* **Paper summary**: Turner et al.'s ActAdd steers a frozen model by adding the activation difference of a contrastive prompt pair into the residual stream at inference — no optimization, no weight change — and shifts topic/sentiment while preserving fluency. Zou et al.'s control experiments use the same linear-combination operator over reading vectors (their `linear_comb` control with `token_pos="end"` is the exact mechanism BLME hooks). The KL-divergence effectiveness/success-rate summary is BLME's own diagnostic, honestly labeled proxy-only.
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `SteeringEffectivenessTask`
* **Critical Info**: The `steering_alpha` parameter controls injection magnitude (default: 1.0). The `steering_threshold` parameter (default: 0.01 KL divergence) determines the minimum effect for a layer to count as "successful." `steering_success_rate` is the fraction of **tested** layers (≤5 on models deeper than 10 layers), not of all layers. Best steering layer identifies where intervention is most effective.
* **Verification** (2026-07, campaign B2): per-layer KL == hand-written fp64 `KL(base‖steered)` reference ≤1.7e-6 on distilgpt2 (direction of the KL verified — matches base‖steered, not the flip); analytic α=0 → KL≈0 (5e-11); hook mechanism == official RepE `linear_comb`/`token_pos="end"`. B2 fixed: the task now actually runs in the default pipeline (the old `dataset is None` fallback was unreachable — core always passes the WikiText corpus — so every study run errored and the feature matrix had zero steering columns) and no longer mutates the caller's shared dataset dicts.

## 4. Refusal Direction
* **What are we measuring**: Whether harmful and harmless prompts separate along a linear hidden-state direction.
* **How are we measuring**: Computing the difference-of-means direction between bundled harmful and harmless prompts, then reporting direction norm, held-out projection AUROC (stratified k-fold: direction fit on train folds, scored on test folds), projection gap, best layer, and AUC at normalized depths 0/25/50/75/100%.
* **Hypothesis**: Refusal behavior in aligned models is often mediated by a comparatively low-dimensional direction; stronger separation suggests a clearer refusal representation.
* **Citation/Paper**: `Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction.` [NeurIPS 2024, ArXiv: 2406.11717]. Related top-down framing: `Zou et al. (2023). Representation Engineering.`
* **Paper summary**: Arditi et al. show that across 13 open chat models, refusal is mediated by a *single* direction: the difference-of-means between mean activations on harmful vs harmless instructions (their §2.3: r = μ_harmful − ν_harmless). Ablating that one direction from the residual stream removes refusal (a white-box jailbreak), while adding it induces refusal on harmless prompts. BLME computes the same difference-of-means direction but substitutes held-out AUROC for the paper's ablation-based direction selection — it measures how linearly accessible the refusal concept is, not the causal effect of removing it.
* **File & Function**: `src/blme/tasks/representation_engineering.py` -> `RefusalDirectionTask`
* **Critical Info**: BLME measures separability and direction strength; it does not perform the paper's full refusal ablation/editing protocol. Output features are architecture-agnostic (depth-quantile AUCs rather than per-layer dicts, which made 99% of refusal columns NaN in cross-model aggregation).
* **Verification** (2026-07, campaign B2): official-code parity — direction kernel == `andyrdt/refusal_direction` `get_mean_diff` exactly (0.0, |cos| = 1 on the pinned fixture); the official forward-pre-hook mechanism run live vs BLME's layer directions agrees ≤5.9e-9 (float32); `direction_norm`/`separability_auc`/`mean_projection_gap` match independent recomputation exactly. B2 fixed the docstring, which promised a `per_layer` key the task deliberately does not emit.
