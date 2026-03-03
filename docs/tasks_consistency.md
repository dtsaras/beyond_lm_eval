# Consistency Tasks

This module evaluates the mathematical reliability and internal logical coherence of the model's likelihood representations, independent of ground-truth accuracy.

---

## 1. Paraphrase Consistency
* **What are we measuring**: Whether the model assigns symmetrically identical likelihoods to sentences that differ in syntax but share exact semantic meaning.
* **How are we measuring**: Computing the variance or difference in the output probabilities of a target concept when conditioned on syntactically varied but semantically equivalent prefixes. 
* **Hypothesis**: A robust latent space should map paraphrases to nearly identical geometric subspaces before tokenization forces them apart. Severe likelihood shifts indicate failure of abstraction.
* **Citation/Paper**: `Elazar, Y., Kassner, N., Ravfogel, S., Abnar, S., Hovy, E., Schütze, H., & Goldberg, Y. (2021). Measuring and improving consistency in pretrained language models.` Transactions of the Association for Computational Linguistics. [ArXiv: 2102.01017]
* **File & Function**: `src/blme/tasks/consistency/paraphrase.py` -> `ParaphraseInvarianceTask`
* **Critical Info**: Generally uses cosine similarity of the final representation layer or simply the KL-divergence of the output logits.

## 2. Likelihood Calibration
* **What are we measuring**: How accurately the model's self-reported mathematical probability (softmax output) correlates with ground-truth correctness.
* **How are we measuring**: Extracting the exact probability assigned to the correct answer versus incorrect answers, and computing Expected Calibration Error (ECE) or Brier Score over a dataset.
* **Hypothesis**: Over-parameterized LLMs often suffer from severe overconfidence, assigning 99% probability even when hallucinating. A well-calibrated geometric space prevents the probabilities from skewing uncontrollably.
* **Citation/Paper**: `Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks.` [ArXiv: 1706.04599]
* **File & Function**: `src/blme/tasks/consistency/calibration.py` -> `CalibrationTask`
* **Critical Info**: Base models are generally better calibrated than RLHF (Instruction-Tuned) models, which often suffer severe calibration collapse during safety training.

## 3. Logical Contradiction / Consistency
* **What are we measuring**: The model's adherence to basic boolean and formal logic equivalencies. (e.g., If A > B is highly probable, then B > A must be highly improbable).
* **How are we measuring**: Generating probability matrices for mutually exclusive statements and checking for constraint violations in the latent probability distribution.
* **Hypothesis**: True internal reasoning requires hard logical constraints within the representation embedding graph. Violation indicates the model is treating tokens as statistically independent sequences rather than connected concepts.
* **Citation/Paper**: Similar properties studied generally in neuro-symbolic and LLM consistency literature.
* **File & Function**: `src/blme/tasks/consistency/logical.py` -> `LogicalConsistencyTask`
* **Critical Info**: Does not evaluate truth, only internal agreement. A model can be logically consistent while being entirely factually wrong.

## 4. Contrastive Evaluation
* **What are we measuring**: The ability of the model's probability distribution to strongly discriminate between a definitively true statement and a subtly corrupted, contrastive counterpart.
* **How are we measuring**: Calculating the log-likelihood ratio (or difference) between a correct sample and a closely matched negative sample.
* **Hypothesis**: A continuous geometry should distinctly separate true from false in its highest-likelihood paths. Failing contrastive tests means false attractors are geographically too close to true attractors.
* **Citation/Paper**: Canonical evaluation methodology.
* **File & Function**: `src/blme/tasks/consistency/contrastive.py` -> `ContrastiveConsistencyTask`
* **Critical Info**: Highly reliant on the quality of the negative sample. If the negative sample is too easy, the test trivially passes.
