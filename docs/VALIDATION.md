# Validation status

Generated at BLME `19b78ca` from the study's verification trackers. Nothing here
is a sign-off; it states what has been checked and how.

**Columns**
- *Released code*: the reference implementation BLME is compared with (`none`: the
  paper released no code, so BLME's implementation of the paper's equations is the only one).
- *Parity* (assessed 2026-10-31, updated where noted): `kernel` = the estimator matches
  released code on shared inputs; `copy` = a
  transcribed copy of the released kernel is tested; `formula` = a closed form;
  `untested` = released code exists but no live comparison runs; `n/a` = no released code.
- *Headline value tests*: headline keys of the study's confirmatory set that have a test
  asserting their value against released code or an analytic anchor, out of all such keys.

## Paper-core tasks

| task | released code | parity | headline value tests |
|---|---|---|---|
| dynamics_coe | Chain-of-Embedding (local) | kernel | 10/10 |
| dynamics_fractal | google-research notebook (local) | untested | 9/9 |
| dynamics_generation_diversity | Texygen (local) | kernel | 11/11 |
| dynamics_gradient_flow | none | n/a | 1/1 |
| dynamics_interpolation | plat (slerp only, local) | n/a | 1/1 |
| dynamics_perturbation_growth | none | n/a | 1/1 |
| dynamics_sharpness | PyHessian (pip+local) | kernel | 7/7 |
| dynamics_stability | none | n/a | 2/2 |
| geometry_categories | none | n/a | 7/7 |
| geometry_cka | Kornblith notebook (local) | copy | 6/6 |
| geometry_cknna | platonic-rep (local) | copy | 9/9 |
| geometry_collapse | none | n/a | 4/4 |
| geometry_contextualization | kawine/contextual (needs allennlp) | untested† | 8/8 |
| geometry_correlation_dimension | none for the headline (nolds = legacy) | n/a | 1/1 |
| geometry_effective_depth | llm_effective_depth, ShortGPT (local) | untested | 3/3 |
| geometry_hsic | Kornblith notebook (local) | copy | 4/4 |
| geometry_hubness | scikit-hubness (local, not installed) | kernel (live scikit-hubness c36a058) | 12/12 |
| geometry_intrinsic_dim | IntrinsicDimDeep (local) | kernel | 4/4 |
| geometry_isoscore | p_eickhoff_isoscore (local; pip absent) | kernel (live IsoScore 2.0.1) | 2/2 |
| geometry_lid | lid_adversarial (local) | untested | 6/6 |
| geometry_lipschitz | lipEstimation (other estimand) | n/a | 4/4 |
| geometry_magnitude | magnipy (pip+local) | kernel | 2/2 |
| geometry_mahalanobis | deep_Mahalanobis (local) | untested | 7/7 |
| geometry_matrix_entropy | Diff-eRank (local) | copy | 2/2 |
| geometry_mp_bulk_deviation | none | n/a | 12/12 |
| geometry_neural_collapse | linguistic-collapse (kernel not local) | copy | 2/2 |
| geometry_perplexity | HF docs snippet | formula | not in the confirmatory set |
| geometry_phd_dimension | GPTID (local) | kernel | 2/2 |
| geometry_positional_decay | none | n/a | 1/1 |
| geometry_prediction_alignment | none | n/a | 2/2 |
| geometry_procrustes_linearity | LLM-Microscope (local) | copy | 17/17 |
| geometry_representation_sensitivity | none | n/a | 2/2 |
| geometry_rsa | rsatoolbox (pip) | untested | 2/2 |
| geometry_schatten | MatrixNuclearNorm (local) | kernel | 6/6 |
| geometry_spectral | WeightWatcher (pip+local) | kernel | 4/4 |
| geometry_superposition_strength | SuperpositionScaling (local) | untested | 6/6 |
| geometry_svd | kawine/contextual (local) | untested | 2/2 |
| geometry_tokenizer_efficiency | none | n/a | not in the confirmatory set |
| geometry_trajectory_curvature | sent_sampling (local) | copy | 3/3 |
| geometry_unembedding | none | n/a | 5/5 |
| geometry_vendi_score | Vendi-Score (pip) | kernel | 4/4 |
| geometry_weight_norms | WeightWatcher (pip+local) | untested | 4/4 |
| repe_concept_separability | representation-engineering (local) | kernel | not in the confirmatory set |
| repe_refusal_direction | refusal_direction (local) | kernel | not in the confirmatory set |
| repe_steering_effectiveness | none | n/a | not in the confirmatory set |
| repe_task_vectors | representation-engineering (direction only) | copy | not in the confirmatory set |
| topology_betti_curve | topnn (Julia, not runnable) | untested† | 1/1 |
| topology_homology | none (ripser is the engine) | n/a | 15/15 |
| topology_persistence_entropy | giotto-tda (not importable) | untested† | 6/6 |
| topology_persistence_landscape | persim, gudhi (pip) | kernel | 15/15 |
| topology_zigzag_persistence | ZigZagLLMs (local, not importable) | copy | 2/2 |

## Methods added in 2026-11

| task | released code | parity |
|---|---|---|
| geometry_alpha_req | Agrawal 2022 / Li et al. 2025 | reptrix@dbf53c5: spectrum, fit, RankMe ≤1e-9, via evaluate() |
| geometry_spectral_signature | Spectral Signatures 2607.03377 / AlphaPruning | Spectral_Signature@d04eacd: names + alphas 1e-12 |
| geometry_gride | Cheng et al. 2025 / Denti 2022 | DADApy@d729b88 (side-installed) 1e-6, via evaluate() |
| geometry_information_imbalance | Glielmo 2022 / Cheng et al. 2025 | DADApy exact, via evaluate() |
| geometry_farms_alpha | Hu et al. ICML 2025 | FARMS@e0a2ded: window eigenvalues, alpha ≤6.3e-7 and the same xmin index; whole tiny models 1e-5; real Qwen3-8B q_proj (450k eigenvalues) same xmin |
| causality_layer_skip | Csordás, Manning & Potts, NeurIPS 2025 | llm_effective_depth@393e5c7: the release's own merge_io / get_future / effect_layer_score, test_effect's loop transcribed onto HF hooks (the nnsight script cannot run on transformers 5); matrices 1e-6, via evaluate() |
| consistency_icl_score | Olsson et al. 2022 | formula only (no code): brute-force cross-entropy exact |
| geometry_representation_dispersion | Li et al., ICLR 2026 | rep_dispersion@dae432d (side-installed) run end to end: exact equality |
| geometry_predictive_information | Chang et al., COLM 2026 (InfoRidge) | InfoRidge@2cde643 utils / analyze_batch, via evaluate() 1e-3 |
| geometry_neighborhood_overlap | Doimo 2020 / Valeriani 2023 | DADApy exact, via evaluate() |
| geometry_ww_randomized | Prakash & Martin 2025; SETOL; Staats et al. | WeightWatcher@a5940f7 (clone): every function and per-matrix column exact |
| interpretability_logit_lens_kl | Hu et al. 2025; Entropy-Lens | model-human-processing@69e405c lens and summaries; KL ≤3e-8 via evaluate() |
| geometry_rogue_dimensions | Timkey & van Schijndel, EMNLP 2021 | rogue-dimensions@84f4009 notebook cos_contrib over every pair 1e-12; exact all-pairs CC replaces the 500k-pair Monte Carlo |
| geometry_lm_head_free_energy | Wilson, NeurIPS 2026 | heavytail@80f7f80 bfe_w / compute_free_energy 1e-12 on the release's float32 spectrum |
| dynamics_entropy_calibration | Cao et al., NeurIPS 2025 | entropy-calibration@d3f62e4 compute_logprobs: entropies 1e-4; log losses equal after fixing the release's one-position slice |
| geometry_dime | Skean et al., ICML 2025 | information_flow@dde387a + representation-itl@e5d8a12: entropies, doe, compute_dime ≤2e-15, augmentations identical |
| geometry_zipfian_symmetry | Yokoi et al., NeurIPS 2024 | zipfian-whitening@fa44d29 scores ≤8e-9 (the release's log d is float32) |
| geometry_delta_hyperbolicity | Yang et al., NeurIPS 2025 (HypLoRA); Khrulkov 2020 | HypLoRA@c888e16 delta_hyp / get_delta exact |
| interpretability_inactive_heads | Sandoval-Segura et al., ICLR 2026 | inactive-heads@9dd884c HONOR scores ≤3.6e-7, masks identical, via evaluate() |

Interpretability, causality and consistency tasks outside these tables are
experimental: they run and are tested for crashes and leaks, but have not been
validated against their papers.
