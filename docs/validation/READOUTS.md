# Library headline and study readout evidence

## Frozen confirmatory readouts

This table follows the exact task and key names in the study's frozen analysis manifest. Each entry gives the reviewed comparison and its linked CPU test cases. A kernel check covers a component. A model or corpus change can affect the end-to-end measurement. [All registered tasks](../VALIDATION.md) · [Exact selectors](metrics.json)

| Task and readout | Comparison | 5.2.0 | 5.10.2 | 5.22.2 | 5.23.0 | Tested scope |
|---|---|---:|---:|---:|---:|---|
| `dynamics_coe.coe_c` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.coe_c_std` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.coe_r` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.coe_r_std` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.coe_r_sum` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.mean_angle_change` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.mean_normalized_angle` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_coe.std_angle_change` [1](../../tests/tasks/parity/test_coe_release_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs released score.py CoEScoreInfo |
| `dynamics_fractal.hurst_exponent` [1](../../tests/tasks/parity/test_fractal_pooled_parity.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | _pooled_fractal_params vs released notebook functions (ast), offsets scripted |
| `dynamics_fractal.hurst_exponent_perdoc` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel: white noise H = 0.5 +- 0.07; nolds.hurst_rs +- 0.1 |
| `dynamics_fractal.hurst_exponent_std` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel: white noise H = 0.5 +- 0.07; nolds.hurst_rs +- 0.1 |
| `dynamics_fractal.joseph_exponent` [1](../../tests/tasks/parity/test_fractal_pooled_parity.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | _pooled_fractal_params vs released notebook functions (ast), offsets scripted |
| `dynamics_fractal.joseph_exponent_perdoc` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel: white noise J = 0.5 +- 0.05 |
| `dynamics_fractal.joseph_exponent_std` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel: white noise J = 0.5 +- 0.05 |
| `dynamics_fractal.self_similarity_exponent` [1](../../tests/tasks/parity/test_fractal_pooled_parity.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | _pooled_fractal_params vs released notebook functions (ast), offsets scripted |
| `dynamics_fractal.self_similarity_exponent_perdoc` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel: white noise S in (0.3, 0.7) (weak band) |
| `dynamics_fractal.self_similarity_exponent_std` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel: white noise S in (0.3, 0.7) (weak band) |
| `dynamics_generation_diversity.distinct_1` [1](../../tests/tasks/parity/test_coverage_gap_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel _distinct_n_corpus, n = 1 only (0.1 exactly) |
| `dynamics_generation_diversity.distinct_2` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | scripted generate(), entropies exactly log k |
| `dynamics_generation_diversity.distinct_3` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | scripted generate(), entropies exactly log k |
| `dynamics_generation_diversity.mean_distinct_1` [1](../../tests/tasks/parity/test_coverage_gap_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel _distinct_n vs Li et al. definition, n = 1, 2, 3 |
| `dynamics_generation_diversity.mean_distinct_2` [1](../../tests/tasks/parity/test_coverage_gap_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel _distinct_n vs Li et al. definition, n = 1, 2, 3 |
| `dynamics_generation_diversity.mean_distinct_3` [1](../../tests/tasks/parity/test_coverage_gap_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel _distinct_n vs Li et al. definition, n = 1, 2, 3 |
| `dynamics_generation_diversity.mean_phrase_repetition_rate` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | scripted generate(), entropies exactly log k |
| `dynamics_generation_diversity.mean_sampling_entropy_first_quarter` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 6/6 | 6/6 | 6/6 | 6/6 | Scripted varying support has entropy log(k) per step, with per-completion quarter means compared in float64 at 1e-12 nats. A second control mixes a point mass and a masked, nonuniform distribution in four score dtypes. Interior padding and EOS exercise alignment with original score positions. |
| `dynamics_generation_diversity.mean_sampling_entropy_last_quarter` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 6/6 | 6/6 | 6/6 | 6/6 | Scripted varying support has entropy log(k) per step, with per-completion quarter means compared in float64 at 1e-12 nats. A second control mixes a point mass and a masked, nonuniform distribution in four score dtypes. Interior padding and EOS exercise alignment with original score positions. |
| `dynamics_generation_diversity.mean_self_bleu` [1](../../tests/tasks/invariance/test_dynamics_generation_diversity.py) [2](../../tests/tasks/test_reference_parity_formulas.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs NLTK sentence_bleu method1, Texygen's calc_bleu |
| `dynamics_generation_diversity.sampling_entropy_collapse_delta` [1](../../tests/tasks/parity/test_generation_diversity_values.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | Scripted varying support has entropy log(k) per step, with per-completion quarter means compared in float64 at 1e-12 nats. Interior padding and EOS exercise alignment with original score positions. |
| `dynamics_gradient_flow.gradient_flow_log_ratio_in_out` [1](../../tests/tasks/parity/test_gradient_flow_log_ratio_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs torch.autograd.grad endpoints |
| `dynamics_interpolation.convexity_gap` [1](../../tests/tasks/parity/test_interpolation_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Gap recomputed at the true midpoint over every distinct pair. |
| `dynamics_perturbation_growth.growth_log_ratio_relative` [1](../../tests/tasks/test_perturbation_growth.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | a stack rescaling by a per layer gives absolute L ln a and relative 0 exactly |
| `dynamics_sharpness.asam_sharpness` [1](../../tests/tasks/parity/test_sharpness_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel on a quadratic |
| `dynamics_sharpness.critical_sharpness` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 3/3 | 3/3 | 3/3 | 3/3 | kernel: eta* = 2/lambda exactly |
| `dynamics_sharpness.critical_step_size` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 3/3 | 3/3 | 3/3 | 3/3 | kernel: eta* = 2/lambda exactly |
| `dynamics_sharpness.hutchinson_trace_estimate` [1](../../tests/tasks/parity/test_sharpness_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel vs PyHessian trace, same probes, rtol 1e-12 |
| `dynamics_sharpness.hutchinson_trace_per_param` [1](../../tests/tasks/parity/test_spectral_sharpness_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Tests the division by the parameter count in the selected scope. The trace estimator has a separate PyHessian comparison. |
| `dynamics_sharpness.sam_sharpness` [1](../../tests/tasks/parity/test_sharpness_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel on a quadratic |
| `dynamics_sharpness.top_eigenvalue_estimate` [1](../../tests/tasks/parity/test_sharpness_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs PyHessian eigenvalues and exact lambda_max |
| `dynamics_stability.stability_mean` [1](../../tests/tasks/test_comprehensive_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs sklearn NearestNeighbors(cosine) + set Jaccard |
| `geometry_cka.early_late_cka` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) [2](../../tests/test_gpu_linalg.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased); on the GPU path the cached Gram norms share the float64 numerator (CKA(X, X) = 1 to 1e-12) |
| `geometry_cka.early_late_cka_debiased` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased) |
| `geometry_cka.min_adjacent_cka` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) [2](../../tests/test_gpu_linalg.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased); on the GPU path the cached Gram norms share the float64 numerator (CKA(X, X) = 1 to 1e-12) |
| `geometry_cka.min_adjacent_cka_debiased` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased) |
| `geometry_cka.min_offdiag_cka` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) [2](../../tests/test_gpu_linalg.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased); on the GPU path the cached Gram norms share the float64 numerator (CKA(X, X) = 1 to 1e-12) |
| `geometry_cka.min_offdiag_cka_debiased` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | evaluate() vs Kornblith Demo.ipynb feature_space_linear_cka (biased + debiased) |
| `geometry_cknna.avg_adjacent_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.early_late_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.first_middle_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.max_adjacent_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.mean_offdiag_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.min_adjacent_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.min_offdiag_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.std_adjacent_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_cknna.std_offdiag_cknna` [1](../../tests/tasks/parity/test_cknna_parity.py) [2](../../tests/tasks/invariance/test_cknna.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | kernel _cknna vs official platonic-rep metrics.cknna; preprocessing bit-exact |
| `geometry_collapse.collapse_ratio` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs independent scipy svdvals Roy-Vetterli erank |
| `geometry_collapse.max_erank` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs independent scipy svdvals Roy-Vetterli erank |
| `geometry_collapse.min_erank` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs independent scipy svdvals Roy-Vetterli erank |
| `geometry_collapse.qdl_entropy_min` [1](../../tests/tasks/invariance/test_collapse.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on injected states vs Eq. 1 written from the paper |
| `geometry_contextualization.mean_anisotropy_baseline` [1](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | weak: collinear ray -> 1.0 |
| `geometry_contextualization.mean_intra_sentence_similarity_corrected` [1](../../tests/tasks/parity/test_coverage_gap_parity.py) [2](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel |
| `geometry_contextualization.mean_mev` [1](../../tests/tasks/test_reference_parity_formulas.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel vs sklearn PCA ratio (kawine variance_explained_by_pc) |
| `geometry_contextualization.mean_mev_baseline` [1](../../tests/tasks/parity/test_contextualization_mev_baselines.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernels vs sklearn TruncatedSVD/PCA (kawine calls); composition via evaluate() |
| `geometry_contextualization.mean_mev_baseline_pool1k` [1](../../tests/tasks/parity/test_contextualization_mev_baselines.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernels vs sklearn TruncatedSVD/PCA (kawine calls); composition via evaluate() |
| `geometry_contextualization.mean_mev_corrected_pool1k` [1](../../tests/tasks/parity/test_contextualization_mev_baselines.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernels vs sklearn TruncatedSVD/PCA (kawine calls); composition via evaluate() |
| `geometry_contextualization.mean_mev_corrected_sizematched` [1](../../tests/tasks/parity/test_contextualization_mev_baselines.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernels vs sklearn TruncatedSVD/PCA (kawine calls); composition via evaluate() |
| `geometry_contextualization.mean_self_similarity_corrected` [1](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernels: pairwise mean (all-pairs regime) and baseline (collinear ray = 1) |
| `geometry_correlation_dimension.correlation_dimension` [1](../../tests/tasks/test_correlation_dimension_protocol.py) [2](../../tests/tasks/test_correlation_dimension_atom_e2e.py) | analytic | 12/12 | 12/12 | 12/12 | 12/12 | kernel: Du & Tanaka-Ishii window, independent polyfit; D2 = 0 for an atomic measure via evaluate() |
| `geometry_effective_depth.effective_depth_90` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | kernel: exactly 1/L |
| `geometry_effective_depth.max_block_influence` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel: per-layer BI = 2 (sign flip), 0 (identity); sum / max are plain reductions |
| `geometry_effective_depth.total_block_influence` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel: per-layer BI = 2 (sign flip), 0 (identity); sum / max are plain reductions |
| `geometry_hubness.hubness_k100_gini` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k100_max` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k100_skew` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k100_top1pct` [1](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | weak: [0,1,1,2] -> 0.5 |
| `geometry_hubness.hubness_k10_gini` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k10_max` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k10_skew` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k10_top1pct` [1](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | weak: [0,1,1,2] -> 0.5 |
| `geometry_hubness.hubness_k50_gini` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k50_max` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k50_skew` [1](../../tests/tasks/parity/test_hubness_parity.py) | reference | 7/7 | 7/7 | 7/7 | 7/7 | evaluate() at k = 5 (k-generic loop) vs float64 kNN + scipy skew (scikit-hubness k_skewness) |
| `geometry_hubness.hubness_k50_top1pct` [1](../../tests/tasks/test_reference_parity_formulas.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | weak: [0,1,1,2] -> 0.5 |
| `geometry_intrinsic_dim.lid_first_layer` [1](../../tests/tasks/test_reference_parity_formulas.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel vs skdim.id.TwoNN (abs 0.05) and ID = 3 recovery (abs 0.3) |
| `geometry_intrinsic_dim.lid_last_layer` [1](../../tests/tasks/test_reference_parity_formulas.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel vs skdim.id.TwoNN (abs 0.05) and ID = 3 recovery (abs 0.3) |
| `geometry_intrinsic_dim.lid_mid_layer` [1](../../tests/tasks/test_reference_parity_formulas.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel vs skdim.id.TwoNN (abs 0.05) and ID = 3 recovery (abs 0.3) |
| `geometry_isoscore.isoscore` [1](../../tests/tasks/parity/test_isoscore_release_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | evaluate() vs released IsoScore.py from the clone |
| `geometry_isoscore.isoscore_effective_dims` [1](../../tests/tasks/parity/test_isoscore_release_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | evaluate() vs released IsoScore.py from the clone |
| `geometry_lid.lid_first_layer` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lid.lid_max` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lid.lid_mean` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lid.lid_median` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lid.lid_mid_layer` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lid.lid_std` [1](../../tests/tasks/parity/test_lid_mle_batch_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | layerwise evaluate() vs released mle_batch (ast) |
| `geometry_lipschitz.e2e_log_norm_growth` [1](../../tests/tasks/parity/test_lipschitz_depth_additive_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() closed forms on scale-rotation stacks |
| `geometry_lipschitz.embedding_relative_change` [1](../../tests/tasks/parity/test_lipschitz_depth_additive_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() closed forms on scale-rotation stacks |
| `geometry_lipschitz.relative_change_max` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate(): analytic 1.9 stack + independent numpy recompute |
| `geometry_lipschitz.total_angular_path` [1](../../tests/tasks/parity/test_lipschitz_depth_additive_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() closed forms on scale-rotation stacks |
| `geometry_magnitude.magnitude_area_tconv` [1](../../tests/tasks/parity/test_magnitude_parity.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | kernel vs live magnipy rel 1e-10 |
| `geometry_magnitude.magnitude_dimension_tconv` [1](../../tests/tasks/parity/test_magnitude_parity.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | kernel vs live magnipy rel 1e-10 |
| `geometry_mahalanobis.auroc` [1](../../tests/tasks/parity/test_mahalanobis_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | mean per-split sklearn roc_auc_score of LedoitWolf-refit distances, rel 1e-12 |
| `geometry_mahalanobis.mahalanobis_sq_per_dim_id` [1](../../tests/tasks/parity/test_mahalanobis_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() splits refit with sklearn LedoitWolf |
| `geometry_mahalanobis.mahalanobis_sq_per_dim_ood` [1](../../tests/tasks/parity/test_mahalanobis_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() splits refit with sklearn LedoitWolf |
| `geometry_mahalanobis.ood_separation_gap_per_dim` [1](../../tests/tasks/parity/test_mahalanobis_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() splits refit with sklearn LedoitWolf |
| `geometry_mahalanobis.ood_separation_ratio` [1](../../tests/tasks/parity/test_mahalanobis_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() splits refit with sklearn LedoitWolf |
| `geometry_matrix_entropy.matrix_entropy` [1](../../tests/tasks/parity/test_matrix_entropy_parity.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | kernel vs the official Wei et al. code |
| `geometry_matrix_entropy.matrix_entropy_raw` [1](../../tests/tasks/parity/test_matrix_entropy_parity.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | kernel vs the official Wei et al. code |
| `geometry_mp_bulk_deviation.mp_ks_distance_g25_first` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_ks_distance_g25_last` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_ks_distance_g25_mean` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_ks_distance_g25_mid` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_outlier_frac_g25_first` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_outlier_frac_g25_last` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_outlier_frac_g25_mean` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_outlier_frac_g25_mid` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_spike_energy_g25_first` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_spike_energy_g25_last` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_spike_energy_g25_mean` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_mp_bulk_deviation.mp_spike_energy_g25_mid` [1](../../tests/tasks/parity/test_mp_bulk_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on exact equicorrelated clouds; scipy kstest vs integrated MP density |
| `geometry_neural_collapse.nc1_cdnv_galanti` [1](../../tests/tasks/invariance/test_neural_collapse.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs independent recompute of linguistic-collapse kernels.py |
| `geometry_phd_dimension.phd_dimension` [1](../../tests/tasks/parity/test_phd_dimension_parity.py) [2](../../tests/tasks/test_phd_dimension_protocol.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel bit-exact vs GPTID; evaluate() mean == kernel on the notebook grid |
| `geometry_phd_dimension.phd_dimension_std` [1](../../tests/tasks/parity/test_phd_dimension_parity.py) [2](../../tests/tasks/test_phd_dimension_protocol.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel bit-exact vs GPTID; evaluate() mean == kernel on the notebook grid |
| `geometry_positional_decay.positional_decay_nosink_write_weighted` [1](../../tests/tasks/parity/test_positional_decay_write_weights.py) | analytic | 3/3 | 3/3 | 3/3 | 3/3 | Separate component tests for per-head write norms and their weighted mean. |
| `geometry_prediction_alignment.prediction_alignment_mean` [1](../../tests/tasks/parity/test_prediction_alignment_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Readout checked against supplied next-token logits and closed-form margins. |
| `geometry_prediction_alignment.prediction_alignment_std` [1](../../tests/tasks/parity/test_prediction_alignment_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Readout checked against supplied next-token logits and closed-form margins. |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_first` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_last` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_max` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_mean` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_min` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_q25` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_q50` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_adjusted_q75` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_first` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_last` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_max` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_mean` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_min` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_q25` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_q50` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_q75` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_procrustes_linearity.procrustes_linearity_slope` [1](../../tests/tasks/parity/test_procrustes_linearity_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs LLM-Microscope procrustes_similarity (ast) + Wherry |
| `geometry_representation_sensitivity.S_unit` [1](../../tests/tasks/parity/test_representation_sensitivity_s_unit_values.py) | analytic | 3/3 | 3/3 | 3/3 | 3/3 | Autograd through affine LayerNorm and RMSNorm heads and the saturated logits of an actual Gemma-2 causal LM. |
| `geometry_rsa.rsa_adjacent_min_corrdist` [1](../../tests/tasks/parity/test_rsa_corrdist_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs rsatoolbox correlation RDM + spearman |
| `geometry_rsa.rsa_early_late_corrdist` [1](../../tests/tasks/parity/test_rsa_corrdist_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs rsatoolbox correlation RDM + spearman |
| `geometry_schatten.rankme_last` [1](../../tests/tasks/test_schatten_task.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel |
| `geometry_schatten.row_normalized_matrix_nuclear_norm_last` [1](../../tests/test_reference_parity_guard.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | kernel bit-exact vs MatrixNuclearNorm MNN.py |
| `geometry_schatten.row_normalized_rankme_last` [1](../../tests/tasks/test_schatten_task.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel |
| `geometry_schatten.row_normalized_schatten_inf_last` [1](../../tests/tasks/test_schatten_task.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | kernel |
| `geometry_spectral.avg_alpha_hat` [1](../../tests/tasks/parity/test_spectral_sharpness_values.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the selected-matrix mean with WeightWatcher's alpha_weighted column. |
| `geometry_spectral.avg_alpha_official` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) [2](../../tests/test_reference_parity_guard.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel _pl_mle_alpha_ks vs live weightwatcher |
| `geometry_spectral.avg_stable_rank` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate(): diag(10..1) -> 3.85 |
| `geometry_spectral.median_alpha_official` [1](../../tests/tasks/parity/test_b8_new_metrics_parity.py) [2](../../tests/test_reference_parity_guard.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel _pl_mle_alpha_ks vs live weightwatcher |
| `geometry_superposition_strength.frac_norm_above_half_max` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_superposition_strength.mean_squared_overlap` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_superposition_strength.mean_squared_overlap_centered` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_superposition_strength.row_norm_cv` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_superposition_strength.superposition_ratio` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_superposition_strength.superposition_ratio_centered` [1](../../tests/tasks/parity/test_superposition_strength_values.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs overlap-0.py loop (verbatim) + Welch bound |
| `geometry_svd.avg_cosine_similarity` [1](../../tests/tasks/parity/test_svd_isotropy_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs scipy svdvals R&V and brute-force pair cosines |
| `geometry_svd.effective_rank` [1](../../tests/tasks/parity/test_svd_isotropy_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs scipy svdvals R&V and brute-force pair cosines |
| `geometry_svd.participation_ratio` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Checks the SVD kernel using (sum of squared singular values)^2 divided by sum of fourth powers. |
| `geometry_trajectory_curvature.curvature_mean_first_layer` [1](../../tests/tasks/parity/test_trajectory_curvature_parity.py) [2](../../tests/tasks/test_trajectory_curvature_protocol.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | kernel vs official curvature_utils; evaluate() == kernel |
| `geometry_trajectory_curvature.curvature_mean_last_layer` [1](../../tests/tasks/parity/test_trajectory_curvature_parity.py) [2](../../tests/tasks/test_trajectory_curvature_protocol.py) | reference | 3/3 | 3/3 | 3/3 | 3/3 | kernel vs official curvature_utils; evaluate() == kernel |
| `geometry_trajectory_curvature.straightening_ratio` [1](../../tests/tasks/parity/test_trajectory_curvature_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() on constant-turning paths |
| `geometry_unembedding.unembedding_eff_rank` [1](../../tests/tasks/parity/test_unembedding_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs scipy svdvals R&V (raw != centred) and F.layer_norm cosines |
| `geometry_unembedding.unembedding_eff_rank_centered` [1](../../tests/tasks/parity/test_unembedding_values.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs scipy svdvals R&V (raw != centred) and F.layer_norm cosines |
| `geometry_vendi_score.vendi_max` [1](../../tests/tasks/parity/test_vendi_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs official vendi score_K |
| `geometry_vendi_score.vendi_mean_first_layer` [1](../../tests/tasks/parity/test_vendi_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs official vendi score_K |
| `geometry_vendi_score.vendi_mean_last_layer` [1](../../tests/tasks/parity/test_vendi_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs official vendi score_K |
| `geometry_vendi_score.vendi_mean_mid_layer` [1](../../tests/tasks/parity/test_vendi_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | kernel vs official vendi score_K |
| `geometry_weight_norms.mean_frobenius_norm` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs textbook formulas |
| `geometry_weight_norms.mean_spectral_norm` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs textbook formulas |
| `geometry_weight_norms.mean_stable_rank` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs textbook formulas |
| `geometry_weight_norms.norm_uniformity` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs textbook formulas |
| `topology_betti_curve.betti_1_curve_mean` [1](../../tests/test_replacement_candidates.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | the mean of the emitted beta_1 curve, checked against it through evaluate() |
| `topology_homology.first_layer_max_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.first_layer_mean_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.first_layer_mean_persistence_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.first_layer_num_loops_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.last_layer_max_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.last_layer_mean_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.last_layer_mean_persistence_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.last_layer_num_loops_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.mid_layer_max_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.mid_layer_mean_persistence_h0_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.mid_layer_mean_persistence_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_homology.mid_layer_num_loops_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.first_layer_pe_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.first_layer_pe_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.last_layer_pe_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.last_layer_pe_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.mid_layer_pe_h1` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_entropy.mid_layer_pe_h1_norm` [1](../../tests/tasks/parity/test_homology_entropy_gudhi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | evaluate() vs GUDHI Rips intervals |
| `topology_persistence_landscape.first_layer_h1_landscape_max_1_mpd` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.first_layer_h1_landscape_norm_1_mpd15` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.first_layer_h1_mean_landscape_integral_mpd2` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.last_layer_h1_landscape_max_1_mpd` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.last_layer_h1_landscape_norm_1_mpd15` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.last_layer_h1_mean_landscape_integral_mpd2` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.mid_layer_h1_landscape_max_1_mpd` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.mid_layer_h1_landscape_norm_1_mpd15` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_persistence_landscape.mid_layer_h1_mean_landscape_integral_mpd2` [1](../../tests/tasks/parity/test_persistence_landscape_values.py) | reference | 5/5 | 5/5 | 5/5 | 5/5 | kernel vs persim exact landscapes; evaluate() closed forms on square clouds |
| `topology_zigzag_persistence.zz_true_adjacent_mean` [1](../../tests/tasks/parity/test_zigzag_scalars_release.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | _bar_scalars vs ZigZagLLMs post_process.py POSTPROCESS; bars pinned by test_zigzag_reference_construction |
| `topology_zigzag_persistence.zz_true_mean_bar_len` [1](../../tests/tasks/parity/test_zigzag_scalars_release.py) | reference | 4/4 | 4/4 | 4/4 | 4/4 | _bar_scalars vs ZigZagLLMs post_process.py POSTPROCESS; bars pinned by test_zigzag_reference_construction |

## Other library headlines

Each declared key or template outside the frozen study set appears here. A task-level entry still has a scientific review in the main dashboard. Its exact headline lacks a separate comparison in this ledger. Component and aggregation checks identify the part exercised.

| Task and headline | Evidence | 5.2.0 | 5.10.2 | 5.22.2 | 5.23.0 | Scope |
|---|---|---:|---:|---:|---:|---|
| `causality_layer_skip.skip_depth_score_future_ratio` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `causality_layer_skip.skip_depth_score_output_ratio` [1](../../tests/tasks/parity/test_layer_skip_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the released output-effect depth score normalized by the number of layers. |
| `causality_layer_skip.skip_output_effect_max` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `causality_layer_skip.skip_output_effect_total` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `consistency_icl_score.icl_score` [1](../../tests/tasks/test_icl_score.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | Brute-force next-token cross-entropy at the selected token or byte-anchored positions. |
| `consistency_icl_score.icl_score_byte_anchored` [1](../../tests/tasks/test_icl_score.py) | analytic | 2/2 | 2/2 | 2/2 | 2/2 | Brute-force next-token cross-entropy at the selected token or byte-anchored positions. |
| `dynamics_entropy_calibration.entropy_calibration_error` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `dynamics_entropy_calibration.entropy_calibration_ratio` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `dynamics_sharpness.baseline_loss` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `dynamics_sharpness.sam_perturbed_loss` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_alpha_req.alpha_req` [1](../../tests/tasks/parity/test_alpha_req_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the reported statistic with the reptrix reference on the same matrix. |
| `geometry_alpha_req.rankme` [1](../../tests/tasks/parity/test_alpha_req_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the reported statistic with the reptrix reference on the same matrix. |
| `geometry_categories.{category}_hub_max` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{category}_hub_skew` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{category}_inter` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{category}_intra` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{category}_purity` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{category}_separation` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_categories.{relation}_consistency` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_delta_hyperbolicity.delta_rel_input_embedding` [1](../../tests/tasks/parity/test_delta_hyperbolicity_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the input-embedding mean with the released delta_hyp loop on matched points. |
| `geometry_delta_hyperbolicity.delta_rel_last_layer` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_delta_hyperbolicity.delta_rel_min` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_dime.dime_last_layer` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_dime.dime_peak` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_dime.dime_peak_depth` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_domain_bpb.domain_bpb_macro` [1](../../tests/tasks/test_domain_lidar.py) | component and study check | 1/1 | 1/1 | 1/1 | 1/1 | The task test checks likelihood, exact bytes and domain aggregation. The study separately recomputed the one-domain GPT-2 macro with explicit BOS. A multi-domain macro has equal domain weights. |
| `geometry_farms_alpha.farms_alpha_mean` [1](../../tests/tasks/parity/test_farms_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the mean exponent with released estimates using the corresponding window. |
| `geometry_farms_alpha.farms_alpha_mean_w512` [1](../../tests/tasks/parity/test_farms_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the mean exponent with released estimates using the corresponding window. |
| `geometry_ffn_spectral_utilization.ffn_edim_median` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_ffn_spectral_utilization.ffn_hard_rank_median` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_ffn_spectral_utilization.ffn_soft_rank_median` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_ffn_spectral_utilization.ffn_sui_median` [1](../../tests/tasks/test_ffn_spectral_utilization.py) | aggregation | 2/2 | 2/2 | 2/2 | 2/2 | Checks the median of the model's SUI profile. The same case separately checks the captured post-activation. |
| `geometry_gride.gride_id_last_layer` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_gride.gride_id_peak` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_hsic.hsic_compression_ratio` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares normalized HSIC calculated from the authors’ demo on matched pairs, with the indicated bias correction. |
| `geometry_hsic.hsic_compression_ratio_debiased` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares normalized HSIC calculated from the authors’ demo on matched pairs, with the indicated bias correction. |
| `geometry_hsic.min_adjacent_hsic` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares normalized HSIC calculated from the authors’ demo on matched pairs, with the indicated bias correction. |
| `geometry_hsic.min_adjacent_hsic_debiased` [1](../../tests/tasks/parity/test_cka_hsic_demo_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares normalized HSIC calculated from the authors’ demo on matched pairs, with the indicated bias correction. |
| `geometry_information_imbalance.ii_asymmetry_first_last` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_information_imbalance.ii_first_to_last` [1](../../tests/tasks/parity/test_gride_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the directional endpoint imbalance with DADApy on the same ranked neighbours. |
| `geometry_information_imbalance.ii_last_to_first` [1](../../tests/tasks/parity/test_gride_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Compares the directional endpoint imbalance with DADApy on the same ranked neighbours. |
| `geometry_lm_head_free_energy.lm_head_bfe_w` [1](../../tests/tasks/parity/test_lm_head_free_energy_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Matches the released free-energy score on the same selected spectrum. |
| `geometry_neighborhood_overlap.nn_overlap_adjacent_min` [1](../../tests/tasks/parity/test_gride_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Minimum adjacent-layer overlap checked against DADApy neighbourhoods. |
| `geometry_neighborhood_overlap.nn_overlap_first_last` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_paraphrase_lidar.lidar_log_rank_fraction` [1](../../tests/tasks/test_domain_lidar.py) | component | 3/3 | 3/3 | 3/3 | 3/3 | The tests pin discriminant rank and the representation readout. The headline further applies log(rank/(pairs−1)) and averages over equal-size panels. This row’s direct numerical comparison covers the rank input. |
| `geometry_perplexity.bits_per_byte` [1](../../tests/tasks/parity/test_perplexity_parity.py) [2](../../tests/tasks/test_perplexity_byte_ownership.py) | analytic | 3/3 | 3/3 | 3/3 | 3/3 | Loss divided by independently counted UTF-8 bytes on ASCII and multibyte text. Includes ambiguous piece ownership, literal and inserted specials, and a tokenizer that appends EOS. |
| `geometry_predictive_information.predictive_info_last_layer` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_predictive_information.predictive_info_peak` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_predictive_information.predictive_info_peak_depth` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_procrustes_linearity.procrustes_linearity_heldout_r2_mean` [1](../../tests/tasks/test_metric_repairs.py) [2](../../tests/tasks/parity/test_procrustes_heldout_rank.py) | independent formula and end-to-end controls | 26/26 | 26/26 | 26/26 | 26/26 | An inner document split selects each truncated-SVD cutoff above the float32 covariance floor. Independent NumPy fits check learned ranks, null directions, repeated rows and refitting on all outer training documents. Changing the outer targets leaves the selected ranks unchanged. Grouped-document public evaluations check native and NumPy parent IDs. A separate 100-passage study replay on Transformers 5.23.0 verified both splits against the original articles through the sweep and repair CLIs. The opt-in float64 Gram solver is compared with direct SVD on tall, wide, weak-rank and repeated-row inputs and two GPU checkpoint replays. |
| `geometry_prompt_id.prompt_id_last_layer` [1](../../tests/tasks/parity/test_prompt_id_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | The endpoint or maximum of the released per-prompt GRIDE profile on matched inputs. |
| `geometry_prompt_id.prompt_id_peak` [1](../../tests/tasks/parity/test_prompt_id_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | The endpoint or maximum of the released per-prompt GRIDE profile on matched inputs. |
| `geometry_representation_dispersion.representation_dispersion` [1](../../tests/tasks/parity/test_dispersion_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Exact scalar agreement with the released evaluation on the same samples. |
| `geometry_rogue_dimensions.rogue_top1_share_last` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_rogue_dimensions.rogue_top1_share_max` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_rogue_dimensions.rogue_top3_share_max` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_spectral_signature.alpha_hill_mean` [1](../../tests/tasks/parity/test_spectral_signature_parity.py) | aggregation | 1/1 | 1/1 | 1/1 | 1/1 | Checks the mean of the exported exponent profile. The release has a separate matrix-by-matrix comparison. |
| `geometry_spectral_signature.alpha_hill_median` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_tokenizer_efficiency.fertility` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Independent count on the fixed tokenizer/corpus fixture. |
| `geometry_tokenizer_efficiency.token_entropy` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Independent count on the fixed tokenizer/corpus fixture. |
| `geometry_tokenizer_efficiency.vocab_size` [1](../../tests/tasks/test_comprehensive_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Independent count on the fixed tokenizer/corpus fixture. |
| `geometry_ww_randomized.frac_below_bulk_min_mean` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_ww_randomized.mp_softrank_mean` [1](../../tests/tasks/parity/test_ww_randomized_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Checks the returned summary against the corresponding WeightWatcher analyze output and stated normalization. |
| `geometry_ww_randomized.rand_distance_mean` [1](../../tests/tasks/parity/test_ww_randomized_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Checks the returned summary against the corresponding WeightWatcher analyze output and stated normalization. |
| `geometry_ww_randomized.spike_fraction_mean` [1](../../tests/tasks/parity/test_ww_randomized_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Checks the returned summary against the corresponding WeightWatcher analyze output and stated normalization. |
| `geometry_ww_randomized.ww_softrank_mean` [1](../../tests/tasks/parity/test_ww_randomized_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Checks the returned summary against the corresponding WeightWatcher analyze output and stated normalization. |
| `geometry_zipfian_symmetry.zipf_sym1_embedding` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_zipfian_symmetry.zipf_sym1_unembedding` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_zipfian_symmetry.zipf_sym2_embedding` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `geometry_zipfian_symmetry.zipf_sym2_unembedding` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `interpretability_inactive_heads.inactive_head_fraction` [1](../../tests/tasks/parity/test_inactive_heads_parity.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | Compares with the released scoring and threshold calculations on the tested architectures. |
| `interpretability_inactive_heads.inactive_head_threshold_5pct` [1](../../tests/tasks/parity/test_inactive_heads_parity.py) | reference | 6/6 | 6/6 | 6/6 | 6/6 | Compares with the released scoring and threshold calculations on the tested architectures. |
| `interpretability_logit_lens_kl.decision_depth` [1](../../tests/tasks/parity/test_logit_lens_kl_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the reported summary with a computation from the released layerwise probabilities. |
| `interpretability_logit_lens_kl.kl_churn` [1](../../tests/tasks/parity/test_logit_lens_kl_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the reported summary with a computation from the released layerwise probabilities. |
| `interpretability_logit_lens_kl.kl_to_final_auc` [1](../../tests/tasks/parity/test_logit_lens_kl_parity.py) | reference | 2/2 | 2/2 | 2/2 | 2/2 | Compares the reported summary with a computation from the released layerwise probabilities. |
| `repe_concept_separability.lat_layer_frac` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_concept_separability.lat_test_acc` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_refusal_direction.direction_norm` [1](../../tests/tasks/parity/test_repe_refusal_direction_arditi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Checks the reported norm or intervention score against the released direction-selection pipeline. |
| `repe_refusal_direction.mean_projection_gap` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_refusal_direction.refusal_score_bypass` [1](../../tests/tasks/parity/test_repe_refusal_direction_arditi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Checks the reported norm or intervention score against the released direction-selection pipeline. |
| `repe_refusal_direction.refusal_score_induce` [1](../../tests/tasks/parity/test_repe_refusal_direction_arditi_parity.py) | reference | 1/1 | 1/1 | 1/1 | 1/1 | Checks the reported norm or intervention score against the released direction-selection pipeline. |
| `repe_refusal_direction.rel_direction_energy` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_refusal_direction.selected_layer_frac` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_refusal_direction.separability_auc` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.best_directional_effect` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.best_directional_layer_frac` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.mean_directional_effect` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.steering_beats_random_frac` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.steering_concept_over_random` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.steering_semantic_effect` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_steering_effectiveness.steering_semantic_success_rate` | Task-level only | — | — | — | — | Key-specific comparison pending. |
| `repe_task_vectors.consensus_energy_rel` [1](../../tests/tasks/parity/test_repe_task_vectors_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Compares the BLME activation-direction summary with a controlled direction or closed-form computation. |
| `repe_task_vectors.max_norm_layer_fraction` [1](../../tests/tasks/parity/test_repe_task_vectors_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Compares the BLME activation-direction summary with a controlled direction or closed-form computation. |
| `repe_task_vectors.pair_direction_cos_mean` [1](../../tests/tasks/parity/test_repe_task_vectors_parity.py) | analytic | 1/1 | 1/1 | 1/1 | 1/1 | Compares the BLME activation-direction summary with a controlled direction or closed-form computation. |
