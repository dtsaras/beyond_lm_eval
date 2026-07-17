# Verification status — all 51 paper tasks (47 structural + 4 RepE)
Compiled 2026-07-17, post-B8 and post-adversarial-audit (commits b35bdc0, a559fe1, 26a173b).
Evidence bar (user decision, campaign 4): official-code parity / analytic anchors /
small-model signatures — NOT full paper-table reproduction.

## A. Numerically verified against OFFICIAL code (27)

| Task | Evidence |
|---|---|
| geometry_cka | official Kornblith Demo.ipynb executed; fixture bit-identical (0.450953881199808); BLME ≤1.8e-8 |
| geometry_cknna | bit-exact vs platonic-rep metrics.cknna @dcd76ba (unbiased+biased HSIC, both orderings); audit re-derivation ≤4.1e-8 fresh |
| geometry_procrustes_linearity | diff 0 vs pip llm_microscope 0.0.7 (float64; see caveat E2) |
| geometry_trajectory_curvature | ≤8.9e-16 vs official curvature_utils.py; paper aggregation (mean-of-per-sentence-means) pinned |
| geometry_matrix_entropy | ≤1e-5 (fp32) vs cloned official Diff-eRank normalize/cal_cov/cal_entropy; audit re-verified vs actual repo |
| geometry_schatten (MNN kernel) | verbatim official MNN.py, diff 0.0 (d^{1/p} normalization correctly attributed to BLME) |
| geometry_vendi_score | bit-exact vs official score_K; ≤9e-15 vs score_X; audit re-verified fresh clouds |
| geometry_isoscore | ≤4e-8 vs official IsoScore 2.x incl. N<d branch; audit: ≤3.3e-8 on 7 fresh clouds |
| geometry_phd_dimension | seed-matched bit-exact vs GPTID @8c8759e; audit re-verified fresh clouds (R2/R4/R7) |
| geometry_magnitude | ≤3.6e-15 vs magnipy (5 paths) incl. duplicate clouds (see caveat E3) |
| geometry_lid | bit-exact vs official mle_batch (≤1.2e-5 fp32 gap); duplicate handling now matches official (audit fix) |
| geometry_intrinsic_dim | bit-identical to skdim _TwoNN on Pareto samples; fresh generic clouds 0.1–0.2% |
| geometry_correlation_dimension | == live nolds.corr_dim (self-match artifact removed); see caveat E4 |
| geometry_hubness | scikit-hubness kernel line-for-line; independent full-argsort e2e pin |
| geometry_mahalanobis | kernel == official deep_Mahalanobis_detector (0.0); adaptation disclosed; held-out AUROC non-degenerate (audit) |
| geometry_spectral (alpha_official) | live WeightWatcher 0.7.7: ALL 49 gpt2 modules, mean dα 3e-4 / max 2.1e-3 (post KS-convention fix) |
| geometry_spectral (Hill alpha) | disclosed proxy; exact identity α_σ=2α_λ−1 vs WW at matched cutoff |
| geometry_weight_norms | WW stable_rank agreement 9.2e-5; analytic planted spectra ≤1.2e-6 |
| dynamics_sharpness (Hessian/SAM) | pip PyHessian <1e-5; exact eigh <1e-9; SAM quadratic exact; signed-Rayleigh pinned |
| dynamics_generation_diversity | Self-BLEU == Texygen calc_bleu 2.8e-17; distinct-n == Li formula (see caveat E11) |
| dynamics_coe | == official score.py convention ≤3.4e-7 (tighter after acos fix); see caveat E12 |
| dynamics_interpolation (slerp kernel) | == official plat interpolate.py 8.9e-16 (task-level: proxy, tier D) |
| topology_homology | ripser vs GUDHI cross-backend ≤6.2e-8; scipy-MST H0; analytic unit-square/hexagon anchors exact |
| topology_persistence_entropy | executed gtda 0.6.2 parity 2.7e-15 over 203 cases; live scipy.stats.entropy(base=2) |
| topology_persistence_landscape | persim parity ~9e-16 incl. K=4 multi-bar |
| repe_task_vectors | == official RepE ClusterMeanRepReader 2.2e-16 |
| repe_refusal_direction | == official andyrdt get_mean_diff exactly (0.0) |

## B. Verified against the PAPER's equations + analytic anchors (no runnable official diagnostic code) (9)

| Task | Evidence | Why no code parity |
|---|---|---|
| geometry_mp_bulk_deviation | analytic MP theory; _mp_cdf vs scipy.quad ≤6.4e-8; scikit-rmt CDF cross-check; BBP anchors | MP is textbook math; no single official repo |
| geometry_effective_depth (BI) | ShortGPT equation re-derived by auditor; independent recompute 7e-8; identity/flip/rotation anchors exact; pre-norm-hook e2e pinned | ShortGPT releases pruning code, not a BI diagnostic |
| geometry_superposition_strength | protocol verified by READING official overlap-0.py (raw rows, all pairs); Welch/simplex-ETF/Gaussian anchors exact; independent Gram-identity recompute on gpt2 | see caveat E1 — not executed head-to-head |
| geometry_svd | Ethayarajh Eq. 4 exact-pinned (raw states); Roy-Vetterli erank exact; IsoScore-pkg cross-checks | equations, not a diagnostic repo |
| geometry_collapse | Roy-Vetterli verbatim + scipy svdvals; analytic anchors | compression-valley proxy vs 2510.06477 disclosed |
| geometry_hsic | Gretton Def. 2 explicit-H; == Kornblith cka() 3.3e-15 | HSIC is a formula |
| geometry_perplexity | independent f64 reconstruction ≤1e-4 (overall/rare/freq/BPC); BOS + truncation regressions pinned | standard CE; no official repo exists |
| dynamics_gradient_flow | independent leaf-replacement autograd recompute: EXACT (audit); entropy==scipy; slope==linregress | official Pascanu code is RNN training code |
| dynamics_fractal | estimators match paper definitions (auditor re-derived from ar5iv); R/S vs live nolds ±0.1; fGn known-H recovery; PAPER-NUMBER check: gpt2 H=0.63 vs language 0.70±0.09 | paper releases no estimator code; see caveat E5 |

## C. Paper-grounded with DISCLOSED protocol deviations (refined adaptations; kernel verified per A/B)

betti_curve (geodesic-kNN/ripser vs Eirene; triple ground-truth anchors + networkx cross-check) ·
zigzag (band merging vs dionysus barcode — anchored on ground-truth constructions; mean-pool vs
paper's last-token, disclosed) · fractal (concatenated stream vs per-doc 4K windows; no Anis-Lloyd) ·
matrix_entropy (512 vs official 2048 tokens) · mahalanobis (single-centroid + Ledoit-Wolf) ·
contextualization (kernels ≤3.3e-16 vs official formulas; MEV reported UNADJUSTED — paper headline
is anisotropy-adjusted, not comparable) · sharpness critical-step (raw-gradient vs Kalra's
optimizer-direction; quadratic anchors exact; NaN-on-cap) · coe (prompt-side default) ·
neural_collapse (NC1 == official rhubarbwu library diff 0.0; NC2 = proxy) ·
rsa (== rsatoolbox 1e-12; Euclidean-RDM deviation stated).

## D. Proxy-only — NOT externally verifiable BY DESIGN (12)
BLME diagnostics validated against their own documented spec with exact analytic/kernel anchors;
cited papers are motivation, not ground truth. No claim of external parity is made anywhere.

geometry_lipschitz (σ_max kernel cross-checked vs live lipEstimation; the diagnostic itself is a
relative-change ratio, disclaimed) · geometry_positional_decay (Spearman kernel 1.1e-16 vs hand
reference) · geometry_prediction_alignment (untied-head behavior pinned) ·
geometry_representation_sensitivity (score-function identity == f64 autograd ≤6.7e-16 — the MATH
is verified; "proxy" as a capability diagnostic; softcap guard added) · geometry_tokenizer_efficiency
(analytic fake-tokenizer anchors exact) · geometry_unembedding (erank exact; purity/alignment BLME) ·
geometry_categories (exact constant-relation anchors; cross_model_comparable=False) ·
dynamics_stability (kernel == independent topk reimpl exactly; monotone transform of Wendlandt
overlap) · dynamics_interpolation (slerp official-exact; convexity summary BLME) ·
dynamics_perturbation_growth (linear-gain anchors exact at fp32 floor; QLE protocol NOT reproduced) ·
repe_steering_effectiveness (hook == RepE linear_comb; KL summary BLME; alpha=0 → KL≈0 pinned) ·
repe_concept_separability (== sklearn cross_val_score 0.0; probe family adaptation of Zou Sec 5.1).

## E. STILL UNSURE — residual caveats (the honest list)

1. **superposition_strength**: CLOSED 2026-07-17 — the official overlap-0.py computation
   (verbatim math, batched all-pairs) executed head-to-head on gpt2: official exact
   E[cos^2] = 0.07465654 vs BLME sampled headline 0.07476744 (0.15% rel, inside the
   ~0.45% 200k-pair sampling error). Task upgraded: protocol AND numbers verified.
2. **procrustes_linearity**: bit-exact vs official only under float64; the official float32
   pipeline diverges up to 0.42 in the ill-conditioned regime. Absolute values are
   conditioning-dependent — only the across-depth PROFILE is trustworthy (documented).
3. **magnitude**: the singular-ζ fallback (LU vs magnipy's Cholesky trigger) can diverge on
   near-singular clouds; BLME's branch is closer to dedup ground truth but is not bit-exact there.
4. **correlation_dimension**: compressive at higher dimension (true 6 → ~3.4 with R²>0.97) —
   a monotone complexity proxy, not a calibrated GP dimension (documented in-task).
5. **fractal**: Hölder exponent has high variance / NaN risk on short single streams; no
   Anis-Lloyd correction (mild upward Hurst bias, within the claimed ±0.1 band).
6. **zigzag**: faithful proxy — anchored to the paper's engine on ground-truth constructions,
   NOT a bit-exact barcode port; identity-blind band merging; mean-pool vs last-token deviation.
7. **MP lower-edge extension (Staats)**: analytic anchors only; the authors' code was not obtained.
8. **critical sharpness (Kalra)**: no official code released; exact on quadratic anchors;
   gradient-direction variant of their optimizer-direction probe (disclosed).
9. **scale-stratified PH (Fay)**: BLME-designed summary MOTIVATED by the paper; constructed-diagram
   anchors only; their pipeline not compared.
10. **effective_depth_90 / frac_low_influence**: BLME summaries (relabeled in audit) — the cited
    papers' own statistics differ (Csordás's importance-weighted depth score not yet implemented;
    proposal M6).
11. **Self-BLEU**: computed over subword ids; Texygen uses word tokens — absolute values are not
    cross-tokenizer comparable (kernel parity is real; unit disclosed).
12. **coe**: matches the official released CODE; the paper itself is 3-way inconsistent
    (prose ≠ Algorithm 2 ≠ code), so "correct" here means "matches the authors' code".
13. **Tier D as a class**: no external ground truth exists by construction; certainty is
    "matches its own spec exactly", nothing stronger — and the paper labels them accordingly.

## Paper-NUMBER reproductions performed (beyond code parity)
fractal H (gpt2 0.63 vs language 0.70±0.09) · procrustes (0.9988–0.99999 vs paper's ~0.99) ·
Hill/official alpha in the HT-SR 2–6 band · trajectory-curvature qualitative signature ·
superposition centered ratio ~1.9 consistent with strong-superposition regime.
Full paper-TABLE reproduction was explicitly descoped (campaign-4 user decision) — it would
require the original papers' exact models and corpora.
