"""Task certification metadata.

This module is the machine-readable source for BLME's publication status
labels. The labels are intentionally conservative:

- ``parity-ready``: focused formula/reference tests exist for the core method.
- ``formula-faithful``: implementation follows the paper formula, but no
  external repo parity fixture is checked in.
- ``refined-adaptation``: paper-derived implementation adapted for BLME's
  architecture-agnostic, label-light setting.
- ``proxy-only``: BLME diagnostic inspired by literature; do not claim paper
  or repository parity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


VALID_CERTIFICATION_STATUSES = {
    "parity-ready",
    "formula-faithful",
    "refined-adaptation",
    "proxy-only",
}


@dataclass(frozen=True)
class TaskCertification:
    status: str
    papers: str
    reference: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _entry(status: str, papers: str, reference: str = "", notes: str = ""):
    if status not in VALID_CERTIFICATION_STATUSES:
        raise ValueError(f"Unknown certification status: {status}")
    return TaskCertification(status, papers, reference, notes)


TASK_CERTIFICATION: Dict[str, TaskCertification] = {
    # Geometry
    "geometry_categories": _entry("proxy-only", "BLME category geometry", "", "Tokenizer/category dependent. B7: relation/analogy kernels now covered (exact constant-relation anchors) + hub kernel vs independent k-occurrence recomputation."),
    "geometry_cka": _entry("parity-ready", "Kornblith et al. 2019 (ICML); Cortes et al. 2012", "google-research/representation_similarity", "CONFIRMED_CORRECT (C3): executed official Demo.ipynb code — fixture 0.450953881199808 bit-identical; BLME <=1.8e-8 (fp32)."),
    "geometry_collapse": _entry("refined-adaptation", "Jing et al. 2021 (ICLR 2022); Roy & Vetterli 2007; Queipo-de-Llano et al. 2025 (arXiv:2510.06477)", "", "erank profile == verbatim Roy-Vetterli Eq(1)/Def 1 + scipy svdvals; analytic anchors exact; B6 pinned max_drop/max_drop_layer. Compression-valley proxy, not the 2510.06477 pipeline."),
    "geometry_contextualization": _entry("refined-adaptation", "Ethayarajh 2019 (EMNLP, arXiv:1909.00512)", "kawine/contextual", "SelfSim/IntraSim/MEV kernels == official formulas <=3.3e-16; B7 pinned the 3 previously uncovered kernels. MEV reported UNADJUSTED (paper headline is anisotropy-adjusted, not comparable — documented)."),
    "geometry_correlation_dimension": _entry("formula-faithful", "Grassberger & Procaccia 1983 (PRL 50(5):346-349)", "CSchoel/nolds (corr_dim, executed live); skdim.id.CorrInt", "Classical GP estimator; matches nolds after removing its diagonal self-match artifact. Kernel = _gp_correlation_dimension (module-level, imported by parity tests)."),
    "geometry_hsic": _entry("parity-ready", "Gretton et al. 2005 (ALT, Def. 2); Kornblith et al. 2019", "", "nHSIC == verbatim Kornblith cka() + explicit-H Gretton Def. 2 (3.3e-15). B6: evaluate() now calls the parity-pinned _nhsic_from_grams kernel; anchor layer = first block output (not input embedding), documented."),
    "geometry_hubness": _entry("parity-ready", "Radovanovic et al. 2010 (JMLR 11); Tomasev et al. 2014 (IEEE TKDE 26(3))", "scikit-hubness", "k-occurrence kernel == scikit-hubness (bincount+skew+Gini, self-hit filter); B4: e2e evaluate() pinned vs independent full-argsort reference (self-mask regression guarded)."),
    "geometry_intrinsic_dim": _entry("formula-faithful", "Facco et al. 2017 (Sci Rep 7:12140); Ansuini et al. 2019", "efacco/TWO-NN; scikit-dimension", "Bit-identical to skdim _TwoNN fit on analytic Pareto samples; label kept conservative (mu<=1 pre-filter deviates on duplicate-heavy data). No sub-1 floor: Pareto support d in (0,inf) — the paper never discusses sub-1 estimates (B5 doc fix)."),
    "geometry_isoscore": _entry("parity-ready", "Rudman et al. 2022 (Findings of ACL, arXiv:2108.07344)", "bcbi-edu/p_eickhoff_isoscore", "Algorithm 1/Eq 4.1 exact; k-of-n analytic anchors to 2.2e-16; <=4e-8 vs official IsoScore 2.0.1 incl. the N<d zero-padding branch (live-package test added B5)."),
    "geometry_lid": _entry("formula-faithful", "Ma et al. 2018 Eq. 4 (-k MLE variant); derived from Levina & Bickel 2004", "xingjunm/lid_adversarial_subspace_detection", "Bit-exact (0.0) vs official mle_batch; -k variant is Ma et al. (biased k/(k-1) vs L-B -(k-1)), so NOT parity-ready with skdim.MLE. Docstring title+attribution fixed B5."),
    "geometry_lipschitz": _entry("proxy-only", "Miyato 2018; Scaman & Virmaux 2018 (motivation only)", "avirmaux/lipEstimation (sigma_max cross-check)", "Hidden-state relative-change diagnostic, NOT an operator-norm bound (disclaimed). lipschitz_max = max over layers of per-layer token MEANS (aggregation documented; C3 DISCREPANCY resolved). Analytic 4-layer case + distilgpt2 vs numpy reference."),
    "geometry_mahalanobis": _entry("refined-adaptation", "Lee et al. 2018 (NeurIPS, Eq. 2; single-centroid C=1 adaptation with Ledoit-Wolf)", "pokaxpoka/deep_Mahalanobis_detector", "Sentence-level held-out OOD proxy; kernel == official sqrt(-2*term_gau) (0.0). Held-out-duplicate bug fixed (dataset=None pool now text-disjoint); B7 pinned the sklearn-free pinv fallback analytically."),
    "geometry_matrix_entropy": _entry("parity-ready", "Wei et al. 2024 (arXiv:2401.17139; v2 = Diff-eRank, NeurIPS 2024)", "waltonfuture/Diff-eRank", "B4: dedicated parity file vs verbatim official normalize/cal_cov/cal_entropy (<=1e-5 fp32, ln2 analytic anchor, mutant check). Cache/fresh truncation unified at 512 (official protocol: 2048, documented)."),
    "geometry_mp_bulk_deviation": _entry("parity-ready", "Marchenko & Pastur 1967; Baik et al. 2005 (BBP)", "scikit-rmt (CDF cross-check)", "Analytic MP bulk-edge/CDF tests + closed-form/quad parity <=2.3e-5. B4: gamma>1 KS atom handling + gamma=1 CDF singularity fixed with regression tests (previously wrong: KS ~1-1/gamma, CDF err 0.59). B8: lower-edge outlier fraction/energy below (1-tol)*MP lower edge for gamma<1 (Staats et al. 2024, arXiv:2410.17770 small-eigenvalue signal); NaN for gamma>=1 (edge=0); iid ~0 / correlated >=0.04 anchors."),
    "geometry_neural_collapse": _entry("refined-adaptation", "Papyan et al. 2020", "neuralcollapse/neuralcollapse", "Bundled topic-label proxy; NC2 proxy only."),
    "geometry_perplexity": _entry("formula-faithful", "Standard LM cross entropy/perplexity (HF guide protocol)", "", "B7: dedicated parity file — overall/rare/freq PPL + BPC reconstructed independently (<=1e-4); BOS-tokenizer BPC bug + empty-text crash fixed with regression tests."),
    "geometry_positional_decay": _entry("proxy-only", "RoFormer (Su et al. 2021) decay premise; ALiBi (Press et al. 2021) context", "", "Long-context-only aggregate decay trend (documented); Spearman kernel == hand-written tie-averaged reference to 1.1e-16."),
    "geometry_prediction_alignment": _entry("proxy-only", "Output-projection geometry / logit-lens motivation (Belrose et al. 2023 tuned-lens as context)", "", "Cosine(final hidden, output-projection row of target token). B7: untied-head behavior numerically pinned (must use lm_head.weight, not the embedding table)."),
    "geometry_representation_sensitivity": _entry("proxy-only", "Amari 1998, Natural Gradient Works Efficiently in Learning (Neural Computation 10(2)) — softmax score-function identity", "", "grad_h log P = W_y - E_p[W] == f64 autograd + finite differences; B7 covered the lm_head-bias branch (was untested). Docstring cited a non-existent title/year pairing — fixed."),
    "geometry_rsa": _entry("refined-adaptation", "Kriegeskorte, Mur & Bandettini 2008 (Front. Syst. Neurosci. 2:4)", "rsatoolbox", "CONFIRMED_CORRECT (C3): all 6 outputs == from-scratch Spearman AND rsatoolbox 0.3.2 to 1.1e-16."),
    "geometry_schatten": _entry("refined-adaptation", "Yusupov et al. 2025 (arXiv:2509.25359); Li et al. 2024 (arXiv:2410.10672); Garrido et al. 2023", "MLGroupJLU/MatrixNuclearNorm", "normalize+MNN kernels == official MNN.py verbatim (0.0); B4: per-sentence pipeline incl. mnn/T pinned e2e vs in-test official transcription. RankMe eps placement deviates O(1e-5), documented. Schatten-p reported /d^(1/p); p=2 dropped (content-free)."),
    "geometry_spectral": _entry("refined-adaptation", "Martin & Mahoney 2019/2021 (HT-SR); Martin, Peng & Mahoney 2021 (Nat Comms 12:4122)", "CalculatedContent/WeightWatcher", "Singular-value Hill proxy on fixed top-20% tail, not exact WW alpha (eigenvalues + KS-optimal xmin); at matched xmin alpha_sigma = 2*alpha_MM - 1 exactly (verified vs weightwatcher 0.7.7). Stable rank def: Rudelson & Vershynin 2007 (Bartlett misattribution fixed, B4). Real-task anti-drift test added. B8: OFFICIAL Clauset PL-MLE alpha with KS-optimal xmin on eigenvalues (alpha_official, alpha_hat = alpha*log10 lambda_max). Audit 2026-07-16 fixed the KS empirical-CDF convention (right->LEFT-continuous, the plfit/WW convention) + raised the xmin scan cap: == live weightwatcher 0.7.7 across ALL 49 gpt2 modules, mean |dalpha| 3e-4, max 2.1e-3 (pre-fix: mean 0.41, max 3.64); planted Pareto alpha=3 -> 2.994."),
    "geometry_svd": _entry("refined-adaptation", "Ethayarajh 2019 (Eq. 4 anisotropy); Roy & Vetterli 2007 (erank)", "", "erank/PR/AUC pinned analytically + vs official IsoScore pkg. avg_cosine_similarity = Eq. 4 on RAW states, self-pairs excluded (historic centered-states deviation read ~0 always; fixed + exact-pinned B5)."),
    "geometry_tokenizer_efficiency": _entry("proxy-only", "Rust et al. 2021 (fertility, after Acs 2019); Ali et al. 2024 (arXiv:2310.08754)", "", "Whitespace-word micro-average fertility proxy (documented). Analytic fake-tokenizer anchors exact. B7: fabricated docstring author list (Rajput/Chamberlain/Reese) corrected to Ali et al."),
    "geometry_trajectory_curvature": _entry("parity-ready", "Hosseini & Fedorenko 2023 (NeurIPS, arXiv:2311.04930)", "eghbalhosseini/sent_sampling", "Angle kernel == official curvature_utils.py (8.9e-16); layer aggregation = paper mean-of-per-sentence-means (pooled-angles deviation fixed; pinned by test_trajectory_curvature_parity.py with length-varying trajectories). First cached layer == paper L0 baseline."),
    "geometry_unembedding": _entry("proxy-only", "Effective rank: Roy & Vetterli 2007 (EUSIPCO); purity/alignment are BLME diagnostics", "", "erank of MEAN-CENTERED W_out == f64 EUSIPCO re-derivation (5.4e-6 rel); B4: exact orthogonal-cluster purity==1.0 anchor added (was bound-checked only)."),
    "geometry_weight_norms": _entry("formula-faithful", "Standard definitions: Frobenius; sigma_max; stable rank (Rudelson & Vershynin 2007)", "CalculatedContent/WeightWatcher (stable_rank cross-check)", "Analytic planted-spectrum parity + WW stable_rank agreement 9.2e-5 rel; 3-D fused-MoE params handled by slicing (B4). norm_uniformity = max(0, 1-CV)."),
    # Topology — campaign-B1 (2026-07) verified: see docs/tasks_topology.md
    "topology_betti_curve": _entry("refined-adaptation", "Naitzat et al. 2020 (JMLR 21(184))", "topnn/topnn_framework", "Geodesic-kNN adaptation (ripser, not Eirene; normalized-persistence beta_1) — deviation disclosed. networkx connected-components cross-check + ground-truth anchors (4 blobs/circle/figure-8/hi-dim noise); derived stats numerically pinned (test_topology.py::test_betti_curve). (An earlier note claimed GF(2)/GUDHI engines that are not in the repo — corrected, audit 2026-07-16.)"),
    "topology_homology": _entry("parity-ready", "Zomorodian & Carlsson 2005; Edelsbrunner & Harer 2008; Naitzat et al. 2020 (JMLR)", "ripser.py; GUDHI", "Persistence-summary reference test vs analytic unit-square + GUDHI; also emits mean-pairwise-distance-normalized summaries (paper tab:normalization). B8: scale-stratified fine/coarse persistence summaries (death vs frac*mean-pairwise-distance; Fay et al. 2026, arXiv:2505.20435 topology-compression signal), anchored on constructed diagrams."),
    "topology_persistence_entropy": _entry("parity-ready", "Rucco et al. 2016 (ECCS 2014); Chintakunta et al. 2015; Atienza et al. 2020 (Pattern Recognition 107:107509)", "giotto-tda", "log2 Def 3.1; executed-gtda-0.6.2 parity 2.7e-15 over 203 cases (campaign 3); live guard = scipy.stats.entropy(base=2), gtda's own kernel. Also emits relative PE (/log2 m)."),
    "topology_persistence_landscape": _entry("parity-ready", "Bubenik 2015 (JMLR 16:77-102)", "persim", "persim parity incl. multi-bar K=4 case (k-selection exercised) at ~9e-16; analytic single/two-bar anchors exact."),
    # Interpretability
    "interpretability_activation_sinks": _entry("refined-adaptation", "Xiao 2023; Gu 2025; Sun 2024; Arroyo 2025", "sail-sg/Attention-Sink; locuslab/massive-activations"),
    "interpretability_attention_effective_rank": _entry("proxy-only", "Elhage 2022; Templeton 2024 inspiration"),
    "interpretability_attention_entropy": _entry("parity-ready", "Clark et al. 2019", "clarkkev/attention-analysis", "Shannon-entropy reference test (uniform->log T, vs scipy)."),
    "interpretability_attention_graph": _entry("proxy-only", "Xiao et al. 2023 (attention sinks)", notes="Damped PageRank centrality on per-head/layer attention (verified vs networkx PageRank in test_comprehensive_parity). NOT Abnar-Zuidema rollout despite the historical name -- rollout is the separate interpretability_attention_rollout task (added 2026-07)."),
    "interpretability_attention_rank": _entry("formula-faithful", "Dong et al. 2021; Roy & Vetterli 2007", "twistedcubic/attention-rank-collapse"),
    "interpretability_attribution": _entry("refined-adaptation", "Simonyan et al. 2014"),
    "interpretability_head_roles": _entry("refined-adaptation", "Clark 2019; Voita 2019; Olsson 2022; Wang 2022 (IOI)", notes="Previous-token score == official TransformerLens prev-token kernel to 8.8e-8 (tests/tasks/parity/test_head_roles_parity.py); duplicate-token/other role scores are content-dependent adaptations, not pinned."),
    "interpretability_induction_heads": _entry("parity-ready", "Olsson et al. 2022", "TransformerLens", "Per-head prefix-matching (induction) score = mean of the induction diagonal (offset 1-N) == official TransformerLens induction_score to <1e-4 (tests/tasks/parity/test_induction_heads_parity.py). Fixed 2026-07 to average the FULL diagonal (was ~0.03 low). Also carries a causal-validation ablation sub-score."),
    "interpretability_logit_lens": _entry("parity-ready", "nostalgebraist 2020; Belrose et al. 2023", "transformer-utils; tuned-lens", "Logit-lens projection W_U(ln_f(h)) == official tuned-lens untrained LogitLens to 1.9e-5 (float32); final-layer lens == the model's real logits (tests/tasks/parity/test_logit_lens_parity.py)."),
    "interpretability_prediction_entropy": _entry("formula-faithful", "Holtzman et al. 2020", "ari-holtzman/degen"),
    "interpretability_probing": _entry("refined-adaptation", "Alain & Bengio 2017"),
    "interpretability_sae_features": _entry("refined-adaptation", "Bricken/Cunningham SAE work", "SAELens", "L0 sparsity stat kernel == numpy reference (0.0) on features encoded by a REAL pretrained SAELens SAE (jbloom/gpt2-small-res-jb, d_sae=24576); trained-SAE pipeline faithful. tests/tasks/parity/test_sae_features_parity.py. CAVEAT: sae_features.py:97 unpacks SAE.from_pretrained() as a 3-tuple; sae-lens >=6.x returns the SAE directly -> update needed to run against current sae-lens."),
    "interpretability_sparsity": _entry("proxy-only", "MoEfication / contextual sparsity literature"),
    "interpretability_superposition": _entry("proxy-only", "Elhage et al. 2022; Templeton et al. 2024"),
    "interpretability_waa": _entry("proxy-only", "Park et al. 2024", "KihoPark/linear_rep_geometry"),
    # Causality
    "causality_ablation": _entry("proxy-only", "Standard mechanistic ablation practice"),
    "causality_attention_knockout": _entry("formula-faithful", "Michel 2019; Voita 2019", "pmichel31415/are-16-heads-really-better-than-1", "Direct head-ablation dNLL importance == an independent faithful Michel reimplementation exactly (0.0) on gpt2; Michel's key claim (Eq.5 gradient proxy tracks |ablation|) reproduced (Spearman 0.82). BLME reports the ablation, not the Eq.5 proxy. tests/tasks/parity/test_attention_knockout_parity.py."),
    "causality_circuit_quality": _entry("proxy-only", "Chan causal scrubbing; Conmy ACDC"),
    "causality_edge_attribution": _entry("proxy-only", "Syed et al. 2024 EAP", "Aaquib111/edge-attribution-patching", "The EAP scoring KERNEL |(clean-corr).grad| is bit-exact vs paper Eq 2/3 and the official repo kernel (0.0; 5.5e-16 on gpt2; exact on a linear model). BLME's registered output is a per-LAYER residual proxy, not the per-EDGE transformer_lens circuit -> proxy at the circuit level. tests/tasks/parity/test_edge_attribution_parity.py."),
    "causality_knowledge_neurons": _entry("proxy-only", "Dai et al. 2022"),
    "causality_tracing": _entry("parity-ready", "Meng et al. 2022 ROME", "kmeng01/rome", "Per-layer Average Indirect Effect == the actual (unmodified) ROME trace_with_patch bit-for-bit (max abs diff 0.0) on gpt2 with shared noise; peaks at early/mid layers (ROME Fig. 2). tests/tasks/parity/test_tracing_rome_parity.py."),
    # Consistency
    "consistency_bias_weat": _entry("parity-ready", "Caliskan 2017; May 2019", "W4ngatang/sent-bias", "WEAT effect size + test statistic bit-exact (abs_diff 0.0) vs official sent-bias weat.py effect_size/s_XYAB @e3559fb; tests/tasks/parity/test_bias_weat_parity.py. p-value path stays stochastic."),
    "consistency_calibration": _entry("parity-ready", "Guo et al. 2017; Brier 1950", "temperature_scaling"),
    "consistency_contamination": _entry("parity-ready", "Shi et al. 2023", "swj0419/detect-pretrain-code"),
    "consistency_contrastive": _entry("proxy-only", "CounterFact-style negative rejection"),
    "consistency_format_robustness": _entry("proxy-only", "Sclar et al. 2023", "msclar/formatspread"),
    "consistency_icl_slope": _entry("proxy-only", "Brown 2020; Min 2022"),
    "consistency_knowledge_capacity": _entry("proxy-only", "Tirumala 2022; Carlini 2023 memorization framing"),
    "consistency_logical": _entry("proxy-only", "General entailment/consistency literature"),
    "consistency_membership_inference": _entry("proxy-only", "Yeom 2018; Carlini 2021"),
    "consistency_paraphrase": _entry("proxy-only", "Paraphrase-invariance literature"),
    "consistency_position_sensitivity": _entry("proxy-only", "Liu et al. 2023", "lost-in-the-middle"),
    "consistency_self_consistency": _entry("proxy-only", "Wang et al. 2022"),
    # Dynamics — campaign-B3 (2026-07) verified: see docs/tasks_dynamics.md
    "dynamics_coe": _entry("formula-faithful", "Wang et al. 2025 (ICLR, arXiv:2410.13640)", "Alsace08/Chain-of-Embedding", "Prompt-side hidden-state variant by default. Follows OFFICIAL score.py (normalized mag AND angle in CoE-C); the paper prose/Algorithm-2/code are 3-way inconsistent. Parity <=1.9e-9 on Z_Mag!=1 chains (test_dynamics.py::test_chain_of_embedding_zmag_not_one...)."),
    "dynamics_generation_diversity": _entry("parity-ready", "Li et al. 2016 (NAACL); Zhu et al. 2018 (SIGIR, Texygen)", "geek-ai/Texygen", "Self-BLEU == verbatim Texygen calc_bleu to 2.8e-17; distinct-n == Li formula (per-completion scope flagged in output). Seeding via torch.manual_seed (generate() has no generator kwarg; 7f9ba4c)."),
    "dynamics_gradient_flow": _entry("refined-adaptation", "Pascanu et al. 2013 (ICML, arXiv:1211.5063)", "", "Depth-for-time adaptation: ||dL_CE/dh_l||_F. Norms verified against float64 central finite differences to 1.1e-7 rel (campaign-B3 session artifact; the committed suite pins entropy == scipy, slope == linregress, and an independent leaf-replacement autograd recomputation matched exactly in the 2026-07-16 audit). Slope over NORMALIZED depth. No repo parity claimed (official code is RNN training code)."),
    "dynamics_interpolation": _entry("proxy-only", "BLME convexity proxy; slerp kernel = White 2016 (arXiv:1609.04468)", "dribnet/plat", "_slerp == official plat interpolate.py to 8.9e-16; e2e entropies == independent f64 recompute <=8.1e-6; endpoint invariant (lerp==slerp at alpha 0/1) pinned. Hang on <2 usable rows fixed (c2911bf)."),
    "dynamics_sharpness": _entry("parity-ready", "Foret 2021; Yao 2020; Hutchinson 1990", "google-research/sam; amirgholami/PyHessian", "Top Hessian eigenvalue = SIGNED Rayleigh quotient (PyHessian convention; negative-dominant anchor pins the sign) == exact eigh <1e-9 rel, == pip PyHessian <1e-5. SAM delta == analytic quadratic exactly. Probe vectors seeded (config seed, B3). tests/tasks/parity/test_sharpness_parity.py drives the REAL helpers. B8: critical step size eta* (raw-gradient doubling+bisection probe theta - eta*g, params restored) with lambda_c = 2/eta* (Kalra et al. 2026, arXiv:2601.16979 edge-of-stability framing); exact on quadratic anchors lambda in {0.25, 4, 37.5}."),
    "dynamics_stability": _entry("proxy-only", "BLME embedding-neighborhood diagnostic; inspired by Wendlandt et al. 2018 (arXiv:1804.09692)", "", "Jaccard = monotone transform of the paper's 10-NN overlap (J = o/(2k-o), verified). Independent torch.topk reimplementation matched exactly in the 2026-07-16 audit (the earlier 'triple route' figure was a session artifact, not a committed test); reference_model branch covered (self-ref -> exactly 1.0/0.0). Global-RNG seeding replaced with a local default_rng honoring config seed (audit fix)."),
    # Representation engineering
    "repe_concept_separability": _entry("refined-adaptation", "Zou et al. 2023", "andyzoujm/representation-engineering", "Held-out LR-CV separability probe (paper Sec 5.1 linear-reading family), NOT the PCA/LAT reading vector. Per-layer AUC/acc == sklearn cross_val_score to 0.0; planted-concept analytic AUC=1.0. B2 (2026-07): nan-aware summaries + binary-int label validation."),
    "repe_refusal_direction": _entry("refined-adaptation", "Arditi et al. 2024; Zou et al. 2023", "andyrdt/refusal_direction", "Refusal direction = difference-of-means (harmful - harmless) == official andyrdt/refusal_direction get_mean_diff exactly (0.0); |cos|=1 (tests/tasks/parity/test_repe_refusal_direction_parity.py)."),
    "repe_steering_effectiveness": _entry("proxy-only", "Zou et al. 2023; Turner et al. 2023 (ActAdd, arXiv:2308.10248)", "", "Hook == official RepE linear_comb @ token_pos='end'; per-layer KL == fp64 reference <=1.7e-6; alpha=0 -> KL~0. KL effectiveness/success-rate summary is BLME's own. B2 (2026-07): fallback now reachable on the default text corpus (was dead in every study run) + no caller-dataset mutation."),
    "repe_task_vectors": _entry("refined-adaptation", "Zou et al. 2023; Ilharco et al. 2023", "andyzoujm/representation-engineering", "Reading-vector direction = class mean-difference == official RepE ClusterMeanRepReader to 2.2e-16; |cos|=1 (tests/tasks/parity/test_repe_task_vectors_parity.py)."),
    # --- Campaign-2 additions (2026-06): new methods, official-code parity-verified ---
    "geometry_vendi_score": _entry("parity-ready", "Friedman & Dieng 2023 (TMLR, arXiv:2210.02410)", "vertaix/Vendi-Score", "Bit-exact vs official score_K/score_X; nonlinear kernel, distinct from erank/RankMe. Subsampling is a size-CAPPED (not unbiased) estimate — VS <= max_points (docstring fixed B5)."),
    "geometry_phd_dimension": _entry("parity-ready", "Tulchinskii et al. 2023 (NeurIPS)", "ArGintum/GPTID", "Persistent-homology dimension from the MST-length (total H0 persistence) power-law E(n)~n^((d-1)/d). Seed-matched bit-exact vs GPTID PHD; stochastic estimator, RNG-pinned; tests/tasks/parity/test_phd_dimension_parity.py."),
    "geometry_cknna": _entry("parity-ready", "Huh et al. 2024 (ICML, Platonic Representation Hypothesis)", "minyoungg/platonic-rep", "Mutual k-NN conditional CKA (local-neighborhood alignment). Bit-exact vs platonic-rep metrics.cknna (unbiased+biased HSIC, both orderings); tests/tasks/parity/test_cknna_parity.py."),
    "geometry_magnitude": _entry("parity-ready", "Limbeck et al. 2024 (NeurIPS, arXiv:2311.16054)", "aidos-lab/magnipy", "Bit-exact vs magnipy cholesky/pinv/no_gpu + Leinster line closed form. B5: median-heuristic scale matches magnipy incl. duplicate clouds (zero-filter deviation fixed); med_scale = |tX| AT t_med; numpy-1.x trapezoid crash fixed + guarded. Scale ladder + log-log AUC are documented BLME choices."),
    "geometry_procrustes_linearity": _entry("parity-ready", "Razzhigaev et al. 2024 (ACL, 'Your Transformer is Secretly Linear')", "AIRI-Institute/LLM-Microscope", "Per-adjacent-layer linear (pseudo-inverse) Procrustes similarity; bit-exact port of llm_microscope.procrustes_similarity. NOTE: the absolute value is conditioning-dependent (unguarded 1/S amplifies ~1e-17 singular values) — interpret the across-depth PROFILE, not a single value; tests/tasks/parity/test_procrustes_linearity_parity.py."),
    "interpretability_activation_kurtosis": _entry("parity-ready", "Akhondzadeh et al. 2025 (KurTail, EMNLP Findings); Sun et al. 2024 (Massive Activations)", "scipy.stats.kurtosis; locuslab/massive-activations", "Per-channel excess (Fisher) kurtosis of activations (outlier/quantizability signal). Bit-exact vs scipy.stats.kurtosis(fisher=True, bias=True); tests/tasks/parity/test_activation_kurtosis_parity.py."),
    "interpretability_attention_rollout": _entry("parity-ready", "Abnar & Zuidema 2020 (ACL, arXiv:2005.00928)", "samiraabnar/attention_flow", "Attention rollout = cumulative product of normalize(A_l + I) over head-averaged per-layer attention. Bit-exact (0.0) vs samiraabnar/attention_flow compute_joint_attention; tests/tasks/parity/test_attention_rollout_parity.py. Added 2026-07 to implement the rollout that attention_graph only cited."),
    "topology_zigzag_persistence": _entry("refined-adaptation", "Gardinazzi et al. 2025 (ICML, arXiv:2410.11042)", "RitAreaSciencePark/ZigZagLLMs", "Across-layer topological-persistence summary (ripser, existing dep). FAITHFUL PROXY: feature layer-lifetimes anchored EXACTLY to the paper's dionysus zigzag engine on ground-truth constructions (loop in a contiguous layer band -> one zigzag bar), not a bit-exact barcode port of raw short-bar multiplicity; tests/tasks/parity/test_zigzag_parity.py."),
    # --- Campaign-4 B8 additions (2026-07): gap-survey metrics ---
    "geometry_superposition_strength": _entry("formula-faithful", "Liu, Liu & Gore 2025 (NeurIPS, arXiv:2505.10465)", "liuyz0/SuperpositionScaling", "Welch-bound-normalized mean squared overlap of unembedding rows. Analytic anchors exact: orthonormal->0, simplex ETF == Welch bound (rel 1e-6), iid Gaussian -> 1/m. PROTOCOL (audit 2026-07-16): headline MSO on RAW rows per the official overlap-0.py (no centering, paper's ~1/m is a scaling trend); mean-centered variant reported as a disclosed BLME auxiliary (gpt2: raw 0.075 = 58x Welch vs centered 0.0025 = 1.9x Welch); pinned by test_task_headline_follows_official_raw_protocol."),
    "geometry_effective_depth": _entry("formula-faithful", "Men et al. 2024 (ShortGPT, arXiv:2403.03853); Csordas et al. 2025 (arXiv:2505.13898)", "", "Block Influence BI_l = 1 - E_t[cos(h_l, h_{l+1})] (ShortGPT Eq. 2, hidden-state form) + effective_depth_90, a BLME cumulative-contribution summary INSPIRED BY Csordas et al.'s depth-utilization analysis (their own metric is an importance-weighted depth score — not this). frac_low_influence threshold 0.05 is a BLME choice (ShortGPT ranks and prunes to a ratio, no threshold). Audit 2026-07-16: final transition now uses the last block's PRE-final-norm output via a forward hook (HF hidden_states[-1] is post-ln_f; on gpt2 ~90% of the old last-layer BI was the ln_f artifact). Analytic anchors exact: identity -> 0, sign flip -> 2, rotation ~ 1, concentrated stack -> 0.25."),
    "dynamics_fractal": _entry("refined-adaptation", "Alabdulmohsin, Tran & Dehghani 2024 (NeurIPS, arXiv:2402.01825)", "nolds (R/S cross-check)", "Fractal parameters of the NLL 'language process' z_t = -log p(x_t|prefix): R/S Hurst, Joseph (sigma_tau ~ tau^J), Holder (p_eps(tau) ~ tau^-S, eps=5e-3). White-noise/trending anchors + nolds.hurst_rs agreement +-0.1. DEVIATION (disclosed): computed on the concatenated eval-corpus stream, not the paper's per-document 4K-token windows. gpt2 H=0.63 vs paper's ~0.70 for language."),
    "dynamics_perturbation_growth": _entry("proxy-only", "Li et al. 2025 (arXiv:2503.13530, quasi-Lyapunov grounding); Fernando & Guitchounts 2025/26 (residual-stream dynamics context)", "", "BLME diagnostic (documented formulation, NOT the official QLE protocol): seeded per-token-relative Gaussian perturbation of input embeddings; growth_slope = LSQ slope of log divergence vs normalized depth. Analytic anchor: linear gain-a stack -> slope/(L-1) = ln|a| exact (fp32 floor rel 1e-5); contracting stack -> negative."),
}


def get_task_certification(task_name: str) -> TaskCertification:
    return TASK_CERTIFICATION[task_name]


def task_metadata_dict(task_name: str) -> dict:
    return get_task_certification(task_name).to_dict()


def validate_certification_coverage(task_names: List[str]) -> tuple[list[str], list[str]]:
    expected = set(task_names)
    actual = set(TASK_CERTIFICATION)
    return sorted(expected - actual), sorted(actual - expected)

