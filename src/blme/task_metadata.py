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
    "geometry_categories": _entry("proxy-only", "BLME category geometry", notes="Tokenizer/category dependent."),
    "geometry_cka": _entry("parity-ready", "Kornblith et al. 2019", "google-research/representation_similarity"),
    "geometry_collapse": _entry("refined-adaptation", "Jing 2021; Roy & Vetterli 2007; Arroyo et al. 2025"),
    "geometry_contextualization": _entry("refined-adaptation", "Ethayarajh 2019", "kawine/contextual"),
    "geometry_correlation_dimension": _entry("formula-faithful", "Grassberger & Procaccia 1983"),
    "geometry_hsic": _entry("parity-ready", "Gretton et al. 2005; Kornblith et al. 2019"),
    "geometry_hubness": _entry("parity-ready", "Radovanovic et al. 2010; Tomasev et al. 2014", "scikit-hubness"),
    "geometry_intrinsic_dim": _entry("formula-faithful", "Facco et al. 2017; Ansuini et al. 2019", "efacco/TWO-NN; scikit-dimension"),
    "geometry_isoscore": _entry("parity-ready", "Rudman et al. 2022", "bcbi-edu/p_eickhoff_isoscore"),
    "geometry_lid": _entry("formula-faithful", "Levina & Bickel 2004; Ma et al. 2018", "xingjunm/lid_adversarial_subspace_detection", "Uses the -k MLE variant (biased k/(k-1) high vs Levina-Bickel -(k-1)); not bit-parity with skdim.MLE, so NOT parity-ready."),
    "geometry_lipschitz": _entry("proxy-only", "Miyato 2018; Virmaux & Scaman 2018"),
    "geometry_mahalanobis": _entry("refined-adaptation", "Lee et al. 2018", "pokaxpoka/deep_Mahalanobis_detector", "Sentence-level held-out OOD proxy."),
    "geometry_matrix_entropy": _entry("parity-ready", "Wei et al. 2024", "waltonfuture/Matrix-Entropy"),
    "geometry_mp_bulk_deviation": _entry("parity-ready", "Marchenko & Pastur 1967; Baik et al. 2005", notes="Analytic MP bulk-edge/CDF reference tests (test_geometry.py::test_mp_bulk_edge_and_cdf_sanity)."),
    "geometry_neural_collapse": _entry("refined-adaptation", "Papyan et al. 2020", "neuralcollapse/neuralcollapse", "Bundled topic-label proxy; NC2 proxy only."),
    "geometry_perplexity": _entry("formula-faithful", "Standard language-model cross entropy/perplexity"),
    "geometry_positional_decay": _entry("proxy-only", "RoFormer / long-context positional literature"),
    "geometry_prediction_alignment": _entry("proxy-only", "Output-projection geometry / logit-lens motivation"),
    "geometry_representation_sensitivity": _entry("proxy-only", "Amari information-geometry motivation"),
    "geometry_rsa": _entry("refined-adaptation", "Kriegeskorte et al. 2008", "rsatoolbox"),
    "geometry_schatten": _entry("refined-adaptation", "Yusupov et al. 2025; Li et al. 2024; Garrido et al. 2023", "MLGroupJLU/MatrixNuclearNorm"),
    "geometry_spectral": _entry("refined-adaptation", "Martin & Mahoney 2019/2021", "CalculatedContent/WeightWatcher", "Singular-value Hill proxy, not exact WeightWatcher alpha."),
    "geometry_svd": _entry("refined-adaptation", "Ethayarajh 2019; Roy & Vetterli 2007"),
    "geometry_tokenizer_efficiency": _entry("proxy-only", "Tokenizer-efficiency literature"),
    "geometry_trajectory_curvature": _entry("parity-ready", "Hosseini & Fedorenko 2023"),
    "geometry_unembedding": _entry("proxy-only", "Unembedding-geometry literature"),
    "geometry_weight_norms": _entry("proxy-only", "Standard weight norm diagnostics"),
    # Topology
    "topology_betti_curve": _entry("refined-adaptation", "Naitzat et al. 2020", "topnn/topnn_framework"),
    "topology_homology": _entry("parity-ready", "Zomorodian & Carlsson 2005; Edelsbrunner & Harer 2008", "ripser.py; GUDHI", "Persistence-summary reference test vs analytic unit-square + GUDHI."),
    "topology_persistence_entropy": _entry("parity-ready", "Rucco et al. 2016; Chintakunta et al. 2015", "giotto-tda"),
    "topology_persistence_landscape": _entry("parity-ready", "Bubenik 2015", "persim"),
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
    # Dynamics
    "dynamics_coe": _entry("formula-faithful", "Wang et al. 2025", "Alsace08/Chain-of-Embedding", "Prompt-side hidden-state variant by default."),
    "dynamics_generation_diversity": _entry("parity-ready", "Li 2016; Zhu 2018", "Texygen"),
    "dynamics_gradient_flow": _entry("refined-adaptation", "Pascanu et al. 2013"),
    "dynamics_interpolation": _entry("proxy-only", "Latent interpolation / slerp motivation"),
    "dynamics_sharpness": _entry("parity-ready", "Foret 2021; Yao 2020", "google-research/sam; amirgholami/PyHessian", "Top Hessian eigenvalue (power iteration via _hvp) converges to exact torch.autograd.functional.hessian+eigh ground truth (<1e-9 rel) and matches official PyHessian.eigenvalues(top_n=1) to ~2e-6; tests/tasks/parity/test_sharpness_parity.py."),
    "dynamics_stability": _entry("proxy-only", "BLME embedding-neighborhood stability diagnostic"),
    # Representation engineering
    "repe_concept_separability": _entry("refined-adaptation", "Zou et al. 2023", "andyzoujm/representation-engineering"),
    "repe_refusal_direction": _entry("refined-adaptation", "Arditi et al. 2024; Zou et al. 2023", "andyrdt/refusal_direction", "Refusal direction = difference-of-means (harmful - harmless) == official andyrdt/refusal_direction get_mean_diff exactly (0.0); |cos|=1 (tests/tasks/parity/test_repe_refusal_direction_parity.py)."),
    "repe_steering_effectiveness": _entry("proxy-only", "Zou et al. 2023; Turner et al. 2023"),
    "repe_task_vectors": _entry("refined-adaptation", "Zou et al. 2023; Ilharco et al. 2023", "andyzoujm/representation-engineering", "Reading-vector direction = class mean-difference == official RepE ClusterMeanRepReader to 2.2e-16; |cos|=1 (tests/tasks/parity/test_repe_task_vectors_parity.py)."),
    # --- Campaign-2 additions (2026-06): new methods, official-code parity-verified ---
    "geometry_vendi_score": _entry("parity-ready", "Friedman & Dieng 2023 (TMLR)", "vertaix/Vendi-Score", "exp(Shannon entropy of kernel-matrix eigenvalues), effective number of distinct samples; nonlinear (cosine/RBF) kernel so it is distinct from effective_rank/RankMe. Bit-exact vs official vendi_score.vendi.score_K on the same kernel; tests/tasks/parity/test_vendi_parity.py."),
    "geometry_phd_dimension": _entry("parity-ready", "Tulchinskii et al. 2023 (NeurIPS)", "ArGintum/GPTID", "Persistent-homology dimension from the MST-length (total H0 persistence) power-law E(n)~n^((d-1)/d). Seed-matched bit-exact vs GPTID PHD; stochastic estimator, RNG-pinned; tests/tasks/parity/test_phd_dimension_parity.py."),
    "geometry_cknna": _entry("parity-ready", "Huh et al. 2024 (ICML, Platonic Representation Hypothesis)", "minyoungg/platonic-rep", "Mutual k-NN conditional CKA (local-neighborhood alignment). Bit-exact vs platonic-rep metrics.cknna (unbiased+biased HSIC, both orderings); tests/tasks/parity/test_cknna_parity.py."),
    "geometry_magnitude": _entry("parity-ready", "Limbeck et al. 2024 (NeurIPS)", "aidos-lab/magnipy", "Metric-space magnitude |tX| = 1^T zeta^-1 1 (zeta_ij = exp(-t d_ij)) across scales + magnitude dimension. Bit-exact vs magnipy cholesky/pinv paths; tests/tasks/parity/test_magnitude_parity.py."),
    "geometry_procrustes_linearity": _entry("parity-ready", "Razzhigaev et al. 2024 (ACL, 'Your Transformer is Secretly Linear')", "AIRI-Institute/LLM-Microscope", "Per-adjacent-layer linear (pseudo-inverse) Procrustes similarity; bit-exact port of llm_microscope.procrustes_similarity. NOTE: the absolute value is conditioning-dependent (unguarded 1/S amplifies ~1e-17 singular values) — interpret the across-depth PROFILE, not a single value; tests/tasks/parity/test_procrustes_linearity_parity.py."),
    "interpretability_activation_kurtosis": _entry("parity-ready", "Akhondzadeh et al. 2025 (KurTail, EMNLP Findings); Sun et al. 2024 (Massive Activations)", "scipy.stats.kurtosis; locuslab/massive-activations", "Per-channel excess (Fisher) kurtosis of activations (outlier/quantizability signal). Bit-exact vs scipy.stats.kurtosis(fisher=True, bias=True); tests/tasks/parity/test_activation_kurtosis_parity.py."),
    "interpretability_attention_rollout": _entry("parity-ready", "Abnar & Zuidema 2020 (ACL, arXiv:2005.00928)", "samiraabnar/attention_flow", "Attention rollout = cumulative product of normalize(A_l + I) over head-averaged per-layer attention. Bit-exact (0.0) vs samiraabnar/attention_flow compute_joint_attention; tests/tasks/parity/test_attention_rollout_parity.py. Added 2026-07 to implement the rollout that attention_graph only cited."),
    "topology_zigzag_persistence": _entry("refined-adaptation", "Gardinazzi et al. 2025 (ICML, arXiv:2410.11042)", "RitAreaSciencePark/ZigZagLLMs", "Across-layer topological-persistence summary (ripser, existing dep). FAITHFUL PROXY: feature layer-lifetimes anchored EXACTLY to the paper's dionysus zigzag engine on ground-truth constructions (loop in a contiguous layer band -> one zigzag bar), not a bit-exact barcode port of raw short-bar multiplicity; tests/tasks/parity/test_zigzag_parity.py."),
}


def get_task_certification(task_name: str) -> TaskCertification:
    return TASK_CERTIFICATION[task_name]


def task_metadata_dict(task_name: str) -> dict:
    return get_task_certification(task_name).to_dict()


def validate_certification_coverage(task_names: List[str]) -> tuple[list[str], list[str]]:
    expected = set(task_names)
    actual = set(TASK_CERTIFICATION)
    return sorted(expected - actual), sorted(actual - expected)

