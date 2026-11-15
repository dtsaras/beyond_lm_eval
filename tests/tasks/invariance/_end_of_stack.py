"""What an identity block at either end of the stack does to every key declared
invariant under identity blocks.

``identity_blocks_interior`` declares a key unchanged by exact identity blocks
inserted BETWEEN existing blocks: each one duplicates a hidden state the model
already has. A block at an end of the stack is just as exact on the logits but
adds a state (``insert_end_identity_block``):

  leading   before block 0. Block 0's output becomes the embedding output, a
            new state for any profile that starts at block 0. A profile that
            already holds the embedding only gains a duplicate.
  trailing  after the last block. HF's last entry splits into the raw
            last-block output and its final norm, a new state for any profile
            that ends in the post-norm entry. A profile that reads the raw last
            block (replace_final_norm) only gains a duplicate.

``OUTCOMES[task][key] = (leading, trailing)``, measured through each task's
real ``evaluate()`` by ``test_end_of_stack_identity_blocks.py`` on the tiny
gpt2 and llama of ``_family_d`` (``llama4``: the 4-block llama, seed 3), each
with its final-norm gain made non-uniform by ``final_norm_unembedding_gauge``,
as a trained model's is (a unit gain makes the final norm a per-token rescale
that scale-free keys cannot see). Each outcome is one of:

  INVARIANT         structural: the end block only duplicates a state the key
                    reads, or the key does not read the hidden-state profile.
                    Bit-identical on gpt2 and llama, and declared as
                    ``identity_blocks_leading`` / ``identity_blocks_trailing``.
  moves(*fixtures)  a known negative: moves by more than ``MIN_REL`` on each
                    listed fixture. Comments give the measured change.
  unchanged(w)      bit-identical on gpt2 and llama only because the state the
                    end block adds is not the extreme the key takes; ``w`` is
                    a witness that the state was added (a profile that gains a
                    value, or a key that moves). Not declared.
  NAN               NaN on both sides on the tiny models. Not declared.
"""
from __future__ import annotations

INVARIANT = "invariant"
NAN = "nan"
MIN_REL = 1e-3


def moves(*fixtures):
    return ("moves",) + fixtures


def unchanged(witness):
    return ("unchanged", witness)


I = INVARIANT
BOTH = moves("gpt2", "llama")

OUTCOMES = {
    # Skipping an identity block changes nothing; a real block's skip is unchanged.
    "causality_layer_skip": {
        "skip_output_effect_max": (I, I),
        "skip_output_effect_total": (I, I),
    },
    # Output distribution only.
    "consistency_icl_score": {
        "icl_score": (I, I),
        "icl_score_byte_anchored": (I, I),
    },
    # The chain holds the embedding (leading: a zero step) and ends post-norm
    # (trailing: the last step splits). Trailing: gpt2 -1.75 -> -1.96, llama
    # -0.69 -> -1.12.
    "dynamics_coe": {
        "coe_r_sum": (I, BOTH),
    },
    "dynamics_entropy_calibration": {
        "entropy_calibration_error": (I, I),
        "entropy_calibration_ratio": (I, I),
    },
    # End-to-end ratios from the embedding to the final residual.
    "dynamics_gradient_flow": {
        "gradient_flow_log_ratio_in_out": (I, I),
    },
    "dynamics_interpolation": {
        "convexity_gap": (I, I),
    },
    "dynamics_perturbation_growth": {
        "growth_log_ratio": (I, I),
    },
    # The post-norm output cloud; its 32 eigenvalues are too few for alpha.
    "geometry_alpha_req": {
        "alpha_req": (NAN, NAN),
        "rankme": (I, I),
    },
    # Raw-last-block profile without the embedding. Leading: early/offdiag
    # gpt2 -12%, llama -4.2%; adjacent gpt2 -3.4% (llama unchanged).
    "geometry_cka": {
        "early_late_cka": (BOTH, I),
        "early_late_cka_debiased": (BOTH, I),
        "min_adjacent_cka": (moves("gpt2"), I),
        "min_adjacent_cka_debiased": (moves("gpt2"), I),
        "min_offdiag_cka": (BOTH, I),
        "min_offdiag_cka_debiased": (BOTH, I),
    },
    # Leading: max_erank and the ratio gpt2 +4.9% / -4.6% (llama 5e-5).
    "geometry_collapse": {
        "collapse_ratio": (moves("gpt2"), I),
        "max_erank": (moves("gpt2"), I),
        "min_erank": (unchanged("_meta_erank_per_layer"), I),
        "qdl_entropy_min": (unchanged("_meta_qdl_entropy_per_layer"), I),
    },
    "geometry_correlation_dimension": {
        "correlation_dimension": (I, I),
    },
    # Leading: min gpt2 -6.8%, llama -4.0%; trailing: llama4 -1.2%.
    "geometry_delta_hyperbolicity": {
        "delta_rel_input_embedding": (I, I),
        "delta_rel_last_layer": (I, I),
        "delta_rel_min": (BOTH, moves("llama4")),
    },
    # The profile holds the embedding and ends post-norm. Trailing: llama +24%.
    "geometry_dime": {
        "dime_last_layer": (I, I),
        "dime_peak": (I, moves("llama")),
    },
    # An identity block's influence is 0.
    "geometry_effective_depth": {
        "max_block_influence": (I, I),
        "total_block_influence": (I, I),
    },
    # Leading: peak gpt2 +1.9%, llama +19%; trailing: llama +12%.
    "geometry_gride": {
        "gride_id_last_layer": (I, I),
        "gride_id_peak": (BOTH, moves("llama")),
    },
    # As geometry_cka.
    "geometry_hsic": {
        "hsic_compression_ratio": (BOTH, I),
        "hsic_compression_ratio_debiased": (BOTH, I),
        "min_adjacent_hsic": (moves("gpt2"), I),
        "min_adjacent_hsic_debiased": (moves("gpt2"), I),
    },
    # Leading makes the first layer the embedding: gpt2 +37% / +45%, llama
    # +190% / +140%.
    "geometry_information_imbalance": {
        "ii_asymmetry_first_last": (BOTH, I),
        "ii_first_to_last": (BOTH, I),
        "ii_last_to_first": (BOTH, I),
    },
    # Leading: gpt2 6.1 -> 16.3; llama -> NaN (repeated tokens coincide).
    "geometry_intrinsic_dim": {
        "lid_first_layer": (BOTH, I),
        "lid_last_layer": (I, I),
    },
    # Single-layer keys read the post-norm output. Leading first layer: gpt2
    # -16%, llama 6.6 -> 0.
    "geometry_lid": {
        "lid_first_layer": (BOTH, I),
        "lid_max": (I, I),
        "lid_mean": (I, I),
        "lid_median": (I, I),
        "lid_std": (I, I),
    },
    # The path starts at block 0 (leading adds the embedding -> block 0 step):
    # gpt2 +34% / +31% / +14%, llama +63% / +52% / +4.7%.
    "geometry_lipschitz": {
        "e2e_log_norm_growth": (BOTH, I),
        "relative_change_max": (BOTH, I),
        "total_angular_path": (BOTH, I),
    },
    "geometry_lm_head_free_energy": {
        "lm_head_bfe_w": (I, I),
    },
    "geometry_magnitude": {
        "magnitude_area_tconv": (I, I),
        "magnitude_dimension_tconv": (I, I),
    },
    "geometry_matrix_entropy": {
        "matrix_entropy_raw": (I, I),
    },
    # Leading: first_last moves on both; on 20-word windows the added
    # embedding-to-block-0 pair is not the adjacent minimum on either fixture.
    # Trailing (adjacent pair raw/post-norm): gpt2 +6.9%, llama +1.3%.
    "geometry_neighborhood_overlap": {
        "nn_overlap_adjacent_min": (unchanged("nn_overlap_adjacent"), BOTH),
        "nn_overlap_first_last": (BOTH, I),
    },
    # An identity block writes nothing, so it has weight 0.
    "geometry_positional_decay": {
        "positional_decay_nosink_write_weighted": (I, I),
    },
    # Leading: gpt2 +2.6%; trailing: llama +0.41%.
    "geometry_predictive_information": {
        "predictive_info_last_layer": (I, I),
        "predictive_info_peak": (moves("gpt2"), moves("llama")),
    },
    # Pairs from the embedding to the raw last block: an end block makes the
    # first (leading) or last (trailing) pair a duplicate, scored 1. Leading
    # gpt2 +1.6%, llama +4.6%; trailing gpt2 +0.8%, llama +2.9%.
    "geometry_procrustes_linearity": {
        "procrustes_linearity_adjusted_first": (BOTH, I),
        "procrustes_linearity_adjusted_last": (I, BOTH),
        "procrustes_linearity_adjusted_min": (I, I),
        "procrustes_linearity_first": (BOTH, I),
        "procrustes_linearity_last": (I, BOTH),
        "procrustes_linearity_min": (I, I),
    },
    # Leading: peak gpt2 +9.1%, llama x5.5; trailing: llama4 +4.1%.
    "geometry_prompt_id": {
        "prompt_id_last_layer": (I, I),
        "prompt_id_peak": (BOTH, moves("llama4")),
    },
    "geometry_representation_dispersion": {
        "representation_dispersion": (I, I),
    },
    # The profile holds the embedding and ends post-norm. Trailing: llama4 +18%.
    "geometry_rogue_dimensions": {
        "rogue_top1_share_last": (I, I),
        "rogue_top1_share_max": (I, moves("llama4")),
        "rogue_top3_share_max": (I, unchanged("rogue_top3_share_profile")),
    },
    # Leading: early_late gpt2 -15%, llama -5.9%; adjacent gpt2 -5.7%.
    "geometry_rsa": {
        "rsa_adjacent_min_corrdist": (moves("gpt2"), I),
        "rsa_early_late_corrdist": (BOTH, I),
    },
    # Leading: first layer llama +0.23% (gpt2 +0.05%); ratio gpt2 +6.1%, llama x2.
    "geometry_trajectory_curvature": {
        "curvature_mean_first_layer": (moves("llama"), I),
        "curvature_mean_last_layer": (I, I),
        "straightening_ratio": (BOTH, I),
    },
    # Leading: first layer gpt2 +1.3%, llama -12%.
    "geometry_vendi_score": {
        "vendi_max": (unchanged("vendi_mean_first_layer"), I),
        "vendi_mean_first_layer": (BOTH, I),
        "vendi_mean_last_layer": (I, I),
    },
    "geometry_zipfian_symmetry": {
        "zipf_sym1_embedding": (I, I),
        "zipf_sym1_unembedding": (I, I),
        "zipf_sym2_embedding": (I, I),
        "zipf_sym2_unembedding": (I, I),
    },
    # Block outputs (raw last block): leading adds the embedding's decode,
    # gpt2 +38%, llama +39%.
    "interpretability_logit_lens_kl": {
        "kl_churn": (BOTH, I),
    },
    "topology_betti_curve": {
        "h1_max_persistence": (unchanged("h1_max_persistence_curve"), I),
    },
}
