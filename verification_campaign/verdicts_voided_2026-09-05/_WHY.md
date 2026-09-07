# Why these 51 verdicts were voided — 2026-09-05

Under the Pink–Ponk protocol a fix voids BOTH prior verdicts: a reviewer signed
off on a code state, and that state no longer exists. 26 tasks changed today.

The trigger was three independent red-team passes over the tasks that carried a
dual GREEN from round 1 and had **never been re-opened**. All three broke the
"ready to publish" claim, and between them they found the campaign's most-repeated
defect in **five more places**. That prompted a repo-wide count instead of another
spot-check, and the count is the reason this list is long:

> **24 of 39** `get_hidden_states`/`collect_hidden_states` call sites had never
> passed `replace_final_norm`, plus ~17 direct `hidden_states[...]` reads with no
> stated intent.

Six review rounds had been sampling a couple of sites at a time out of that
population. The class was never fixed, only locally patched, which is exactly why
it kept "recurring".

`tests/test_post_final_norm_declaration.py` now makes the default illegal: every
call site must pass `replace_final_norm` explicitly, and every direct index must
carry a `post-final-norm:` note saying which representation it wants. The class
cannot silently reappear.

## What changed per task

See `RERUN_LEDGER.md` for the measurements. In brief:

| task | change |
|---|---|
| geometry_trajectory_curvature | hooked; straightening_ratio +66.6% (pythia-70m) |
| geometry_contextualization | hooked; gpt2 self_sim_corrected +1513%, pythia-160m MEV sign flip |
| geometry_intrinsic_dim | hooked; gpt2 lid_layer_L +27.0%; its final-layer ID expansion was the LayerNorm |
| geometry_cka / cknna | hooked; min_offdiag_cka +313% (gpt2) / +1114% (distilgpt2) |
| topology_homology | hooked; mean_persistence_h0 +562% (gpt2), −69% (pythia-70m); depth-aligned column names added |
| geometry_matrix_entropy | fallback route hooked so it matches the cache route |
| geometry_rmt_bulk, interpretability activation_kurtosis, topology_zigzag, repe probe sweep | hooked |
| geometry_vendi_score | hooked; also refuses an undersized shared cache (slope SIGN FLIP across co-runner sets) |
| geometry_isoscore / lid / magnitude / phd_dimension | post-norm DECLARED with reasons (single-layer estimands) |
| geometry_mahalanobis | fit/score split was a TOPIC SPLIT; interleaved. gpt2 mahalanobis_sq_per_dim_id −63% |
| geometry_superposition_strength | MSO returned NaN below scale 1e-25 and NEGATIVE −1/(V−1) above 1e19 |
| geometry_unembedding | `UnboundLocalError` killed the whole task on any SVD failure |
| dynamics_interpolation | +26.5% co-runner coupling; `seed` was dead; duplicate columns demoted |
| dynamics_fractal | study config re-overrode the fixed `holder_eps`; headline was −92.3% |
| dynamics_perturbation_growth | rel_eps below bf16 resolution; leaked forward hook on early return |
| geometry_phd_dimension | per-document layer index off by one vs the pooled path |
| repe_refusal_direction | the fix's own justification number was wrong (in-sample 0.9583 vs shipped CV 0.625) |
| geometry_tokenizer_efficiency, weight_norms, repe_task_vectors, categories | architecture-proxy / exactly-collinear columns demoted to `_meta_` |
