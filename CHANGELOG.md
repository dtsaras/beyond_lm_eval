# Changelog

## Unreleased

### Fixed
- **bfloat16 storage no longer crashes.** numpy has no bfloat16 dtype, so `.numpy()` raised `TypeError` on bf16 hidden states in three places:
  - disk offload in `collect_hidden_states` (`BLME_DISK_OFFLOAD=1`), which now stores the bf16 bit pattern and reads it back as bf16, bit for bit;
  - the self-collect routes of `geometry_lipschitz` and `geometry_rsa`, which now widen bf16 exactly to float32 (new helper `geometry.utils.as_numpy`), so they match the cache route bit for bit.

  Shipped and study numbers are unchanged: those paths raised before.
- **`geometry_phd_dimension` with `pooling: mean`** no longer raises `AttributeError` (`.double()` was called on numpy arrays). Document means are now taken in float64 on all three routes. Shipped and study configs use `pooling: all_tokens`, so no shipped number moves.
- **`geometry_hubness`**: a repeated entry in `k_values` doubled every occurrence count for that k. Duplicates are now dropped in order and counted in `_meta_k_values_duplicates_dropped`. Non-positive k is rejected. The default `[10, 50, 100]` is unaffected.
- **`repe_refusal_direction`** returns an error when scikit-learn is missing instead of raising `NameError`, as `repe_concept_separability` already did.
