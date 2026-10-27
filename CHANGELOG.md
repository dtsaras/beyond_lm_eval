# Changelog

## Unreleased

### Changed
- **Python ≥ 3.10 is now required.** numpy 2.x and transformers 5.x need it, and the package evaluates PEP 604 annotations (`bool | None`) at import time. The CI matrix is now 3.10–3.12.
- **scikit-learn is now a core dependency**, because five paper-core tasks need it.
- **The default corpus changed** (`load_default_corpus`, used when no dataset is passed). Numbers move for runs that relied on it; runs that pass their own dataset are unaffected.
  - It loads `Salesforce/wikitext` (`wikitext-103-raw-v1`, validation) and keeps passages of at least 64 characters after stripping, returned stripped. The trailing newline of every WikiText row is a massive-activation sink on pythia and TinyLlama.
  - `diverse=True` is now the default: draws follow one fixed van der Corput order, so every draw is spread across the split and `load_default_corpus(n)` is a prefix of `load_default_corpus(m)` for `n < m`. `diverse=False` gives the split-order prefix.
  - Requesting more passages than exist (1691) raises `ValueError`, and a dataset that cannot be loaded raises `RuntimeError`. The three hardcoded fallback sentences are gone. `evaluate()` caps its corpus request at the corpus size, because some `num_samples` count vocabulary rows.
  - New `load_long_documents(n, min_tokens, tokenizer)` returns whole articles from `EleutherAI/wikitext_document_level` in the same nested order.

### Fixed
- **bfloat16 storage no longer crashes.** numpy has no bfloat16 dtype, so `.numpy()` raised `TypeError` on bf16 hidden states in three places:
  - disk offload in `collect_hidden_states` (`BLME_DISK_OFFLOAD=1`), which now stores the bf16 bit pattern and reads it back as bf16, bit for bit;
  - the self-collect routes of `geometry_lipschitz` and `geometry_rsa`, which now widen bf16 exactly to float32 (new helper `geometry.utils.as_numpy`), so they match the cache route bit for bit.

  Shipped and study numbers are unchanged: those paths raised before.
- **`geometry_phd_dimension` with `pooling: mean`** no longer raises `AttributeError` (`.double()` was called on numpy arrays). Document means are now taken in float64 on all three routes. Shipped and study configs use `pooling: all_tokens`, so no shipped number moves.
- **`geometry_hubness`**: a repeated entry in `k_values` doubled every occurrence count for that k. Duplicates are now dropped in order and counted in `_meta_k_values_duplicates_dropped`. Non-positive k is rejected. The default `[10, 50, 100]` is unaffected.
- **`repe_refusal_direction`** returns an error when scikit-learn is missing instead of raising `NameError`, as `repe_concept_separability` already did.
