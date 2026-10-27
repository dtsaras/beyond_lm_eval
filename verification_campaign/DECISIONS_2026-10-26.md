# Decisions needed from the user (one batch) — 2026-10-26

These block parts of Phases 3–4 of the hardening plan
(`~/.claude/plans/i-previously-asked-another-tender-brooks.md`). Each item
carries a recommended default, so an approval can be given by number. Nothing
here has been implemented yet.

| # | Question | Evidence | Recommended default |
|---|---|---|---|
| 1 | `dynamics_coe` degenerate-chain floor (PENDING §30): keep BLME's floor, or match the released `score.py` bit-for-bit? | On `H=[(1,0),(1e13,1e13),(0,1)]` the release returns ~1e13 from a denominator below float32 resolution of the coordinates; every commensurate floor rejects it. Margin on 300 real chains: 11.98 orders, so no shipped number moves either way. | **Keep the floor**, and disclose the divergence in `docs/VALIDATION.md` |
| 2 | Keys that cannot discriminate between models (11 of 386), incl. `topology_persistence_entropy`'s whole H0 family = log2(N−1)·0.994: demote, normalise, or keep? | Sweep 7 in `OPUS_ROUND13_FINDINGS.md`: identical on gpt2 and distilgpt2 for structural reasons. | **Demote to `_meta_`** in the same schema change as the config echoes (readers patched together). Additionally emit H0 / log2(N−1) only if it then discriminates on the 3-model panel. |
| 3a | Position-0 exception — `geometry_collapse`: its source (Queipo-de-Llano 2025) shows the BOS massive activation CAUSES the compression valley. Excluding position 0 removes the paper's phenomenon. | Measured on gpt2 L6: position-0 rows = 96.3% of the centred variance. | **Keep the inclusive headline for collapse** (paper protocol), and emit the excluded value as `_meta_` |
| 3b | Position-0 exception — `geometry_positional_decay`: it already emits a sink-excluded `_meta_..._nosink`. Which value is the headline? | The headline currently mixes distance decay with sink mass. | **Headline = nosink** (non-LLM decay premise), with the inclusive value kept as `_meta_` |
| 4a | `topology_zigzag_persistence` mean-pools, but Gardinazzi 2025 uses the LAST token. Align to the paper? | Disclosed at `zigzag.py:66-69`. | **Align to last token** (paper protocol), keep mean-pool as `_meta_` |
| 4b | `repe_concept_separability` mean-pools, but RepE uses `rep_token=-1`. Align? | `representation_engineering.py:598` vs `:640`. | **Align to last token**, keep mean-pool as `_meta_` |
| 5 | Corpus loader details for the approved diverse default. | 1) `diverse(N)` draws are not nested: diverse(100) shares 20 of 100 docs with diverse(500)[:100]. 2) The library filters ≥50 characters and the study ≥64, so 353 of the first 500 passages differ. 3) Offline, the `wikitext` id silently falls back to 3 hardcoded sentences, while `Salesforce/wikitext` resolves. | **One fixed evenly-spaced ordering sliced by prefix** (draws nest); a **single 64-character filter** shared by library and study; the `Salesforce/wikitext` id; and an **error** instead of the silent fallback |
| 6 | Delimiter tokens may also be massive-activation sinks (Sun et al. 2024). Is a position-0 rule enough? | **Measured 2026-10-26** (60 evenly spaced WikiText docs × 128 tokens, bf16, layers L/4, L/2, 3L/4; script and raw output `scratchpad/sink_census.py` → `sink_census.jsonl`, copied to `~/hara-artifacts/blme/sink_census_2026-10-26.jsonl`). Position 0 (1% of rows) has median norm ×35–222 and carries **91–99% of the centred sum of squares** on both Qwen3-4B and Qwen3-8B. Delimiters (first or later) sit at ×0.9–1.0 and carry ≤0.3%. Other >10× outliers: none in Qwen3-8B; in Qwen3-4B, 8 rows (a bare-space token and ' well') carry ~6% at L18/L27. gpt2 L6: position 0 = 96.3%. Not measured: a BOS-adding model (Llama-3.1-8B is not cached on pod1). | **Position-0 rule only; no delimiter rule.** Re-check on one BOS model (Llama/Gemma) once it can be downloaded, and report the residual Qwen3-4B outlier share as `_meta_` rather than excluding by norm |
| 7 | fp32 promotion for `dynamics_gradient_flow` / `perturbation_growth` on the 24–34B study models (fp32 params ≈ 136 GB before gradients). | `sharpness.py:440-450` promotes every parameter. | **Promote only the analysis tensors** (loss, logits, collected activations), with no size gate, and disclose the result in `_meta_` |

---

# Resolution — 2026-10-28 (user's method-fidelity rule applied)

The user's rule, verbatim in substance:

1. **A method with a proposing paper or a published GitHub implementation runs
   exactly as that paper/repo does** (token selection, layer, pooling,
   estimator, hyper-parameters, edge cases).
2. **A method adapted to LLMs** does what is mathematically sound *and* what
   other LLM papers applying it commonly do (cite the precedent).
3. **A concept with no clear LLM metric** (BLME construct) is allowed, but the
   reason for choosing this adaptation over the alternatives is written down
   next to the task.
4. **Models of different sizes are compared**, so each headline needs a
   version that does not depend mechanically on width, depth, sample count,
   vocabulary or parameter count.

Mapping onto `task_metadata.py`: `parity-ready` / `formula-faithful` → rule 1
(25 tasks); `refined-adaptation` → rule 2 (15); `proxy-only` → rule 3 (11).

| # | Resolution | Rule |
|---|---|---|
| 1 | **Follow the released `score.py`.** The headline divides by Z_Mag as the release does; NaN only where the release itself would produce a non-finite value. Emit a derived `_meta_` margin (`log10(Z_Mag / (eps_storage · chain_max))`) so a chain whose normaliser sits below storage resolution is visible instead of silently filtered. No shipped number moves (11.98 orders of margin on 300 real chains). | 1 |
| 2 | **Normalise, then demote what is still constant.** H0 persistence entropy: use the reference's own normalisation if giotto-tda / Atienza et al. define one, otherwise E/log2(N−1); raw goes to `_meta_`. The other 10 keys are decided per key in the methods dossier. | 1, 4 |
| 3a | **collapse: per-key policy.** Keys from Queipo-de-Llano 2025 (the BOS sink is the phenomenon) keep position 0, as the paper does; keys from Jing 2021 / Roy–Vetterli 2007 (non-LLM erank/spectrum) exclude it. | 1, 2 |
| 3b | **positional_decay: sink-excluded headline**, with the inclusive value and sink mass in `_meta_`. Rationale note: RoPE's decay premise concerns relative distance, whereas the sink is distance-independent mass parked on position 0. The 2026-09-04 identity-block finding is a **depth-aggregation** defect that affects both twins; the fix goes through the size-normalisation work, not through keeping the sink. | 3, 4 |
| 4a | **zigzag: last token**, as in Gardinazzi 2025 / `ZigZagLLMs` (to be confirmed against the repo); mean-pool goes to `_meta_`. | 1 |
| 4b | **concept_separability: `rep_token=-1`**, as in RepE; mean-pool goes to `_meta_`. | 1 |
| 5 | **Recommended default approved**: nested ordering, one 64-character filter, `Salesforce/wikitext`, error instead of fallback. | — |
| 6 | **Position-0 rule only**, matching the massive-activation and attention-sink literature. Re-measure on a BOS-adding model (internet now works from pod1: download an ungated one such as TinyLlama). | 2 |
| 7 | **Promote the analysis tensors only**, with the effective dtypes disclosed in `_meta_`. | — |

**Normalisation (rule 4) — operational definition.** "Independent of
dimension" means the *mechanism test* already used by the 2026-09 campaign
(`src/blme/normalization.py` header): the value must not move under a
transformation that leaves the geometry or function fixed but changes the
size variable. That covers isometric embedding or zero-padding (width d), a
function-preserving re-decomposition or identity-block insertion (depth L),
sample count N at a fixed distribution, and vocabulary padding with unused
rows (V). Per headline key, there are three outcomes:
(a) already invariant, so no twin is needed;
(b) a known structural factor, so emit the exact inverse (the precedent is IsoScore's `iso·(d−1)+1`);
(c) size-linked by meaning (loss, perplexity), so disclose and do not normalise.

A "fraction of capacity used" (erank/d) is a different quantity: it is
reported only under its own name. Normalised variants are emitted **by the
task itself**, declared in `task_metadata.py` and pinned by the invariance
test. `normalization.py`'s divide-by-arch registry is empty and moves to
beyond_benchmarks with `is_feature`.
