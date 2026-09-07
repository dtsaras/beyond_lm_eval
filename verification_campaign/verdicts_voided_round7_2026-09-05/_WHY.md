# Voided in round 7 — 2026-09-05, after the Opus re-reviews

These 13 tasks were changed **in response to the round-7 reviews themselves**, so
their fresh verdicts describe a code state that no longer exists. Under
Pink–Ponk that voids them, however recent they are.

The two that matter most are the ones the reviews caught me on:

* **`geometry_matrix_entropy` — RED against my own sweep.** Wei et al.'s released
  recipe is `AutoModel(...)[0]`, bit-identical to `hidden_states[-1]`, so the
  headline is a single-layer estimand the reference defines as POST-norm.
  Hooking it moved gpt2 +93.3% off the reference and turned a published
  70m-vs-160m gap from +6.3% into +105.5%. The flag is now split: profile
  hooked, headline not.
* **`geometry_contextualization` — YELLOW, same class.** `kawine/contextual`'s
  vendored GPT-2 literally does `all_hidden_states.pop()  # ignore the last
  layer prior to layer norm` and appends `ln_f(...)`. The reference defines its
  deepest layer as post-norm. Hook reverted.

Both taught the sweep a third clause it did not have: **the released reference
wins.** It is now in the guard's own docstring.

The other eleven are column demotions and one real bug fix (`perturbation_growth`
was still leaking a forward hook on the EXCEPTION path — `_forward_hooks` 0 → 1
after a raise — now wrapped in try/finally and covered by
`tests/tasks/test_forward_hook_leaks.py`).

## A note on precision

Four tasks share `representation_engineering.py`, so a change for one of them
moves the file for all four. Only `repe_task_vectors` was changed here;
`repe_refusal_direction`, `repe_steering_effectiveness` and
`repe_concept_separability` were left alone and their verdicts stand.

`verification_campaign/source_hash.py` now exists so this stops being a
judgement call: it hashes each task's AST with comments and docstrings stripped,
stamps the hash into every verdict, and reports which verdicts no longer match.
A `post-final-norm:` marker added to satisfy a test no longer looks like a
semantic change, and a real one can no longer hide behind one.

---

## Second wave — 2026-09-05, after the round-7 reviews returned

Ten more tasks voided, because the round-7 reviews found defects **in the
round-7 fixes** and I acted on them:

* `dynamics_coe` — YELLOW. `Alsace08/Chain-of-Embedding`'s `inference.py`
  builds the chain from the raw HF tuple, all L+1 entries. The hook flipped the
  headline's SIGN on 3/3 models. Reverted; the substituted value is disclosed as
  `_meta_coe_r_blockout`.
* `geometry_matrix_entropy` — still leaked a forward hook on the exception path
  after its RED was fixed. Wrapped.
* `geometry_lid` — the round-7 fix ADDED an exact duplicate (`lid_last_layer`
  == `lid_mean` bit-for-bit on 10/10). Removed; `_meta_lid_mode` added because
  the layerwise and single-layer modes ship different estimands under one name
  (1,271x apart on pythia-160m).
* `geometry_contextualization` — `layers_reported` was `np.int64`, which strict
  JSON encoders reject.
* `geometry_categories` — four `<cat>_analogy_acc_top{1,5}` columns were
  reaching the feature matrix. Word-analogy accuracy is a CAPABILITY measure,
  i.e. what the study predicts.
* `geometry_schatten`, `topology_homology`, `repe_concept_separability`,
  `repe_refusal_direction`, `repe_steering_effectiveness` — forward-hook leaks
  on the exception path, found by a leak test that now DISCOVERS the tasks that
  hook rather than naming two by hand. Ten tasks leaked; all ten are wrapped.

`repe_steering_effectiveness` and `repe_refusal_direction` are voided only
because they share `representation_engineering.py` with tasks that changed;
neither's own code was touched in this wave.

## The freeze

The code is frozen at this point. Each round of fixes has voided that round's
verdicts, which is correct under Pink–Ponk and also means the campaign cannot
converge while editing continues. Nothing further changes until a review runs
against this state.
