# Opus verdicts invalidated by the red-team pass, 2026-09-05

Two RED-TEAM reviewers were asked to BREAK the Opus GREENs on the 13 tasks
awaiting a codex sign-off. They found two defects that six prior rounds had
missed, plus two stale claims. The four verdicts here are void because their
task source changed afterwards.

* `geometry_schatten` — the shipped `max_length: 128` was DEAD whenever a
  shared cache existed, because `use_cache` is absent from its config (so the
  code default True fires) and the shared cache tokenises at a hard-coded 512.
  EVERY ARCHIVED SCHATTEN NUMBER was computed at 512 tokens, not 128:
  `rankme_last` +159.1% on gpt2 and +175.7% on pythia-70m, `schatten_1_last`
  +133.4%. Nothing in the output recorded the window.
* `geometry_correlation_dimension` — the atomic branch overwrote `slope` and
  `r_squared`, which are exported as `_meta_correlation_dimension_percentile_
  window`, so a key named for the percentile window shipped the PLATEAU
  headline: 6.604087299227721 against a true 1.784666709122511 (+270%), R^2 NaN
  where it was 0.9034. Reachable with ONE repeated document, and it breaks this
  repo's own assertion at `test_comprehensive_parity.py:3188`, whose fixtures
  cannot enter the branch. `num_radii` was also bit-identically inert on the
  promoted kernel.
* `geometry_spectral` — `_meta_n_modules_excluded_unembedding` reported 1 while
  ZERO modules are excluded; renamed to `_meta_n_head_modules_tagged`.
* `topology_zigzag_persistence` — the comment justifying the
  `zz_h1_survival_rate` demotion claimed it was BIT-IDENTICAL to
  `zz_true_adjacent_mean`. That was measured before the footnote-6 fix; they
  now differ 0.5916666666666666 vs 0.5378787878787878 on gpt2 (9.1%).
