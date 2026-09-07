# Round-3 codex verdicts, quarantined 2026-09-04

5 GREEN / 12 YELLOW / 11 RED. The rise in REDs against round 2 is not a
regression: round 3 GENERALISED round 2's finding -- that removing the
absolute epsilons made the formulas scale-invariant while the arithmetic
still was not -- to every remaining norm in the repo, probing at 2^-80,
1e-50, 1e-109 and 2.8e38. Nearly every RED is that one class in a new
place, and every one was then fixed (see RERUN_LEDGER.md, 'Round-4
corrections').

One RED was a genuine defect introduced by the round-3 fixes themselves:
`correlation_dimension`'s atomic branch was recomputed and OVERWRITTEN by
a later block, so the fix was inert in production.

cknna, procrustes_linearity, svd, magnitude and generation_diversity
reached GREEN here.
