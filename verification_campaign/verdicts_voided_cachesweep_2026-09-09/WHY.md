# Why these verdicts are void

The 2026-09-09 execution sweep of the shared-cache class fixed 15 sites
(a read-only sweep had reported 2, then 5). Those fixes plus the round-14
repairs to geometry_spectral, geometry_perplexity and geometry_intrinsic_dim
touched `geometry/utils.py`, `tasks/common.py` and ten task modules.

Under the Pink-Ponk protocol a fix voids BOTH reviewers' verdicts for every
task whose source it touches. Two of the voided tasks -- geometry_collapse and
geometry_lid -- had been DUAL-SIGNED while carrying a live cache defect that an
executed check found in minutes.

Full suite on the repaired tree: 2635 passed, 158 skipped, 0 failed (1:18:03).
