# Round-8 codex verdicts quarantined — 2026-09-07

All 50 describe the PRE-FIX tree. Since they were written, 63 source files
changed: five confirmed fidelity fixes (four of them from these very verdicts),
seven tasks reverted to the raw hidden-states tuple after their references were
read, a forward-hook leak in `cache.py` closed, and two provenance flags that
reported the opposite of the truth corrected.

They are kept, not deleted: four of the REDs here were adjudicated correct by
direct measurement and drove the fixes, and the six still-open ones are carried
forward into the round-9 prompt so they are not lost.
