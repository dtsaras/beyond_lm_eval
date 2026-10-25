# Why these 102 verdicts are void

Round 12 (gpt-6-astra, max effort) reviewed 51 of 51 tasks and returned
21 GREEN / 20 YELLOW / 10 RED. Every one of the 10 REDs was
`defect_location: CODE`, and all 10 were fixed on 2026-09-08.

Under the Pink-Ponk protocol a fix voids BOTH reviewers' verdicts for every
task whose source it touches. These fixes changed `geometry/utils.py`,
`tasks/common.py` and other shared dependencies, so all 51 tasks -- and
therefore all 102 verdicts, opus5's included -- fall.

This is the protocol working, not a regression. Round 13 must re-review from
the frozen post-fix state. The full suite is green on that state:
**2610 passed, 158 skipped, 0 failed** (1:36:14, 2026-09-08).
