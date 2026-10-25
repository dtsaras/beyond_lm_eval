# Why these verdicts are void

Round 13 (gpt-6-astra) returned 20 GREEN / 23 YELLOW / 8 RED. All 8 REDs were
`defect_location: CODE` and all 8 were fixed on 2026-09-08, each verified
against the reviewer's own counterexample. Under the Pink-Ponk protocol a fix
voids BOTH reviewers' verdicts for every task whose source it touches.

Tasks voided here (9): the 8 REDs plus geometry_intrinsic_dim, whose call-site
narrowing was repaired in the same pass.

Notably SIX of the eight were classes I had already 'fixed' in round 12 at one
site and failed to sweep:
  - the attention-only-cache defect reached geometry_isoscore, geometry_svd and
    geometry_prediction_alignment -- five sites in total, not the two I claimed;
    I had cleared isoscore and svd by READING the code rather than running it.
  - the duplicate quotient was applied to topology_betti_curve and never swept
    to topology_homology.
  - the SVD-residue envelope was applied to alpha_official and not to the Hill
    alpha in the same file.
  - repe_refusal_direction's repair was itself wrong, and the regression test I
    wrote for it asserted the wrong thing and PASSED.
