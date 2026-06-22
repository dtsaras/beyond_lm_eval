# BLME Task Certification Status

BLME task certification status is machine-readable via:

```bash
blme list-tasks --json
```

The source of truth is `src/blme/task_metadata.py`. Each task is assigned
one conservative publication label:

| Status | Meaning |
|---|---|
| `parity-ready` | Focused formula/reference tests exist for the task's core method. |
| `formula-faithful` | The implementation follows the paper formula, but no external repo parity fixture is checked in. |
| `refined-adaptation` | Paper-derived method adapted for BLME's architecture-agnostic, label-light setting. |
| `proxy-only` | BLME diagnostic inspired by literature; do not claim paper or repository parity. |

Publication language should use these labels directly. A GitHub reference in
`docs/REPOSITORIES.md` means reviewers have a comparison target; it does not
imply line-for-line parity unless the task is marked `parity-ready` and the
covering test is named.

Checked-in reference fixtures live in
`tests/fixtures/reference_parity/formula_fixtures.json`. They include the
paper/reference source, upstream repository URL, observed upstream HEAD where
available, toy inputs, and expected outputs for compact formulas.

Current high-level posture:

- `parity-ready`: small set of heavily tested core formulas such as IsoScore,
  matrix entropy, trajectory curvature, and selected reference helpers.
- `formula-faithful`: direct implementations of compact paper formulas such
  as CKA/HSIC, persistent-homology summaries, Min-K%, Distinct-n, Self-BLEU,
  and paper-correct CoE-C.
- `refined-adaptation`: methods adapted to BLME's single-model diagnostic
  setting, for example causal tracing, RepE reading vectors, refusal direction,
  contextualization, and several interpretability metrics.
- `proxy-only`: useful BLME diagnostics that should not be described as full
  reproductions of their motivating papers.

Before any paper or release claims broad reference parity, run:

```bash
PYTHONPATH=src pytest tests/test_task_metadata.py tests/test_publication_docs.py -q
PYTHONPATH=src pytest tests/tasks/test_reference_parity_formulas.py -q
blme list-tasks --json
```

