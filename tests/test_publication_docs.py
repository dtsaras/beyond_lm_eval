"""Publication-readiness drift tests for paper and reference docs."""

from pathlib import Path


DOCS = Path("docs")


def _docs_text() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(DOCS.glob("*.md"))
    )


def test_no_stale_publication_identifiers():
    """Guard paper IDs, task names, and output keys corrected in audit."""
    text = _docs_text()

    banned = {
        "arXiv:2207.10341": "IsoScore arXiv ID; use arXiv:2108.07344",
        "Wei et al. 2025 — From Internal Representations to Text Quality": (
            "Text-quality paper author label; use Yusupov et al. 2025"
        ),
        "geometry_schatten.matrix_nuclear_norm": (
            "Old MNN output path; use row_normalized_matrix_nuclear_norm"
        ),
        "geometry_schatten.schatten_{1,2,4,inf}_last": (
            "Schatten-2 is intentionally omitted after row-L2 normalization"
        ),
        "dynamics_trajectories": "Registered task is dynamics_interpolation",
        "geometry_information_fisher": (
            "Registered task is geometry_representation_sensitivity"
        ),
        "geometry_consistency": "Registered task is geometry_prediction_alignment",
        "contamination_score": "Use min_k_score / contamination_ratio",
        "mean_persistance_h0": "Spelling is mean_persistence_h0",
        "component_coherence_mean": "Old attribution proxy key",
        "mean_attention_svd_entropy": (
            "Use mean_attention_output_effective_rank_entropy"
        ),
        "FisherInformationTraceTask": "Class is RepresentationSensitivityTask",
        "BettiCurveSimplificationTask": "Class is BettiCurveTask",
        "Lyapunov Exponents": "dynamics_stability is kNN Jaccard stability",
        "Unembedding Dark Matter": "Fabricated citation (no such paper); geometry_unembedding uses Roy & Vetterli effective rank",
        "arXiv:1209.6425": "Wrong hubness id (= Deng & Runger gene selection); Tomasev 2014 is IEEE TKDE, no arXiv",
        "arXiv:1705.10933": "Wrong Facco Two-NN id; correct is arXiv:1803.06992",
        "Choe, Y. J., Wattenberg": "Linear Rep. Hypothesis authors are Park, Choe, Veitch (arXiv:2311.03658)",
        "Park, Choe, Wattenberg": "Linear Rep. Hypothesis authors are Park, Choe, Veitch (arXiv:2311.03658)",
        "Novikova et al. 2017 (paraphrase perturbation)": "dynamics_stability is Wendlandt et al. 2018 kNN embedding instability (arXiv:1804.09692)",
        "Loshchilov & Hutter 2019 slerp": "slerp is Shoemake 1985; 1711.05101 is AdamW, not interpolation",
        "Loshchilov-Hutter 2019 slerp": "slerp is Shoemake 1985; 1711.05101 is AdamW, not interpolation",
        "37 / 71": "Stale citation-count fraction; registry has 74 tasks",
        "29 / 71": "Stale citation-count fraction; registry has 74 tasks",
        "5 / 71": "Stale citation-count fraction; registry has 74 tasks",
        "54 intrinsic": "Stale task-count language; registry has 74 tasks",
        "implements all of the above": "Overclaim; docs must separate implemented vs discussed literature",
        "one forward pass": "Cache docs must mention only cache-aware tasks share cached tensors",
        "1 forward pass": "Cache docs must mention only cache-aware tasks share cached tensors",
    }
    offenders = {
        pattern: reason for pattern, reason in banned.items() if pattern in text
    }
    assert not offenders


def test_reference_repo_docs_are_conservative():
    """The repo index must distinguish available repos from parity claims."""
    text = (DOCS / "REPOSITORIES.md").read_text(encoding="utf-8")

    assert "## BLME reference-check status" in text
    assert "Do not claim" in text
    assert "line-for-line reference-code parity" in text
    assert "reviewer comparison targets" in text
    assert "all URLs verified" not in text

