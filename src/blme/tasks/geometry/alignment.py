from ...registry import register_task
from ..gem.alignment import (
    AlignmentResidualTask as _GemAlignmentResidualTask,
    SubstitutionConsistencyTask as _GemSubstitutionConsistencyTask,
)


@register_task("geometry_alignment")
class AlignmentResidualTask(_GemAlignmentResidualTask):
    """Backward-compatible alias for GEM alignment metrics."""

    def evaluate(self, model, tokenizer, dataset):
        results = super().evaluate(model, tokenizer, dataset)
        translated = dict(results)
        for key, value in results.items():
            if key.startswith("alignment_k") and key.endswith("_l2_dist_mean"):
                k = key[len("alignment_k") : -len("_l2_dist_mean")]
                translated[f"k{k}_l2_mean"] = value
            elif key.startswith("alignment_k") and key.endswith("_cosine_sim_mean"):
                k = key[len("alignment_k") : -len("_cosine_sim_mean")]
                translated[f"k{k}_cos_mean"] = value
        return translated


@register_task("geometry_substitution")
class SubstitutionConsistencyTask(_GemSubstitutionConsistencyTask):
    """Backward-compatible alias for GEM substitution metrics."""

    def evaluate(self, model, tokenizer, dataset):
        results = super().evaluate(model, tokenizer, dataset)
        if "substitution_top1_match" in results and "substitution_top1_agreement" not in results:
            results["substitution_top1_agreement"] = results["substitution_top1_match"]
        return results
