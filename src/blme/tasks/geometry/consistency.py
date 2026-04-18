from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_embeddings, get_lm_head
from .utils import collect_prediction_stats
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger("blme")


def _get_output_projection_weight(model) -> torch.Tensor:
    """Return the weight matrix actually used to project the final
    hidden state into logits, preferring ``lm_head.weight`` over the
    input embedding table.

    For tied LM heads (GPT-2, Pythia, Llama 1/2, Qwen 2) these are the
    same tensor, so the choice doesn't matter. For **untied** LM heads
    (Gemma 3/4, some recent Llama checkpoints) they differ and using
    the input embedding would measure alignment with the token's
    *input* representation, not with the projection that actually
    produces the next-token logit — a different geometric object. This
    task is advertised as "prediction alignment", so the output
    projection is the correct tensor.
    """
    head = get_lm_head(model)
    if head is not None and hasattr(head, "weight"):
        return head.weight.detach()
    # Fallback to input embeddings (the old behaviour).
    return get_embeddings(model)


@register_task("geometry_prediction_alignment")
class PredictionAlignmentTask(DiagnosticTask):
    """
    Measures how well the final hidden state aligns with the target token
    embedding via cosine similarity.  When the LM head is tied to the input
    embeddings this is essentially a normalized logit — a high value means
    the representation already points toward the correct next token.

    Uses the output projection (``lm_head.weight``) rather than the input
    embedding table, so the metric retains its "prediction alignment"
    interpretation for untied-head architectures (Gemma 3/4, etc.). Falls
    back to input embeddings if ``lm_head`` is unavailable.
    """
    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Prediction Alignment Analysis...")
        if dataset is None:
            dataset = [{"text": "The quick brown fox jumps over the lazy dog."} for _ in range(50)]

        num_samples = self.config.get("num_samples", 100)
        use_cache = self.config.get("use_cache", True)

        if cache is not None and cache.is_populated and use_cache:
            stats, _ = cache.get_prediction_stats(num_samples=num_samples)
        else:
            stats, _ = collect_prediction_stats(model, tokenizer, dataset, num_samples=num_samples)

        # Resolve the projection weight the model actually uses for
        # logits (lm_head.weight). Ignore the cache's ``embeddings``
        # return value, which is the input-embedding table.
        embeddings = _get_output_projection_weight(model)
        if embeddings is None:
            return {"error": "Could not access output projection"}

        embeddings = embeddings.cpu()
        cosine_sims = []

        for h, labels in zip(stats["hidden"], stats["labels"]):
            # Normalize shapes: h → (N, D), labels → (N,). The cache
            # historically stores hidden flattened but labels with a batch
            # dim, so we need to reconcile them before F.embedding.
            if hasattr(h, "dim") and h.dim() == 3:
                B, T, D = h.shape
                h = h.reshape(-1, D)
            if hasattr(labels, "dim") and labels.dim() >= 2:
                labels = labels.reshape(-1)

            target_embs = F.embedding(labels, embeddings)
            # At this point both should be 2D of shape (N, D).

            h_norm = F.normalize(h.float(), p=2, dim=-1)
            e_norm = F.normalize(target_embs.float(), p=2, dim=-1)

            cos = (h_norm * e_norm).sum(dim=-1)
            # Ensure 1D before extending to avoid nested lists.
            if cos.dim() > 1:
                cos = cos.reshape(-1)
            cosine_sims.extend(cos.tolist())

        return {
            "prediction_alignment_mean": float(np.mean(cosine_sims)),
            "prediction_alignment_std": float(np.std(cosine_sims))
        }
