"""Helpers shared by the representation-engineering tasks."""
import logging

import torch.nn as nn

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger("blme")

__all__ = ["HAS_SKLEARN", "LogisticRegression", "StratifiedKFold", "roc_auc_score",
           "accuracy_score", "logger", "find_final_norm"]


def find_final_norm(model):
    """The normalisation applied after the last block, before the unembedding.

    Tries the known attribute names (``ln_f``, ``final_layer_norm``, ``norm``,
    ...) on the base model, including OPT's ``model.decoder``; None if absent.
    """
    roots = []
    for attr in ("transformer", "gpt_neox", "model", "base_model"):
        r = getattr(model, attr, None)
        if r is not None:
            roots.append(r)
            d = getattr(r, "decoder", None)
            if d is not None:
                roots.append(d)
    roots.append(model)
    for r in roots:
        for name in ("ln_f", "final_layer_norm", "norm", "final_norm",
                     "ln_final"):
            m = getattr(r, name, None)
            if isinstance(m, nn.Module):
                return m
    return None
