"""Domain-conditioned code lengths and a paired-view adaptation of LiDAR for causal LMs."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from ...registry import register_task
from ..base import DiagnosticTask
from ..common import get_max_positions


def _inputs(config, dataset, field):
    if config.get("panel_path"):
        raw = Path(config["panel_path"]).read_bytes()
        rows = json.loads(raw).get(field)
        identity = hashlib.sha256(raw).hexdigest()
    else:
        rows = list(dataset or ())
        identity = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if not rows:
        raise ValueError(f"Supply {field} via panel_path or a structured dataset")
    fields = {"id", "block"} | ({"domain", "text"} if field == "documents" else {"anchor", "positive", "negative"})
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not fields <= row.keys():
            raise ValueError(f"Invalid {field} row {i}: expected fields {sorted(fields)}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)) or any(not isinstance(r["block"], int) for r in rows):
        raise ValueError("Expected unique item IDs and integer block IDs")
    return rows, identity


def start_token(tokenizer):
    for name in ("bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int) and value >= 0:
            return value, name
    raise ValueError("A BOS or EOS token is required for the common document-start policy")


def _batches(model, tokenizer, texts, config, lossless):
    start, _ = start_token(tokenizer)
    ids = tokenizer(list(texts), add_special_tokens=False, return_attention_mask=False)["input_ids"]
    limit = min(int(config.get("max_tokens", 256)), get_max_positions(model, 2048))
    if any(not seq or len(seq) + 1 > limit for seq in ids):
        raise ValueError(f"Empty or overlong content at context limit {limit}; no clipping allowed")
    if lossless:
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if any(a != b for a, b in zip(texts, decoded)):
            raise ValueError("Tokenizer cannot round-trip the common raw passage")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else start
    size = int(config.get("batch_size", 8))
    if size < 1:
        raise ValueError("batch_size must be positive")
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    device = model.get_input_embeddings().weight.device
    for off in range(0, len(order), size):
        ix = order[off:off + size]
        lengths = torch.tensor([len(ids[i]) for i in ix], device=device)
        seq = torch.full((len(ix), int(lengths.max()) + 1), pad, dtype=torch.long, device=device)
        seq[:, 0] = start
        for j, i in enumerate(ix):
            seq[j, 1:len(ids[i]) + 1] = torch.tensor(ids[i], device=device)
        mask = torch.arange(seq.shape[1], device=device)[None, :] <= lengths[:, None]
        yield ix, seq, mask, lengths


def _effective_rank(values):
    values = np.asarray(values, dtype=np.float64)
    top = values.max(initial=0.)
    if top <= 0 or values.min(initial=0.) < -1e-9 * top:
        raise ValueError("No positive spectrum or a materially indefinite covariance")
    weights = np.maximum(values, 0.) / top
    p = weights[weights > 0]
    p /= p.sum()
    return float(np.exp(-p @ np.log(p)))


def paired_lidar(first, second, ridge_fraction=0.01):
    """Effective rank of (W+lambda I)^(-1/2) B (W+lambda I)^(-1/2) in sample space."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or len(a) < 3 or a.shape[1] == 0:
        raise ValueError("Expected at least three matched pairs of feature vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not 0 < ridge_fraction < math.inf:
        raise ValueError("Nonfinite representations or invalid ridge")
    n = len(a)
    means = (a + b) / 2
    between = (means - means.mean(axis=0)) / np.sqrt(n - 1)
    within = (a - b) / np.sqrt(2 * n)
    lam = ridge_fraction * (np.square(between).sum() + np.square(within).sum()) / (n - 1)
    if lam <= 0:
        raise ValueError("Both covariance traces vanish")
    # Woodbury applied in pair space. The common eigenvalue multiplier 1/lambda
    # cancels during entropy normalization. This avoids a width-by-width inverse.
    ww = within @ within.T
    ww.flat[::n + 1] += lam
    bw = between @ within.T
    gram = between @ between.T - bw @ np.linalg.solve(ww, bw.T)
    return _effective_rank(np.linalg.eigvalsh((gram + gram.T) * .5))


def _rankme(rows):
    # RankMe uses singular values of the uncentered embedding matrix.
    gram = rows @ rows.T
    return _effective_rank(np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0.)))


def _document_nll(logits, targets):
    # Sum over the vocabulary in float64. Bound the temporary for wide
    # vocabularies without copying the complete batch of logits to double.
    chunk_tokens = max(1, 4_194_304 // logits.shape[-1])
    totals = []
    for document, labels in zip(logits[:, :-1], targets):
        partial = []
        for scores, ids in zip(document.split(chunk_tokens), labels.split(chunk_tokens)):
            scores = scores.double()
            selected = scores.gather(1, ids.clamp_min(0)[:, None]).squeeze(1)
            token_losses = scores.logsumexp(dim=1) - selected
            partial.append(token_losses.masked_fill(ids == -100, 0).sum())
        totals.append(torch.stack(partial).sum())
    return torch.stack(totals)


def _bpb(values):
    return sum(v["nll_nats"] for v in values) / (math.log(2) * sum(v["bytes"] for v in values))


@register_task("geometry_domain_bpb")
class DomainBitsPerByteTask(DiagnosticTask):
    """Teacher-forced per-domain bits per UTF-8 byte, with all document-start tokens scored."""
    HEADLINE_KEYS = ("domain_bpb_macro",)
    SOURCES = {"paper": "arXiv:2409.05816 (domain-specific LM losses as baselines)"}

    @torch.inference_mode()
    def evaluate(self, model, tokenizer, dataset, cache=None):
        rows, identity = _inputs(self.config, dataset, "documents")
        texts = [r["text"] for r in rows]
        if any(not isinstance(t, str) or not t for t in texts) or any(not r["domain"] for r in rows):
            raise ValueError("Each document needs nonempty text and a domain")
        details = [None] * len(rows)
        for ix, seq, mask, lengths in _batches(model, tokenizer, texts, self.config, True):
            logits = model(input_ids=seq, attention_mask=mask.long(), use_cache=False).logits
            targets = seq[:, 1:].masked_fill(~mask[:, 1:], -100)
            nll = _document_nll(logits, targets)
            for i, ntok, value in zip(ix, lengths.tolist(), nll.tolist()):
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite likelihood for document {rows[i]['id']}")
                details[i] = {k: rows[i][k] for k in ("id", "domain", "block")}
                details[i].update(nll_nats=value, bytes=len(texts[i].encode("utf-8")), tokens=ntok)
        domains = sorted({r["domain"] for r in rows})
        blocks = sorted({r["block"] for r in rows})
        by_domain = {d: _bpb([r for r in details if r["domain"] == d]) for d in domains}
        per_block = {}
        for block in blocks:
            sub = [r for r in details if r["block"] == block]
            if {r["domain"] for r in sub} != set(domains):
                raise ValueError("Each block must cover every domain")
            per_block[str(block)] = {d: _bpb([r for r in sub if r["domain"] == d]) for d in domains}
        return {"domain_bpb_macro": float(np.mean(list(by_domain.values()))), "by_domain": by_domain,
                "blocks": per_block, "documents": details, "_meta_panel_sha256": identity,
                "_meta_start": start_token(tokenizer)}


@register_task("geometry_paraphrase_lidar")
class ParaphraseLiDARTask(DiagnosticTask):
    """Discriminant effective rank on paired sentence views and matched negative controls."""
    HEADLINE_KEYS = ("lidar_log_rank_fraction",)
    SOURCES = {"paper": "arXiv:2312.04000 (LiDAR); arXiv:1904.01130 (PAWS)"}

    @torch.inference_mode()
    def evaluate(self, model, tokenizer, dataset, cache=None):
        rows, identity = _inputs(self.config, dataset, "triples")
        texts = [r[key] for r in rows for key in ("anchor", "positive", "negative")]
        if any(not isinstance(t, str) or not t for t in texts) or len(set(texts)) != len(texts):
            raise ValueError("Expected distinct, nonempty views across all supplied triples")
        features = [None] * len(texts)
        for ix, seq, mask, lengths in _batches(model, tokenizer, texts, self.config, True):
            # The LM wrapper's base model yields the states before the vocabulary head.
            state = model.base_model(input_ids=seq, attention_mask=mask.long(), use_cache=False,
                                     return_dict=True).last_hidden_state
            if state.ndim != 3 or state.shape[:2] != seq.shape:
                raise ValueError("Expected one final text state per input token")
            content = mask.clone()
            content[:, 0] = False
            pooled = (state.float() * content.unsqueeze(-1)).sum(dim=1) / lengths[:, None]
            for i, vector in zip(ix, pooled.cpu().numpy()):
                features[i] = vector
        matrix = np.asarray(features, dtype=np.float64).reshape(len(rows), 3, -1)
        blocks = np.array([r["block"] for r in rows])
        ridge = float(self.config.get("ridge_fraction", .01))

        def scores(data):
            a, p, neg = data.transpose(1, 0, 2)
            n = len(a)
            pos_rank = paired_lidar(a, p, ridge)
            neg_rank = paired_lidar(a, neg, ridge)
            rankme = _rankme(np.vstack([a, p]))
            return {"n": n, "positive_rank": pos_rank, "negative_rank": neg_rank, "rankme": rankme,
                    "lidar": math.log(pos_rank / (n - 1)), "negative": math.log(neg_rank / (n - 1)),
                    "log_rankme": math.log(rankme)}
        per_block = {str(i): scores(matrix[blocks == i]) for i in sorted(set(blocks))}
        if len({v["n"] for v in per_block.values()}) != 1:
            raise ValueError("Comparable rank panels require equal numbers of pairs")
        return {"lidar_log_rank_fraction": float(np.mean([v["lidar"] for v in per_block.values()])),
                "blocks": per_block, "pooled": scores(matrix), "_meta_panel_sha256": identity,
                "_meta_width": matrix.shape[-1], "_meta_start": start_token(tokenizer),
                "_meta_ridge_fraction": ridge}
