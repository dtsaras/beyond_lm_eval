"""Entropy calibration error: the model's entropy on its own generations
minus its log loss on human text (Cao et al., NeurIPS 2025).

Reference: "On the Entropy Calibration of Language Models", arXiv:2511.11966;
code stevenxcao/entropy-calibration (``do_generation.py``,
``compute_ents.py``, ``modeling.py``, ``load_datasets.py``).

Each WikiText-103 train article gives a prompt (its first 128 tokens, as the
model tokenizes it with special tokens) and a human continuation (the rest).
The model samples a continuation of up to 1024 tokens at temperature 1. The
per-step entropy H_t of its next-token distribution is read along its own
sample, the log loss L_t along the human continuation, and

    EntCE = mean_t H_t - mean_t L_t

over all steps of all examples (nats per token). Positive means the model's
generations are more uncertain than its fit to human text warrants: errors
accumulate and the samples drift.
"""
import logging
import math

import numpy as np
import torch

from ...registry import register_task
from ...tasks.base import DiagnosticTask
from ..common import get_max_positions, stop_ids
from .generation_diversity import _ignored_generation_settings, _no_model_generation_defaults

logger = logging.getLogger("blme")

# do_generation.py: prompt_len 128, min_len 1024, context_len 4096; the paper
# limits samples to 1024 tokens (the code's max_len is 2048).
_PROMPT_LEN = 128
_MAX_NEW = 1024
_MIN_RESPONSE = 1024
# Articles kept by bytes so every tokenizer scores the same ones: 10000 bytes
# hold 128 + 1024 tokens at up to 8 bytes per token.
_MIN_BYTES = 10000


def prompt_and_response(tokenizer, text, prompt_len=_PROMPT_LEN):
    """``Tokenizer.encode_prompt`` (special tokens, a trailing EOS dropped),
    split into the first ``prompt_len`` ids and the rest."""
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids = ids[:-1]
    return ids[:prompt_len], ids[prompt_len:]


def _step_logprobs(model, ids, device):
    """float32 log-softmax of the next-token logits at every position."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0]
    return torch.log_softmax(logits.float(), dim=-1)


def generation_entropies(model, prompt, generated, device):
    """Entropy of the model's distribution when predicting each generated
    token (``compute_logprobs(compute_ent=True)``)."""
    lp = _step_logprobs(model, prompt + generated, device)
    steps = lp[len(prompt) - 1:len(prompt) - 1 + len(generated)]
    return (-(steps * steps.exp()).sum(dim=-1)).cpu().numpy().astype(np.float64)


def response_log_losses(model, prompt, response, device):
    """-log p(response token | prefix) for every response token. The release
    slices one position late (it drops the first response token and, in a
    padded batch, scores a pad token); this is the aligned version."""
    lp = _step_logprobs(model, prompt + response, device)
    pos = torch.arange(len(prompt) - 1, len(prompt) - 1 + len(response), device=lp.device)
    tgt = torch.tensor(response, device=lp.device)
    return (-lp[pos, tgt]).cpu().numpy().astype(np.float64)


def _sample(model, tokenizer, prompts, max_new, batch_size, device):
    """Pure sampling at temperature 1 (top_k 0, top_p 1, as vLLM's defaults),
    left-padded batches; each continuation is cut after its first stop id.
    The task's own GenerationConfig, with the model's decoding defaults unset
    for the call: generate() fills every field the passed config leaves unset
    from model.generation_config, so its processors (suppress_tokens,
    repetition_penalty, no_repeat_ngram_size, min_p, ...) would otherwise shape
    the samples (test_model_generation_config_does_not_apply)."""
    from transformers import GenerationConfig
    stops = stop_ids(model, tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad = 0 if pad is None else pad
    gen_cfg = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                               max_new_tokens=max_new, pad_token_id=pad,
                               eos_token_id=sorted(stops) or None)
    out = []
    for s in range(0, len(prompts), batch_size):
        batch = prompts[s:s + batch_size]
        width = max(len(p) for p in batch)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in batch], device=device)
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch], device=device)
        with _no_model_generation_defaults(model):
            seqs = model.generate(input_ids=ids, attention_mask=mask, generation_config=gen_cfg)
        for row in seqs[:, width:].tolist():
            cut = next((i for i, t in enumerate(row) if t in stops), None)
            out.append(row if cut is None else row[:cut + 1])
    return out


@register_task("dynamics_entropy_calibration")
class EntropyCalibrationTask(DiagnosticTask):
    """Entropy calibration error EntCE (Cao et al. 2025)."""

    TOKEN_POLICY = ("paper_protocol:Cao2025 every generated step and every human "
                    "continuation token after a 128-token prompt")
    SOURCES = {
        "paper": "arXiv:2511.11966 (Cao et al., NeurIPS 2025, On the Entropy Calibration of "
                 "Language Models)",
        "code": "stevenxcao/entropy-calibration@d3f62e4b506c08c70be8e16f39e091133d0f2592",
        "llm_precedent": ("arXiv:2511.11966 (Qwen2.5 0.5-72B, Llama 3 1-70B, Llama 2 7-70B, "
                          "Pythia 0.41-12B on WikiText-103 train)"),
    }
    ADAPTATION_NOTE = (
        "As released: WikiText-103 train articles, the first 128 tokens (special tokens "
        "included, a trailing EOS dropped) as the prompt and the rest as the human "
        "continuation, pure sampling at temperature 1, the per-step entropy along the sample "
        "and the log loss along the human continuation, both from a separate float32 "
        "log-softmax forward pass. Departures: samples of up to 1024 tokens, as the paper "
        "states (the code allows 2048), with the human continuation cut to the same length; "
        "500 articles, where the release uses 5000 (class 2; _meta_ reports a bootstrap "
        "interval); articles kept by bytes (>= 10000) so every model scores the same ones, "
        "in the library's article order; the model as loaded scores both, where the release "
        "rescores with a 4-bit NF4 copy; and the log loss aligned to the response tokens, "
        "which the release reads one position late. entropy_calibration_ratio = H / L - 1 "
        "is a BLME twin (class 2) that removes the per-token units of each tokenizer to "
        "first order. Both are functions of the model's distribution, so identity blocks "
        "leave them unchanged at a fixed seed."
    )
    HEADLINE_KEYS = ("entropy_calibration_error", "entropy_calibration_ratio")
    SIZE_INVARIANT = dict.fromkeys(
        HEADLINE_KEYS,
        (("identity_blocks_interior", "identity_blocks_leading", "identity_blocks_trailing"),
         "tests/tasks/invariance/test_dynamics_entropy_calibration.py::test_headline_invariant"))

    def evaluate(self, model, tokenizer, dataset, cache=None):
        logger.info("Running Entropy Calibration Error...")
        num_samples = int(self.config.get("num_samples", 500))
        prompt_len = int(self.config.get("prompt_len", _PROMPT_LEN))
        max_new = int(self.config.get("max_new_tokens", _MAX_NEW))
        min_response = int(self.config.get("min_response_tokens", _MIN_RESPONSE))
        min_bytes = int(self.config.get("min_bytes", _MIN_BYTES))
        batch_size = int(self.config.get("batch_size", 16))
        seed = int(self.config.get("generation_seed", 0))
        max_new = min(max_new, get_max_positions(model, prompt_len + max_new) - prompt_len)
        min_response = min(min_response, max_new)
        if max_new < 1:
            return {"error": "The model's context does not hold the prompt"}

        if self.config.get("use_dataset", False):
            pool = list(dataset)
        else:
            from ...cache import load_long_documents
            pool = load_long_documents(4 * num_samples, split="train")
        texts = [d["text"] if isinstance(d, dict) else str(d) for d in pool]
        texts = [t for t in texts if len(t.encode("utf-8")) >= min_bytes][:num_samples]
        prompts, responses = [], []
        for t in texts:
            p, r = prompt_and_response(tokenizer, t, prompt_len)
            if len(p) == prompt_len and len(r) >= min_response:
                prompts.append(p)
                responses.append(r[:max_new])
        if not prompts:
            return {"error": "No article has a long enough human continuation"}
        device = next(model.parameters()).device

        from ...utils import isolated_torch_rng
        with torch.no_grad(), isolated_torch_rng(model):
            torch.manual_seed(seed)
            generations = _sample(model, tokenizer, prompts, max_new, batch_size, device)
            ents = [generation_entropies(model, p, g, device) for p, g in zip(prompts, generations)]
            losses = [response_log_losses(model, p, r, device) for p, r in zip(prompts, responses)]

        H_all, L_all = np.concatenate(ents), np.concatenate(losses)
        H, L = math.fsum(H_all) / H_all.size, math.fsum(L_all) / L_all.size
        # Bootstrap over examples (pooled means of resampled examples).
        rng = np.random.default_rng(seed)
        ent_sum = np.array([e.sum() for e in ents]); ent_n = np.array([e.size for e in ents])
        loss_sum = np.array([x.sum() for x in losses]); loss_n = np.array([x.size for x in losses])
        boot = []
        for _ in range(int(self.config.get("n_bootstrap", 1000))):
            i = rng.integers(0, len(ents), len(ents))
            boot.append(ent_sum[i].sum() / ent_n[i].sum() - loss_sum[i].sum() / loss_n[i].sum())
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
        return {
            "entropy_calibration_error": H - L,
            "entropy_calibration_ratio": H / L - 1.0,
            "_meta_mean_generation_entropy": H,
            "_meta_mean_human_log_loss": L,
            "_meta_entce_ci95": [float(lo), float(hi)],
            "_meta_entce_per_example_mean": float(np.mean([e.mean() for e in ents if e.size])
                                                  - np.mean([x.mean() for x in losses])),
            "_meta_n_examples": len(prompts),
            "_meta_n_generated_tokens": int(H_all.size),
            "_meta_n_generations_ended_by_eos": int(sum(
                1 for g in generations if g and g[-1] in stop_ids(model, tokenizer))),
            # The batched sampler draws different samples at another batch size.
            "_meta_batch_size": batch_size,
            "_meta_max_new_tokens": max_new,
            "_meta_generation_seed": seed,
            # The model's own decoding defaults, which the sampling keeps out.
            "_meta_ignored_generation_config": _ignored_generation_settings(model),
        }
