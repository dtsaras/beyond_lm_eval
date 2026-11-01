"""repe_refusal_direction formats instructions as andyrdt/refusal_direction does:
the family's own template with no system prompt, so the prompt never carries the
run date or a vendor system prompt, and keeps only prompts whose refusal score
agrees with their label (filter_data). The released code itself is compared in
tests/tasks/parity/test_repe_refusal_direction_arditi_parity.py."""
import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
transformers = pytest.importorskip("transformers")

from blme.tasks.repe import refusal as R  # noqa: E402

QWEN = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
THINK_OFF = "<think>\n\n</think>\n\n"
LLAMA3 = ("<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")
CHAT = ("{% for m in messages %}<|user|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def _tok(repo):
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:
        pytest.skip(f"{repo} not cached")


# The study's instruct checkpoints. Their own templates inject, respectively: the
# run date, "You are Qwen ...", nothing, an open <think>, and an open <think> after
# a "Reasoning effort is set to xhigh" system prompt.
@pytest.mark.parametrize("repo,label,template,injected", [
    ("unsloth/Llama-3.2-1B-Instruct", "refusal_direction:llama-3", LLAMA3, "Today Date"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "refusal_direction:qwen", QWEN, "You are Qwen"),
    ("Qwen/Qwen3-32B", "refusal_direction:qwen", QWEN + THINK_OFF, None),
    ("Qwen/Qwen3.5-27B", "refusal_direction:qwen", QWEN + THINK_OFF, None),
    ("Qwen/Qwen3.8-27B", "refusal_direction:qwen", QWEN + THINK_OFF, "Reasoning effort"),
])
def test_release_family_gets_the_release_template(repo, label, template, injected):
    tok = _tok(repo)
    if injected is not None:      # known positive: the model's own template injects it
        own = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                      add_generation_prompt=True)
        assert injected in own
    fmt = R._prompt_format(tok)
    assert fmt.label == label and fmt.template == template
    assert fmt.add_special_tokens is True
    inst = R._HARMFUL_PROMPTS[0]
    assert R._format(tok, inst) == template.format(instruction=inst)
    assert R._encode(tok, inst, "cpu")[0].tolist() == tok.encode(template.format(instruction=inst))


# Thinking on: Qwen3 renders no think block (the model opens it), Qwen3.5+ an open
# one; Qwen3.8's reasoning-effort system prompt is dropped as any system prompt is.
@pytest.mark.parametrize("repo,template", [
    ("Qwen/Qwen3-32B", QWEN),
    ("Qwen/Qwen3.5-27B", QWEN + "<think>\n"),
    ("Qwen/Qwen3.8-27B", QWEN + "<think>\n"),
])
def test_thinking_on_keeps_the_release_turn(repo, template):
    tok = _tok(repo)
    own = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=True)
    assert not own.endswith(THINK_OFF)       # known positive: the switch reaches the template
    fmt = R._prompt_format(tok, enable_thinking=True)
    assert fmt.label == "refusal_direction:qwen" and fmt.template == template
    assert R._prompt_format(tok).template == QWEN + THINK_OFF    # the default is off


@pytest.mark.parametrize("repo", ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "microsoft/phi-4"])
def test_other_families_keep_their_own_template(repo):
    tok = _tok(repo)
    fmt = R._prompt_format(tok)
    own = tok.apply_chat_template([{"role": "user", "content": "{instruction}"}],
                                  tokenize=False, add_generation_prompt=True)
    assert fmt.label == "model_chat_template" and fmt.template == own
    assert fmt.add_special_tokens is False and fmt.refusal == ("I", "As")


def test_refusal_tokens_are_the_release_family_set():
    assert R._refusal_token_ids(_tok("unsloth/Llama-3.2-1B-Instruct")) == [40]   # ['I']
    assert R._refusal_token_ids(_tok("Qwen/Qwen2.5-1.5B-Instruct")) == [40, 2121]  # ['I', 'As']


def test_refusal_tokens_are_the_tokens_a_response_starts_with():
    """TinyLlama-Chat's template ends in a newline, after which its SentencePiece
    vocabulary starts a response with "I" (29902) and "As" (2887); "I" on its own
    tokenises to the word-start "▁I" (306), which the model does not emit there."""
    tok = _tok("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert tok.encode("I", add_special_tokens=False) == [306]     # known positive
    assert R._refusal_token_ids(tok) == [29902, 2887]
    # The release's Llama-2 turn ends in a space; its hard-coded refusal_toks is [306].
    llama2 = R._Format("[INST] {instruction} [/INST] ", True, ("I",), "refusal_direction:llama")
    assert R._refusal_token_ids(tok, llama2) == [306]


@pytest.fixture(scope="module")
def tiny():
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        attn_implementation="eager")
    return cfg, tok


def test_a_template_that_reads_the_date_is_not_run(tiny):
    cfg, tok = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = "{{ strftime_now('%d %b %Y') }}" + CHAT
    res = R.RefusalDirectionTask({}).evaluate(transformers.LlamaForCausalLM(cfg).eval(), tok, None)
    assert res["_meta_prompt_format"] == "model_chat_template"
    assert "current date" in res["_meta_refusal_skipped_reason"]
    assert all(np.isnan(res[k]) for k in R.RefusalDirectionTask.HEADLINE_KEYS)


class _Rigged(transformers.LlamaForCausalLM):
    """Refuses exactly the prompts that contain "refuse": +30 on the "I" logit."""
    tok = None

    def forward(self, input_ids=None, **kw):
        out = super().forward(input_ids=input_ids, **kw)
        if "refuse" in self.tok.decode(input_ids[0]):
            i = self.tok.encode("I", add_special_tokens=False)[0]
            out.logits[..., i] = out.logits[..., i] + 30.0
        return out


class _Loose(R.RefusalDirectionTask):
    KL_THRESHOLD, INDUCE_THRESHOLD = 1e9, -1e9


def _labelled(harmful, harmless):
    return ([{"text": t, "label": "harmful"} for t in harmful]
            + [{"text": t, "label": "harmless"} for t in harmless])


def test_filter_data_keeps_the_prompts_the_model_classifies_as_labelled(tiny):
    """run_pipeline.py:filter_data: harmful kept when the refusal score is > 0,
    harmless when it is < 0, in train and validation. Each class has 4 train and 2
    validation prompts; one train prompt per class is mislabelled for the model."""
    cfg, tok = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    torch.manual_seed(0)
    model = _Rigged(cfg).eval()
    model.tok = tok
    data = _labelled(
        ["refuse to say a", "refuse to say b", "refuse to say c", "comply with d",
         "refuse to say e", "refuse to say f"],
        ["comply with g", "comply with h", "refuse to say i", "comply with j",
         "comply with k", "comply with l"])
    on = _Loose({}).evaluate(model, tok, data)
    assert on["_meta_filter_data"] is True and on["_meta_enable_thinking"] is False
    assert on["_meta_n_train_per_class"] == [4, 4]
    assert on["_meta_n_train_per_class_filtered"] == [3, 3]
    assert on["_meta_n_val_per_class_filtered"] == [2, 2]
    assert np.isfinite(on["direction_norm"])

    class _NoFilter(_Loose):
        FILTER_DATA = False
    off = _NoFilter({}).evaluate(model, tok, data)
    assert off["_meta_n_train_per_class_filtered"] == [4, 4]
    assert off["direction_norm"] != on["direction_norm"]


def test_filter_data_that_empties_a_class_gives_nan(tiny):
    """A random model refuses nothing, so every harmful prompt is filtered out."""
    cfg, tok = tiny
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = CHAT
    torch.manual_seed(0)
    res = _Loose({}).evaluate(transformers.LlamaForCausalLM(cfg).eval(), tok, None)
    assert res["_meta_n_train_per_class_filtered"][0] == 0
    assert res["_meta_refusal_token_ids"] == R._refusal_token_ids(tok)   # what was scored
    assert "filter_data" in res["_meta_refusal_skipped_reason"]
    assert all(np.isnan(res[k]) for k in R.RefusalDirectionTask.HEADLINE_KEYS)
