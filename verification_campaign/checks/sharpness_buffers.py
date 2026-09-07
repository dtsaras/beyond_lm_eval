"""Does the sharpness probe leave buffers untouched?"""
import torch, blme.tasks.dynamics
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
from blme.cache import load_default_corpus
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
name = "EleutherAI/pythia-160m"
tok = AutoTokenizer.from_pretrained(name); ds = load_default_corpus(20)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                         attn_implementation="eager").eval()
before = {n: (b.dtype, b.detach().float().clone()) for n, b in m.named_buffers()}
ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")
with torch.no_grad():
    nll_before = float(m(**ids, labels=ids["input_ids"]).loss.item())
get_task("dynamics_sharpness")(config={"num_samples": 3}).evaluate(m, tok, ds, cache=None)
bad = []
for n, b in m.named_buffers():
    dt0, v0 = before[n]
    if b.dtype != dt0:
        bad.append(f"{n}: {dt0} -> {b.dtype}")
    elif not torch.allclose(b.detach().float(), v0, atol=0, rtol=0):
        bad.append(f"{n}: values changed")
with torch.no_grad():
    nll_after = float(m(**ids, labels=ids["input_ids"]).loss.item())
print(f"RESULT buffers changed: {bad if bad else 'NONE'}", flush=True)
print(f"RESULT NLL before={nll_before:.10f} after={nll_after:.10f} "
      f"identical={nll_before == nll_after}", flush=True)
print(f"RESULT param dtype restored: {next(m.parameters()).dtype}", flush=True)
