"""Does critical sharpness survive the dtype the study actually runs in?"""
import torch, blme.tasks.dynamics
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
from blme.cache import load_default_corpus
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
tok = AutoTokenizer.from_pretrained("gpt2"); ds = load_default_corpus(20)
for dt, lab in [(torch.float32, "fp32"), (torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
    m = AutoModelForCausalLM.from_pretrained("gpt2", dtype=dt,
                                             attn_implementation="eager").eval()
    r = get_task("dynamics_sharpness")(config={"num_samples": 4}).evaluate(
        m, tok, ds, cache=None)
    print(f"RESULT {lab}: eta*={r.get('critical_step_size')} "
          f"lambda_c={r.get('critical_sharpness')} "
          f"dtype_restored={next(m.parameters()).dtype}", flush=True)
    del m
