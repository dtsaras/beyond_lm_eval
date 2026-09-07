import torch, blme.tasks.geometry
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
from blme.cache import load_default_corpus
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
tok=AutoTokenizer.from_pretrained("gpt2"); ds=load_default_corpus(20)
for dt,lab in [(torch.float32,"fp32"),(torch.bfloat16,"bf16")]:
    m=AutoModelForCausalLM.from_pretrained("gpt2",dtype=dt,attn_implementation="eager").eval()
    r=get_task("geometry_representation_sensitivity")(config={"num_samples":4}).evaluate(m,tok,ds,cache=None)
    print(f"RESULT {lab}: error={'error' in r}  value={r.get('representation_sensitivity')}", flush=True)
    del m
