import torch, blme.tasks.geometry
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
for name in ["gpt2", "EleutherAI/pythia-70m"]:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name).eval()
    r = get_task("geometry_mahalanobis")(config={"num_samples": 40}).evaluate(m, tok, None, cache=None)
    print(f"RESULT {name:24s} AUROC={r.get('ood_auroc')} gap={r.get('ood_separation_gap')}", flush=True)
    del m
