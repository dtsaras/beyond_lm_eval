"""Does the scaling-region estimator return a value at shipped defaults?"""
import torch, blme.tasks.geometry
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
from blme.cache import load_default_corpus
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
tok = AutoTokenizer.from_pretrained("gpt2")
m = AutoModelForCausalLM.from_pretrained("gpt2").eval()
if torch.cuda.is_available(): m = m.cuda()
ds = load_default_corpus(120)
r = get_task("geometry_correlation_dimension")().evaluate(m, tok, ds, cache=None)
print(f"RESULT legacy  = {r.get('correlation_dimension')}", flush=True)
print(f"RESULT scaling = {r.get('correlation_dimension_scaling_region')} "
      f"status={r.get('_meta_correlation_dimension_scaling_status')} "
      f"window={r.get('_meta_correlation_dimension_scaling_window')} "
      f"N={r.get('_meta_num_points')}", flush=True)
