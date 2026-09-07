"""Exact all-pairs MSO against the official overlap-0.py value on gpt2."""
import torch, blme.tasks.geometry
from transformers import AutoModelForCausalLM
from blme.tasks.geometry.superposition_strength import _mean_squared_overlap
m = AutoModelForCausalLM.from_pretrained("gpt2").eval()
W = m.get_output_embeddings().weight.detach().float()
mso = _mean_squared_overlap(W)
print(f"RESULT gpt2 exact all-pairs MSO = {mso:.10f}", flush=True)
print(f"RESULT official overlap-0.py    = 0.0746565427   (reviewer-executed)", flush=True)
print(f"RESULT rel = {abs(mso-0.0746565427)/0.0746565427:.2e}", flush=True)
print(f"RESULT V={W.shape[0]}  ordered pairs = {W.shape[0]*(W.shape[0]-1):,}", flush=True)
