"""Is repe_concept_separability still degenerate (AUC=1.0 everywhere)?"""
import torch, blme.tasks, blme.tasks.representation_engineering
from transformers import AutoModelForCausalLM, AutoTokenizer
from blme.registry import get_task
import logging; logging.getLogger("blme").setLevel(logging.WARNING)
for name in ["gpt2", "EleutherAI/pythia-70m", "gpt2-large"]:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name).eval()
    if torch.cuda.is_available(): m = m.cuda()
    r = get_task("repe_concept_separability")(config={"num_samples": 40}).evaluate(
        m, tok, None, cache=None)
    aucs = r.get("layer_concept_auc") or []
    uniq = sorted({round(float(a), 6) for a in aucs}) if aucs else []
    print(f"RESULT {name:26s} max_auc={r.get('max_auc'):.4f} mean_auc={r.get('mean_auc'):.4f} "
          f"distinct_layer_aucs={len(uniq)} range=[{min(uniq) if uniq else 0:.3f},{max(uniq) if uniq else 0:.3f}]",
          flush=True)
    del m
