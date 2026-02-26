from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_args_str: str, device=None):
    """
    Simple parser for model_args string like "pretrained=gpt2,dtype=float16"
    """
    args = {}
    if model_args_str:
        for pair in model_args_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                args[k.strip()] = v.strip()
    
    model_name = args.get("pretrained", "gpt2")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: extensive arg handling
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if str(device).startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    
    return model, tokenizer
