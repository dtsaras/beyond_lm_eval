import torch
import torch.nn.functional as F
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers, get_embeddings

@register_task("causality_tracing")
class CausalTracingTask(DiagnosticTask):
    """
    Implements a simplified Causal Tracing (ROME-style) analysis.
    Corrupts the embedding of a token and measures how much restoring a
    specific hidden state at a specific layer rescues the original prediction.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Causal Tracing...")
        num_samples = self.config.get("num_samples", 3)
        noise_std = self.config.get("noise_std", 0.1)
        
        device = next(model.parameters()).device
        layers = get_layers(model)
        num_layers = len(layers)
        
        # Determine the number of layers to sample for tracing (to speed up)
        # We trace early, middle, and late layers.
        if num_layers > 10:
            trace_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
        else:
            trace_layers = list(range(num_layers))
            
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("NeelNanda/counterfact-tracing", split="train")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    item = dset[i]
                    # We just need text that the model can process and predict on.
                    # The prompt + target_true provides a factual statement.
                    dataset.append({"text": item["prompt"] + item["target_true"]})
            except ImportError:
                print("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [
                    {"text": "The Space Needle is located in the city of Seattle"},
                    {"text": "Eiffel Tower is located in the city of Paris"},
                ] * num_samples
        
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}
            
        embeddings = model.get_input_embeddings()
        
        results_by_layer = {layer_idx: [] for layer_idx in trace_layers}
        
        with torch.no_grad():
            for s in samples:
                if isinstance(s, dict) and "text" in s:
                    text = s["text"]
                else:
                    continue
                    
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
                seq_len = input_ids.shape[1]
                
                if seq_len < 3:
                    continue
                    
                # We corrupt the subject (heuristically, the second to last token before the target prediction)
                # Actually, standard causal tracing corrupts the subject tokens and looks at the prediction 
                # at the last token. Here we'll just corrupt the *first half* of the prompt to see how
                # the model recovers.
                corrupt_idx_start = 0
                corrupt_idx_end = seq_len // 2
                target_idx = seq_len - 1
                
                # 1. Clean Run
                clean_out = model(input_ids, output_hidden_states=True)
                clean_logits = clean_out.logits[0, target_idx]
                clean_probs = F.softmax(clean_logits, dim=-1)
                
                # The token the model *actually* wanted to predict
                target_token_id = torch.argmax(clean_probs).item()
                clean_prob_target = clean_probs[target_token_id].item()
                
                # Cache the clean hidden states
                clean_states = [h.detach() for h in clean_out.hidden_states]
                
                # 2. Corrupted Run
                # We need to manually embed and add noise
                inputs_embeds = embeddings(input_ids).clone()
                noise = torch.randn_like(inputs_embeds[:, corrupt_idx_start:corrupt_idx_end, :]) * noise_std
                inputs_embeds[:, corrupt_idx_start:corrupt_idx_end, :] += noise
                
                corrupted_out = model(inputs_embeds=inputs_embeds)
                corrupted_logits = corrupted_out.logits[0, target_idx]
                corrupted_probs = F.softmax(corrupted_logits, dim=-1)
                corrupted_prob_target = corrupted_probs[target_token_id].item()
                
                # Calculate the maximum possible restoration
                max_restoration = clean_prob_target - corrupted_prob_target
                if max_restoration <= 0:
                    continue # Skip if noise didn't hurt the prediction
                    
                # 3. Restored Runs (Layer by Layer Patching)
                for layer_idx in trace_layers:
                    
                    # We create a forward hook to patch the clean state back in
                    def get_patch_hook(clean_state_to_patch):
                        def patch_hook(module, input, output):
                            # Patch the hidden states at the corrupt indices
                            if isinstance(output, tuple):
                                out_tensor = output[0]
                                out_tensor[:, corrupt_idx_start:corrupt_idx_end, :] = clean_state_to_patch[:, corrupt_idx_start:corrupt_idx_end, :]
                                return (out_tensor,) + output[1:]
                            else:
                                output[:, corrupt_idx_start:corrupt_idx_end, :] = clean_state_to_patch[:, corrupt_idx_start:corrupt_idx_end, :]
                                return output
                        return patch_hook
                        
                    # hidden_states stores [embedding_out, layer_0_out, ...]
                    # So clean_states[layer_idx + 1] corresponds to the output of layers[layer_idx]
                    hook = layers[layer_idx].register_forward_hook(get_patch_hook(clean_states[layer_idx + 1]))
                    
                    try:
                        restored_out = model(inputs_embeds=inputs_embeds)
                        restored_logits = restored_out.logits[0, target_idx]
                        restored_probs = F.softmax(restored_logits, dim=-1)
                        restored_prob_target = restored_probs[target_token_id].item()
                        
                        # Calculate Average Indirect Effect (AIE)
                        # How much of the lost probability did we get back?
                        aie = (restored_prob_target - corrupted_prob_target) / max_restoration
                        results_by_layer[layer_idx].append(aie)
                        
                    finally:
                        hook.remove()
                        
        results = {}
        aie_list = []
        for l_idx, aies in results_by_layer.items():
            if aies:
                mean_aie = float(np.mean(aies))
                results[f"layer_{l_idx}_aie"] = mean_aie
                # We clamp negative AIE to a small positive epsilon for entropy calculation
                aie_list.append(max(mean_aie, 1e-10))
                
        # Find the center of causal effect and evaluate the entropy (localization vs distribution)
        if aie_list:
            max_layer = max((k for k in results if "_aie" in k), key=results.get)
            results["max_causal_layer"] = max_layer
            results["max_aie"] = results[max_layer]
            
            # Causal Entropy
            aie_arr = np.array(aie_list)
            # Normalize to form a probability distribution
            if np.sum(aie_arr) > 0:
                p_aie = aie_arr / np.sum(aie_arr)
                causal_entropy = -np.sum(p_aie * np.log(p_aie + 1e-12))
            else:
                causal_entropy = 0.0
            
            results["causal_entropy"] = float(causal_entropy)
            
        return results
