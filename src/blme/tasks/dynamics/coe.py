import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

from ...tasks.base import DiagnosticTask
from ...registry import register_task
from ..common import get_layers

@register_task("dynamics_coe")
class ChainOfEmbeddingTask(DiagnosticTask):
    """
    Implements Chain-of-Embedding (CoE) analysis (Wang et al., ICLR 2025).
    Measures Magnitude Change and Angle Change between adjacent hidden states
    during the generation trajectory, serving as an output-free self-evaluation.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Chain-of-Embedding (CoE)...")
        num_samples = self.config.get("num_samples", 5)
        generation_steps = self.config.get("generation_steps", 10)
        
        device = next(model.parameters()).device
        
        if dataset is None:
            try:
                from datasets import load_dataset
                dset = load_dataset("EleutherAI/lambada_openai", "en", split="test")
                dataset = []
                for i in range(min(num_samples, len(dset))):
                    dataset.append({"text": dset[i]["text"]})
            except ImportError:
                print("Warning: `datasets` library not found. Falling back to default examples.")
                dataset = [{"text": "The capital of France is Paris."}] * num_samples        
        samples = list(dataset)[:num_samples]
        if len(samples) < 1:
            return {"error": "Need at least 1 sample"}

        magnitude_changes = []
        angle_changes = []
        
        with torch.no_grad():
            for s in samples:
                if isinstance(s, dict) and "text" in s:
                    text = s["text"]
                else:
                    text = str(s)
                    
                input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
                
                # Generate step-by-step to capture the sequence of final layer hidden states
                trajectory_states = []
                
                # Pre-fill
                out = model(input_ids, output_hidden_states=True)
                # Take the last layer, last token hidden state
                current_state = out.hidden_states[-1][0, -1].float()
                trajectory_states.append(current_state)
                
                current_ids = input_ids
                
                for _ in range(generation_steps):
                    # For simplicity, do greedy decoding step
                    logits = out.logits[0, -1, :]
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    current_ids = torch.cat([current_ids, next_token], dim=-1)
                    
                    out = model(current_ids, output_hidden_states=True)
                    next_state = out.hidden_states[-1][0, -1].float()
                    trajectory_states.append(next_state)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                        
                # Compute CoE features: Magnitude & Angle changes
                sample_mags = []
                sample_angles = []
                
                for i in range(len(trajectory_states) - 1):
                    h_t = trajectory_states[i]
                    h_t_next = trajectory_states[i+1]
                    
                    # 1. Magnitude Change (L2 norm distance)
                    # Use norm of the difference as specified in some CoE formulations
                    # Or relative norm difference
                    mag_diff = torch.norm(h_t_next - h_t, p=2).item()
                    sample_mags.append(mag_diff)
                    
                    # 2. Angle Change (Arc Cosine of dot product between normalized states)
                    h_t_norm = F.normalize(h_t, p=2, dim=-1)
                    h_t_next_norm = F.normalize(h_t_next, p=2, dim=-1)
                    # Clamp to avoid NaN in acos due to numerical instability
                    cos_sim = torch.clamp(torch.dot(h_t_norm, h_t_next_norm), -1.0 + 1e-7, 1.0 - 1e-7)
                    angle = torch.acos(cos_sim).item()
                    sample_angles.append(angle)
                    
                if sample_mags and sample_angles:
                    magnitude_changes.append(sample_mags)
                    angle_changes.append(sample_angles)
                    
        # Pad sequences for nice matrix output if needed, or simply average
        # Since lengths can vary (due to EOS), we'll provide the mean across steps for each sample,
        # and then the global mean.
        
        sample_mean_mags = [np.mean(m) for m in magnitude_changes]
        sample_mean_angles = [np.mean(a) for a in angle_changes]
        
        # Flatten for global stats
        flat_mags = [item for sublist in magnitude_changes for item in sublist]
        flat_angles = [item for sublist in angle_changes for item in sublist]
        
        results = {}
        if flat_mags and flat_angles:
            results["mean_magnitude_change"] = float(np.mean(flat_mags))
            results["std_magnitude_change"] = float(np.std(flat_mags))
            results["mean_angle_change"] = float(np.mean(flat_angles))
            results["std_angle_change"] = float(np.std(flat_angles))
            
            # For deeper analysis, return the per-sample average changes
            results["per_sample_mean_magnitude"] = sample_mean_mags
            results["per_sample_mean_angle"] = sample_mean_angles
        else:
            results["error"] = "Failed to extract trajectory"
            
        return results
