"""
Information Geometry (Empirical Fisher Trace) Task
──────────────────────────────────────────────────────────────────────
Evaluates the local curvature and sharpness of the learned representation space
by computing the Trace of the Empirical Fisher Information Matrix (FIM).

In information geometry, the FIM acts as a Riemannian metric tensor on the
statistical manifold of probability distributions. A high trace indicates 
sharpness (high sensitivity to parameter/representation changes), while a low
trace indicates a flatter, robust, and often better-generalizing manifold. 

Since computing the full FIM across all parameters is intractable, we compute
the empirical Fisher trace with respect to the continuous hidden states at the
final layer, before the unembedding projection.

References:
- "Information Geometry of Neural Networks" (Amari, 1998)
- "Understanding Deep Learning Requires Rethinking Generalization" (Zhang et al., 2017)
- 2024-2025 LLM Information Geometry literature.
"""

import torch
import numpy as np

from ...tasks.base import DiagnosticTask
from ...registry import register_task


@register_task("geometry_information_fisher")
class FisherInformationTraceTask(DiagnosticTask):
    """
    Computes the Trace of the Empirical Fisher Information Matrix (FIM)
    with respect to the final layer representations.
    
    A lower trace generally correlates with flatter minima, better generalization,
    and a robust topological manifold.
    """
    def evaluate(self, model, tokenizer, dataset):
        print("Running Information Geometry (Fisher Trace) Analysis...")
        num_samples = self.config.get("num_samples", 20)
        
        if dataset is None:
             dataset = [
                 {"text": "Information geometry studies probability distributions as a Riemannian manifold."}
             ] * num_samples
             
        samples = list(dataset)[:num_samples]
        if not samples:
             return {"error": "Need at least 1 sample."}
             
        device = next(model.parameters()).device
        
        fisher_traces = []
        
        # We need gradients with respect to the hidden states
        # so we cannot use torch.no_grad()
        model.eval()
        
        for s in samples:
            text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
            
            # Forward pass to get hidden states
            # We want to intercept the final hidden state before the LM head
            out = model(**inputs, output_hidden_states=True)
            
            # The last hidden state
            final_hidden = out.hidden_states[-1] # shape: (1, seq_len, hidden_dim)
            
            # We must detach, require grad, and pass through the LM head manually
            # to compute gradients of the log probabilities w.r.t the hidden state.
            h = final_hidden.detach().requires_grad_(True)
            
            # Reconstruct LM head forward pass (varies by model architecture)
            if hasattr(model, "lm_head"):
                logits = model.lm_head(h)
            elif hasattr(model, "cls"): # Some BERT variants
                logits = model.cls(h)
            else:
                # Fallback: just use the raw logits from the original output, 
                # but we can't backprop to our detached `h` easily unless we 
                # use a backward hook on the hidden states. 
                # Let's use a backward hook on the full model pass instead to be architecture agnostic.
                break
                
            # Efficient Empirical Fisher Trace computation:
            # Empirical Fisher Matrix F = (1/N) * sum_i (\nabla log P_i)(\nabla log P_i)^T
            # Trace(F) = (1/N) * sum_i ||\nabla log P_i||_2^2
            
            # We compute gradients w.r.t the representation `h`
            # For each token t, P is the predicted probability of the *true* or *argmax* token
            probs = torch.softmax(logits, dim=-1) # (1, seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Expected Fisher (sampling from model's own distribution) or Empirical Fisher (using argmax)
            # Typically, Empirical Fisher Trace uses the argmax token to avoid sampling variance
            preds = torch.argmax(logits, dim=-1) # (1, seq_len)
            
            trace_sum = 0.0
            seq_len = preds.shape[1]
            valid_tokens = 0
            
            for t in range(seq_len):
                if h.grad is not None:
                    h.grad.zero_()
                    
                target_log_prob = log_probs[0, t, preds[0, t]]
                
                # Backpropagate this single token's log probability
                # retain_graph=True because we iterate over sequence
                target_log_prob.backward(retain_graph=True)
                
                # The gradient w.r.t the specific token's hidden state
                grad_h = h.grad[0, t, :] # (hidden_dim,)
                
                # Add squared L2 norm of the gradient
                trace_sum += torch.sum(grad_h ** 2).item()
                valid_tokens += 1
                
            if valid_tokens > 0:
                fisher_traces.append(trace_sum / valid_tokens)
                
        # If the direct lm_head access failed (e.g. custom architecture), use a slower backward hook
        if not fisher_traces:
            for s in samples:
                text = s["text"] if isinstance(s, dict) and "text" in s else str(s)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                
                # Keep track of gradients of the last hidden layer
                h_grad = []
                def hook(module, grad_input, grad_output):
                    h_grad.append(grad_output[0].detach())
                    
                # Register hook to the actual backbone's last layer
                # (Highly architecture dependent, using a generic heuristic)
                if hasattr(model.base_model, "hidden_dropout_prob"): # BERT/RoBERTa
                    handle = getattr(model.base_model, 'encoder').layer[-1].register_backward_hook(hook)
                else: 
                     handle = model.base_model.register_backward_hook(hook) # Fallback, might not be exact last layer
                     
                out = model(**inputs)
                logits = out.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                trace_sum = 0.0
                seq_len = preds.shape[1]
                
                for t in range(seq_len):
                     model.zero_grad()
                     h_grad.clear()
                     target_log_prob = log_probs[0, t, preds[0, t]]
                     target_log_prob.backward(retain_graph=True)
                     
                     if h_grad:
                          grad = h_grad[0][0, t, :]
                          trace_sum += torch.sum(grad ** 2).item()
                          
                if seq_len > 0:
                     fisher_traces.append(trace_sum / seq_len)
                handle.remove()
                
        if not fisher_traces:
             return {"error": "Could not compute Fisher Trace (architecture incompatibility)."}
             
        mean_trace = float(np.mean(fisher_traces))
        
        return {
            "empirical_fisher_trace": mean_trace,
            "fisher_trace_std": float(np.std(fisher_traces)),
            "num_samples_analyzed": len(fisher_traces)
        }
