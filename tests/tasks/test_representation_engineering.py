

def test_concept_separability_imbalanced_labels_degrade_to_error():
    """Audit 2026-07-16: labels [1,1,1,0] previously crashed evaluate()
    (StratifiedKFold produced a one-class train fold and LogisticRegression
    raised out of the task, killing the study run). Must return an error
    dict instead."""
    import numpy as np
    import torch
    from blme.tasks.representation_engineering import ConceptSeparabilityTask

    task = ConceptSeparabilityTask(config={"num_samples": 4})
    ds = [
        {"text": "alpha", "label": 1},
        {"text": "beta", "label": 1},
        {"text": "gamma", "label": 1},
        {"text": "delta", "label": 0},
    ]

    class TinyLM(torch.nn.Module):
        def __init__(self, D=8):
            super().__init__()
            self.emb = torch.nn.Embedding(50, D)

        def forward(self, input_ids=None, attention_mask=None,
                    output_hidden_states=False, **kw):
            h = self.emb(input_ids)
            class Out: pass
            o = Out()
            o.hidden_states = (h, h + 1.0)
            return o

        def get_input_embeddings(self):
            return self.emb

    class Tok:
        def __call__(self, text, **kw):
            ids = torch.tensor([[1, 2, 3]])
            class Enc(dict):
                def to(self, device): return self
            e = Enc(); e["input_ids"] = ids
            return e

    out = task.evaluate(TinyLM(), Tok(), dataset=ds)
    assert "error" in out and "imbalanced" in out["error"]
