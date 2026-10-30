"""geometry_phd_dimension: precision of the private forward and the GPTID protocol."""
import types

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
]


@pytest.fixture(scope="module")
def tiny64():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float64).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


def test_private_forward_keeps_float64_states(tiny64):
    """The per-text and pooled collectors take the states as the model emits
    them: a float64 model's rows reach the estimator unrounded (they were cast
    to float32 first)."""
    from blme.tasks.geometry.phd_dimension import _gather_point_cloud, _iter_per_document_clouds
    model, tok = tiny64
    data = [{"text": t} for t in TEXTS]
    with torch.no_grad():
        ref = [model(**tok(t, return_tensors="pt"), output_hidden_states=True)
               .hidden_states[-1][0].numpy() for t in TEXTS]
    assert any(not np.array_equal(r, r.astype(np.float32).astype(np.float64)) for r in ref)
    docs = list(_iter_per_document_clouds(model, tok, data, num_samples=2, max_length=64,
                                          layer_idx=-1, pooling="all_tokens"))
    assert all(np.array_equal(d, r) for d, r in zip(docs, ref))
    pooled = _gather_point_cloud(model, tok, data, num_samples=2, max_length=64, layer_idx=-1,
                                 pooling="all_tokens", cache=None, use_cache=False)
    assert np.array_equal(pooled, np.concatenate(ref))


# ── A stub model with a fixed per-text cloud ─────────────────────────────
# Text "doc i" is T_TOK tokens whose final hidden states are a fixed random
# cloud for document i; ``ends`` rescales the first and last row only.
T_TOK = 120


class _Tok:
    eos_token_id = None
    sep_token_id = None

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None, **kw):
        i = int(str(text).split()[-1])
        T = T_TOK if max_length is None else min(T_TOK, int(max_length))
        ids = torch.full((1, T), i, dtype=torch.long)
        return transformers.BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


class _CloudModel(torch.nn.Module):
    def __init__(self, ends=1.0, d=24):
        super().__init__()
        self.ends, self.d = ends, d
        self._anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        i, T = int(input_ids[0, 0]), input_ids.shape[1]
        g = torch.Generator().manual_seed(7 + i)
        # A curved low-dimensional cloud in R^d, so PHD is well defined.
        z = torch.rand(T, 3, generator=g, dtype=torch.float64)
        h = torch.cat([z, torch.sin(3 * z), torch.zeros(T, self.d - 6, dtype=torch.float64)], 1)
        h[0] *= self.ends
        h[-1] *= self.ends
        h = h.unsqueeze(0)
        return types.SimpleNamespace(hidden_states=(0.5 * h, h))


CFG = {"documents": "dataset", "num_samples": 6, "max_length": 512, "n_reruns": 1}


def _run(model, cfg=CFG):
    from blme.tasks.geometry.phd_dimension import PHDimensionTask
    data = [{"text": f"doc {i}"} for i in range(int(cfg["num_samples"]))]
    return PHDimensionTask(dict(cfg)).evaluate(model, _Tok(), data, cache=None)


def test_first_and_last_token_are_dropped():
    """GPTID's notebook estimates each text on outp[0][0][1:-1]: rescaling the
    first and last row of every text leaves the per-text PHD unchanged."""
    base, moved = _run(_CloudModel()), _run(_CloudModel(ends=50.0))
    assert np.isfinite(base["phd_dimension"])
    assert base["phd_dimension"] == moved["phd_dimension"]
    assert base["phd_dimension_std"] == moved["phd_dimension_std"]
    assert base["_meta_phd_n_end_rows_dropped"] == 2 * CFG["num_samples"]


def test_per_text_grid_is_the_notebook_grid():
    """GPTID's notebook grid on a 120-token text: mx = 118 rows, step =
    (118 - 40) // 7 = 11, sizes range(40, 107, 11) -> 7 sizes, 9 subsamples
    each. The value equals the kernel run on exactly that grid. (The old fixed
    range(50, n, 40) grid gave 2 sizes.)"""
    from blme.tasks.geometry.phd_dimension import _phd_dimension
    out = _run(_CloudModel())
    assert out["_meta_phd_fit_points_min"] == 7 and out["_meta_phd_fit_points_median"] == 7
    model = _CloudModel()
    vals = []
    for i in range(CFG["num_samples"]):
        ids = torch.full((1, T_TOK), i, dtype=torch.long)
        h = model(input_ids=ids).hidden_states[-1][0].numpy()[1:-1]
        vals.append(_phd_dimension(h, n_reruns=1, n_points=9, n_points_min=3, min_points=40,
                                   max_points=118 - 11, point_jump=11, seed=42))
    assert out["phd_dimension"] == pytest.approx(float(np.mean(vals)), rel=1e-12)
