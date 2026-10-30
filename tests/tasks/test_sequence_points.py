"""The one-point-per-sequence reader shared by the sequence-level geometry tasks."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from blme.cache import ModelOutputCache  # noqa: E402
from blme.tasks._tokens import select_rows, sink_keep_masks  # noqa: E402
from blme.tasks.geometry._sequence_points import _norm_columns, sequence_clouds  # noqa: E402

TEXTS = [
    "The cat sat on the mat and looked out of the window at the rain.",
    "Stock markets fell sharply on Tuesday after the central bank raised rates.",
    "In 1905 Einstein published four papers that changed physics forever.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the old wooden door and stepped into the dusty library.",
    "The recipe calls for two cups of flour, one egg and a pinch of salt.",
]


@pytest.fixture(scope="module")
def tiny():
    name = "hf-internal-testing/tiny-random-gpt2"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    except OSError as e:  # pragma: no cover - environment dependent
        pytest.skip(f"{name} not cached: {e}")
    return model, tok


def _cache(model, tok, n):
    c = ModelOutputCache(model, tok, dataset=[{"text": t} for t in TEXTS[:n]], num_samples=n)
    c.populate(need_hidden=True, need_logits=False)
    return c


@pytest.mark.parametrize("layer,rfn", [("all", True), (-1, False)])
def test_cache_and_forward_routes_agree_bit_for_bit(tiny, layer, rfn):
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    a = sequence_clouds(model, tok, data, _cache(model, tok, 6), n=6, layer=layer,
                        replace_final_norm=rfn, token_docs=4)
    b = sequence_clouds(model, tok, data, None, n=6, layer=layer,
                        replace_final_norm=rfn, token_docs=4)
    assert a.info["source"] == "shared_cache" and b.info["source"] == "self_collected"
    assert a.last.keys() == b.last.keys()
    for k in a.last:
        assert torch.equal(a.last[k], b.last[k])
        assert all(torch.equal(x, y) for x, y in zip(a.tokens[k], b.tokens[k]))
    assert all(torch.equal(x, y) for x, y in zip(a.keep, b.keep))
    assert a.info["last_is_block"] == b.info["last_is_block"] == rfn
    assert a.meta() == {**b.meta(), "_meta_points_source": "shared_cache"}


def test_one_row_per_document_is_its_last_token(tiny):
    model, tok = tiny
    c = _cache(model, tok, 6)
    sc = sequence_clouds(model, tok, None, c, n=6, layer="all", replace_final_norm=True)
    per_doc = c.get_hidden_states(layer_idx="all", per_sample=True, replace_final_norm=True)
    for k, rows in sc.last.items():
        assert rows.shape[0] == 6
        assert torch.equal(rows, torch.stack([d[-1] for d in per_doc[k]]))
    assert sc.meta()["_meta_n_points"] == 6


def test_a_cache_with_fewer_documents_than_n_is_not_used(tiny):
    """The cache cap cannot shrink N: a 3-document cache is bypassed for n=6."""
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    sc = sequence_clouds(model, tok, data, _cache(model, tok, 3), n=6, layer=-1,
                         replace_final_norm=False)
    assert sc.info["source"] == "self_collected"
    assert sc.info["cache_rejected"] == "too_few_docs"
    assert sc.n_points == 6


def test_a_dataset_shorter_than_n_is_logged(tiny, caplog):
    model, tok = tiny
    data = [{"text": t} for t in TEXTS]
    with caplog.at_level("WARNING", logger="blme"):
        sc = sequence_clouds(model, tok, data, None, n=1000, layer=-1, replace_final_norm=False)
    assert sc.n_points == 6 and sc.meta()["_meta_n_points_requested"] == 1000
    assert "6 of the 1000 requested sequences" in caplog.text


def test_norm_columns_give_the_full_row_masks():
    g = torch.Generator().manual_seed(0)
    depths = [[torch.randn(T, 16, generator=g, dtype=torch.float64) for T in (7, 5, 9)]
              for _ in range(3)]
    depths[1][2][4] *= 400.0
    depths[2][0][3] *= 400.0
    full, info_full = sink_keep_masks(depths)
    cols, info_cols = sink_keep_masks([_norm_columns(d) for d in depths])
    assert all(torch.equal(a, b) for a, b in zip(full, cols))
    assert info_full == info_cols and info_full["n_massive_dropped"] == 2


def test_token_cloud_drops_the_sink_rows(tiny):
    model, tok = tiny
    sc = sequence_clouds(model, tok, None, _cache(model, tok, 6), n=6, layer="all",
                         replace_final_norm=True, token_docs=6)
    for k in sc.tokens:
        rows, meta = sc.token_rows(k)
        assert meta.policy == "exclude_sinks"
        assert not bool((meta.position == 0).any())
        ref, _ = select_rows(sc.tokens[k], "exclude_sinks", keep=sc.keep)
        assert torch.equal(rows, ref)
