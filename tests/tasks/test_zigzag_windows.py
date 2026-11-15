"""topology_zigzag_persistence reads the last token of a 20-word window of each
passage, as the four Vietoris-Rips topology tasks do (blme.tasks._windows);
whole passages are ``points_readout: passage``. The sites, the tokenisation
without special tokens and the barcode are the paper's on either route.
"""
import statistics
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("ripser")
pytest.importorskip("dionysus")
sys.path.insert(0, str(Path(__file__).resolve().parent / "invariance"))

import _family_d as fd  # noqa: E402
from blme.cache import load_default_corpus  # noqa: E402
from blme.tasks._windows import window_set  # noqa: E402
from blme.tasks.config_loader import resolve_task_config  # noqa: E402
from blme.tasks.topology.zigzag import ZigzagPersistenceTask  # noqa: E402

N = 40
READOUT_KEYS = ("_meta_points_readout", "_meta_window_")


def _run(data, **cfg):
    cfg = resolve_task_config("topology_zigzag_persistence", dict(cfg, num_samples=N, seed=0))
    return ZigzagPersistenceTask(cfg).evaluate(fd.tiny_gpt2(), fd.tokenizer(), data)


@pytest.fixture(scope="module")
def data():
    return load_default_corpus(60)


@pytest.fixture(scope="module")
def default(data):
    return _run(data)


def test_default_route_reads_the_window_set(default, data):
    ws = window_set(data, N)
    tok = fd.tokenizer()
    counts = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in ws.texts]
    assert "error" not in default, default
    assert default["_meta_points_readout"] == "last_token_of_20_word_window"
    assert default["_meta_window_set_hash"] == ws.digest
    assert default["_meta_window_words"] == 20 and default["_meta_window_seed"] == 0
    assert default["_meta_zz_pool_size"] == N == len(ws.texts)
    assert default["_meta_window_tokens_mean"] == pytest.approx(statistics.fmean(counts))
    assert default["_meta_window_tokens_sd"] == pytest.approx(statistics.pstdev(counts))
    assert default["_meta_zz_n_docs_starting_with_bos"] == 0      # no special tokens


def test_nothing_outside_the_window_is_read(default, data):
    """The default result equals the passage route run on the window texts
    themselves: every key but the readout record, bit for bit."""
    only = [{"text": t} for t in window_set(data, N).texts]
    alone = _run(only, points_readout="passage")

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith(READOUT_KEYS)}
    a, b = strip(default), strip(alone)
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], float) and a[k] != a[k] and b[k] != b[k]:
            continue
        assert a[k] == b[k], (k, a[k], b[k])


def test_whole_passages_are_opt_in(default, data):
    passage = _run(data, points_readout="passage")
    assert passage["_meta_points_readout"] == "last_content_token_of_passage"
    assert not any(k.startswith("_meta_window_") for k in passage)
    assert passage["_meta_zz_pool_size"] == N
    moved = [k for k in ZigzagPersistenceTask.HEADLINE_KEYS if passage[k] != default[k]]
    assert moved, "the passage route read the same points as the windows"


def test_passages_shorter_than_the_window_are_an_error_that_says_so():
    res = _run(fd.dataset())
    assert set(res) == {"error"}
    assert "20-word window" in res["error"] and "points_readout" in res["error"]


def test_the_policy_names_the_window():
    """A paper_protocol policy (the norm2 sites are the paper's own), so the
    task is not one of test_word_windows.WINDOW_TASKS (last_per_sequence)."""
    tp = ZigzagPersistenceTask.TOKEN_POLICY
    assert tp.startswith("paper_protocol:Gardinazzi2025") and "20-word window" in tp
