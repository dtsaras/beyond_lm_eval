"""geometry_spectral: a module whose analysis raises is counted, not hidden.

The per-module loop leaves a failing module out of every average. It was
logged at INFO only, so a run whose averages covered fewer modules looked
complete; _meta_n_modules_failed derives the count from what happened.
"""
import pytest

transformers = pytest.importorskip("transformers")


def test_a_failing_module_is_counted(monkeypatch):
    import blme.tasks.geometry.spectral as S
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    _register_all_tasks()
    name = "hf-internal-testing/tiny-random-gpt2"
    model = transformers.AutoModelForCausalLM.from_pretrained(name)
    tok = transformers.AutoTokenizer.from_pretrained(name)
    clean = get_task("geometry_spectral")({}).evaluate(model, tok, None)
    assert clean["_meta_n_modules_failed"] == 0

    real, calls = S.safe_svdvals, []

    def flaky(W, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected failure")
        return real(W, *a, **k)

    monkeypatch.setattr(S, "safe_svdvals", flaky)
    out = get_task("geometry_spectral")({}).evaluate(model, tok, None)
    assert out["_meta_n_modules_failed"] == 1
    assert out["_meta_n_modules_scanned"] == clean["_meta_n_modules_scanned"]
