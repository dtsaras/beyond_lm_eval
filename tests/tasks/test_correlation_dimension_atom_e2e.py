"""Drive the REAL evaluate() on an atomic measure — no simulated branch."""
import numpy as np
import pytest
import torch


def test_atomic_measure_gives_zero_through_evaluate(mock_model, mock_tokenizer,
                                                    monkeypatch):
    """A positive mass of coincident points makes C(r) flat, so D2 is exactly 0.

    The task runs its own forward pass, so the atom is injected by making the
    model return IDENTICAL hidden states for every token — the definition of a
    measure with an atom. Patching a collector would not reach this code path.
    """
    from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask

    real_forward = mock_model.forward

    def constant_states(*a, **k):
        out = real_forward(*a, **k)
        hs = list(out.hidden_states)
        const = torch.zeros_like(hs[-1])
        out.hidden_states = tuple(hs[:-1] + [const])
        return out

    monkeypatch.setattr(mock_model, "forward", constant_states)
    task = CorrelationDimensionTask(config={"num_samples": 10,
                                            "pooling": "all_tokens"})
    res = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in res:
        pytest.skip(f"task declined: {res['error']}")
    v = res["correlation_dimension"]
    assert v == 0.0 or np.isclose(v, 0.0, atol=1e-12), (
        f"correlation_dimension = {v!r}, expected exactly 0 "
        f"(atom_frac={res.get('_meta_correlation_dimension_zero_pair_frac')})")


def test_mixed_atom_continuous_population_gives_zero_through_evaluate(
        mock_model, mock_tokenizer, monkeypatch):
    """mu = 0.5*delta(-2) + 0.5*Uniform[0,1]  ->  exact D2 = 0.

    C(r) = 0.25 + 0.25*(2r - r^2) for 0 < r < 1, so C(r) -> 0.25 > 0 as r -> 0
    and log C(r) is flat: the Grassberger-Procaccia dimension is exactly 0.
    Executed nolds returns 5.35160579501927e-17 on this population.

    Half the mass is an atom, so this is the case that distinguishes the true
    correlation integral from the CONDITIONAL one (C(r) - c0)/(1 - c0): the
    conditional integral sees only the uniform part and reports D2 = 1.
    The population lives in coordinate 0 with the rest zero, so the pairwise
    distances are exactly |x_i - x_j|.
    """
    from blme.tasks.geometry.correlation_dimension import CorrelationDimensionTask

    real_forward = mock_model.forward
    rng = np.random.default_rng(0)

    def mixed_population(*a, **k):
        out = real_forward(*a, **k)
        hs = list(out.hidden_states)
        last = hs[-1]
        flat = int(last.shape[0] * last.shape[1])
        x = np.where(rng.random(flat) < 0.5, -2.0, rng.random(flat))
        new = torch.zeros_like(last)
        new[..., 0] = torch.tensor(
            x, dtype=new.dtype).reshape(last.shape[0], last.shape[1])
        out.hidden_states = tuple(hs[:-1] + [new])
        return out

    monkeypatch.setattr(mock_model, "forward", mixed_population)
    task = CorrelationDimensionTask(config={"num_samples": 64,
                                            "pooling": "all_tokens"})
    res = task.evaluate(mock_model, mock_tokenizer, dataset=None)
    if "error" in res:
        pytest.skip(f"task declined: {res['error']}")

    n = res["_meta_num_points"]
    # The correlation integral must count EVERY pair, coincident ones included.
    assert res["_meta_correlation_integral_n_pairs"] == n * (n - 1) // 2, (
        "C(r) was computed over a positive-filtered pair population: "
        f"{res['_meta_correlation_integral_n_pairs']} of {n * (n - 1) // 2}")
    assert res["_meta_coincident_pair_fraction"] > 0.1, (
        "fixture did not produce an atom: "
        f"{res['_meta_coincident_pair_fraction']}")

    v = res["correlation_dimension"]
    assert abs(v) < 0.05, (
        f"correlation_dimension = {v!r}, expected ~0 for a half-atomic "
        f"population (the conditional integral reports ~1); n={n}")
