"""End-of-stack identity blocks for every key declared identity-block invariant.

``identity_blocks_interior`` covers blocks inserted between existing blocks.
This file runs every declaring task's real ``evaluate()`` with one exact
identity block before block 0 and one after the last block, and checks the
outcome recorded in ``_end_of_stack.OUTCOMES``: structural invariance, a known
negative, or a coincidence. The declaring tasks are enumerated from the
registry, so a new declaration fails here until its outcomes are recorded.
"""
from __future__ import annotations

import functools
import math

import pytest
import torch

import _family_d as fd
from _end_of_stack import INVARIANT, MIN_REL, NAN, OUTCOMES

inv = fd.inv

from blme.cache import load_default_corpus, load_long_documents  # noqa: E402
from blme.core import _register_all_tasks  # noqa: E402
from blme.registry import get_task, list_tasks  # noqa: E402

ENDS = ("leading", "trailing")
BUILDERS = {
    "gpt2": fd.tiny_gpt2,
    "llama": lambda: fd.tiny_llama(n_layers=3, seed=0),
    "llama4": lambda: fd.tiny_llama(n_layers=4, seed=3),
}


def _declared_identity_keys():
    _register_all_tasks()
    out = {}
    for name in sorted(list_tasks()):
        for key, (transforms, _node) in getattr(get_task(name), "SIZE_INVARIANT", {}).items():
            if any("identity_blocks" in t for t in transforms):
                out.setdefault(name, set()).add(key)
    return out


@functools.lru_cache(maxsize=None)
def _tok():
    return fd.tokenizer()


def _verify_ids():
    return [_tok()(t, return_tensors="pt")["input_ids"] for t in fd.TEXTS[:3]]


@functools.lru_cache(maxsize=None)
def _model(fixture, end=None):
    if end is None:
        return inv.final_norm_unembedding_gauge(BUILDERS[fixture](), D=None, seed=0)
    return inv.insert_end_identity_block(_model(fixture), end, verify_on=_verify_ids())


# --------------------------------------------------------------------------
# One runner per task, at the configuration of its interior test
# --------------------------------------------------------------------------

def _states(task, texts, config=None, **kw):
    def run(model):
        S = inv.states_from_model(model, _tok(), texts())
        return inv.run_task_on_states(task, S, config, **kw)
    return run


def _cached(task, texts, config=None, **kw):
    def run(model):
        return inv.run_task_on_model(task, model, _tok(), texts(), config=config, **kw)
    return run


def _evaluate(module, cls, cfg, data):
    def run(model):
        import importlib
        task_cls = getattr(importlib.import_module(f"blme.tasks.{module}"), cls)
        with inv.limit_threads(1):
            return task_cls(dict(cfg)).evaluate(model, _tok(), data())
    return run


def _corpus(n):
    return lambda: [d["text"] for d in load_default_corpus(n)]


def _docs(n):
    return lambda: [{"text": d["text"]} for d in load_default_corpus(n)]


def _long(n):
    return lambda: load_long_documents(n, split="validation")


def _lid(model):
    S = inv.states_from_model(model, _tok(), fd.TEXTS[:12])
    out = dict(inv.run_task_on_states("geometry_lid", S, {"k": 5}))
    out["lid_first_layer"] = inv.run_task_on_states(
        "geometry_lid", S, {"k": 5, "layerwise": True})["lid_first_layer"]
    return out


def _betti(model):
    S = inv.states_from_model(model, _tok(), fd.TEXTS)
    return inv.run_task_on_states("topology_betti_curve", S,
                                  {"num_samples": S.n_docs, "n_neighbors": 4},
                                  force=True, store_dtype=torch.float64)


def _dispersion_rows():
    from blme.tasks.geometry.dispersion import wikitext_validation_rows
    return [{"text": t} for t in wikitext_validation_rows()[:200]]


def _dime(model):
    from _nlpaug_side import require_nlpaug
    require_nlpaug()
    return _evaluate("geometry.dime", "DiMETask", {"num_samples": 48, "max_length": 64},
                     lambda: load_default_corpus(48))(model)


TWELVE = lambda: fd.TEXTS[:12]  # noqa: E731
RUNNERS = {
    "causality_layer_skip": _evaluate("causality.layer_skip", "LayerSkipEffectTask",
                                      {"num_samples": 3, "max_length": 64}, _docs(3)),
    "consistency_icl_score": _evaluate(
        "consistency.icl_score", "ICLScoreTask",
        {"num_samples": 3, "use_long_documents": False, "early_byte": 100, "late_byte": 800},
        _long(3)),
    "dynamics_coe": _evaluate("dynamics.coe", "ChainOfEmbeddingTask",
                              {"num_samples": 3, "max_new_tokens": 8}, fd.dataset),
    "dynamics_entropy_calibration": _evaluate(
        "dynamics.entropy_calibration", "EntropyCalibrationTask",
        {"num_samples": 3, "prompt_len": 16, "max_new_tokens": 24, "min_response_tokens": 24,
         "min_bytes": 0, "use_dataset": True, "batch_size": 2, "n_bootstrap": 20}, _long(3)),
    "dynamics_gradient_flow": _evaluate("dynamics.gradient_flow", "GradientFlowTask",
                                        {"num_samples": 4}, fd.dataset),
    "dynamics_interpolation": _evaluate(
        "dynamics.trajectories", "LatentInterpolationTask",
        {"num_samples": 8, "num_pairs": 6, "steps": 3, "seed": 0}, fd.dataset),
    "dynamics_perturbation_growth": _evaluate(
        "dynamics.perturbation_growth", "PerturbationGrowthTask",
        {"num_samples": 4, "rel_eps": 1e-2, "seed": 0}, fd.dataset),
    "geometry_alpha_req": _cached("geometry_alpha_req", _corpus(128)),
    "geometry_cka": _states("geometry_cka", TWELVE),
    "geometry_collapse": _states("geometry_collapse", TWELVE),
    "geometry_correlation_dimension": _states("geometry_correlation_dimension", _corpus(64),
                                              force=True),
    "geometry_delta_hyperbolicity": _evaluate(
        "geometry.delta_hyperbolicity", "DeltaHyperbolicityTask",
        {"num_samples": 5, "n_points": 24, "min_bytes": 300, "use_dataset": True},
        lambda: load_default_corpus(40)),
    "geometry_dime": _dime,
    "geometry_effective_depth": _evaluate("geometry.effective_depth", "EffectiveDepthTask",
                                          {"num_samples": 4}, lambda: fd.dataset(4)),
    "geometry_gride": _cached("geometry_gride", _corpus(96)),
    "geometry_hsic": _states("geometry_hsic", TWELVE),
    "geometry_information_imbalance": _cached("geometry_information_imbalance", _corpus(96)),
    "geometry_intrinsic_dim": _states("geometry_intrinsic_dim", TWELVE),
    "geometry_lid": _lid,
    "geometry_lipschitz": _states("geometry_lipschitz", lambda: fd.TEXTS[:6]),
    "geometry_lm_head_free_energy": _evaluate("geometry.lm_head_free_energy",
                                              "LMHeadFreeEnergyTask", {}, lambda: None),
    "geometry_magnitude": _states("geometry_magnitude", TWELVE),
    "geometry_matrix_entropy": _states("geometry_matrix_entropy", TWELVE),
    "geometry_neighborhood_overlap": _cached("geometry_neighborhood_overlap", _corpus(96)),
    "geometry_positional_decay": _evaluate("geometry.positional_decay",
                                           "PositionalAttentionDecayTask", {"num_samples": 3},
                                           lambda: fd.dataset(3)),
    "geometry_predictive_information": _evaluate(
        "geometry.predictive_information", "PredictiveInformationTask",
        {"num_samples": 80, "max_length": 64}, _docs(80)),
    "geometry_procrustes_linearity": _states("geometry_procrustes_linearity", TWELVE,
                                             store_dtype=torch.float64),
    "geometry_prompt_id": _evaluate("geometry.prompt_id", "PromptIntrinsicDimTask",
                                    {"num_samples": 4, "max_length": 96, "split": "validation"},
                                    lambda: None),
    "geometry_representation_dispersion": _evaluate(
        "geometry.dispersion", "RepresentationDispersionTask",
        {"chunk_len": 32, "n_chunks": 24, "use_dataset": True}, _dispersion_rows),
    "geometry_rogue_dimensions": _evaluate(
        "geometry.rogue_dimensions", "RogueDimensionsTask",
        {"num_samples": 4, "context_length": 32, "use_dataset": True}, _long(4)),
    "geometry_rsa": _states("geometry_rsa", TWELVE),
    "geometry_trajectory_curvature": _states("geometry_trajectory_curvature", lambda: fd.TEXTS),
    "geometry_vendi_score": _states("geometry_vendi_score", TWELVE),
    "geometry_zipfian_symmetry": _evaluate("geometry.zipfian_symmetry", "ZipfianSymmetryTask",
                                           {"use_dataset": True},
                                           lambda: load_default_corpus(20)),
    "interpretability_logit_lens_kl": _cached("interpretability_logit_lens_kl", _corpus(24),
                                              use_cache=False),
    "topology_betti_curve": _betti,
}


@functools.lru_cache(maxsize=None)
def _result(task, fixture, end=None):
    out = RUNNERS[task](_model(fixture, end))
    assert "error" not in out, out
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_outcomes_cover_every_identity_block_declaration():
    """The record and the registry's identity-block declarations agree, key
    for key, and every recorded task has a runner."""
    declared = _declared_identity_keys()
    recorded = {t: set(keys) for t, keys in OUTCOMES.items()}
    assert recorded == declared
    assert set(RUNNERS) == set(OUTCOMES)


def _gained_a_state(base, out, witness):
    a, b = base[witness], out[witness]
    if isinstance(a, list):
        return any(x not in a for x in b)
    return inv.rel_change(a, b) >= MIN_REL


@pytest.mark.parametrize("end", ENDS)
@pytest.mark.parametrize("task", sorted(OUTCOMES))
def test_end_of_stack_outcome(task, end):
    i = ENDS.index(end)
    for key, outcome in sorted(OUTCOMES[task].items()):
        outcome = outcome[i]
        if outcome == INVARIANT or outcome[0] == "unchanged":
            for fx in ("gpt2", "llama"):
                inv.assert_invariant(_result(task, fx), _result(task, fx, end), [key], rtol=0.0)
            if outcome != INVARIANT:
                witness = outcome[1]
                assert any(_gained_a_state(_result(task, fx), _result(task, fx, end), witness)
                           for fx in ("gpt2", "llama")), (key, witness)
        elif outcome == NAN:
            for fx in ("gpt2", "llama"):
                assert math.isnan(_result(task, fx)[key])
                assert math.isnan(_result(task, fx, end)[key])
        else:
            assert outcome[0] == "moves" and outcome[1:], outcome
            for fx in outcome[1:]:
                inv.assert_moves(_result(task, fx), _result(task, fx, end), key, MIN_REL)
