"""Tests for blme.tasks.config_loader — default loading and merge logic."""

import pytest
from blme.tasks.config_loader import get_default_configs, resolve_task_config, _DEFAULTS_CACHE
import blme.tasks.config_loader as config_loader_mod


@pytest.fixture(autouse=True)
def _reset_defaults_cache():
    """Clear the module-level cache before each test so file is re-read."""
    config_loader_mod._DEFAULTS_CACHE = None
    yield
    config_loader_mod._DEFAULTS_CACHE = None


def test_get_default_configs_returns_dict():
    defaults = get_default_configs()
    assert isinstance(defaults, dict)
    # Spot-check a few known task keys
    assert "geometry_lid" in defaults
    assert "geometry_svd" in defaults
    assert "consistency_calibration" in defaults


def test_default_configs_all_values_are_dicts():
    """Null values in YAML should be normalized to empty dicts."""
    defaults = get_default_configs()
    for task_name, cfg in defaults.items():
        assert isinstance(cfg, dict), f"Config for '{task_name}' is {type(cfg)}, expected dict"


def test_resolve_task_config_defaults_only():
    cfg = resolve_task_config("geometry_lid")
    assert isinstance(cfg, dict)
    assert cfg.get("k") == 20
    assert cfg.get("num_samples") == 50


def test_resolve_task_config_with_override():
    cfg = resolve_task_config("geometry_lid", user_overrides={"k": 10, "extra_param": True})
    assert cfg["k"] == 10
    assert cfg["extra_param"] is True
    # Original default for num_samples should still be present
    assert cfg.get("num_samples") == 50


def test_resolve_task_config_unknown_task():
    cfg = resolve_task_config("nonexistent_task_xyz")
    assert isinstance(cfg, dict)
    assert cfg == {}
