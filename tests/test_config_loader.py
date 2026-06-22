"""Tests for blme.tasks.config_loader — default loading and merge logic."""

from pathlib import Path

import pytest
import yaml
from blme.core import _register_all_tasks
from blme.registry import list_tasks
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


def test_default_configs_cover_all_registered_tasks():
    """README promises every registered diagnostic task has bundled defaults."""
    _register_all_tasks()
    registered = set(list_tasks())
    defaults = set(get_default_configs())

    missing = sorted(registered - defaults)
    extra = sorted(defaults - registered)

    assert not missing, f"Registered tasks missing defaults.yaml entries: {missing}"
    assert not extra, f"defaults.yaml contains unregistered task entries: {extra}"


def test_default_all_recipe_matches_registered_tasks():
    """The exhaustive default recipe should not drift from the registry."""
    _register_all_tasks()
    registered = set(list_tasks())
    recipe_path = Path(__file__).resolve().parents[1] / "examples/recipes/default_all.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe_tasks = set(recipe["tasks"])

    missing = sorted(registered - recipe_tasks)
    extra = sorted(recipe_tasks - registered)

    assert not missing, f"default_all recipe is missing registered tasks: {missing}"
    assert not extra, f"default_all recipe contains unregistered tasks: {extra}"


def test_demo_recipe_uses_registered_tasks():
    """The public demo recipe should stay runnable with registered tasks."""
    _register_all_tasks()
    registered = set(list_tasks())
    recipe_path = Path(__file__).resolve().parents[1] / "examples/recipes/demo_recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe_tasks = set(recipe["tasks"])

    extra = sorted(recipe_tasks - registered)
    assert not extra, f"demo recipe contains unregistered tasks: {extra}"

    assert recipe.get("global", {}).get("device") == "cpu"
    assert recipe.get("model", {}).get("args", "").startswith("pretrained=")


def test_demo_recipe_dry_run_is_machine_readable(capsys):
    """YAML dry-run should emit only JSON so automation can parse it."""
    from blme.runner import run_from_yaml

    recipe_path = Path(__file__).resolve().parents[1] / "examples/recipes/demo_recipe.yaml"
    plan = run_from_yaml(str(recipe_path), dry_run=True, strict=True)
    captured = capsys.readouterr()

    parsed = yaml.safe_load(captured.out)
    assert parsed["recipe"].endswith("demo_recipe.yaml")
    assert parsed["unknown_tasks"] == []
    assert plan["diagnostic_tasks"] == parsed["diagnostic_tasks"]
