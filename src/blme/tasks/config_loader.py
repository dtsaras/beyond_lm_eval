"""
Task configuration loading for BLME.

Loads default task configs from the bundled YAML file and merges
them with user-provided overrides from recipe files or CLI.
"""

import os
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger("blme")

_DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "configs", "defaults.yaml")
_DEFAULTS_CACHE: Optional[Dict[str, dict]] = None


def get_default_configs() -> Dict[str, dict]:
    """
    Load default task configurations from the bundled defaults.yaml.

    Returns:
        Dict mapping task_name -> config dict.
    """
    global _DEFAULTS_CACHE
    if _DEFAULTS_CACHE is not None:
        return _DEFAULTS_CACHE

    if not os.path.exists(_DEFAULTS_PATH):
        logger.warning(f"Default task configs not found at {_DEFAULTS_PATH}")
        _DEFAULTS_CACHE = {}
        return _DEFAULTS_CACHE

    with open(_DEFAULTS_PATH, "r") as f:
        _DEFAULTS_CACHE = yaml.safe_load(f) or {}

    # Ensure every value is a dict (some tasks have `{}` or `null`)
    for k, v in _DEFAULTS_CACHE.items():
        if v is None:
            _DEFAULTS_CACHE[k] = {}

    logger.debug(f"Loaded default configs for {len(_DEFAULTS_CACHE)} tasks")
    return _DEFAULTS_CACHE


def resolve_task_config(
    task_name: str,
    user_overrides: Optional[dict] = None,
) -> dict:
    """
    Merge default config with user overrides for a specific task.

    Priority: user overrides > defaults.yaml

    Args:
        task_name: Registered task name (e.g., 'geometry_svd').
        user_overrides: Dict of user-provided config values.

    Returns:
        Merged config dict.
    """
    defaults = get_default_configs()
    base = dict(defaults.get(task_name, {}))  # copy so we don't mutate cache

    if user_overrides:
        base.update(user_overrides)

    return base
