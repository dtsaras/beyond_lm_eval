"""Packaging configuration guards."""

from pathlib import Path


def test_package_data_includes_defaults_and_assets():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in text
    assert '"assets/*.json"' in text
    assert '"tasks/configs/*.yaml"' in text


def test_test_extra_bundles_optional_test_deps():
    """Campaign-3 release blocker: CI installed only `.[dev]` + scikit-learn, so
    ripser/nltk/vendi_score were missing and collection errored / tests failed.
    A `test` extra must bundle the optional deps the full suite needs."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.9/3.10
        import tomli as tomllib

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "test" in extras, "missing a `test` optional-dependencies group"
    joined = " ".join(extras["test"]).lower()
    for dep in ("pytest", "scikit-learn", "ripser", "nltk", "vendi"):
        assert dep in joined, f"`test` extra should include {dep}"

