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


def _requirement_names(requirements):
    import re
    return {re.split(r"[\s;<>=!~\[]", r, maxsplit=1)[0].lower() for r in requirements}


def test_dionysus_is_an_optional_extra_that_the_paper_and_ci_install():
    """dionysus serves only topology_zigzag_persistence, so it is the `topology`
    extra rather than a core dependency; `paper` installs everything the
    paper-core tasks need, and `test` keeps it in CI so the zigzag tests run."""
    try:
        import tomllib
    except ModuleNotFoundError:  # py3.10
        import tomli as tomllib

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert "dionysus" not in _requirement_names(project["dependencies"])
    for extra in ("topology", "paper", "test", "all"):
        assert "dionysus==2.1.1" in extras[extra], f"`{extra}` extra should pin dionysus==2.1.1"
    assert set(extras["topology"]) | set(extras["diversity"]) <= set(extras["paper"])

