"""Packaging configuration guards."""

from pathlib import Path


def test_package_data_includes_defaults_and_assets():
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[tool.setuptools.package-data]" in text
    assert '"assets/*.json"' in text
    assert '"tasks/configs/*.yaml"' in text

