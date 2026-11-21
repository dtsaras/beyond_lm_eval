"""A changed source tree must not report the last release's version.

Between v0.2.5 and v0.2.6, seven commits changed ``src/`` while
``_version.py`` still said 0.2.5, so their result envelopes claimed a release
they did not contain. The check compares ``src/`` with the tree of the newest
``v*`` tag reachable from HEAD.

Marked ``release``: ordinary development commits between releases are expected
to fail it until the version is bumped, so it runs in the release workflow
(``pytest -m release``), not on every commit. It skips where there is no git
checkout or no tag, as in an sdist or a shallow clone.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.release

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src" / "blme" / "_version.py"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _checkout_version() -> str:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', VERSION_FILE.read_text(), re.M)
    assert match, f"no __version__ in {VERSION_FILE}"
    return match.group(1)


def _newest_reachable_tag():
    """The highest-versioned ``v*`` tag reachable from HEAD, or a skip."""
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    try:
        if _git("rev-parse", "--is-inside-work-tree") != "true":
            pytest.skip("not a git checkout")
        if Path(_git("rev-parse", "--show-toplevel")).resolve() != ROOT:
            pytest.skip("the tests are not at the root of this git checkout")
        tags = _git("tag", "--merged", "HEAD", "--list", "v*", "--sort=-v:refname").split()
    except subprocess.CalledProcessError:
        pytest.skip("not a git checkout")
    if not tags:
        pytest.skip("no v* tag is reachable from HEAD (no tags, or a shallow clone)")
    return tags[0]


def _src_changes(tag: str) -> list[str]:
    """Paths under src/ whose working-tree content differs from the tag's tree."""
    tracked = _git("diff", "--name-only", tag, "--", "src").splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", "src").splitlines()
    return sorted(set(tracked) | set(untracked))


def test_changed_source_does_not_keep_the_released_version():
    tag = _newest_reachable_tag()
    released = tag.removeprefix("v")
    if _checkout_version() != released:
        return  # already bumped past the tag
    changed = _src_changes(tag)
    assert not changed, (
        f"src/ differs from {tag} but src/blme/_version.py still says {released}; "
        f"bump the version (and pyproject.toml) before releasing. Changed: {changed[:20]}")


def test_a_tagged_commit_reports_its_tag():
    tag = _newest_reachable_tag()
    at_head = _git("tag", "--points-at", "HEAD", "--list", "v*").split()
    if tag not in at_head:
        pytest.skip(f"HEAD is not the {tag} commit")
    assert _checkout_version() == tag.removeprefix("v"), (
        f"HEAD is tagged {tag} but src/blme/_version.py says {_checkout_version()}")
