#!/usr/bin/env python3
"""Content hash of each task's source, with comments and docstrings stripped.

A verdict is a statement about a CODE STATE. Until now "is this verdict stale?"
was answered by file mtime or by the maintainer's memory of what he edited,
which is exactly the kind of judgement call this campaign keeps getting wrong:
on 2026-09-05 thirty verdicts went stale by mtime, most of them because a
one-line `post-final-norm:` marker was added to satisfy a new test, and there
was no way to tell those apart from the six that were real semantic changes.

So the hash ignores comments, docstrings and formatting, and covers only what
can change a number. Add `--write` to stamp the current hashes into each verdict
file; run without it to list which verdicts no longer match their source.
"""
import ast
import glob
import hashlib
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from blme.tasks import geometry, dynamics, consistency, interpretability  # noqa: E402,F401
from blme.tasks import causality, topology, representation_engineering  # noqa: E402,F401
from blme.registry import _TASK_REGISTRY  # noqa: E402


class _Strip(ast.NodeTransformer):
    """Drop docstrings; comments are already absent from the AST."""

    def _strip(self, node):
        self.generic_visit(node)
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


def source_hash(path):
    tree = _Strip().visit(ast.parse(open(path).read()))
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


# Modules every task's numbers can depend on without importing them by name.
# A change to `safe_unit_rows_torch` moves results in a dozen tasks whose own
# files are untouched, and the per-task hash cannot see that -- so it is tracked
# separately rather than left implicit.
SHARED = ("tasks/geometry/utils.py", "tasks/common.py", "cache.py",
          "normalization.py", "tasks/config_loader.py",
          # Not code, but merged INTO self.config and therefore able to move
          # every number a task reports. `dynamics_fractal`'s headline fell 79.5%
          # to a one-line config override, so a hash that ignored configs would
          # have called that verdict fresh.
          "tasks/configs/defaults.yaml")

# The study's own overrides win over `defaults.yaml`, so they belong here too.
SHARED_EXTERNAL = (
    os.path.expanduser("~/projects/beyond_benchmarks/configs/tasks.yaml"),
)


def shared_hash():
    h = hashlib.sha256()
    for rel in SHARED:
        path = os.path.join(ROOT, "src", "blme", rel)
        if not os.path.exists(path):
            continue
        if path.endswith(".py"):
            h.update(source_hash(path).encode())
        else:
            # Not parseable as Python, so hash the bytes -- which means a
            # COMMENT in a config counts as a change. That is the conservative
            # direction and configs are edited rarely.
            h.update(hashlib.sha256(open(path, "rb").read()).hexdigest().encode())
    for path in SHARED_EXTERNAL:
        if os.path.exists(path):
            h.update(hashlib.sha256(open(path, "rb").read()).hexdigest().encode())
    return h.hexdigest()[:16]


def task_hashes():
    out = {}
    for name, cls in _TASK_REGISTRY.items():
        try:
            out[name] = source_hash(inspect.getfile(cls))
        except Exception:
            pass
    return out


def main():
    write = "--write" in sys.argv
    hashes = task_hashes()
    shared = shared_hash()
    stale, ok, unstamped = [], [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "verification_campaign",
                                           "verdicts", "*.json"))):
        d = json.load(open(f))
        task = d.get("task") or os.path.basename(f).split(".")[0]
        cur = hashes.get(task)
        if cur is None:
            continue
        rec = d.get("source_hash")
        if write:
            d["source_hash"] = cur
            d["shared_hash"] = shared
            json.dump(d, open(f, "w"), indent=2)
            continue
        if d.get("shared_hash") not in (None, shared):
            stale.append((os.path.basename(f) + "  [shared helpers]",
                          d.get("shared_hash"), shared))
            continue
        if rec is None:
            unstamped.append(os.path.basename(f))
        elif rec != cur:
            stale.append((os.path.basename(f), rec, cur))
        else:
            ok.append(os.path.basename(f))

    if write:
        print(f"stamped {len(hashes)} task hashes into the verdict files")
        return
    print(f"{len(ok)} verdicts match their source")
    print(f"{len(unstamped)} verdicts carry no hash (written before stamping)")
    if stale:
        print(f"\n{len(stale)} verdicts are STALE -- the source changed "
              f"semantically since they were written:")
        for n, r, c in stale:
            print(f"  {n:46s} {r} -> {c}")


if __name__ == "__main__":
    main()
