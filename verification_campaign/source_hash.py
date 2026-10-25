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
import time

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


def task_source_files(task):
    """Every file whose change invalidates a verdict for `task`.

    The task's own module plus the shared helpers, since a change to
    `geometry/utils.py` can alter a task that never mentions it.
    """
    files = []
    cls = _TASK_REGISTRY.get(task)
    if cls is not None:
        try:
            files.append(inspect.getfile(cls))
        except Exception:
            pass
    for rel in SHARED:
        cand = os.path.join(ROOT, "src", "blme", rel)   # same base as shared_hash()
        if os.path.exists(cand):
            files.append(cand)
    for path in SHARED_EXTERNAL:
        if os.path.exists(path):
            files.append(path)
    return files


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
    stamped, skipped = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "verification_campaign",
                                           "verdicts", "*.json"))):
        d = json.load(open(f))
        task = d.get("task") or os.path.basename(f).split(".")[0]
        cur = hashes.get(task)
        if cur is None:
            continue
        rec = d.get("source_hash")
        if write:
            # REFUSE TO STAMP A VERDICT OLDER THAN THE CODE IT COVERS.
            #
            # `--write` used to stamp unconditionally. Run after a green suite
            # on 2026-09-08 it turned the accounting from "0 of 102 verified"
            # into "102 of 102 verified against current source" and reported
            # success — while 51 Opus verdicts predated ~49 fixes and 51 codex
            # verdicts predated 17 more. Neither reviewer had seen the tree it
            # certified them against. That is worse than carrying no hash: it
            # manufactures provenance, and it silently inverts the campaign's
            # headline number.
            #
            # A verdict may only be stamped if the source it covers has not
            # changed since the verdict was written. Stamping belongs at the
            # START of a round, against the tree the reviewer is about to read.
            _vm = os.path.getmtime(f)
            _srcs = task_source_files(task)
            _newest = max([os.path.getmtime(x) for x in _srcs] or [0.0])
            if _newest > _vm:
                skipped.append((os.path.basename(f),
                                time.strftime("%m-%d %H:%M", time.localtime(_vm)),
                                time.strftime("%m-%d %H:%M", time.localtime(_newest))))
                continue
            d["source_hash"] = cur
            d["shared_hash"] = shared
            json.dump(d, open(f, "w"), indent=2)
            stamped.append(os.path.basename(f))
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
        print(f"stamped {len(stamped)} verdicts")
        if skipped:
            print(f"REFUSED to stamp {len(skipped)} verdicts whose source changed "
                  f"after they were written:")
            for n, vm, sm in skipped:
                print(f"  {n:46s} verdict {vm}  source {sm}")
        if not stamped:
            print("nothing stamped — a verdict written before its own source "
                  "cannot be certified against it")
            sys.exit(1)
        return
    print(f"{len(ok)} verdicts VERIFIED against their source")
    # UNSTAMPED IS NOT VERIFIED.
    #
    # These used to be reported on their own line and then quietly excluded
    # from the stale list, so a consumer counting "stale" saw a smaller number
    # than the number of unverified verdicts -- 32 of them, on 2026-09-07,
    # sitting in neither bucket. An absent hash is not evidence that the source
    # is unchanged; it is the absence of evidence, and the conservative reading
    # is the only safe one when the whole point of the mechanism is to stop
    # unverified claims being counted as verified.
    print(f"{len(unstamped)} verdicts carry NO HASH -> counted as UNVERIFIED "
          f"(written before stamping; absence of a hash is not a match)")
    print(f"{len(ok)} of {len(ok) + len(unstamped) + len(stale)} verdicts are "
          f"verified against current source")
    if stale:
        print(f"\n{len(stale)} verdicts are STALE -- the source changed "
              f"semantically since they were written:")
        for n, r, c in stale:
            print(f"  {n:46s} {r} -> {c}")


if __name__ == "__main__":
    main()
