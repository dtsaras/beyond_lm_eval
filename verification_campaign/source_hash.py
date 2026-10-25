#!/usr/bin/env python3
"""Fingerprint of everything that can move a task's numbers.

A verdict is a statement about a CODE STATE, so "is this verdict stale?" must be
answered mechanically. The fingerprint of a task covers:

  * the task's own module and its in-package IMPORT CLOSURE (every ``blme``
    module it imports, transitively, including imports nested in functions),
    with comments, docstrings and formatting stripped, so only what can change
    a number counts;
  * the RUNTIME modules every task runs through without importing them
    (``core.py`` decides dtype casts and forced-eager attention, ``cache.py``
    decides what a task receives, ``models/wrapper.py`` loads the model);
  * the task's RESOLVED CONFIG: shipped ``defaults.yaml`` merged with the
    study's overrides for that task, exactly as a study run sees it, rather
    than whole-file hashes that void all 51 verdicts on any edit;
  * the versions of the numerical libraries the reference pins depend on.

Usage::

    python source_hash.py                  # check every verdict in verdicts/
    python source_hash.py --launch TASK..  # fingerprints to embed at round START
    python source_hash.py --launch all
    python source_hash.py --self-test      # known-positive checks of the hasher

Verdicts are stamped at LAUNCH, against the tree the reviewer is about to read
(``fingerprint_at_launch``), never afterwards: stamping after the fact once
turned "0 of 102 verified" into "102 of 102" while every verdict predated the
code it was certified against. File mtimes are not used, because copying or
moving a verdict file refreshes them.

Package files are located from ``blme.__file__`` (set PYTHONPATH to test a
worktree), and a file that should exist but does not is an ERROR, never a skip.
"""
import ast
import glob
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICTS = os.path.join(HERE, "verdicts")
STUDY_TASKS_YAML = os.path.expanduser("~/projects/beyond_benchmarks/configs/tasks.yaml")

# Modules on every task's execution path that tasks do not import by name.
RUNTIME_MODULES = ("blme.core", "blme.cache", "blme.registry", "blme.results",
                   "blme.models.wrapper", "blme.tasks.base",
                   "blme.tasks.config_loader")
LIBRARIES = ("torch", "transformers", "numpy", "scipy", "scikit-learn",
             "ripser", "dionysus")


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


def source_hash(path, text=None):
    """Hash of a module's AST with docstrings removed (comments never reach it)."""
    if text is None:
        with open(path) as fh:
            text = fh.read()
    tree = _Strip().visit(ast.parse(text))
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _pkg_root():
    import blme
    return os.path.dirname(os.path.abspath(blme.__file__))


def module_file(modname):
    """File for an in-package module name, or None if it is not one."""
    if modname != "blme" and not modname.startswith("blme."):
        return None
    root = _pkg_root()
    rel = modname.split(".")[1:]
    base = os.path.join(root, *rel)
    for cand in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(cand):
            return cand
    return None


def _module_name(path):
    root = _pkg_root()
    rel = os.path.relpath(path, os.path.dirname(root))[:-3].split(os.sep)
    if rel[-1] == "__init__":
        rel = rel[:-1]
    return ".".join(rel)


def imports_of(path):
    """In-package modules imported anywhere in ``path`` (nested imports too)."""
    tree = ast.parse(open(path).read())
    me = _module_name(path)
    pkg = me if path.endswith("__init__.py") else me.rsplit(".", 1)[0]
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                base = parts[: len(parts) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            for a in node.names:            # `from pkg import submodule`
                out.add(f"{mod}.{a.name}")
    files = set()
    for m in out:
        f = module_file(m)
        if f is not None and f != path:
            files.add(f)
    return files


def _registration_only(path):
    """A package ``__init__`` that only imports task modules to register them.

    Following it would put every task into every closure (``core.py`` imports
    them all), so any edit anywhere would void all 51 verdicts.
    """
    if not path.endswith("__init__.py"):
        return False
    tree = ast.parse(open(path).read())
    return not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                  ast.Assign, ast.AnnAssign))
                   for n in tree.body)


def import_closure(path, skip=frozenset()):
    """``path`` plus every in-package module it reaches, not entering ``skip``."""
    seen, todo = set(), [path]
    while todo:
        p = todo.pop()
        if p in seen or (p != path and (p in skip or _registration_only(p))):
            continue
        seen.add(p)
        todo.extend(imports_of(p) - seen)
    return seen


def _registry():
    from blme.core import _register_all_tasks
    from blme.registry import _TASK_REGISTRY
    _register_all_tasks()
    return _TASK_REGISTRY


def task_files(task, registry=None):
    """Every package file whose change can move ``task``'s numbers."""
    reg = registry or _registry()
    cls = reg.get(task)
    if cls is None:
        raise KeyError(f"unknown task {task!r}")
    own = inspect.getfile(cls)
    files = import_closure(own)
    # The runtime path imports every task module to register it; those are
    # other tasks' code, not this task's dependencies, so do not enter them.
    other_tasks = {inspect.getfile(c) for c in reg.values()} - files
    for m in RUNTIME_MODULES:
        f = module_file(m)
        if f is None:
            raise FileNotFoundError(f"runtime module {m} not found under {_pkg_root()}")
        files |= import_closure(f, skip=frozenset(other_tasks))
    return sorted(files)


def resolved_config(task):
    from blme.tasks.config_loader import resolve_task_config
    overrides = {}
    if os.path.exists(STUDY_TASKS_YAML):
        import yaml
        with open(STUDY_TASKS_YAML) as fh:
            overrides = (yaml.safe_load(fh).get("task_configs") or {}).get(task) or {}
    return resolve_task_config(task, dict(overrides))


def library_versions():
    out = {}
    for lib in LIBRARIES:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = None
    return out


def fingerprint(task, registry=None, overrides_text=None):
    """``(digest, parts)``. ``overrides_text`` maps path -> replacement source
    (used only by the self-test to simulate an edit without touching disk)."""
    overrides_text = overrides_text or {}
    root = os.path.dirname(_pkg_root())
    files = task_files(task, registry)
    parts = {
        "files": {os.path.relpath(f, root): source_hash(f, overrides_text.get(f))
                  for f in files},
        "config": hashlib.sha256(json.dumps(resolved_config(task), sort_keys=True,
                                            default=repr).encode()).hexdigest()[:16],
        "libraries": library_versions(),
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return digest, parts


def paper_core_tasks():
    """The 51 paper-core task names, from the package repo's test fixtures."""
    repo = os.path.dirname(os.path.dirname(_pkg_root()))     # <repo>/src/blme
    with open(os.path.join(repo, "tests", "fixtures", "paper_core_tasks.json")) as fh:
        return json.load(fh)


def check_verdicts():
    reg = _registry()
    cache = {}
    ok, stale, unstamped = [], [], []
    paths = sorted(glob.glob(os.path.join(VERDICTS, "*.json")))
    for f in paths:
        d = json.load(open(f))
        task = d.get("task") or os.path.basename(f).split(".")[0]
        if task not in reg:
            stale.append((os.path.basename(f), "unknown task"))
            continue
        if task not in cache:
            cache[task] = fingerprint(task, reg)[0]
        rec = d.get("fingerprint_at_launch")
        if rec is None:
            unstamped.append(os.path.basename(f))
        elif rec != cache[task]:
            stale.append((os.path.basename(f), f"{rec} -> {cache[task]}"))
        else:
            ok.append(os.path.basename(f))
    n = len(paths)
    print(f"{len(ok)} of {n} verdicts VERIFIED against the current fingerprint")
    # Absence of a fingerprint is absence of evidence, so it counts as UNVERIFIED.
    print(f"{len(unstamped)} of {n} carry no fingerprint_at_launch -> UNVERIFIED")
    print(f"{len(stale)} of {n} are STALE")
    for name, why in stale:
        print(f"  {name:50s} {why}")
    return 0 if len(ok) == n else 1


def self_test():
    """Known positives and negatives: a hasher that misses an edit is worse
    than none, because it reports a stale verdict as verified."""
    reg = _registry()
    task = "geometry_svd"
    base, parts = fingerprint(task, reg)
    root = os.path.dirname(_pkg_root())
    files = task_files(task, reg)
    rels = set(parts["files"])
    must = {"blme/tasks/geometry/utils.py", "blme/core.py", "blme/cache.py",
            "blme/tasks/base.py", "blme/tasks/config_loader.py"}
    missing = must - rels
    assert not missing, f"closure misses {missing}"
    # 1. A semantic edit to a TRANSITIVE dependency must change the fingerprint.
    dep = os.path.join(root, "blme/tasks/geometry/utils.py")
    edited = open(dep).read() + "\n_SELF_TEST_SENTINEL = 1\n"
    assert fingerprint(task, reg, {dep: edited})[0] != base, "missed a dependency edit"
    # 2. A comment-only edit must NOT change it.
    commented = open(dep).read() + "\n# self-test comment\n"
    assert fingerprint(task, reg, {dep: commented})[0] == base, "comment changed hash"
    # 3. A runtime-module edit must change it.
    core = os.path.join(root, "blme/core.py")
    assert fingerprint(task, reg, {core: open(core).read() + "\nX_ = 2\n"})[0] != base
    # 4. An edit to an UNRELATED task's module must not change it.
    other = os.path.join(root, "blme/tasks/topology/zigzag.py")
    assert other not in files, "closure leaked into an unrelated task module"
    assert fingerprint(task, reg, {other: open(other).read() + "\nY_ = 3\n"})[0] == base
    # 5. Every paper-core task resolves, and every closure file exists.
    core_tasks = paper_core_tasks()
    sizes = {}
    for t in core_tasks:
        fs = task_files(t, reg)
        assert all(os.path.exists(f) for f in fs)
        sizes[t] = len(fs)
    print(f"self-test OK: {task} closure = {len(files)} files; "
          f"{len(core_tasks)} paper-core tasks resolved "
          f"(closure size min {min(sizes.values())}, max {max(sizes.values())})")


def main():
    argv = sys.argv[1:]
    if "--write" in argv:
        sys.exit("--write is retired: stamp at round START with --launch and "
                 "store the value as fingerprint_at_launch in each verdict.")
    if "--self-test" in argv:
        self_test()
        return
    if argv and argv[0] == "--launch":
        reg = _registry()
        names = argv[1:]
        if names == ["all"]:
            names = paper_core_tasks()
        out = {t: {"fingerprint_at_launch": fingerprint(t, reg)[0]} for t in names}
        print(json.dumps(out, indent=1, sort_keys=True))
        return
    sys.exit(check_verdicts())


if __name__ == "__main__":
    main()
