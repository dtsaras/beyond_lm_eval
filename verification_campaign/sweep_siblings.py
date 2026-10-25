#!/usr/bin/env python3
"""Given a line you just changed, find every structurally similar line.

    python3 verification_campaign/sweep_siblings.py src/blme/tasks/topology/homology.py:191

The most repeated process failure of this campaign is fixing one instance of a
defect and not sweeping for the same shape elsewhere. Written down twice on
2026-09-07 and violated twice the same day, so it is now a command instead of a
rule:

  * the ripser unit-max rescale reached `homology` and not its 2 siblings
  * `safe_svdvals` was wired at 28 sites and not at `schatten`
  * `hidden.mean(dim=0).float()` was fixed in `representation_engineering` and
    left in 6 other files, including the family under review that hour

Matching is on AST SHAPE, not text: call/attribute structure with identifiers
erased, so a grep that guessed the wrong variable name cannot miss a site.

TWO passes, because exact shape is not enough. `hidden.mean(dim=0).float()` and
`h.float().mean(dim=0)` are the SAME defect in different order, and their
skeletons differ -- the first version of this tool claimed otherwise in this
docstring and found only 3 of the 6 known sites. The second pass therefore
matches the method-name MULTISET of the chain, which is order-insensitive.
"""
from __future__ import annotations

import ast
import pathlib
import sys


def skeleton(node: ast.AST) -> str:
    """Structural fingerprint: call/attribute shape, identifiers erased."""
    if isinstance(node, ast.Call):
        return f"C({skeleton(node.func)};{len(node.args)};{','.join(sorted(k.arg or '' for k in node.keywords))})"
    if isinstance(node, ast.Attribute):
        return f"A({skeleton(node.value)}.{node.attr})"
    if isinstance(node, ast.Name):
        return "N"
    if isinstance(node, ast.Constant):
        return f"K({type(node.value).__name__})"
    if isinstance(node, ast.Subscript):
        return f"S({skeleton(node.value)})"
    if isinstance(node, ast.BinOp):
        return f"B({skeleton(node.left)}{type(node.op).__name__}{skeleton(node.right)})"
    return type(node).__name__


def chain_methods(node: ast.AST) -> frozenset:
    """Multiset of method/attribute names in a call chain, order erased.

    Catches the reordering that exact-shape matching misses.
    """
    names = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Call):
            stack.append(n.func); stack.extend(n.args)
        elif isinstance(n, ast.Attribute):
            names.append(n.attr); stack.append(n.value)
    return frozenset(names)


def nodes_on_line(tree: ast.AST, line: int):
    return [n for n in ast.walk(tree)
            if getattr(n, "lineno", None) == line
            and isinstance(n, (ast.Call, ast.Attribute, ast.BinOp))]


def main() -> int:
    if len(sys.argv) != 2 or ":" not in sys.argv[1]:
        print(__doc__)
        return 2
    path, lineno = sys.argv[1].rsplit(":", 1)
    src = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    targets = nodes_on_line(tree, int(lineno))
    if not targets:
        print(f"no Call/Attribute/BinOp on {path}:{lineno}")
        return 1
    # the largest node on the line is the most specific shape
    _big = max(targets, key=lambda n: len(ast.dump(n)))
    want = {skeleton(_big)}
    want_chain = chain_methods(_big)
    print(f"shape : {sorted(want)[0]}")
    print(f"chain : {sorted(want_chain)}\n")

    hits = []
    for f in sorted(pathlib.Path("src/blme").rglob("*.py")):
        try:
            t = ast.parse(f.read_text())
        except SyntaxError:
            continue
        lines = f.read_text().splitlines()
        for n in ast.walk(t):
            if not isinstance(n, (ast.Call, ast.Attribute, ast.BinOp)):
                continue
            exact = skeleton(n) in want
            loose = bool(want_chain) and chain_methods(n) == want_chain
            if exact or loose:
                ln = getattr(n, "lineno", 0)
                hits.append((str(f), ln, lines[ln - 1].strip()[:74],
                             "exact" if exact else "reordered"))
    seen, out = set(), []
    for h in hits:
        if (h[0], h[1]) not in seen:
            seen.add((h[0], h[1]))
            out.append(h)
    same = sum(1 for f, _, _, _ in out if f == path)
    print(f"{len(out)} site(s) share this shape ({same} in the file you changed):\n")
    for f, ln, text, kind in out:
        mark = "  <-- the one you changed" if (f == path and ln == int(lineno)) else ""
        print(f"  [{kind:9s}] {f}:{ln}  {text}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
