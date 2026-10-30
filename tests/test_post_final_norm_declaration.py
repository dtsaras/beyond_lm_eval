"""EVERY hidden-state read must DECLARE what it does with HF's last entry.

This is the most-repeated defect in the whole verification campaign. HuggingFace's
final `output_hidden_states` entry is ``ln_f(b_{L-1})`` -- a LayerNorm output, not
a block output -- and the true final block output is **not in the tuple at all**.
A task that treats entry L as "the deepest layer" is comparing a normalised
vector against L unnormalised ones.

It was found, fixed, and then found somewhere ELSE, sixteen times across six
review rounds. A repo-wide sweep on 2026-09-05 finally counted the population:
**24 of 39** `get_hidden_states`/`collect_hidden_states` call sites had never
opted in. Every previous round had been sampling from that population a couple
of sites at a time, which is why it kept "recurring" -- it was never fixed, only
locally patched.

So the default is no longer allowed to decide. Every call site must pass
``replace_final_norm`` explicitly:

  * ``True``  -- the site builds a PER-LAYER PROFILE, or otherwise compares
    entries against each other. Mixing one LayerNorm output into a profile of
    block outputs is a category error under either convention.
  * ``False`` -- the site wants THE MODEL'S OUTPUT REPRESENTATION, which is
    genuinely post-norm (this is what most papers mean by "the final layer").
    Legitimate, but it must be a decision on the record, not a default.

**And a third clause, which the first pass of this sweep got wrong twice: THE
RELEASED REFERENCE WINS.** Two tasks look like per-layer profiles and are still
``False``, because the code they reproduce made the choice already:

  * ``geometry_matrix_entropy`` -- Wei et al.'s recipe is
    ``AutoModel(...)[0]``, which is bit-identical to ``hidden_states[-1]``.
    Hooking it moved gpt2 +93.3% off the reference and turned a published
    70m-vs-160m gap from +6.3% into +105.5%.
  * ``geometry_contextualization`` -- ``kawine/contextual``'s vendored GPT-2
    literally does ``all_hidden_states.pop()  # ignore the last layer prior to
    layer norm`` and appends ``ln_f(...)``.

Matching a released reference IS faithfulness; a mixed-convention profile that
the reference itself has is a ``comparable`` concern, not a ``verdict`` one.
Both tasks now collect the hooked profile too, where they need one.

Direct ``out.hidden_states[...]`` indexing that can reach the last entry is
covered by the second test, which requires an explicit marker comment.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "blme"
# The shared gates wrap the two collectors and REQUIRE the keyword; tasks that
# moved onto them are still hidden-state reads that must declare it.
COLLECTORS = {"get_hidden_states", "collect_hidden_states", "hidden_states_for_task",
              "sequence_clouds"}

# The two functions that IMPLEMENT the option, plus the cache's own internals.
IMPLEMENTATION_SITES = {
    ("cache.py", "get_hidden_states"),
    ("tasks/geometry/utils.py", "collect_hidden_states"),
}


def _call_sites():
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        src = path.read_text()
        try:
            tree = ast.parse(src)
        except SyntaxError:  # pragma: no cover
            continue
        # Skip the bodies of the two functions that define the parameter.
        defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if name not in COLLECTORS:
                continue
            if (rel, name) in IMPLEMENTATION_SITES and name in defs:
                # A recursive/self call inside the implementing module.
                pass
            yield rel, node.lineno, name, {k.arg for k in node.keywords}


def test_every_hidden_state_collector_call_declares_replace_final_norm():
    missing = [f"{rel}:{line}  {name}(...)"
               for rel, line, name, kw in _call_sites()
               if "replace_final_norm" not in kw]
    assert not missing, (
        "These hidden-state reads leave the post-final-norm question to a "
        "default. Pass `replace_final_norm=True` for a per-layer profile, or "
        "`replace_final_norm=False` with a one-line reason for a site that "
        "deliberately wants the model's post-norm output representation:\n  "
        + "\n  ".join(missing))


def test_the_sweep_still_sees_every_call_site():
    """A guard on the guard: if the collectors are renamed, the test above
    silently passes on an empty set. The shared gates absorbed most of the 39
    original sites (28 remain), so the bar is loose and per name: it only has
    to catch "the AST walk found nothing"."""
    sites = list(_call_sites())
    assert len(sites) >= 20, f"only {len(sites)} collector call sites found; did they get renamed?"
    unseen = COLLECTORS - {name for _, _, name, _ in sites}
    assert not unseen, f"no call site found for {sorted(unseen)}; renamed?"


# ── Direct `out.hidden_states[...]` indexing ──────────────────────────────
# A site that indexes the tuple itself bypasses the collectors entirely. It is
# only safe if it (a) never reaches the last entry, or (b) says out loud that
# post-norm is what it wants. `MARKER` is that statement.
# Either punctuation counts as a declaration: sites that SUBSTITUTE tend to
# write "post-final-norm: SUBSTITUTED", sites that keep the post-norm state
# tend to write "post-final-norm=False, DELIBERATELY".
MARKERS = ("post-final-norm:", "post-final-norm=")
MARKER = MARKERS[0]   # what the failure message tells the author to write

# AST, not text: prose in docstrings and comments mentions `hidden_states[...]`
# constantly, and a grep-based guard would demand markers on documentation.
#
# ALIASES ARE FOLLOWED. The first version of this guard matched only the
# literal name `hidden_states`, and an independent review immediately defeated
# it: **48 subscripts** reach the last entry through a rename, and the idiom
# that does it is the one the fixes themselves introduced --
#
#     hs = list(out.hidden_states)
#     ...
#     hs[-1] = _h_raw
#
# `geometry_phd_dimension` was the live example: it declares
# `replace_final_norm=False` on its cache route while its two private-forward
# routes index `hs[li]` / `hs[idx]` with no note anywhere in the file. A guard
# that the fix pattern evades is not a guard.
def _alias_names(tree):
    """Local names bound DIRECTLY to a `hidden_states` tuple.

    "Directly" is the whole point: propagating through any expression that
    merely mentions an alias makes every downstream dict and list look like a
    hidden-state tuple. Only these forms bind an alias:

        hs = out.hidden_states          Attribute
        hs = other_alias                Name
        hs = list(alias) / tuple(alias) Call to list/tuple
        hs = alias[1:]                  Subscript with a Slice
        hs = tuple(alias[:-1]) + (h,)   BinOp of the above
    """
    names = {"hidden_states"}

    def _derives(node):
        if isinstance(node, ast.Attribute):
            return node.attr in names
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Subscript):
            # `out["hidden_states"]` -- the HF mapping idiom, and the one the
            # RepE reference itself uses. Quoted in this repo at
            # `representation_engineering.py`, and it defeated the guard until
            # 2026-09-06.
            sl = node.slice
            if (isinstance(sl, ast.Constant) and isinstance(sl.value, str)
                    and sl.value in names):
                return True
        if isinstance(node, ast.IfExp):
            # `out.hidden_states if x else []`
            return _derives(node.body) or _derives(node.orelse)
        if isinstance(node, ast.BoolOp):
            # `out.hidden_states or []` -- the same defensive family as the
            # `getattr` form that defeated an earlier version.
            return any(_derives(v) for v in node.values)
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in (
                    "list", "tuple", "reversed", "sorted", "iter", "enumerate"):
                # `reversed` and `sorted` REORDER, which is why index 0 is no
                # longer treated as provably safe -- see `_cannot_reach_last`.
                return any(_derives(a) for a in node.args)
            # `getattr(out, "hidden_states", None)` -- the defensive idiom, and
            # the one that defeated the FIRST alias fix. An independent review
            # found `interpretability/activation_sinks.py` building a per-layer
            # profile over `hs[1:]` through exactly this form, with no hook and
            # no declaration, while the guard reported all-clear.
            if isinstance(fn, ast.Name) and fn.id == "getattr" and node.args:
                return any(isinstance(a, ast.Constant) and a.value in names
                           for a in node.args[1:2])
            return False
        if isinstance(node, ast.Subscript):
            return isinstance(node.slice, ast.Slice) and _derives(node.value)
        if isinstance(node, ast.BinOp):
            return _derives(node.left) or _derives(node.right)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp,
                             ast.DictComp)):
            # `[h.detach() for h in out.hidden_states]` -- the form that hid
            # `causality/ablation.py` from the previous version of this guard.
            return any(_derives(g.iter) for g in node.generators)
        return False

    def _bind(target, value):
        """Add `target` if `value` derives from a tuple, unpacking as needed."""
        if isinstance(target, ast.Name) and _derives(value):
            names.add(target.id)
        elif isinstance(target, ast.Starred) and _derives(value):
            # `first, *rest = out.hidden_states`
            if isinstance(target.value, ast.Name):
                names.add(target.value.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # `a, b = out.hidden_states, x` binds `a`; `for i, h in
            # enumerate(hs)` binds `h`. Conservative: bind every element.
            for elt in target.elts:
                if _derives(value):
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
                    elif isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                        names.add(elt.value.id)

    # Several passes: an alias can be bound from another alias.
    #
    # ASSIGNMENTS ARE NOT THE ONLY BINDING FORM, and assuming they were is how
    # this guard was defeated a third time. An independent review found
    # `interpretability/probing.py` iterating `for layer_idx, hs in
    # enumerate(hidden_states)` -- a per-layer profile that includes the `ln_f`
    # entry -- and `causality/ablation.py` binding through a list
    # comprehension, both invisible to an `ast.Assign`-only walk (2026-09-05).
    for _ in range(3):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # EVERY target, not just a single one: `keep = hs = out.hidden_states`
                # has two, and requiring exactly one let it through.
                for _t in node.targets:
                    _bind(_t, node.value)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                # `hs: tuple = out.hidden_states` is an AnnAssign, not an Assign.
                _bind(node.target, node.value)
            elif isinstance(node, ast.AugAssign):
                # `hs += out.hidden_states`
                _bind(node.target, node.value)
            elif isinstance(node, ast.NamedExpr):          # walrus
                _bind(node.target, node.value)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                it = node.iter
                # `for h in hs:` and `for i, h in enumerate(hs):`
                if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                        and it.func.id in ("enumerate", "reversed", "list",
                                           "tuple", "sorted") and it.args):
                    it = it.args[0]
                _bind(node.target, it)
            elif isinstance(node, (ast.ListComp, ast.GeneratorExp,
                                   ast.SetComp, ast.DictComp)):
                for gen in node.generators:
                    it = gen.iter
                    if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                            and it.func.id in ("enumerate", "reversed", "list",
                                               "tuple", "sorted") and it.args):
                        it = it.args[0]
                    _bind(gen.target, it)
    return names


def _alias_resolver(tree):
    """Return `lookup(node) -> alias set in force at that node`.

    Extracted so `_subscripts` and `_iterations` genuinely share one alias
    model. They did not: `_iterations` built its set from MODULE-level bindings
    only while `_subscripts` tracked them per function, so
    `hs = list(out.hidden_states)` bound INSIDE a function and then only
    ITERATED was invisible to both -- untracked by one and unexamined by the
    other. 16 shipping sites already use that shape (Opus round 9, 2026-09-07).
    `_iterations`' own docstring claimed it detected "over the same alias set
    as `_subscripts`", which was simply untrue.
    """
    module_only = ast.Module(
        body=[n for n in tree.body
              if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))],
        type_ignores=[])
    base = _alias_names(module_only)

    # Parent chain, so a CLOSURE sees the aliases of the function enclosing it.
    parent = {}
    def _walk_fns(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent[id(child)] = list(chain)
                _walk_fns(child, chain + [child])
            else:
                _walk_fns(child, chain)
    _walk_fns(tree, [])

    # innermost enclosing function for every node that has one
    owner = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(fn):
            prev = owner.get(id(n))
            if prev is None or (
                    getattr(fn, "end_lineno", 0) - fn.lineno
                    < getattr(prev, "end_lineno", 0) - prev.lineno):
                owner[id(n)] = fn

    cache = {}

    def lookup(node):
        fn = owner.get(id(node))
        if fn is None:
            return base
        if id(fn) not in cache:
            a = base | _alias_names(fn)
            for anc in parent.get(id(fn), []):
                a |= _alias_names(anc)
            cache[id(fn)] = a
        return cache[id(fn)]

    return lookup


def _subscripts(tree):
    """Hidden-state subscripts, with aliases tracked PER ENCLOSING FUNCTION.

    Module-wide alias tracking taints a name everywhere it appears: `chain` is
    bound from a hidden-state tuple inside `evaluate` and is an ordinary
    PARAMETER of `_coe_from_chain`, and a module-scoped tracker demanded a
    marker on the second. Requiring declarations at sites that never touch the
    tuple is how a guard trains people to add the marker without thinking.
    """
    # Per-function aliases, incl. the enclosing-function chain, so a CLOSURE
    # sees the outer function's bindings. Shared with `_iterations` -- see
    # `_alias_resolver`.
    lookup = _alias_resolver(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        aliases = lookup(node)
        v = node.value
        name = (v.attr if isinstance(v, ast.Attribute)
                else v.id if isinstance(v, ast.Name) else None)
        if name not in aliases:
            continue
        if (isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            continue
        yield node


def _cannot_reach_last(node):
    """True only for an index that provably excludes the final entry.

    INDEX 0 IS NOT SUCH AN INDEX. It was whitelisted as "obviously the embedding
    entry" until `hs = list(reversed(out.hidden_states)); hs[0]` walked through
    the hole -- doubly invisible, because the alias was untracked AND the index
    was waved past. An alias can be reversed, sorted, or rebuilt in any order,
    so position 0 says nothing about which entry it holds. Only a slice with a
    negative constant upper bound (`[:-1]`) provably drops the tail.
    """
    sl = node.slice
    # `hidden_states[1:]` still reaches the last entry; `[:-1]` does not.
    if isinstance(sl, ast.Slice) and sl.upper is not None:
        up = sl.upper
        if (isinstance(up, ast.UnaryOp) and isinstance(up.op, ast.USub)
                and isinstance(up.operand, ast.Constant)):
            return True
    return False


def _enclosing_spans(tree):
    """(start, end) line spans of every function/method, innermost first."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if end:
                spans.append((node.lineno, end))
    spans.sort(key=lambda s: s[1] - s[0])
    return spans


def _iterations(tree):
    """`for h in out.hidden_states:` / `enumerate(...)` over an alias.

    Iterating the tuple touches the `ln_f` entry just as surely as indexing it
    -- `interpretability/probing.py` built a whole per-layer profile that way
    and the guard saw nothing, because it only looked for subscripts.

    Detected over the same alias set as `_subscripts` -- which this claimed
    before it was true. It used MODULE-level bindings only, so a
    function-local `hs = list(out.hidden_states)` that was merely iterated
    evaded both detectors at once. Both now call `_alias_resolver`.
    """
    lookup = _alias_resolver(tree)
    for node in ast.walk(tree):
        it = None
        if isinstance(node, (ast.For, ast.AsyncFor)):
            it = node.iter
        elif isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp,
                               ast.DictComp)):
            it = node.generators[0].iter if node.generators else None
        if it is None:
            continue
        if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id in ("enumerate", "reversed", "list", "tuple",
                                   "sorted") and it.args):
            it = it.args[0]
        # A slice that drops the tail cannot reach `ln_f`.
        if isinstance(it, ast.Subscript) and isinstance(it.slice, ast.Slice):
            up = it.slice.upper
            if (isinstance(up, ast.UnaryOp) and isinstance(up.op, ast.USub)
                    and isinstance(up.operand, ast.Constant)):
                continue
            it = it.value
        name = (it.attr if isinstance(it, ast.Attribute)
                else it.id if isinstance(it, ast.Name) else None)
        if name in lookup(node) or name == "hidden_states":
            yield node


def _direct_sites():
    """A site is declared if `post-final-norm:` appears in its ENCLOSING
    FUNCTION, not merely in the 12 lines above it.

    Function scope, not a line window: the substitution blocks these fixes
    introduce carry long measurement notes, so the note that declares the
    intent frequently sits 20-40 lines above the line that reads the tuple. A
    function states its intent once; requiring it per line would push authors
    to repeat the marker rather than to think about it.
    """
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        lines = path.read_text().splitlines()
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:  # pragma: no cover
            continue
        spans = _enclosing_spans(tree)
        for node in list(_subscripts(tree)) + list(_iterations(tree)):
            if isinstance(node, ast.Subscript) and _cannot_reach_last(node):
                continue
            i = node.lineno
            scope = None
            for a, b in spans:
                if a <= i <= b:
                    scope = "\n".join(lines[a - 1:b])
                    break
            if scope is None:                      # module level
                scope = "\n".join(lines[max(0, i - 13):i])
            if any(m in scope for m in MARKERS):
                continue
            yield rel, i, lines[i - 1].strip()[:78]


def test_direct_hidden_state_indexing_is_annotated():
    unmarked = [f"{rel}:{line}  {text}" for rel, line, text in _direct_sites()]
    assert not unmarked, (
        "These lines index `hidden_states` directly and can reach the "
        f"post-`ln_f` entry without a `{MARKER}` note in the 12 lines above "
        "saying which representation is intended:\n  " + "\n  ".join(unmarked))


# ── The adversarial set, pinned ──────────────────────────────────────────────
# This guard has been defeated SIXTEEN times across five review rounds, each
# time by a Python idiom nobody had thought of — and several times by the very
# idiom the fixes themselves introduce (`hs = list(out.hidden_states)`), or the
# one the reference code uses (`out["hidden_states"]`).
#
# So the evasions are now a fixture. A future change that reintroduces any of
# them fails here rather than in a review six weeks later.
_ADVERSARIAL = {
    "plain attribute":    "hs = out.hidden_states\n    x = hs[-1]",
    "list()":             "hs = list(out.hidden_states)\n    x = hs[-1]",
    "getattr()":          "hs = getattr(out, 'hidden_states', None)\n    x = hs[-1]",
    "for-loop target":    "for i, hs in enumerate(out.hidden_states):\n        x = hs[0]",
    "list comprehension": "hs = [h for h in out.hidden_states]\n    x = hs[-1]",
    "AnnAssign":          "hs: tuple = out.hidden_states\n    x = hs[-1]",
    "multi-target":       "keep = hs = out.hidden_states\n    x = hs[-1]",
    "starred unpack":     "first, *rest = out.hidden_states\n    x = rest[-1]",
    "mapping subscript":  "hs = out['hidden_states']\n    x = hs[-1]",
    "IfExp":              "hs = out.hidden_states if flag else []\n    x = hs[-1]",
    "BoolOp":             "hs = out.hidden_states or []\n    x = hs[-1]",
    "AugAssign":          "hs = []\n    hs += out.hidden_states\n    x = hs[-1]",
    "reversed()+[0]":     "hs = list(reversed(out.hidden_states))\n    x = hs[0]",
    "dict comprehension": "hs = {i: h for i, h in enumerate(out.hidden_states)}\n    x = hs[len(hs) - 1]",
    "closure":            "hs = out.hidden_states\n    def inner():\n        return hs[-1]",
    # #16: a function-local alias that is only ITERATED, never subscripted.
    # It evaded BOTH detectors at once -- `_subscripts` never saw it because
    # there is no subscript, and `_iterations` never saw it because that
    # detector built its alias set from module-level bindings only while
    # claiming in its own docstring to share `_subscripts`' set. 16 shipping
    # sites already used this shape (Opus round 9, 2026-09-07); closing the
    # hole took the enforced iteration population from 3 to 19, and all 16
    # already carried their declaration.
    "local alias, iterated": (
        "hs = list(out.hidden_states)\n    for h in hs:\n        x = h.mean()"),
    "local alias, comprehension": (
        "hs = out.hidden_states\n    xs = [h.mean() for h in hs]"),
}

# Forms that genuinely cannot reach the final entry. A guard that flags these
# teaches authors to add the marker without thinking, which is worse than not
# having it.
_SAFE = {
    "tail-dropping slice": "for h in out.hidden_states[:-1]:\n        x = h",
    "unrelated mapping":   "d = {'a': 1}\n    x = d['a']",
}

# DELIBERATELY NOT in `_SAFE`: `hs[0:1]`. It cannot reach the final entry for
# any tuple longer than one, but proving that needs the length, which is not
# available statically. The guard flags it, and that is the right direction to
# be wrong in: a spurious marker costs one comment, while a missed site cost
# this campaign a RED and three YELLOWs. The whitelist is therefore limited to
# slices with a NEGATIVE constant upper bound, which are safe unconditionally.


def _reaches(body: str) -> bool:
    src = "def f(out, flag=True):\n    " + body + "\n"
    tree = ast.parse(src)
    subs = [n for n in _subscripts(tree) if not _cannot_reach_last(n)]
    return bool(subs or list(_iterations(tree)))


@pytest.mark.parametrize("label", sorted(_ADVERSARIAL))
def test_every_known_evasion_is_detected(label):
    assert _reaches(_ADVERSARIAL[label]), (
        f"{label!r} reaches the post-ln_f entry undetected — this is evasion "
        f"#{sorted(_ADVERSARIAL).index(label) + 1} of the set that has defeated "
        f"this guard before")


@pytest.mark.parametrize("label", sorted(_SAFE))
def test_provably_safe_forms_are_not_flagged(label):
    assert not _reaches(_SAFE[label]), (
        f"{label!r} cannot reach the final entry but was flagged; a guard that "
        f"cries wolf trains people to add the marker without thinking")
