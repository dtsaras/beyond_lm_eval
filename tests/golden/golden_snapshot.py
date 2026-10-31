#!/usr/bin/env python3
"""Golden-output snapshot: every output key of every registered task, exactly.

A behaviour-preserving refactor must reproduce a snapshot bit for bit. Any
change to a number is a fix, not a refactor, and needs its own ledger entry.

Each (model, dtype, task) runs in a FRESH PROCESS, so global RNG calls and hook
leaks in one task cannot move another. It runs on CPU with CUDA hidden and the
thread count pinned and recorded, because fp32 reductions change bits with
both. It runs through ``blme.core.evaluate``, so the eager-attention and dtype
logic of a real run applies (calling ``Task.evaluate`` directly skips it). The
corpus is fixed and independent of the library's own loader, which falls back
to three hardcoded sentences when ``wikitext`` cannot be resolved offline.

Usage::

    python tests/golden/golden_snapshot.py snapshot --out SNAP.json \\
        [--profile shipped|study] [--route cache|nocache|offload] \\
        [--models gpt2:float32,gpt2:bfloat16] [--tasks t1,t2] \\
        [--num-samples 24] [--no-cap] [--workers 24]
    python tests/golden/golden_snapshot.py compare A.json B.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

THREADS = 2
DEFAULT_MODELS = ("gpt2:float32", "gpt2:bfloat16",
                  "hf-internal-testing/tiny-random-gpt2:float32")
STUDY_TASKS_YAML = os.path.expanduser("~/projects/beyond_benchmarks/configs/tasks.yaml")
CORPUS_POOL = 64          # fixed evenly spaced ordering; --num-samples slices it
CORPUS_MAX_CHARS = 600
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _pinned_env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(THREADS)
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Two runs of the SAME code at the same pinned thread count differed in 20
    # of 6184 values (3 of 258 entries, rel 3e-8..2e-6; 2026-10-26): MKL picks
    # kernels by memory alignment unless Conditional Numerical Reproducibility
    # is on, and dynamic threading can change the split.
    env["MKL_CBWR"] = "COMPATIBLE"
    env["MKL_DYNAMIC"] = "FALSE"
    env["OMP_DYNAMIC"] = "FALSE"
    # The tokenizers (Rayon) pool starts one thread per CPU in every worker;
    # with many workers beside a test run that exhausts the container's
    # threads and shows up as task errors.
    env["RAYON_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env["TRANSFORMERS_VERBOSITY"] = "error"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env["PYTHONHASHSEED"] = "0"
    # The editable install points at the MAIN tree; a worktree must test its
    # own src, so put it first and record what was actually imported.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (SRC_DIR, env.get("PYTHONPATH", "")) if p)
    return env


def fixed_corpus(n):
    """The first ``n`` of a fixed, evenly spaced draw over WikiText-103 val.

    Draws nest: the corpus for n is a prefix of the corpus for any m > n.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    rows = [r["text"] for r in ds if isinstance(r.get("text"), str) and len(r["text"]) >= 50]
    step = len(rows) / float(CORPUS_POOL)
    pool = [rows[int(i * step)][:CORPUS_MAX_CHARS] for i in range(CORPUS_POOL)]
    if n > CORPUS_POOL:
        raise ValueError(f"num_samples {n} > fixed pool {CORPUS_POOL}")
    return [{"text": t} for t in pool[:n]]


def _jsonable(x):
    import numpy as np
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch is not None and isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().double().numpy() if x.is_floating_point()
                         else x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        # repr round-trips exactly; JSON cannot carry nan/inf as numbers.
        return x if math.isfinite(x) else {"__float__": repr(x)}
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return {"__repr__": repr(x)[:200]}


def task_configs_for(task, profile, n, route):
    """The ``task_configs`` entry a snapshot passes for ``task``.

    ``n`` only ever LOWERS the sample count, so a full snapshot finishes: a
    resolved ``num_samples`` above ``n`` becomes ``n``, and one at or below it
    is left alone. A task whose resolved config has no ``num_samples`` gets
    none. ``n=None`` caps nothing. Everything else (layerwise, k, windows,
    thresholds) stays as the profile sets it.
    """
    from blme.tasks.config_loader import resolve_task_config
    cfg = {}
    if profile == "study":
        import yaml
        with open(STUDY_TASKS_YAML) as fh:
            cfg = dict((yaml.safe_load(fh).get("task_configs") or {}).get(task) or {})
    resolved = resolve_task_config(task, cfg).get("num_samples")
    if n is not None and resolved is not None and int(resolved) > n:
        cfg["num_samples"] = n
    if route == "nocache":
        cfg["use_cache"] = False
    return cfg


def resolved_num_samples(task, config):
    from blme.tasks.config_loader import resolve_task_config
    return resolve_task_config(task, config).get("num_samples")


def _worker(spec):
    import torch
    torch.set_num_threads(THREADS)
    if spec["route"] == "offload":
        os.environ["BLME_DISK_OFFLOAD"] = "1"
    import blme
    from blme.core import evaluate
    if not os.path.abspath(blme.__file__).startswith(SRC_DIR + os.sep):
        raise RuntimeError(f"imported {blme.__file__}, not the tree under test {SRC_DIR}")
    model, dtype = spec["model"].rsplit(":", 1)
    corpus = fixed_corpus(spec["num_samples"])
    t0 = time.time()
    try:
        env = evaluate(
            model_args=f"pretrained={model},dtype={dtype}",
            tasks=[spec["task"]], device="cpu", dataset=corpus,
            task_configs={spec["task"]: spec["config"]},
            cache_num_samples=spec["num_samples"], seed=42,
            task_timeout=spec.get("timeout", 1800))
        res = (env.get("results") or {}).get(spec["task"])
        err = (env.get("errors") or {}).get(spec["task"])
        status = "error" if (res is None or (isinstance(res, dict) and "error" in res)
                             or err) else "ok"
        out = {"status": status, "result": _jsonable(res), "error": err}
    except Exception as e:  # noqa: BLE001 - a raise is a recorded outcome too
        out = {"status": "raised", "result": None, "error": f"{type(e).__name__}: {e}"[:500]}
    out["secs"] = round(time.time() - t0, 1)
    out["num_samples"] = resolved_num_samples(spec["task"], spec["config"])
    return out


def _run_one(spec, env):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        path = fh.name
    try:
        p = subprocess.run([sys.executable, __file__, "_worker", path], env=env,
                           capture_output=True, text=True,
                           timeout=spec.get("timeout", 1800) + 300)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("GOLDEN_RESULT ")]
        if line:
            return json.loads(line[-1][len("GOLDEN_RESULT "):])
        return {"status": "crashed", "result": None,
                "error": (p.stderr or p.stdout)[-800:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "result": None, "error": "worker timeout"}
    finally:
        os.unlink(path)


def snapshot(args):
    env = _pinned_env()
    sys.path.insert(0, SRC_DIR)
    from blme.core import _register_all_tasks
    from blme.registry import _TASK_REGISTRY
    import blme
    _register_all_tasks()
    tasks = args.tasks.split(",") if args.tasks else sorted(_TASK_REGISTRY)
    models = args.models.split(",") if args.models else list(DEFAULT_MODELS)
    corpus = fixed_corpus(args.num_samples)
    cap = None if args.no_cap else args.num_samples
    specs = [{"model": m, "task": t, "route": args.route, "num_samples": args.num_samples,
              "config": task_configs_for(t, args.profile, cap, args.route),
              "timeout": args.timeout}
             for m in models for t in tasks]
    t0 = time.time()
    entries = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_run_one, s, env): s for s in specs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            s = futs[fut]
            r = fut.result()
            entries[f"{s['model']}|{s['task']}"] = r
            print(f"[{i:3d}/{len(specs)}] {r['status']:7s} {r.get('secs', 0):7.1f}s "
                  f"{s['model']} {s['task']}", flush=True)
    meta = {
        "blme_file": blme.__file__, "profile": args.profile, "route": args.route,
        "num_samples": args.num_samples, "cap": cap,
        "capped_tasks": sorted(s["task"] for s in specs[:len(tasks)]
                               if "num_samples" in s["config"]),
        "threads": THREADS, "models": models,
        "env": {k: env.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                        "OPENBLAS_NUM_THREADS", "MKL_CBWR", "MKL_DYNAMIC",
                                        "OMP_DYNAMIC", "CUDA_VISIBLE_DEVICES")},
        "corpus_sha256": hashlib.sha256(json.dumps(corpus).encode()).hexdigest(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd=os.path.dirname(__file__)).stdout.strip(),
        "wall_secs": round(time.time() - t0, 1),
        "status_counts": {k: sum(1 for e in entries.values() if e["status"] == k)
                          for k in ("ok", "error", "raised", "crashed", "timeout")},
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "entries": dict(sorted(entries.items()))}, fh,
                  indent=1, sort_keys=True)
    print(json.dumps(meta, indent=1))


def _flatten(x, prefix=""):
    if isinstance(x, dict) and not (set(x) <= {"__float__", "__repr__"}):
        for k, v in x.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def _spread(base0, base1):
    """Per-leaf relative spread between two baseline runs of the SAME code.

    Only leaves that actually differed between the two runs get a tolerance,
    so the exclusion is derived from a control, never from a hand-kept list.
    """
    A0 = json.load(open(base0))["entries"]
    A1 = json.load(open(base1))["entries"]
    tol = {}
    for name in set(A0) & set(A1):
        f0, f1 = dict(_flatten(A0[name]["result"])), dict(_flatten(A1[name]["result"]))
        for k in set(f0) & set(f1):
            x, y = f0[k], f1[k]
            if x != y and isinstance(x, float) and isinstance(y, float):
                tol[(name, k)] = abs(x - y) / max(abs(x), abs(y), 1e-300)
    return tol


def compare(a_path, b_path, max_show=40, tolerate_from=None):
    A = json.load(open(a_path))["entries"]
    B = json.load(open(b_path))["entries"]
    tol = _spread(*tolerate_from) if tolerate_from else {}
    diffs, n_keys, n_tolerated = [], 0, 0
    for name in sorted(set(A) | set(B)):
        a, b = A.get(name), B.get(name)
        if a is None or b is None:
            diffs.append(f"{name}: present only in {'B' if a is None else 'A'}")
            continue
        if a["status"] != b["status"] or a.get("error") != b.get("error"):
            diffs.append(f"{name}: status {a['status']} -> {b['status']} ({b.get('error')})")
        fa, fb = dict(_flatten(a["result"])), dict(_flatten(b["result"]))
        n_keys += len(fa)
        for k in sorted(set(fa) | set(fb)):
            x, y = fa.get(k, "<missing>"), fb.get(k, "<missing>")
            if x == y:
                continue
            t = tol.get((name, k))
            if (t is not None and isinstance(x, float) and isinstance(y, float)
                    and abs(x - y) <= 10 * t * max(abs(x), abs(y))):
                n_tolerated += 1
                continue
            diffs.append(f"{name}: {k}: {x!r} -> {y!r}")
    print(f"compared {len(A)} entries / {n_keys} leaf values; {len(diffs)} differences"
          + (f"; {n_tolerated} within 10x the baseline pair's own spread "
             f"({len(tol)} nondeterministic leaves)" if tolerate_from else ""))
    for d in diffs[:max_show]:
        print("  " + d)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out", required=True)
    s.add_argument("--profile", choices=("shipped", "study"), default="shipped")
    s.add_argument("--route", choices=("cache", "nocache", "offload"), default="cache")
    s.add_argument("--models", default="")
    s.add_argument("--tasks", default="")
    s.add_argument("--num-samples", type=int, default=24,
                   help="corpus size, and the cap on each task's num_samples")
    s.add_argument("--no-cap", action="store_true",
                   help="leave every num_samples as the profile resolves it; a task "
                        "asking for more than --num-samples sees a shorter dataset")
    s.add_argument("--workers", type=int, default=24)
    s.add_argument("--timeout", type=int, default=1800)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tolerate-from", nargs=2, metavar=("BASE0", "BASE1"),
                   help="two baseline runs of the same code; leaves that differ "
                        "between them are compared within 10x their spread")
    w = sub.add_parser("_worker")
    w.add_argument("spec")
    args = ap.parse_args()
    if args.cmd == "snapshot":
        snapshot(args)
    elif args.cmd == "compare":
        sys.exit(1 if compare(args.a, args.b, tolerate_from=args.tolerate_from) else 0)
    else:
        spec = json.load(open(args.spec))
        print("GOLDEN_RESULT " + json.dumps(_worker(spec)), flush=True)


if __name__ == "__main__":
    main()
