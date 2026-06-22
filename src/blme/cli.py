"""
BLME Command-Line Interface.

Usage:
    blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd geometry_cka
    blme evaluate --recipe examples/recipes/default_all.yaml
    blme list-tasks
    blme list-tasks --group geometry
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="blme",
        description="Beyond LM Eval — Intrinsic diagnostics for language models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----- evaluate subcommand -----
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run diagnostic tasks on a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Minimal run
  blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd

  # Full run with dtype + device_map
  blme evaluate --model-args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16,device_map=auto \\
                --tasks geometry_svd geometry_cka interpretability_attention_entropy \\
                --output-dir results/llama2

  # Run from a YAML recipe
  blme evaluate --recipe examples/recipes/default_all.yaml
""",
    )
    eval_parser.add_argument(
        "--recipe", type=str,
        help="Path to a YAML recipe file. If provided, other arguments are ignored.",
    )
    eval_parser.add_argument(
        "--model-args", type=str,
        help="Model loading arguments (e.g., pretrained=gpt2,dtype=bfloat16,device_map=auto)",
    )
    eval_parser.add_argument(
        "--tasks", type=str, nargs="+",
        help="List of task names to evaluate",
    )
    eval_parser.add_argument(
        "--task-group", type=str,
        choices=["geometry", "interpretability", "causality", "consistency",
                 "dynamics", "topology", "repe"],
        help="Run all tasks in a group (can combine with --tasks)",
    )
    eval_parser.add_argument(
        "--device", type=str, default=None,
        help="Device to run on (e.g., cuda, cuda:0, cpu). Ignored if device_map is set in --model-args.",
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (passed to lm_eval benchmark tasks)",
    )
    eval_parser.add_argument(
        "--cache-samples", type=int, default=None,
        help="Global sample count for shared cache (overrides per-task num_samples)",
    )
    eval_parser.add_argument(
        "--limit", type=float, default=None,
        help="Limit number of samples for lm_eval benchmark tasks",
    )
    eval_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results.json",
    )
    eval_parser.add_argument(
        "--output-format", type=str, choices=["json", "csv"], default="json",
        help="Output format (default: json)",
    )
    eval_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    eval_parser.add_argument(
        "--task-timeout", type=int, default=600,
        help="Per-task timeout in seconds (default: 600). Unix only.",
    )
    eval_parser.add_argument(
        "--verbosity", type=str, choices=["DEBUG", "INFO", "WARNING"], default="INFO",
        help="Logging verbosity (default: INFO)",
    )
    eval_parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate tasks/recipe and print resolved plan without loading a model.",
    )
    eval_parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any requested task is unknown or any task errors during evaluation.",
    )

    # ----- list-tasks subcommand -----
    list_parser = subparsers.add_parser(
        "list-tasks",
        help="List all available diagnostic tasks",
    )
    list_parser.add_argument(
        "--group", type=str, default=None,
        choices=["geometry", "interpretability", "causality", "consistency",
                 "dynamics", "topology", "repe"],
        help="Filter tasks by group",
    )
    list_parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable task metadata.",
    )

    # ----- Parse -----
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "list-tasks":
        _cmd_list_tasks(args)
    elif args.command == "evaluate":
        _cmd_evaluate(args)


# ---------------------------------------------------------------------------
# list-tasks
# ---------------------------------------------------------------------------

def _cmd_list_tasks(args):
    """Print registered tasks, optionally filtered by group."""
    # Force task registration
    from blme.core import _register_all_tasks  # noqa
    from blme.registry import list_tasks, task_group
    from blme.task_metadata import TASK_CERTIFICATION

    _register_all_tasks()
    all_tasks = sorted(list_tasks())

    if args.group:
        all_tasks = [t for t in all_tasks if task_group(t) == args.group]

    if not all_tasks:
        print(f"No tasks found{' for group: ' + args.group if args.group else ''}.")
        return

    if args.json:
        payload = {
            "count": len(all_tasks),
            "tasks": [
                {
                    "name": task_name,
                    "group": task_group(task_name),
                    **TASK_CERTIFICATION.get(task_name).to_dict(),
                }
                for task_name in all_tasks
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"\nAvailable BLME tasks ({len(all_tasks)}):\n")
    current_group = None
    for task_name in all_tasks:
        group = task_group(task_name)
        if group != current_group:
            current_group = group
            print(f"  [{current_group}]")
        print(f"    {task_name}")
    print()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _cmd_evaluate(args):
    """Run the evaluation pipeline."""
    import logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.verbosity),
    )

    if args.recipe:
        import os
        if not os.path.isfile(args.recipe):
            print(f"Error: recipe file not found: {args.recipe}")
            sys.exit(1)
        from blme.runner import run_from_yaml
        run_from_yaml(args.recipe, dry_run=args.dry_run, strict=args.strict)
        return

    if not args.model_args:
        print("Error: --model-args is required (e.g., --model-args pretrained=gpt2)")
        sys.exit(1)

    # Resolve tasks
    tasks = list(args.tasks) if args.tasks else []
    if args.task_group:
        tasks.extend(_expand_task_group(args.task_group))

    if not tasks:
        print("Error: specify --tasks or --task-group")
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_tasks = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            unique_tasks.append(t)

    plan = _validate_task_names(unique_tasks, strict=args.strict)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    from blme.core import evaluate
    evaluate(
        model_args=args.model_args,
        tasks=unique_tasks,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        output_dir=args.output_dir,
        output_format=args.output_format,
        cache_num_samples=args.cache_samples,
        seed=args.seed,
        task_timeout=args.task_timeout,
        fail_on_task_error=args.strict,
        strict_task_validation=args.strict,
    )


def _expand_task_group(group: str):
    """Expand a group name into all registered task names in that group."""
    from blme.core import _register_all_tasks
    from blme.registry import list_tasks

    _register_all_tasks()
    prefix = group + "_"
    return [t for t in sorted(list_tasks()) if t.startswith(prefix)]


def _validate_task_names(tasks, strict: bool = True):
    """Validate BLME/lm_eval task names before loading a model."""
    from blme.core import _register_all_tasks
    from blme.registry import get_task
    from blme.task_metadata import TASK_CERTIFICATION

    _register_all_tasks()
    diagnostic = []
    lm_eval = []
    unknown = []
    for task_name in tasks:
        if get_task(task_name):
            diagnostic.append(task_name)
            continue
        try:
            from blme.tasks.benchmarks import is_lm_eval_task
            if is_lm_eval_task(task_name):
                lm_eval.append(task_name)
                continue
        except Exception:
            pass
        unknown.append(task_name)

    if unknown and strict:
        raise SystemExit(
            "Unknown task(s): "
            + ", ".join(unknown)
            + ". Use `blme list-tasks --json` to inspect registered tasks."
        )

    return {
        "diagnostic_tasks": diagnostic,
        "lm_eval_tasks": lm_eval,
        "unknown_tasks": unknown,
        "task_metadata": {
            task: TASK_CERTIFICATION[task].to_dict()
            for task in diagnostic
            if task in TASK_CERTIFICATION
        },
    }


if __name__ == "__main__":
    main()
