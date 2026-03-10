"""
BLME Command-Line Interface.

Usage:
    blme evaluate --model-args pretrained=gpt2 --tasks geometry_svd geometry_cka
    blme evaluate --recipe examples/recipes/default_all.yaml
    blme list-tasks
    blme list-tasks --group geometry
"""

import argparse
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
        "--verbosity", type=str, choices=["DEBUG", "INFO", "WARNING"], default="INFO",
        help="Logging verbosity (default: INFO)",
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
    from blme.registry import list_tasks

    _register_all_tasks()
    all_tasks = sorted(list_tasks())

    if args.group:
        all_tasks = [t for t in all_tasks if t.startswith(args.group + "_") or
                     t.startswith(args.group.replace("-", "_") + "_")]

    if not all_tasks:
        print(f"No tasks found{' for group: ' + args.group if args.group else ''}.")
        return

    print(f"\nAvailable BLME tasks ({len(all_tasks)}):\n")
    current_group = None
    for task_name in all_tasks:
        group = task_name.split("_")[0]
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
        run_from_yaml(args.recipe)
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

    from blme.core import evaluate
    evaluate(
        model_args=args.model_args,
        tasks=unique_tasks,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        output_dir=args.output_dir,
        output_format=args.output_format,
    )


def _expand_task_group(group: str):
    """Expand a group name into all registered task names in that group."""
    from blme.core import _register_all_tasks
    from blme.registry import list_tasks

    _register_all_tasks()
    prefix = group + "_"
    return [t for t in sorted(list_tasks()) if t.startswith(prefix)]


if __name__ == "__main__":
    main()
