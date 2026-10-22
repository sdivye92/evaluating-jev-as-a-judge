"""Command line: build-data, run."""

from __future__ import annotations

import argparse
import os
import sys

from .data import DEFAULT_COUNTS, build_all
from .evaluate import summarize
from .judges import DEFAULT_JEV_MODEL, make_judge
from .runner import run_judge
from .schema import DATA_DIR, load_binary, load_rubric

ALL_JUDGES = ["jev", "haiku", "sonnet"]


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _check_keys(judges: list[str]) -> None:
    missing = []
    if "jev" in judges and not os.environ.get("TYPESAFE_API_KEY"):
        missing.append("TYPESAFE_API_KEY")
    if any(j != "jev" for j in judges) and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        sys.exit(f"Missing environment variable(s): {', '.join(missing)} (see .env.example)")


def cmd_build(args: argparse.Namespace) -> None:
    counts = {k: getattr(args, k) for k in DEFAULT_COUNTS}
    for name, n in sorted(build_all(DATA_DIR, counts, args.seed, tuple(args.tasks)).items()):
        print(f"{name}: {n} items")


def cmd_run(args: argparse.Namespace) -> None:
    _check_keys(args.judges)
    fanout = max(8, args.workers * max(args.rubric_samples, args.binary_samples) * 3)
    names = []
    for key in args.judges:
        judge = make_judge(key, args.jev_model, args.binary_samples, args.rubric_samples, fanout)
        names.append(judge.name)
        for task in args.tasks:
            items = (load_binary if task == "binary" else load_rubric)()[: args.limit or None]
            if not items:
                sys.exit(f"No {task} data; run `jevjudge build-data` first")
            run_judge(judge, task, items, workers=args.workers)
    summarize(names, args.tasks, args.limit or None)


def main(argv: list[str] | None = None) -> None:
    _load_env()
    ap = argparse.ArgumentParser(prog="jevjudge", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data", help="download datasets and write data/binary.jsonl and data/rubric.jsonl")
    b.add_argument("--tasks", nargs="+", default=["binary", "rubric"], choices=["binary", "rubric"])
    for name, n in DEFAULT_COUNTS.items():
        b.add_argument(f"--{name}", type=int, default=n,
                       help="items" if name == "rubric" else "questions (2 items each)")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="call the judges (results are cached; reruns resume) and print metrics")
    r.add_argument("--judges", nargs="+", default=ALL_JUDGES,
                   help="jev, haiku, sonnet, haiku-reasoning, sonnet-reasoning")
    r.add_argument("--tasks", nargs="+", default=["binary", "rubric"], choices=["binary", "rubric"])
    r.add_argument("--jev-model", default=DEFAULT_JEV_MODEL)
    r.add_argument("--binary-samples", type=int, default=1,
                   help="Claude samples per binary item (use an odd number >1 to get a graded probability)")
    r.add_argument("--rubric-samples", type=int, default=5, help="Claude samples per criterion")
    r.add_argument("--workers", type=int, default=4, help="items judged concurrently")
    r.add_argument("--limit", type=int, default=0, help="only the first N items (smoke test)")
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
