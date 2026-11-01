"""Command line entrypoint: `qam <command> SPEC [--env local] [--start DATE] [--end DATE]`.

Parameters come from the spec YAML. Only time parameters can be given here; they win over
the spec, with a warning when the two differ.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from qam import registry
from qam.config import load_config


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="experiment spec, e.g. configs/experiments/baseline_hybrid.yaml")
    parser.add_argument("--env", default="local", help="overlay configs/env/<env>.yaml (default: local)")
    parser.add_argument("--start", type=dt.date.fromisoformat, metavar="YYYY-MM-DD", help="data_spec.period.start (wins over the spec)")
    parser.add_argument("--end", type=dt.date.fromisoformat, metavar="YYYY-MM-DD", help="data_spec.period.end (wins over the spec)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qam", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a config and print the resolved version")
    _common(p)
    p = sub.add_parser("build-dataset", help="build or refresh the dataset tables")
    _common(p)
    p.add_argument("--force", action="store_true", help="rebuild every table")
    p = sub.add_parser("train", help="new run: train all folds and write predictions")
    _common(p)
    for name in ("evaluate", "backtest"):
        p = sub.add_parser(name, help=f"run the {name} stage on an existing run directory")
        _common(p)
        p.add_argument("--run-dir", required=True, type=Path)
    p = sub.add_parser("run", help="new run: build -> train -> evaluate -> backtest")
    _common(p)
    p.add_argument("--force-build", action="store_true")
    p = sub.add_parser("gc", help="list (or drop, with --yes) dataset table versions no run uses")
    _common(p)
    p.add_argument("--yes", action="store_true", help="drop them instead of only listing")
    sub.add_parser("components", help="list registered components")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "components":
        for kind in registry.KINDS:
            print(f"{kind:10s} {', '.join(registry.names(kind)) or '-'}")
        return 0

    cfg = load_config(args.config, env=args.env, times={"start": args.start, "end": args.end})
    from qam import pipeline

    if args.command == "validate":
        import yaml

        print(yaml.safe_dump(cfg.to_plain(), sort_keys=False))
        print(f"# config ok, hash {cfg.hash()}", file=sys.stderr)
    elif args.command == "build-dataset":
        hashes = pipeline.build_dataset(cfg, force=args.force)
        for table, value in hashes.items():
            print(f"{table:9s} {value}")
    elif args.command == "train":
        ctx = pipeline.run(cfg, stages=("train",))
        print(ctx.run_dir)
    elif args.command in ("evaluate", "backtest"):
        ctx = pipeline.run(cfg, stages=(args.command,), run_dir=args.run_dir)
        print(ctx.run_dir)
    elif args.command == "run":
        ctx = pipeline.run(cfg, force_build=args.force_build)
        print(ctx.run_dir)
    elif args.command == "gc":
        unused = pipeline.gc(cfg, apply=args.yes)
        for table, config_hash in unused:
            print(f"{'dropped' if args.yes else 'unused '} {table:9s} {config_hash}")
        if unused and not args.yes:
            print("# nothing dropped; pass --yes to drop these", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
