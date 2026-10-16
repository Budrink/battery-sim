from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .io import load_scenario
from .model import ModelError
from .output import write_result
from .solver import Simulator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battery-sim")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser("run", help="run a canonical JSON scenario")
    run.add_argument("scenario", type=Path)
    run.add_argument("--output", "-o", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            scenario = load_scenario(args.scenario)
            result = Simulator(scenario).run()
            output = write_result(result, args.output)
            print(json.dumps(result.summary(), indent=2, ensure_ascii=False, sort_keys=True))
            print(f"wrote results to {output}", file=sys.stderr)
            return 0
    except (ModelError, OSError, RuntimeError) as exc:
        print(f"battery-sim: {exc}", file=sys.stderr)
        return 2
    return 1
