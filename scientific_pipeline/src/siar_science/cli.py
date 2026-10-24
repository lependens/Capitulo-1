"""Non-interactive CLI. All writes require an explicit new output path."""
import argparse
import sys
from pathlib import Path

from .datasets import build_dataset, validate_dataset
from .pipeline import execute_run, json_bytes, safe_new_output


def main(argv=None):
    parser = argparse.ArgumentParser(prog="siar-science", description="SIAR observational ET0 baselines v1")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "calculate-pm", "calculate-hargreaves", "calculate-valiantzas", "calculate-hargreaves-rs", "evaluate", "run-baselines", "run-classical-baselines", "evaluate-classical", "run-four-baselines", "evaluate-four"):
        operation = sub.add_parser(command)
        operation.add_argument("--dataset", type=Path, required=True)
        operation.add_argument("--output", type=Path, required=True, help="New run directory")
        operation.add_argument("--pipeline-sha")
        if command in ("evaluate", "evaluate-classical", "evaluate-four"):
            operation.add_argument("--predictions", type=Path, required=True, action="append")
    build = sub.add_parser("build-dataset", help="Future authorized POST handoff only")
    build.add_argument("--handoff", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True, help="New dataset JSON file")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-dataset":
            dataset = build_dataset(args.handoff)
            validate_dataset(dataset)
            path = safe_new_output(args.output, directory=False)
            with path.open("xb") as stream:
                stream.write(json_bytes(dataset))
            print("SUCCESS: dataset built from authorized verified Quality POST handoff")
            return 0
        result = execute_run(args.dataset, args.output, args.command, args.pipeline_sha, getattr(args, "predictions", ()))
        print(result["status"] + ": " + str(args.output))
        if result["status"] != "SUCCESS":
            print(result.get("error", "Run failed"), file=sys.stderr)
            return 1
        return 0
    except (ValueError, OSError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
