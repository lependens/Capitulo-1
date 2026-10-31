"""Non-interactive CLI. All writes require an explicit new output path."""
import argparse
import sys
from pathlib import Path

from .datasets import build_dataset, validate_dataset
from .pipeline import execute_run, json_bytes, safe_new_output
from .ahc_pipeline import execute_ahc
from .ml_pipeline import execute_ml, execute_ml_smoke


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
    for stage in ("train", "validate", "test", "closeout"):
        operation = sub.add_parser("ahc-"+stage, help="Frozen chronological AHC " + stage)
        operation.add_argument("--dataset", type=Path, required=True)
        operation.add_argument("--baseline-manifest", type=Path, required=True)
        operation.add_argument("--protocol", type=Path, required=True)
        operation.add_argument("--output", type=Path, required=True)
        operation.add_argument("--pipeline-sha")
        if stage != "train":
            operation.add_argument("--parent-manifest", type=Path, required=True)
    for stage in ("train", "validate", "test"):
        operation = sub.add_parser("ml-"+stage, help="Frozen chronological ML v1 " + stage)
        operation.add_argument("--dataset", type=Path, required=True)
        operation.add_argument("--ahc-manifest", type=Path, required=True)
        operation.add_argument("--protocol", type=Path, required=True)
        operation.add_argument("--output", type=Path, required=True)
        operation.add_argument("--pipeline-sha")
        if stage != "train":
            operation.add_argument("--parent-manifest", type=Path, required=True)
    smoke = sub.add_parser("ml-linux-smoke", help="Non-selecting rolling-origin LINEAR/RIDGE server smoke")
    smoke.add_argument("--dataset", type=Path, required=True)
    smoke.add_argument("--ahc-manifest", type=Path, required=True)
    smoke.add_argument("--protocol", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--pipeline-sha")
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
        if args.command == "ml-linux-smoke":
            result = execute_ml_smoke(args.dataset, args.ahc_manifest, args.protocol, args.output,
                                      pipeline_sha=args.pipeline_sha)
        elif args.command.startswith("ml-"):
            result = execute_ml(args.dataset, args.ahc_manifest, args.protocol, args.output,
                                args.command[3:], parent_manifest=getattr(args, "parent_manifest", None), pipeline_sha=args.pipeline_sha)
        elif args.command.startswith("ahc-"):
            result = execute_ahc(args.dataset, args.baseline_manifest, args.protocol, args.output,
                                 args.command[4:], parent_manifest=getattr(args, "parent_manifest", None), pipeline_sha=args.pipeline_sha)
        else:
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
