import argparse
import json
import sys

from .capture import capture_run
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aemet-forecast-raw")
    parser.add_argument("command", choices=["capture"])
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        path = capture_run(load_config(args.config))
        run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    except Exception as exc:
        # Exception text can contain a URL, request detail, or credential.
        print(f"AEMET capture failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"status={run['status']} run_id={run['run_id']} path={path}")
    return 0 if run["status"] == "complete" else (2 if run["status"] == "partial" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
