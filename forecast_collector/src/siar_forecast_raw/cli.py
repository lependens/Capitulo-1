import argparse
import sys

from .capture import capture_run
from .config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(prog="siar-forecast-raw")
    parser.add_argument("command", choices=["capture"])
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        path = capture_run(load_config(args.config))
    except Exception as exc:
        print(f"capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
