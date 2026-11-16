"""Check a private V2 preparation capsule on Kaggle CPU without fitting models."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform


def _meminfo():
    content = Path("/proc/meminfo").read_text(encoding="ascii")
    return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in content.splitlines()
            if line.split(":")[0] in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}}


def _cgroup_memory():
    group = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                  if line.startswith("0::")), "/")
    mount = Path("/sys/fs/cgroup")
    bounds = []
    for directory in [mount, *(mount / group.lstrip("/")).parents, mount / group.lstrip("/")]:
        limit = directory / "memory.max"
        current = directory / "memory.current"
        if not limit.is_file() or not current.is_file():
            continue
        maximum = limit.read_text().strip()
        if maximum != "max":
            bounds.append(int(maximum) - int(current.read_text().strip()))
    return min(bounds) if bounds else None


def verify_capsule(capsule, manifest_sha256):
    manifest_path = capsule / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("V2 capsule manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["status"] != "PREPARATION_ONLY" or any(
        manifest[key] != 0 for key in ("fitted_models", "test_predictions", "test_metrics")
    ):
        raise ValueError("Expected a complete V2 preparation capsule without fitted models or TEST outcomes")
    if set(manifest["outputs"]) != {"development.json.gz", "support-inventory.json"}:
        raise ValueError("Unexpected V2 capsule inventory")
    for name, item in manifest["outputs"].items():
        path = capsule / name
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Capsule artifact mismatch: {name}")


def preflight(capsule, manifest_sha256):
    import resource  # Linux/Kaggle resource accounting

    verify_capsule(capsule, manifest_sha256)
    packages = {name: version(name) for name in ("numpy", "scipy", "joblib", "scikit-learn")}
    if packages["scikit-learn"] != "1.9.1":
        raise ValueError("V2 requires scikit-learn==1.9.1")
    memory = _meminfo()
    cgroup_headroom = _cgroup_memory()
    return {
        "status": "CAPSULE_AND_ENVIRONMENT_VERIFIED_NO_FIT",
        "manifest_sha256": manifest_sha256,
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "available_logical_cpus": len(os.sched_getaffinity(0)),
        "host_memory_bytes": memory, "cgroup_v2_headroom_bytes": cgroup_headroom,
        "preflight_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "training_resource_gate": "PENDING_AUTHORIZED_SINGLE_SHARD_SMOKE",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(preflight(args.capsule, args.manifest_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
