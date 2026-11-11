"""Run resumable V2 DEVELOPMENT stages and freeze their exact outputs."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter, process_time

import joblib
from threadpoolctl import threadpool_limits

from .et0_v2 import load_protocol
from .et0_v2_ib09 import require, verified_bytes
from .et0_v2_kaggle import verify_capsule
from .et0_v2_models import candidates, fit_ahc, fit_candidate, make_cell, refit, selections
from .serialization import digest, json_bytes


STAGES = ("AHC", "ML", "ANN_A", "ANN_E")
CAPSULE_SHA = "0c43f1020a04b0dedeeff84ccb1cbae4285ff00ac20c48473ff6c81b57ecd3e6"
PILOT_CELL = ("IB02", "VALIANTZAS_V1", "monthly")


def sha_file(path):
    from hashlib import file_digest

    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    """Publish a complete file in one replace on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(json_bytes(value))
        temp_path = Path(temp.name)
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def identity(repo, capsule, mode):
    protocol, protocol_sha = load_protocol(repo)
    verify_capsule(capsule, CAPSULE_SHA)
    manifest = read_json(capsule / "manifest.json")
    require(manifest["protocol_sha256"] == protocol_sha
            and manifest["dataset_v1_sha256"] == protocol["dataset_v1_sha256"]
            and manifest["ib09"]["admission"]["handoff_sha256"] == protocol["ib09_handoff"]["sha256"],
            "Dataset, protocol or IB09 handoff identity mismatch")
    verified_bytes(repo / protocol["ib09_handoff"]["path"], protocol["ib09_handoff"]["sha256"])
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                    text=True).strip()
    require(not dirty, "Use a clean committed checkout for execution")
    packages = {pkg: version(pkg) for pkg in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    require(packages["scikit-learn"] == "1.9.1", "Expected scikit-learn 1.9.1")
    return {"schema": "et0-v2-run-v1", "mode": mode, "pipeline_sha": sha, "protocol_sha256": protocol_sha,
            "dataset_v1_sha256": manifest["dataset_v1_sha256"], "capsule_sha256": CAPSULE_SHA,
            "development_view_sha256": manifest["outputs"]["development.json.gz"]["sha256"],
            "ib09": manifest["ib09"], "python": platform.python_version(), "packages": packages}


def run_context(repo, capsule, run, mode):
    identifier = identity(repo, capsule, mode)
    path = run / "identity.json"
    if path.exists():
        require(read_json(path) == identifier, "Run identity changed on resume")
    else:
        require(not run.exists(), "Existing run directory is missing its identity")
        write_json(path, identifier)
    return identifier


def cells(inventory, stage, protocol, mode):
    if mode == "pilot":
        return [PILOT_CELL]
    return [(station, family, calendar)
            for station, item in sorted(inventory["stations"].items())
            for family in protocol["families"] for calendar in protocol["calendars"]
            if stage == "AHC" or item["support"][family]["calendars"][calendar]["selection_supported"]]


def _station_rows(capsule):
    with gzip.open(capsule / "development.json.gz", "rt", encoding="utf-8") as handle:
        view = json.load(handle)
    station_rows = defaultdict(list)
    seen = set()
    for row in view["rows"]:
        key = (row["station_code"], row["date"])
        require(key not in seen and "2004-01-01" <= key[1] <= "2019-12-31",
                "Duplicate or non-DEVELOPMENT row in fitting view")
        seen.add(key)
        station_rows[key[0]].append(row)
    return station_rows


def _resource(run):
    disk = shutil.disk_usage(run)
    total = 0
    for path in run.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue  # Another shard just atomically published a staged file.
    memory = None
    if platform.system() == "Linux":
        import resource
        from .et0_v2_kaggle import _cgroup_memory, _meminfo

        memory = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "cgroup_headroom_bytes": _cgroup_memory(), "host": _meminfo()}
    return {"disk_free_bytes": disk.free, "disk_total_bytes": disk.total, "run_bytes": total, "memory": memory}


def cell_path(run, stage, key):
    return run / stage / "__".join(key)


def _verify_refit(path, info):
    for name, expected in info["models"].items():
        require(sha_file(path / name) == expected["sha256"], "Frozen model hash mismatch")
    for name, expected in info["checkpoints"].items():
        require(sha_file(path / "candidates" / name) == expected, "Candidate checkpoint hash mismatch")


def _store_models(path, models):
    output = {}
    with TemporaryDirectory(dir=path) as staging:
        for label, model, info in models:
            filename = f"{label}.joblib"
            staged = Path(staging) / filename
            joblib.dump(model, staged, compress=3)
            output[filename] = {**info, "sha256": sha_file(staged), "bytes": staged.stat().st_size}
            staged.replace(path / filename)
    return output


def _fit_cell(stage, cell, path, ml_protocol, mode, remaining, deadline=float("inf")):
    completed = path / "frozen.json"
    if completed.exists():
        item = read_json(completed)
        _verify_refit(path, item)
        return 0, 0, True
    if mode == "pilot" and len(cell.plans) > 2:
        cell = replace(cell, plans=[cell.plans[0], cell.plans[-1]])
    if stage == "AHC":
        started = perf_counter()
        fit = fit_ahc(cell, cell.plans)
        write_json(completed, {"mode": mode, "stage": stage, "key": [cell.station, cell.family, cell.calendar],
                               "models": {}, "checkpoints": {}, "ahc": fit, "wall_seconds": perf_counter() - started})
        return 1, 0, True
    summaries, checksums = [], {}
    fitted, resumed = 0, 0
    for config in candidates(stage, cell, ml_protocol):
        checkpoint = path / "candidates" / f"{config['candidate_id']}.json"
        if checkpoint.exists():
            summary = read_json(checkpoint)
            require(summary["config"] == config and summary["plans"] ==
                    [cell.plan_hashes[plan["year"]] for plan in cell.plans], "Stale candidate checkpoint")
            require(digest(json_bytes(summary["summary"])) == summary["summary_sha256"],
                    "Candidate checkpoint data corrupted")
            summary = summary["summary"]
            resumed += 1
        elif (remaining is not None and fitted >= remaining) or perf_counter() >= deadline:
            return fitted, resumed, False
        else:
            summary = fit_candidate(stage, cell, cell.plans, config, ml_protocol)
            write_json(checkpoint, {"config": config, "plans": [fold["plan_sha256"] for fold in summary["folds"]],
                                    "summary": summary, "summary_sha256": digest(json_bytes(summary))})
            fitted += 1
        summaries.append(summary)
        checksums[checkpoint.name] = sha_file(checkpoint)
    principal, chosen = selections(stage, cell, summaries)
    started = perf_counter()
    models = _store_models(path, refit(stage, cell, chosen, ml_protocol))
    write_json(completed, {"mode": mode, "stage": stage, "key": [cell.station, cell.family, cell.calendar],
                           "paired": len(cell.rows), "models": models, "checkpoints": checksums,
                           "principal": principal, "selected": chosen, "refit_wall_seconds": perf_counter() - started})
    return fitted, resumed, True


def _verify_stage(run, stage, keys, mode):
    index_path = run / stage / "index.json"
    index = read_json(index_path)
    require(index["stage"] == stage and index["mode"] == mode
            and set(index["cells"]) == {"__".join(key) for key in keys}, "Incomplete or foreign stage index")
    for key in keys:
        path = cell_path(run, stage, key)
        item = json.loads(verified_bytes(path / "frozen.json", index["cells"][path.name]))
        require(item["key"] == list(key) and item["stage"] == stage and item["mode"] == mode,
                "Foreign frozen cell")
        _verify_refit(path, item)
    return sha_file(index_path)


def shard_keys(keys, shard):
    index, count = shard
    require(0 <= index < count, "Invalid V2 shard")
    return keys[index::count]


def fit_batch(repo, capsule, run, stage, mode, max_candidates=None, *, shard=(0, 1), wall_seconds=None):
    protocol, _ = load_protocol(repo)
    ml_protocol = read_json(repo / protocol["ml"]["protocol_path"])
    identifier = run_context(repo, capsule, run, mode)
    inventory = read_json(capsule / "support-inventory.json")
    for prior in STAGES[:STAGES.index(stage)]:
        _verify_stage(run, prior, cells(inventory, prior, protocol, mode), mode)
    require(mode == "official" or shard == (0, 1), "Pilot runs in one process")
    keys = shard_keys(cells(inventory, stage, protocol, mode), shard)
    indices = {}
    station_rows = None
    before = _resource(run)
    wall, cpu = perf_counter(), process_time()
    deadline = wall + wall_seconds if wall_seconds is not None else float("inf")
    fitted = resumed = 0
    with threadpool_limits(limits=1):
        for key in keys:
            path = cell_path(run, stage, key)
            completed = path / "frozen.json"
            if completed.is_file():
                _verify_refit(path, read_json(completed))
                indices[path.name] = sha_file(completed)
                continue
            if (max_candidates is not None and fitted >= max_candidates) or perf_counter() >= deadline:
                break
            if station_rows is None:
                station_rows = _station_rows(capsule)
            station, family, _ = key
            cell = make_cell(station_rows[station], key, protocol, inventory["stations"][station]["support"][family])
            left = max_candidates - fitted if max_candidates is not None else None
            done, reused, complete = _fit_cell(stage, cell, path, ml_protocol, mode, left, deadline)
            fitted += done
            resumed += reused
            if not complete:
                break
            indices[path.name] = sha_file(completed)
    complete = len(indices) == len(keys)
    if complete and shard == (0, 1):
        write_json(run / stage / "index.json", {"mode": mode, "stage": stage, "cells": indices})
    report = {"stage": stage, "mode": mode, "complete": complete, "completed_cells": len(indices),
              "shard": list(shard),
              "planned_cells": len(keys), "candidate_checkpoints_written": fitted, "candidate_checkpoints_reused": resumed,
              "wall_seconds": perf_counter() - wall, "cpu_seconds": process_time() - cpu, "before": before,
              "after": _resource(run), "pipeline_sha": identifier["pipeline_sha"]}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(run / "telemetry" / f"{stage}-{stamp}.json", report)
    return report


def seal_stage(repo, capsule, run, stage):
    run_context(repo, capsule, run, "official")
    protocol, _ = load_protocol(repo)
    inventory = read_json(capsule / "support-inventory.json")
    for prior in STAGES[:STAGES.index(stage)]:
        _verify_stage(run, prior, cells(inventory, prior, protocol, "official"), "official")
    keys = cells(inventory, stage, protocol, "official")
    paths = [cell_path(run, stage, key) for key in keys]
    entries = {}
    for key, path in zip(keys, paths, strict=True):
        item = read_json(path / "frozen.json")
        require(item["key"] == list(key) and item["mode"] == "official" and item["stage"] == stage,
                "Foreign cell in stage seal")
        _verify_refit(path, item)
        entries[path.name] = sha_file(path / "frozen.json")
    entry = {"mode": "official", "stage": stage, "cells": entries}
    output = run / stage / "index.json"
    if output.exists():
        require(read_json(output) == entry, "Stage seal changed")
    else:
        write_json(output, entry)
    return sha_file(output)


def freeze(repo, capsule, run):
    identifier = run_context(repo, capsule, run, "official")
    protocol, _ = load_protocol(repo)
    inventory = read_json(capsule / "support-inventory.json")
    stage_hashes = {stage: _verify_stage(run, stage, cells(inventory, stage, protocol, "official"), "official")
                    for stage in STAGES}
    path = run / "freeze.json"
    entry = {"status": "FROZEN_BEFORE_RETROSPECTIVE_ACCESS", "identity": identifier, "stages": stage_hashes,
             "stage_order": list(STAGES)}
    if path.exists():
        require(read_json(path) == entry, "Frozen campaign identity changed")
    else:
        write_json(path, entry)
    return sha_file(path)


def pilot(repo, capsule, run):
    """Measure all candidates on two outer years and exercise checkpoint reuse."""
    reports = [fit_batch(repo, capsule, run, "AHC", "pilot"),
               fit_batch(repo, capsule, run, "ML", "pilot", max_candidates=1)]
    first = next((cell_path(run, "ML", PILOT_CELL) / "candidates").glob("*.json"))
    pinned = sha_file(first)
    reports.append(fit_batch(repo, capsule, run, "ML", "pilot"))
    require(reports[1]["candidate_checkpoints_written"] == 1 and not reports[1]["complete"]
            and reports[2]["candidate_checkpoints_reused"] == 1
            and reports[2]["complete"] and sha_file(first) == pinned,
            "Pilot did not demonstrate candidate resumption; use a fresh pilot path")
    reports.extend(fit_batch(repo, capsule, run, stage, "pilot") for stage in ("ANN_A", "ANN_E"))
    inventory = read_json(capsule / "support-inventory.json")
    budget = inventory["fit_budget_if_launched"]
    totals = {}
    for stage in STAGES:
        path = cell_path(run, stage, PILOT_CELL)
        result = read_json(path / "frozen.json")
        if stage == "AHC":
            totals[stage] = {"measured_fit_seconds": result["wall_seconds"], "pilot_cells": 1}
            continue
        candidates_results = [read_json(item)["summary"] for item in (path / "candidates").glob("*.json")]
        measured = sum(item["wall_seconds"] for item in candidates_results)
        measured_folds = len(candidates_results[0]["folds"])
        model_bytes = sum(item["bytes"] for item in result["models"].values())
        totals[stage] = {
            "measured_candidate_seconds": measured, "measured_refit_seconds": result["refit_wall_seconds"],
            "measured_folds_per_candidate": measured_folds, "measured_candidates": len(candidates_results),
            "measured_model_bytes_per_cell": model_bytes,
            "projected_single_worker_seconds": measured * budget["supported_outer_folds"] / measured_folds
            + result["refit_wall_seconds"] * budget["station_family_calendar_cells"],
            "projected_model_bytes": model_bytes * budget["station_family_calendar_cells"],
        }
    result = {"status": "PASS_DEVELOPMENT_PILOT", "cell": list(PILOT_CELL),
              "notes": "Single-cell estimate using first and last supported years; not a runtime guarantee",
              "resume": {"reused_candidates": 1, "checkpoint_sha256": pinned},
              "campaign_fit_budget": budget, "stages": totals, "batches": reports,
              "test_access": "NONE", "identity": read_json(run / "identity.json")}
    write_json(run / "pilot.json", result)
    return result


def drive(repo, capsule, run, *, workers, wall_seconds, through):
    """Continue the official sequence in a bounded Linux/Kaggle session."""
    run_context(repo, capsule, run, "official")
    deadline = perf_counter() + wall_seconds
    completed = []
    outcomes = {}
    for stage in STAGES[:STAGES.index(through) + 1]:
        if (run / stage / "index.json").is_file():
            seal_stage(repo, capsule, run, stage)
            completed.append(stage)
            continue
        remaining = int(deadline - perf_counter())
        if remaining <= 0:
            break
        print(f"ET0 V2 {stage}: {workers} shard workers, soft limit {remaining}s", flush=True)

        def worker(index):
            command = [sys.executable, "-m", "siar_science.et0_v2_campaign",
                       "--repo", str(repo), "--capsule", str(capsule),
                       "--run", str(run), "fit", "--stage", stage, "--mode", "official",
                       "--shard", str(index), str(workers), "--wall-seconds", str(remaining)]
            result = subprocess.check_output(command, text=True)
            return json.loads(result)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch = list(pool.map(worker, range(workers)))
        outcomes[stage] = batch
        if not all(item["complete"] for item in batch):
            break
        seal_stage(repo, capsule, run, stage)
        completed.append(stage)
    frozen = freeze(repo, capsule, run) if completed == list(STAGES) else None
    result = {"completed_stages": completed, "worker_count": workers, "soft_wall_seconds": wall_seconds,
              "through": through, "batches": outcomes, "freeze_sha256": frozen,
              "test_access": "NONE", "resource": _resource(run)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(run / "telemetry" / f"drive-{stamp}.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--stage", required=True, choices=STAGES)
    fit.add_argument("--mode", choices=("pilot", "official"), default="pilot")
    fit.add_argument("--max-candidates", type=int)
    fit.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("INDEX", "COUNT"))
    fit.add_argument("--wall-seconds", type=int, help="Stop starting candidates when this soft deadline is reached")
    seal = commands.add_parser("seal")
    seal.add_argument("--stage", required=True, choices=STAGES)
    commands.add_parser("init")
    commands.add_parser("freeze")
    commands.add_parser("pilot")
    sequence = commands.add_parser("drive")
    sequence.add_argument("--workers", type=int, required=True)
    sequence.add_argument("--wall-seconds", type=int, required=True)
    sequence.add_argument("--through", choices=STAGES, default=STAGES[-1])
    args = parser.parse_args(argv)
    if args.command == "fit":
        if args.max_candidates is not None and args.max_candidates < 1:
            parser.error("--max-candidates must be positive")
        if args.wall_seconds is not None and args.wall_seconds < 1:
            parser.error("--wall-seconds must be positive")
        result = fit_batch(args.repo, args.capsule, args.run, args.stage, args.mode, args.max_candidates,
                           shard=tuple(args.shard), wall_seconds=args.wall_seconds)
    elif args.command == "init":
        result = run_context(args.repo, args.capsule, args.run, "official")
    elif args.command == "seal":
        result = {"stage_sha256": seal_stage(args.repo, args.capsule, args.run, args.stage)}
    elif args.command == "freeze":
        result = {"freeze_sha256": freeze(args.repo, args.capsule, args.run)}
    elif args.command == "drive":
        allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        if not 1 <= args.workers <= allowed or args.wall_seconds < 1:
            parser.error("Invalid worker count or soft wall budget")
        result = drive(args.repo, args.capsule, args.run, workers=args.workers,
                       wall_seconds=args.wall_seconds, through=args.through)
    else:
        result = pilot(args.repo, args.capsule, args.run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
