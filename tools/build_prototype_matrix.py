#!/usr/bin/env python3
"""Build and audit the four-protocol morphology prototype matrix on selected GPUs."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from training.main_prototype import parser as fold_parser
from utils.experiment_config import (
    CANCERS, canonical_cancer, file_sha256, load_data_paths, prototype_spec,
    resolve_protocol, validate_prototype,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def selection(values, allowed, name, convert=str):
    result = [convert(part) for value in values for part in re.split(r"[,\s]+", value) if part]
    if not result or len(set(result)) != len(result) or any(value not in allowed for value in result):
        raise ValueError(f"{name} must contain distinct values from {list(allowed)}")
    return result


def parser():
    defaults = fold_parser()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--gpu", required=True, help="Comma-separated GPU indices or UUIDs; each child sees one GPU")
    result.add_argument("--jobs", type=int, default=1, help="Concurrent subprocesses, default 1; slots use GPUs round-robin")
    result.add_argument("--cancers", nargs="+", default=list(CANCERS))
    result.add_argument("--endpoints", nargs="+", default=["dss", "os"])
    result.add_argument("--split-modes", nargs="+", default=["train_test", "train_val_test"])
    result.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--data-config")
    result.add_argument("--mode", choices=["faiss", "kmeans"], default=defaults.get_default("mode"))
    for option in ("n_proto", "in_dim", "n_proto_patches", "n_init", "n_iter", "seed"):
        result.add_argument("--" + option.replace("_", "-"), type=int, default=defaults.get_default(option))
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_arguments(args):
    args.gpus = [gpu.strip() for gpu in args.gpu.split(",")]
    if (not all(args.gpus) or len(set(args.gpus)) != len(args.gpus)
            or any(re.search(r"\s", gpu) for gpu in args.gpus)):
        raise ValueError("--gpu must select distinct GPU indices or UUIDs")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.num_workers < 0 or args.threads < 1:
        raise ValueError("num-workers must be non-negative and threads must be positive")
    if min(args.n_proto, args.in_dim, args.n_proto_patches, args.n_init, args.n_iter) < 1 or args.seed < 0:
        raise ValueError("Prototype dimensions/counts must be positive and seed non-negative")
    args.cancers = selection(args.cancers, CANCERS, "cancers", canonical_cancer)
    args.endpoints = selection(args.endpoints, ("dss", "os"), "endpoints", str.lower)
    args.split_modes = selection(args.split_modes, ("train_test", "train_val_test"), "split-modes")
    args.folds = selection(args.folds, range(5), "folds", int)
    return args


def child_environment(args, gpu=None):
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpus[0] if gpu is None else gpu
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        environment[key] = str(args.threads)
    return environment


def build_plan(args, paths):
    targets = []
    for endpoint in args.endpoints:
        for split_mode in args.split_modes:
            protocol = resolve_protocol(endpoint, split_mode, paths=paths)
            for cancer in args.cancers:
                for fold in args.folds:
                    target = {"endpoint": endpoint, "split_mode": split_mode, "cancer": cancer,
                              "fold": fold, "status": "pending"}
                    target["id"] = f"{endpoint}_{split_mode}_{cancer}_fold_{fold}"
                    command = [sys.executable, "-B", "-u", "-m", "training.main_prototype",
                               "--cancer", cancer, "--fold", str(fold), "--endpoint", endpoint,
                               "--split-mode", split_mode, "--mode", args.mode,
                               "--num-workers", str(args.num_workers)]
                    for option in ("n_proto", "in_dim", "n_proto_patches", "n_init", "n_iter", "seed"):
                        command.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
                    if args.data_config:
                        command.extend(["--data-config", str(Path(args.data_config).resolve())])
                    target["command"] = command
                    try:
                        target["spec"] = prototype_spec(
                            protocol, cancer, fold, paths=paths, n_proto=args.n_proto,
                            in_dim=args.in_dim, mode=args.mode, n_proto_patches=args.n_proto_patches,
                            n_init=args.n_init, n_iter=args.n_iter, seed=args.seed,
                        )
                    except Exception as exc:
                        target["planning_error"] = f"{type(exc).__name__}: {exc}"
                    targets.append(target)
    destinations = [target["spec"]["path"] for target in targets if "spec" in target]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Requested matrix contains duplicate prototype destinations")
    return targets


def validate_expected(spec):
    """Validate the artifact, every expected recipe field, and observed train counts."""
    expected = spec["metadata"]
    actual = validate_prototype(
        spec["path"], expected["train_csv"], expected["feature_dir"],
        expected["n_proto"], expected["in_dim"], expected["endpoint"], expected["split_mode"],
    )
    differences = {key: {"actual": actual.get(key), "expected": value}
                   for key, value in expected.items() if actual.get(key) != value}
    if differences:
        raise ValueError(f"Prototype recipe metadata mismatch: {differences}")
    with open(expected["train_csv"], newline="") as handle:
        rows = list(csv.DictReader(handle))
    train_counts = {"training_cases": len({row["case_id"] for row in rows}), "training_slides": len(rows)}
    for key, count in train_counts.items():
        if actual.get(key) != count:
            raise ValueError(f"Prototype {key} mismatch: {actual.get(key)} != {count}")
    sampled = actual.get("sampled_patches")
    if (not isinstance(sampled, int) or isinstance(sampled, bool)
            or not expected["n_proto"] <= sampled <= expected["n_proto"] * expected["n_proto_patches"]):
        raise ValueError(f"Invalid sampled_patches in prototype metadata: {sampled}")
    return actual


def atomic_text(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def record_progress(manifest, destination):
    statuses = [target["status"] for target in manifest["targets"]]
    manifest["counts"] = {name: statuses.count(name) for name in ("pending", "running", "built", "reused", "failed")}
    manifest["validated_count"] = manifest["counts"]["built"] + manifest["counts"]["reused"]
    manifest["updated_at"] = utc_now()
    atomic_json(destination, manifest)


def write_index(run_dir, entries):
    atomic_json(run_dir / "index.json", entries)
    fields = sorted(set().union(*(entry.keys() for entry in entries)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(entries)
    atomic_text(run_dir / "index.csv", buffer.getvalue())


def execute_target(target, args, gpu):
    """Workers return new records; only the controller writes the shared manifest."""
    target = dict(target)
    started = time.monotonic()
    target["gpu"] = gpu
    try:
        with open(target["log_path"], "w", buffering=1) as log:
            log.write(json.dumps({"command": target["command"], "gpu": gpu}) + "\n")
            if "planning_error" in target:
                raise ValueError(target["planning_error"])
            spec = target["spec"]
            if Path(spec["path"]).exists():
                metadata = validate_expected(spec)
                target["status"] = "reused"
                log.write(f"Reusing validated prototype: {spec['path']}\n")
            else:
                subprocess.run(target["command"], cwd=REPO / "src", env=child_environment(args, gpu),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
                metadata = validate_expected(spec)
                target["status"] = "built"
            target["validated_metadata"] = metadata
            log.write("Artifact and complete recipe metadata validation passed.\n")
    except Exception as exc:
        target.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        with open(target["log_path"], "a") as log:
            log.write(target["error"] + "\n")
    target.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - started)
    return target


def execute(args, paths, targets):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = Path(paths["prototype_root"]) / "build_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    for target in targets:
        target["log_path"] = str(run_dir / "logs" / f"{target['id']}.log")
    environment = child_environment(args)
    # Validation imports numpy in this process; apply the same CPU thread budget first.
    for key, value in environment.items():
        if key.endswith("_NUM_THREADS") or key in {"VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE"}:
            os.environ[key] = value
    manifest = {"schema_version": 1, "status": "running", "started_at": utc_now(),
                "run_id": run_id, "requested_count": len(targets), "configuration": vars(args),
                "worker_gpus": [args.gpus[slot % len(args.gpus)] for slot in range(args.jobs)],
                "environment": {key: environment[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                "source_sha256": {str(path.relative_to(REPO)): file_sha256(path) for path in (
                    Path(__file__), REPO / "src/training/main_prototype.py",
                    REPO / "src/utils/proto_utils.py", REPO / "src/wsi_datasets/wsi_prototype.py")},
                "targets": targets}
    manifest_path = run_dir / "matrix_manifest.json"
    record_progress(manifest, manifest_path)
    print(f"Matrix manifest: {manifest_path}", flush=True)
    interrupted = False
    position = 0
    active = {}
    pool = ThreadPoolExecutor(max_workers=args.jobs)

    def launch(slot):
        nonlocal position
        index = position
        position += 1
        target = targets[index]
        gpu = manifest["worker_gpus"][slot]
        target.update(status="running", started_at=utc_now(), gpu=gpu, worker_slot=slot)
        record_progress(manifest, manifest_path)
        print(f"[{index + 1}/{len(targets)}] {target['id']} GPU={gpu}", flush=True)
        active[pool.submit(execute_target, target, args, gpu)] = (index, slot)

    def collect(future):
        index, slot = active.pop(future)
        try:
            targets[index] = future.result()
        except Exception as exc:
            targets[index].update(status="failed", error=f"{type(exc).__name__}: {exc}")
        target = targets[index]
        print(f"  {target['id']}: {target['status']}" + (f" ({target['error']})" if "error" in target else ""), flush=True)
        record_progress(manifest, manifest_path)
        return slot

    try:
        for slot in range(min(args.jobs, len(targets))):
            launch(slot)
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                slot = collect(future)
                if position < len(targets):
                    launch(slot)
    except KeyboardInterrupt:
        # Do not launch further work. Already running folds finish and are audited.
        interrupted = True
        manifest["status"] = "interrupting"
        record_progress(manifest, manifest_path)
        print("Interrupted: waiting for already-running folds; no further targets will start.", flush=True)
    finally:
        pool.shutdown(wait=True)
        for future in list(active):
            collect(future)
    entries = [{**target["validated_metadata"], "prototype_path": target["spec"]["path"],
                "metadata_path": target["spec"]["metadata_path"], "build_status": target["status"],
                "log_path": target["log_path"], "gpu": target["gpu"]}
               for target in targets if target["status"] in {"built", "reused"}]
    complete = len(entries) == len(targets)
    if complete:
        write_index(run_dir, entries)
        manifest.update(index_csv=str(run_dir / "index.csv"), index_json=str(run_dir / "index.json"))
    manifest.update(status="complete" if complete else "interrupted" if interrupted else "failed",
                    finished_at=utc_now(), complete_requested_matrix=complete,
                    complete_default_matrix=complete and len(targets) == 160)
    record_progress(manifest, manifest_path)
    print(f"Validated {len(entries)}/{len(targets)} prototypes; status={manifest['status']}", flush=True)
    return 0 if complete else 130 if interrupted else 1


def main(argv=None):
    args = validate_arguments(parser().parse_args(argv))
    paths = load_data_paths(args.data_config)
    targets = build_plan(args, paths)
    if args.dry_run:
        destinations = [target["spec"]["path"] for target in targets if "spec" in target]
        print(json.dumps({"dry_run": True, "requested_count": len(targets),
                          "unique_destinations": len(set(destinations)),
                          "worker_gpus": [args.gpus[slot % len(args.gpus)] for slot in range(args.jobs)],
                          "environment": {key: value for key, value in child_environment(args).items()
                                          if key in {"CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                                                     "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"}},
                          "targets": targets}, indent=2))
        return 1 if any("planning_error" in target for target in targets) else 0
    return execute(args, paths, targets)


if __name__ == "__main__":
    raise SystemExit(main())
