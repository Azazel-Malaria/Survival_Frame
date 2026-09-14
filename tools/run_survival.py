#!/usr/bin/env python3
"""Run one isolated cross-validation experiment with explicit fold manifests."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from utils.experiment_config import (CANCERS, MODELS, MODEL_LABELS, PROTOTYPE_MODELS,
    canonical_cancer, default_batch_size, file_sha256, load_data_paths, prototype_spec,
    resolve_data_recipe, resolve_protocol, split_directory, stable_hash, validate_prototype)


def parse_folds(value):
    try:
        folds = [int(part) for part in re.split(r"[,\s]+", value.strip()) if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be distinct integers in 0..4") from exc
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be distinct integers in 0..4")
    return sorted(folds)


def layer_list(value, allow_empty=False):
    if allow_empty and value.lower() == "none":
        return "none"
    try:
        layers = [int(part) for part in re.split(r"[,\s]+", value.strip()) if part]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be distinct zero-based integers") from exc
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
        raise argparse.ArgumentTypeError("layers must be distinct zero-based integers")
    return ",".join(map(str, sorted(layers)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cancer", required=True, type=canonical_cancer)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--endpoint", choices=["dss", "os"], default="dss")
    p.add_argument("--split-mode", choices=["train_test", "train_val_test"], default="train_test")
    p.add_argument("--loss", choices=["nll", "cox"], default="nll")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--checkpoint", choices=["last", "best"], default="last")
    p.add_argument("--checkpoint-metric", choices=["c_index", "loss"], default="c_index")
    p.add_argument("--early-stopping", type=int, choices=[0, 1], default=0)
    p.add_argument("--es-metric", choices=["loss", "c_index"], default="loss")
    p.add_argument("--es-min-epochs", type=int, default=3)
    p.add_argument("--es-patience", type=int, default=5)
    p.add_argument("--rna-set", choices=["mmp_set", "surv_set", "slotspe", "none"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--folds", type=parse_folds, default=[0, 1, 2, 3, 4])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-bag-size", type=int, default=4096)
    p.add_argument("--starpath-patches", type=int, default=512)
    p.add_argument("--inject-layers", type=layer_list, default="2,4")
    p.add_argument("--trainable-layers", type=lambda x: layer_list(x, True), default="2,3,4,5")
    p.add_argument("--prototype-mode", choices=["faiss", "kmeans"], default="faiss")
    p.add_argument("--prototype-seed", type=int, default=1)
    p.add_argument("--prototype-patches", type=int, default=100000)
    p.add_argument("--prototype-inits", type=int, default=3)
    p.add_argument("--prototype-iterations", type=int, default=50)
    p.add_argument("--data-config")
    p.add_argument("--results-root")
    p.add_argument("--run-id")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("training_args", nargs=argparse.REMAINDER,
                   help="Additional model hyperparameters after --; included in configuration identity")
    return p


def resolve_experiment(args):
    paths = load_data_paths(args.data_config)
    protocol = resolve_protocol(args.endpoint, args.split_mode, args.early_stopping, args.checkpoint, paths)
    recipe = resolve_data_recipe(args.model, args.cancer, args.rna_set, paths)
    batch = args.batch_size if args.batch_size is not None else default_batch_size(args.model, args.loss)
    if batch < 1 or args.max_epochs < 1 or args.lr <= 0:
        raise ValueError("batch-size, max-epochs and lr must be positive")
    if args.num_workers < 0 or args.es_min_epochs < 1 or args.es_patience < 1:
        raise ValueError("Invalid worker or early-stopping setting")
    if min(args.prototype_patches, args.prototype_inits, args.prototype_iterations) < 1:
        raise ValueError("Prototype sampling, initialization and iteration counts must be positive")
    for name in ("train_bag_size", "starpath_patches"):
        if getattr(args, name) == 0 or getattr(args, name) < -1:
            raise ValueError(f"{name} must be -1 or a positive integer")
    extra = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    managed = {"survival_model", "cancer_type", "data_source", "split_dir", "split_names", "split_mode",
               "results_dir", "fold_summary_path", "task", "target_col", "survival_endpoint", "rna_set",
               "omics_path", "signature_path", "loss_fn", "batch_size", "checkpoint_selection", "checkpoint_metric",
               "early_stopping", "es_metric", "es_min_epochs", "es_patience", "seed", "max_epochs", "lr",
               "num_workers", "train_bag_size", "val_bag_size", "starpath_bag_size", "starpath_variant",
               "starpath_titan_inject_layers", "starpath_titan_trainable_layers", "proto_path", "load_proto",
               "starpath_titan_model_path", "starpath_titan_package_root", "titan_embeddings_path", "overwrite",
               "data_paths"}
    for arg in extra:
        if arg.startswith("--") and arg[2:].split("=", 1)[0].replace("-", "_") in managed:
            raise ValueError(f"{arg} is managed by the launcher; use its named setting")
    config = {key: value for key, value in vars(args).items()
              if key not in {"dry_run", "run_id", "python", "results_root", "training_args", "data_config"}}
    config.update(batch_size=batch, rna_set=recipe["rna_set"], split_mode=protocol["split_mode"],
                  protocol=protocol, data=recipe, extra_training_args=extra)
    config["rna_sha256"] = {key: file_sha256(recipe[key]) for key in ("omics_path", "signature_path")
                            if recipe[key] is not None}
    if args.model == "s_mlp":
        config["rna_sha256"]["composition_path"] = file_sha256(recipe["composition_path"])
    config["split_hashes"] = {
        str(fold): {name: file_sha256(split_directory(protocol, args.cancer, fold) / f"{name}.csv")
                    for name in protocol["split_names"].split(",")} for fold in args.folds
    }
    config["source_sha256"] = stable_hash({str(path.relative_to(REPO)): file_sha256(path)
        for path in sorted((REPO / "src").rglob("*.py"))}, 64)
    config_id = f"{args.loss}_bs{batch}_{args.checkpoint}_{stable_hash(config)}"
    run_id = args.run_id or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id must contain only letters, numbers, dot, underscore or hyphen")
    result_root = Path(args.results_root or paths["results_root"]).resolve()
    run_dir = result_root / protocol["result_group"] / args.cancer / MODEL_LABELS[args.model] / config_id / run_id
    return config, paths, run_dir


def fold_command(args, config, paths, run_dir, fold):
    protocol, recipe = config["protocol"], config["data"]
    fold_dir = run_dir / f"fold_{fold}"
    split_dir = split_directory(protocol, args.cancer, fold)
    values = {
        "survival_model": args.model, "cancer_type": recipe["cohort_code"],
        "data_source": recipe["data_source"], "split_dir": split_dir,
        "split_names": protocol["split_names"], "split_mode": protocol["split_mode"],
        "results_dir": fold_dir, "fold_summary_path": fold_dir / "summary.csv",
        "task": f"TCGA_{recipe['cohort_code']}_{args.endpoint}_survival",
        "target_col": protocol["target_col"], "survival_endpoint": args.endpoint,
        "rna_set": recipe["rna_set"], "loss_fn": args.loss, "batch_size": config["batch_size"],
        "checkpoint_selection": args.checkpoint, "checkpoint_metric": args.checkpoint_metric,
        "early_stopping": args.early_stopping, "es_metric": args.es_metric,
        "es_min_epochs": args.es_min_epochs, "es_patience": args.es_patience,
        "max_epochs": args.max_epochs, "lr": args.lr, "seed": args.seed,
        "num_workers": args.num_workers, "in_dim": 768,
        "model_histo_type": "MIL", "model_histo_config": "MIL_default", "model_mm_type": "coattn",
        "train_bag_size": -1, "val_bag_size": -1,
    }
    flags = []
    if args.data_config:
        values["data_paths"] = str(Path(args.data_config).resolve())
    if recipe["omics_path"]:
        values.update(omics_path=recipe["omics_path"], signature_path=recipe["signature_path"])
    if args.model in {"abmil", "transmil", "survpath", "mcat", "slotspe"}:
        values["train_bag_size"] = args.train_bag_size
    if args.model == "survpath":
        values["model_mm_type"] = "survpath"
    if args.model in PROTOTYPE_MODELS:
        spec = prototype_spec(protocol, args.cancer, fold, paths, mode=args.prototype_mode,
                              seed=args.prototype_seed, n_proto_patches=args.prototype_patches,
                              n_init=args.prototype_inits, n_iter=args.prototype_iterations)
        if not args.dry_run:
            validate_prototype(spec["path"], split_dir / "train.csv", recipe["data_source"],
                               endpoint=args.endpoint, split_mode=protocol["split_mode"])
        values.update(proto_path=spec["path"], n_proto=16)
        flags += ["--load_proto", "--fix_proto"]
    if args.model in {"mmp_trans", "mmp_ot", "dimaf"}:
        values.update(model_histo_type="PANTHER", model_histo_config="PANTHER_default",
                      out_type="allcat", em_iter=1, tau=0.001, ot_eps=0.1)
    if args.model in {"mmp_trans", "mmp_ot"}:
        values["append_embed"] = "random"
        flags.append("--net_indiv")
    if args.model == "mmp_ot":
        values["model_mm_type"] = "coattn_mot"
    if args.model == "s_mlp":
        values["composition_path"] = recipe["composition_path"]
    if args.model == "titan":
        values["titan_embeddings_path"] = recipe["titan_embeddings_path"]
    if args.model == "starpath":
        values.update(st_dir=recipe["st_dir"], starpath_bag_size=args.starpath_patches,
                      train_bag_size=args.starpath_patches,
                      starpath_titan_model_path=recipe["starpath_titan_model_path"],
                      starpath_titan_inject_layers=args.inject_layers,
                      starpath_titan_trainable_layers=args.trainable_layers)
    command = [args.python, "-m", "training.main_survival"]
    for key, value in values.items():
        if value is not None:
            command.extend([f"--{key}", str(value)])
    return command + flags + config["extra_training_args"]


def aggregate_run(run_dir, requested_folds):
    """Read only this run's explicitly requested fold files; sample SD uses ddof=1."""
    rows, metrics = [], None
    for fold in requested_folds:
        source = run_dir / f"fold_{fold}" / "summary.csv"
        with source.open() as handle:
            records = list(csv.DictReader(handle))
        if len(records) != 1:
            raise ValueError(f"Expected exactly one row in {source}")
        record = records[0]
        if "fold" in record and int(record["fold"]) != fold:
            raise ValueError(f"Wrong fold identifier in {source}")
        excluded = {"fold", "epoch", "best_epoch", "last_epoch", "selected_epoch", "checkpoint_epoch"}
        current = []
        for key, raw in record.items():
            if key in excluded or "epoch" in key:
                continue
            try:
                float(raw)
            except (ValueError, TypeError):
                continue
            current.append(key)
        current.sort()
        if not any(key.startswith("c_index_") for key in current) or (metrics is not None and current != metrics):
            raise ValueError(f"Inconsistent or missing C-index metrics in {source}")
        metrics = current
        row = {"fold": fold}
        for metric in metrics:
            value = float(record[metric])
            if not math.isfinite(value) or (metric.startswith("c_index_") and not 0 <= value <= 1):
                raise ValueError(f"Invalid {metric} in {source}: {value}")
            row[metric] = value
        rows.append(row)
    mean = {"fold": "mean"}
    std = {"fold": "std"}
    for metric in metrics:
        values = [row[metric] for row in rows]
        mean[metric] = statistics.mean(values)
        std[metric] = statistics.stdev(values) if len(values) > 1 else ""
    with (run_dir / "cv_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold"] + metrics)
        writer.writeheader()
        writer.writerows(rows + [mean, std])
    status = {"requested_folds": requested_folds, "completed_folds": requested_folds,
              "complete_requested_run": True, "complete_five_fold_cv": requested_folds == [0, 1, 2, 3, 4],
              "n_folds": len(rows), "std_ddof": 1,
              "std_note": "Sample standard deviation across folds; undefined for one fold."}
    status["metrics"] = {metric: {"mean": mean[metric], "std": std[metric] if std[metric] != "" else None,
                                  "fold_values": {str(row["fold"]): row[metric] for row in rows}}
                         for metric in metrics}
    (run_dir / "cv_summary.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main(argv=None):
    args = parser().parse_args(argv)
    config, paths, run_dir = resolve_experiment(args)
    commands = {fold: fold_command(args, config, paths, run_dir, fold) for fold in args.folds}
    print(f"Results: {run_dir}")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        for command in commands.values():
            print(shlex.join(command))
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {"status": "running", "requested_folds": args.folds, "completed_folds": [],
                "run_id": run_dir.name, "config_id": run_dir.parent.name,
                "commands": commands, "std_ddof": 1, "n_folds": 0,
                "complete_requested_run": False, "complete_five_fold_cv": False}
    manifest["prototypes"] = {}
    for fold, command in commands.items():
        if "--proto_path" in command:
            artifact = command[command.index("--proto_path") + 1]
            manifest["prototypes"][str(fold)] = {"path": artifact, "sha256": file_sha256(artifact)}
    manifest_path = run_dir / "run_manifest.json"
    try:
        for fold, command in commands.items():
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Starting {args.cancer}/{args.model} fold {fold}", flush=True)
            subprocess.run(command, cwd=REPO / "src", check=True)
            manifest["completed_folds"].append(fold)
            manifest["n_folds"] = len(manifest["completed_folds"])
        manifest.update(aggregate_run(run_dir, args.folds))
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run_dir


if __name__ == "__main__":
    main()
