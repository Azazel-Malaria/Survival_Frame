#!/usr/bin/env python3
"""Read-only audit of frozen cohorts, RNA coverage and feature availability."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from utils.experiment_config import (CANCERS, canonical_cancer, load_data_paths,
                                    resolve_data_recipe, resolve_protocol, split_directory)


def rna_patients(path):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        # SlotSPE is gene-by-patient; the two other recipes are sample-by-gene.
        patient_columns = [value[:12] for value in header if re.match(r"^TCGA-..-....", value)]
        if patient_columns:
            return set(patient_columns)
        id_column = next((header.index(key) for key in ("sample", "case_id") if key in header), 0)
        patients = set()
        for row in reader:
            sample = row[id_column]
            if not re.match(r"^TCGA-..-....", sample):
                raise ValueError(f"Invalid TCGA RNA sample in {path}: {sample!r}")
            if len(sample) > 12 and sample[13:15] != "01":
                continue
            patients.add(sample[:12])
        return patients


def read_split(path, endpoint):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "slide_id", "tissue_source_site", f"{endpoint}_survival_days",
                    f"{endpoint}_censorship"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing split columns in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty split {path}")
    labels = {}
    for row in rows:
        time = float(row[f"{endpoint}_survival_days"])
        censor = float(row[f"{endpoint}_censorship"])
        if not math.isfinite(time) or not time > 0 or censor not in (0, 1):
            raise ValueError(f"Invalid endpoint in {path}: {row['case_id']}")
        key = row["case_id"]
        pair = (time, censor)
        if key in labels and labels[key] != pair:
            raise ValueError(f"Inconsistent case labels in {path}: {key}")
        labels[key] = pair
    fields = {key: {row[key] for row in rows} for key in ("case_id", "slide_id", "tissue_source_site")}
    if len(fields["slide_id"]) != len(rows):
        raise ValueError(f"Duplicate slide rows in {path}")
    return fields


def audit(paths, cancers=CANCERS, check_features=False):
    report = {"cohorts": {}, "protocols": {}, "features_checked": check_features}
    cache = {}
    for cancer in cancers:
        rna = {model: rna_patients(resolve_data_recipe(model, cancer, paths=paths)["omics_path"])
               for model in ("mmp_trans", "mcat", "slotspe")}
        cases = None
        for endpoint in ("dss", "os"):
            for mode in ("train_test", "train_val_test"):
                protocol = resolve_protocol(endpoint, mode, paths=paths)
                key = f"{endpoint}/{mode}/{cancer}"
                report["protocols"][key] = []
                test_cases = set()
                for fold in range(5):
                    directory = split_directory(protocol, cancer, fold)
                    splits = {name: read_split(directory / f"{name}.csv", endpoint)
                              for name in protocol["split_names"].split(",")}
                    cache[(endpoint, mode, cancer, fold)] = splits
                    for a, b in itertools.combinations(splits, 2):
                        for field in splits[a]:
                            if splits[a][field] & splits[b][field]:
                                raise ValueError(f"{field} overlap in {key}/fold{fold}: {a}/{b}")
                    current = set().union(*(split["case_id"] for split in splits.values()))
                    if cases is not None and current != cases:
                        raise ValueError(f"Different cohort in {key}/fold{fold}")
                    cases = current
                    for model, available in rna.items():
                        missing = cases - available
                        if missing:
                            raise ValueError(f"{cancer}/{model} misses {len(missing)} frozen RNA cases: {sorted(missing)[:5]}")
                    if test_cases & splits["test"]["case_id"]:
                        raise ValueError(f"Repeated outer test cases in {key}")
                    test_cases |= splits["test"]["case_id"]
                    report["protocols"][key].append({name: len(value["case_id"]) for name, value in splits.items()})
                if test_cases != cases:
                    raise ValueError(f"Outer folds do not cover the cohort in {key}")
        for endpoint in ("dss", "os"):
            for fold in range(5):
                two = cache[(endpoint, "train_test", cancer, fold)]
                three = cache[(endpoint, "train_val_test", cancer, fold)]
                if two["test"] != three["test"]:
                    raise ValueError(f"Validation variant changes outer test in {endpoint}/{cancer}/{fold}")
                for field in two["train"]:
                    if two["train"][field] != three["train"][field] | three["val"][field]:
                        raise ValueError(f"Validation variant is not a training partition in {endpoint}/{cancer}/{fold}")
        recipe = resolve_data_recipe("starpath", cancer, paths=paths)
        full = cache[("dss", "train_test", cancer, 0)]
        slides = set().union(*(split["slide_id"] for split in full.values()))
        if check_features:
            features = {path.stem.lower() for path in Path(recipe["data_source"]).iterdir()
                        if path.suffix.lower() in {".h5", ".pt"}}
            st = {path.name.removesuffix("_stpath_pred.h5ad").lower()
                  for path in Path(recipe["st_dir"]).glob("*_stpath_pred.h5ad")}
            for label, available in (("CONCH", features), ("ST", st)):
                missing = {value.removesuffix(".svs").lower() for value in slides} - available
                if missing:
                    raise ValueError(f"{cancer}: {len(missing)} missing {label} slides: {sorted(missing)[:5]}")
        report["cohorts"][cancer] = {"patients": len(cases), "slides": len(slides),
                                     "RNA_patients": {key: len(value) for key, value in rna.items()}}
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config")
    parser.add_argument("--cancer", action="append", type=canonical_cancer)
    parser.add_argument("--check-features", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(load_data_paths(args.data_config), args.cancer or CANCERS, args.check_features), indent=2))


if __name__ == "__main__":
    main()
