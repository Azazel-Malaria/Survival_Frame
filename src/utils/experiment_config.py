"""Shared data, protocol and artifact contracts; no training dependencies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CANCERS = ("BRCA", "BLCA", "STAD", "HNSC", "LUAD", "LUSC", "CRC", "KIRC")
MODELS = ("mmp_trans", "mmp_ot", "survpath", "abmil", "transmil", "mcat",
          "mlp", "snn", "s_mlp", "titan", "dimaf", "starpath", "slotspe")
MODEL_LABELS = {name: name.upper() for name in MODELS}
MODEL_LABELS.update(mmp_trans="MMP_TRANS", mmp_ot="MMP_OT", survpath="SurvPath",
                    starpath="STARPath", slotspe="SlotSPE", s_mlp="S_MLP")
PROTOTYPE_MODELS = frozenset({"mmp_trans", "mmp_ot", "dimaf", "starpath"})
FEATURE_COHORTS = {"CRC": "CRC", "LUAD": "NSCLC", "LUSC": "NSCLC", "KIRC": "RCC"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload, length=12):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()[:length]


def canonical_cancer(value):
    cancer = str(value).upper()
    cancer = "CRC" if cancer == "COADREAD" else cancer
    if cancer not in CANCERS:
        raise ValueError(f"Unknown cancer {value!r}; expected {CANCERS}")
    return cancer


def cohort_code(cancer):
    cancer = canonical_cancer(cancer)
    return "COADREAD" if cancer == "CRC" else cancer


def load_data_paths(path=None):
    source = Path(path) if path else REPO_ROOT / "src/configs/data_paths.json"
    data = json.loads(source.read_text())
    for key in ("rna_root", "feature_root", "st_root", "prototype_root", "results_root",
                "titan_root", "starpath_titan_root"):
        value = Path(data[key]).expanduser()
        data[key] = str((REPO_ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    data["split_roots"] = {
        key: str((REPO_ROOT / Path(value).expanduser()).resolve())
        for key, value in data["split_roots"].items()
    }
    if Path(data["titan_root"]) == Path(data["starpath_titan_root"]):
        raise ValueError("TITAN and TITAN_STARPath must have separate roots")
    return data


def resolve_protocol(endpoint="dss", split_mode="train_test", early_stopping=False,
                     checkpoint="last", paths=None):
    paths = load_data_paths() if paths is None else paths
    endpoint = str(endpoint).lower()
    if endpoint not in {"dss", "os"}:
        raise ValueError("endpoint must be dss or os")
    if split_mode not in {"train_test", "train_val_test"}:
        raise ValueError("split_mode must be train_test or train_val_test")
    if checkpoint not in {"last", "best"}:
        raise ValueError("checkpoint must be last or best")
    if isinstance(early_stopping, str):
        if early_stopping.lower() not in {"0", "1", "false", "true"}:
            raise ValueError("early_stopping must be boolean")
        early_stopping = early_stopping.lower() in {"1", "true"}
    if early_stopping or checkpoint == "best":
        split_mode = "train_val_test"
    group = f"{endpoint.upper()}_{'earlystop' if early_stopping else 'standard'}_"
    group += "val_test" if split_mode == "train_val_test" else "test"
    return {
        "endpoint": endpoint, "split_mode": split_mode,
        "early_stopping": bool(early_stopping), "checkpoint": checkpoint,
        "split_root": paths["split_roots"][f"{endpoint}/{split_mode}"],
        "split_names": "train,val,test" if split_mode == "train_val_test" else "train,test",
        "target_col": f"{endpoint}_survival_days", "censorship_col": f"{endpoint}_censorship",
        "result_group": group,
    }


def split_directory(protocol, cancer, fold):
    if fold not in range(5):
        raise ValueError("fold must be in 0..4")
    return Path(protocol["split_root"]) / f"TCGA_{cohort_code(cancer)}_overall_survival_k={fold}"


def resolve_data_recipe(model, cancer, rna_set=None, paths=None):
    paths = load_data_paths() if paths is None else paths
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}")
    cancer = canonical_cancer(cancer)
    code = cohort_code(cancer)
    if model in {"abmil", "transmil", "titan"}:
        expected = "none"
    elif model in {"mcat", "mlp", "snn", "s_mlp"}:
        expected = "surv_set"
    elif model == "slotspe":
        expected = "slotspe"
    else:
        expected = "mmp_set"
    rna_set = expected if rna_set is None else rna_set
    allowed = {"mmp_set", "surv_set"} if model == "starpath" else {expected}
    if rna_set not in allowed:
        raise ValueError(f"{model} requires RNA recipe {sorted(allowed)}, received {rna_set}")
    root = Path(paths["rna_root"])
    result = {
        "cancer": cancer, "cohort_code": code, "rna_set": rna_set,
        "data_source": str(Path(paths["feature_root"]) / FEATURE_COHORTS.get(cancer, cancer)),
        "st_dir": str(Path(paths["st_root"]) / cancer),
        "titan_embeddings_path": str(Path(paths["titan_root"]) / "TCGA_TITAN_features.pkl"),
        "starpath_titan_model_path": paths["starpath_titan_root"],
        "omics_path": None, "signature_path": None,
        "composition_path": None,
    }
    if rna_set == "mmp_set":
        result["omics_path"] = str(root / "mmp_set/hallmarks" / code / "rna_clean.csv")
        result["signature_path"] = str(root / "mmp_set/metadata/hallmarks_signatures.csv")
    elif rna_set == "surv_set":
        layout = "hallmarks" if model == "starpath" else "combine"
        result["omics_path"] = str(root / "surv_set/raw_rna_data" / layout / code.lower() / "rna_clean.csv")
        if model in {"starpath", "mcat"}:
            signature = "hallmarks_signatures.csv" if model == "starpath" else "signatures.csv"
            result["signature_path"] = str(root / "surv_set/metadata" / signature)
        if model == "s_mlp":
            result["composition_path"] = str(root / "surv_set/pathway_compositions/combine_comps.csv")
    elif rna_set == "slotspe":
        result["omics_path"] = str(root / "slotspe" / f"{code.lower()}_rna_inter.csv")
        result["signature_path"] = str(root / "surv_set/metadata/combine_signatures.csv")
    return result


def default_batch_size(model, loss):
    if loss not in {"nll", "cox"}:
        raise ValueError("loss must be nll or cox")
    # DIMAF's released main.py uses 64 for both losses.
    return 64 if loss == "cox" or model == "dimaf" else 1


def prototype_metadata_path(path):
    return Path(str(path) + ".metadata.json")


def prototype_spec(protocol, cancer, fold, paths=None, n_proto=16, in_dim=768,
                   feature_tag="conch_v15", mode="faiss", n_proto_patches=100000,
                   n_init=3, n_iter=50, seed=1):
    paths = load_data_paths() if paths is None else paths
    train_csv = split_directory(protocol, cancer, fold) / "train.csv"
    cancer = canonical_cancer(cancer)
    feature_dir = Path(paths["feature_root"]) / FEATURE_COHORTS.get(cancer, cancer)
    metadata = {
        "schema_version": 1, "endpoint": protocol["endpoint"],
        "split_mode": protocol["split_mode"], "cancer": canonical_cancer(cancer), "fold": fold,
        "train_csv": str(train_csv.resolve()), "train_sha256": file_sha256(train_csv),
        "feature_dir": str(feature_dir.resolve()), "feature_tag": feature_tag,
        "n_proto": n_proto, "in_dim": in_dim, "mode": mode,
        "n_proto_patches": n_proto_patches, "n_init": n_init, "n_iter": n_iter, "seed": seed,
    }
    destination = (Path(paths["prototype_root"]) / protocol["endpoint"] / protocol["split_mode"] /
                   canonical_cancer(cancer) / f"fold_{fold}" / stable_hash(metadata) / "prototypes.pkl")
    return {"path": str(destination), "metadata_path": str(prototype_metadata_path(destination)),
            "metadata": metadata}


def validate_prototype(path, train_csv, feature_dir, n_proto=16, in_dim=768,
                       endpoint=None, split_mode=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Prototype missing: {path}; run scripts/prototype/cancer.sh first")
    sidecar = prototype_metadata_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Prototype provenance missing: {sidecar}")
    metadata = json.loads(sidecar.read_text())
    expected = {"schema_version": 1, "train_sha256": file_sha256(train_csv),
                "feature_dir": str(Path(feature_dir).resolve()), "n_proto": n_proto, "in_dim": in_dim,
                "prototype_sha256": file_sha256(path)}
    if endpoint is not None:
        expected["endpoint"] = endpoint
    if split_mode is not None:
        expected["split_mode"] = split_mode
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items()
                  if metadata.get(key) != value}
    if mismatches:
        raise ValueError(f"Prototype does not match this experiment: {mismatches}")
    # Import only while validating an artifact; configuration/dry runs stay light.
    import pickle
    import numpy as np
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "prototypes" not in payload:
        raise ValueError(f"Invalid prototype payload: {path}")
    values = np.asarray(payload["prototypes"])
    if values.shape == (1, n_proto, in_dim):
        values = values[0]
    if values.shape != (n_proto, in_dim) or not np.isfinite(values).all():
        raise ValueError(f"Invalid prototype dimensions or nonfinite values in {path}: {values.shape}")
    return metadata
