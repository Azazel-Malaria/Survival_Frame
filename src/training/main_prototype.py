"""Fit MMP morphology prototypes from the selected external fold's train.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.experiment_config import (canonical_cancer, file_sha256, load_data_paths,
                                     prototype_spec, resolve_protocol, validate_prototype)


def main(args):
    if min(args.n_proto, args.in_dim, args.n_proto_patches, args.n_init, args.n_iter) < 1:
        raise ValueError("Prototype dimensions, sampling and clustering settings must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    paths = load_data_paths(args.data_config)
    protocol = resolve_protocol(args.endpoint, args.split_mode, paths=paths)
    spec = prototype_spec(protocol, args.cancer, args.fold, paths=paths,
                          n_proto=args.n_proto, in_dim=args.in_dim, mode=args.mode,
                          n_proto_patches=args.n_proto_patches, n_init=args.n_init,
                          n_iter=args.n_iter, seed=args.seed)
    metadata = spec["metadata"]
    destination = Path(spec["path"])
    if args.dry_run:
        print(json.dumps(spec, indent=2))
        return spec
    if destination.exists():
        validate_prototype(destination, metadata["train_csv"], metadata["feature_dir"],
                           args.n_proto, args.in_dim, protocol["endpoint"], protocol["split_mode"])
        print(f"Reusing validated prototype: {destination}")
        return spec

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from wsi_datasets.wsi_prototype import WSIProtoDataset
    from utils.proto_utils import cluster
    from utils.utils import seed_torch
    from utils.file_utils import save_pkl

    seed_torch(args.seed)
    train = pd.read_csv(metadata["train_csv"])
    dataset = WSIProtoDataset({"histo": train}, [metadata["feature_dir"]])
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    sampled, weights = cluster(loader, n_proto=args.n_proto, n_iter=args.n_iter,
                               n_init=args.n_init, feature_dim=args.in_dim,
                               n_proto_patches=args.n_proto_patches, mode=args.mode,
                               use_cuda=torch.cuda.is_available(), seed=args.seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_pkl(str(destination), {"prototypes": weights})
    metadata.update(prototype_sha256=file_sha256(destination), sampled_patches=sampled,
                    training_cases=int(train.case_id.nunique()), training_slides=len(train))
    Path(spec["metadata_path"]).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved train-only prototypes: {destination}")
    return spec


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cancer", required=True, type=canonical_cancer)
    result.add_argument("--fold", required=True, type=int, choices=range(5))
    result.add_argument("--endpoint", choices=["dss", "os"], default="dss")
    result.add_argument("--split-mode", choices=["train_test", "train_val_test"], default="train_test")
    result.add_argument("--data-config")
    result.add_argument("--mode", choices=["faiss", "kmeans"], default="faiss")
    result.add_argument("--n-proto", type=int, default=16)
    result.add_argument("--in-dim", type=int, default=768)
    result.add_argument("--n-proto-patches", type=int, default=100000)
    result.add_argument("--n-init", type=int, default=3)
    result.add_argument("--n-iter", type=int, default=50)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
