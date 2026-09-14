#!/usr/bin/env python3
"""Build prototypes for an explicit fold set using the shared split resolver."""
import argparse
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from training.main_prototype import parser as fold_parser
from run_survival import parse_folds


def main(argv=None):
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--folds", type=parse_folds, default=[0, 1, 2, 3, 4])
    args, rest = selection.parse_known_args(argv)
    if "--fold" in rest or any(item.startswith("--fold=") for item in rest):
        selection.error("Use --folds in the multi-fold prototype launcher")
    # Validate all remaining settings before launching any fold.
    fold_parser().parse_args(["--fold", str(args.folds[0]), *rest])
    for fold in args.folds:
        subprocess.run([sys.executable, "-m", "training.main_prototype", "--fold", str(fold), *rest],
                       cwd=REPO / "src", check=True)


if __name__ == "__main__":
    main()
