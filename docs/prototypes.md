# Morphology prototype protocol

MMP's official procedure fits K-means to training WSI patches for each fold.
This is also the STARPath default for `/data2/lama/SPILT_DSS_test`: use that fold's
current train.csv. Old MMP centroids are not interchangeable because the patients
and fold assignments differ. Validation variants fit from their smaller train.csv;
neither validation nor test slides participate.

```bash
bash src/scripts/prototype/cancer.sh BRCA --dry-run
bash src/scripts/prototype/cancer.sh CRC
bash src/scripts/prototype/cancer.sh LUSC --endpoint os --split-mode train_val_test
```

The launcher accepts all eight task names, including CRC/COADREAD aliases.
Defaults: 16 prototypes, CONCH-v1.5 dimension 768, up to 100000 patches per
prototype, FAISS, 3 initializations, 50 iterations, seed 1. GPU FAISS is explicit;
`--mode kmeans` selects sklearn, with no silent algorithm substitution. Both
algorithms now honor the supplied initialization count and seed.
For a nondefault vocabulary, pass matching settings to the survival launcher:
`--prototype-mode`, `--prototype-seed`, `--prototype-patches`,
`--prototype-inits`, and `--prototype-iterations`.

Outputs are isolated under
`artifacts/prototypes/<dss|os>/<train_test|train_val_test>/<CANCER>/fold_<k>/<identity>/`.
The identity includes the train CSV hash, feature directory, dimension, prototype
count, algorithm and sampling settings. `prototypes.pkl.metadata.json` additionally
stores the resulting artifact hash and training case/slide counts. Loaders reject
missing provenance, modified artifacts, different training splits, feature sources,
dimensions or protocols. MMP, DIMAF and STARPath can share a validated prototype
within the same protocol. There is no search in another experiment or legacy MMP folder.

External split datasets are read-only. Prototype outputs never get written into
SPILT_* directories. `tools/build_prototypes.py` orchestrates folds;
`training.main_prototype` handles one fold; `wsi_datasets.wsi_prototype` validates
and reads the slide features; `utils.proto_utils` contains the MMP clustering algorithm.

For all four protocols × eight cancers × five folds (160 vocabularies), use the
sequential matrix builder. It keeps the same clustering defaults and selects one GPU:

```bash
python tools/build_prototype_matrix.py --gpu 5 --dry-run
python tools/build_prototype_matrix.py --gpu 5
python tools/build_prototype_matrix.py --gpu 5 --cancers BRCA CRC --endpoints dss --split-modes train_val_test --folds 0 1
```

Use the configured MIL Python environment for real builds. CPU threads default to
4 and data-loader workers to 2; `--threads` and `--num-workers` control them.
Each invocation writes a separate `artifacts/prototypes/build_runs/<timestamp_uuid>/`
with per-fold logs and an atomically updated `matrix_manifest.json`. Existing
prototypes are reused only after artifact validation, an exact comparison of every
recipe metadata field, and checking the training/sample counts. Failures are recorded
and remaining targets continue; the process exits nonzero if any target fails.
Only a fully validated requested matrix gets `index.csv` and `index.json`, containing
all metadata and artifact paths. A dry run writes nothing and starts no subprocess.

The default is `--jobs 1`. Optional concurrency uses fixed worker slots and exposes
exactly one GPU to each child. For example, `--gpu 5,6,7 --jobs 3` uses one job per
GPU; `--gpu 5 --jobs 2` explicitly permits two jobs on GPU 5. The parent process
alone updates the shared manifest. Select only GPUs authorized for this experiment.
Interrupting the runner stops dispatching new targets and waits for active folds
to finish and be audited. Current survival/prototype runs do not use STARPath's
local `src/splits`; the retained original MMP embedding examples still reference it.
