# Data and experiment protocol

All external paths are configured in `src/configs/data_paths.json`. Training never
rebuilds a split or filters a model's missing RNA cases. A missing modality is an error.
`python tools/audit_data.py --check-features` checks the frozen cohort, RNA coverage,
CONCH/ST availability, case/slide/TSS separation, all five test folds, and the exact
relationship between the train/test and train/val/test versions.

## RNA sources

The default root is `/data2/lama/self_unify_RNA`. Relative to `unify_RNA`, only the
three surv_set RNA matrices for five cohorts were extended. Existing rows are
byte-for-byte preserved, including gene order and expression values.

| Cancer | Unique patients before → after | Added |
|---|---:|---:|
| BRCA | 935 → 1088 | 153 |
| BLCA | 360 → 405 | 45 |
| STAD | 336 → 411 | 75 |
| HNSC | 394 → 518 | 124 |
| CRC / COADREAD | 317 → 371 | 54 |
| LUAD | 512 → 512 | 0 |
| LUSC | 498 → 498 | 0 |
| KIRC | 530 → 530 | 0 |

These are RNA availability counts, not final survival cohort sizes. Duplicate
samples explain why some CSV row counts exceed patient counts. The gene counts
remain combine=4999, hallmarks=4241, xena=1577. All mmp_set, SlotSPE RNA, metadata,
and pathway-composition files are unchanged (SHA-256 verified).

| Model | RNA recipe |
|---|---|
| MMP_TRANS, MMP_OT, SurvPath, DIMAF | mmp_set/hallmarks |
| MCAT, MLP, SNN, S_MLP | surv_set/raw_rna_data/combine |
| SlotSPE | slotspe/*_rna_inter.csv |
| STARPath | mmp_set/hallmarks by default; explicit surv_set/hallmarks supported |
| ABMIL, TransMIL, TITAN | No RNA |

STARPath keeps the original normalization attached to each RNA source. Scalers
and default NLL time bins are fitted on the selected training fold only.
CRC is the public task name; RNA/split names use COADREAD. CONCH directories map
CRC→CRC, LUAD/LUSC→NSCLC, KIRC→RCC; ST directories use the public cancer names.

## Frozen split selection

| Endpoint | Mode | External root |
|---|---|---|
| DSS | train_test (default) | /data2/lama/SPILT_DSS_test |
| DSS | train_val_test | /data2/lama/SPILT_DSS_val_test |
| OS | train_test | /data2/lama/SPILT_OS_test |
| OS | train_val_test | /data2/lama/SPILT_OS_val_test |

Within each root, use only `src/splits/survival/TCGA_<COHORT>_overall_survival_k=0..4`.
The historical `overall_survival` directory name does not specify the endpoint.
Each CSV contains both DSS and OS time/censorship columns. All four roots use the
same 3707 patients / 3981 slides: BRCA 1001, BLCA 366, STAD 330, HNSC 417,
LUAD 411, LUSC 402, CRC 305, KIRC 475. OS and DSS have different outer fold assignments.
Within either endpoint, adding validation preserves test and partitions only train.

## Launching and results

```bash
bash src/scripts/survival/BRCA/dimaf.sh --dry-run
bash src/scripts/survival/CRC/starpath.sh --loss cox
bash src/scripts/survival/LUSC/slotspe.sh --checkpoint best
bash src/scripts/survival/KIRC/mmp_trans.sh --endpoint os --early-stopping 1
```

Default settings: DSS, train/test, NLL, last checkpoint, no early stopping.
NLL batch size is 1 except DIMAF, which uses a true batch of 64 by default,
consistent with the official training entry point's batch setting. `BATCH_SIZE`
or `--batch-size` can override supported defaults, but STARPath, ABMIL, TransMIL,
MCAT, SurvPath and SlotSPE require NLL batch size 1; use `-- --accum_steps 8`, for
example, to accumulate gradients across patients.

All Cox training loaders use `EventAwareRiskSetBatchSampler`, targeting 64
patients per risk set by default. A singleton tail is regrouped with the previous
batch, so every patient appears exactly once per epoch without dropping or
duplicating patients. The single-patient WSI models perform two forward passes
per patient within each logical risk set: obtain the joint Cox risk gradients,
then restore RNG state and recompute for parameter gradients. Ordinary NLL
gradient accumulation does not substitute for a Cox risk set.

Best checkpoint or early stopping automatically selects train/val/test.
Best selection defaults to validation C-index; early stopping defaults to validation
loss, and both criteria are independently configurable. Validation without early
stopping is supported with `--split-mode train_val_test`.

Each of the 8 cancer directories has the same 13 thin model wrappers. Shared
options can be set in the wrapper/environment or passed on the CLI. Extra model
hyperparameters follow `--`, e.g. `-- --starpath_num_regions 16`; managed data,
split, loss and checkpoint arguments cannot be replaced there.

Runs are stored as
`results/DSS_standard_test/BRCA/DIMAF/<config-id>/<run-id>/fold_0/`.
Other protocol groups include `DSS_standard_val_test`, `DSS_earlystop_val_test`,
and the corresponding OS groups. Configuration identity includes loss, batch,
RNA, checkpoints, layers, hyperparameters, source-code digest and split hashes.
RNA/signature content hashes are included, so extending a CSV at the same path
also creates a new configuration identity. Used prototype hashes are recorded in
the run manifest.
Every invocation creates a fresh run; explicit reuse of an existing run ID fails.
`config.json` and `run_manifest.json` record the exact commands and completed folds.
`cv_summary.csv` contains each requested fold, mean and sample standard deviation
(ddof=1). `cv_summary.json` distinguishes a complete five-fold experiment from a
completed subset and contains the same numeric evaluation means/standard deviations.
Checkpoint/epoch identifiers are not averaged. A one-fold run has an undefined,
blank CSV standard deviation (`null` in JSON).
