# Supporting tools

Training and data preparation have separate entry points. None of these tools
rebuilds the frozen SPILT datasets during training.

| Current file | Purpose |
|---|---|
| tools/run_survival.py | Resolve one experiment, run its requested folds, write config/manifest and aggregate C-index |
| tools/build_prototypes.py | Build morphology vocabularies for an explicit endpoint, split protocol and fold set |
| tools/build_prototype_matrix.py | Sequentially build/audit all four protocol matrices on one GPU; per-fold logs, atomic progress manifest, and complete artifact index |
| tools/audit_data.py | Read-only cohort/RNA/feature audit, including case/slide/TSS separation |
| src/utils/experiment_config.py | Shared data paths, RNA recipes, protocol selection and prototype provenance |
| src/scripts/survival/<CANCER>/*.sh | Editable, thin entry points for all 13 model recipes |
| src/scripts/prototype/cancer.sh | Thin entry point for prototype construction |

## Earlier experiment utilities

The earlier Path2Space experiment contains exploratory data builders. They are
documented here rather than copied into the default training workflow:

| Earlier filename | Role and disposition |
|---|---|
| scripts/data/build_blca_common_splits.py | BLCA-only modality intersection experiment; superseded by the frozen eight-cohort datasets |
| scripts/data/build_tcga_survival_cohort_splits.py | Earlier RNA-intersection cohorts and availability/fold reports; historical provenance |
| scripts/data/build_tcga_cdr_rna_inter_splits.py | CDR-labelled, common RNA/CONCH/ST cohorts; authoritative frozen build records live with SPILT_DSS_test |
| scripts/data/build_survival_threeway_splits.py | Earlier three-way and SlotSPE-specific split families; replaced by the selected SPILT_*_val_test roots |
| training/summarize_survival_folds.py | Earlier fold summaries using population SD; replaced by per-run aggregation with explicitly documented sample SD |
| scripts/prototype/clustering.sh | Earlier feature-layout probing and command construction; replaced by shared configuration and explicit feature directories |

RNA expansion provenance remains beside the external data in
`self_unify_RNA/surv_set/raw_rna_expansion_{manifest.csv,summary.json}` and
`expand_raw_rna_from_toil.py`; these are not training dependencies.
The original MMP embedding/visualization examples remain examples for cached slide
representations. Current survival wrappers do not invoke them or reuse their old caches.
