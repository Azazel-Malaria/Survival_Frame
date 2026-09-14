"""Patient-level data for all survival models, with explicit RNA recipes.

Each split keeps its frozen cases and ordered slides. Training transforms are
reused for validation/test; STARPath summarizes full slides before sampling.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from collections import Counter
from typing import Dict, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, default_collate

from .dataset_utils import apply_aligned_sampling, apply_sampling


_EXPECTED_STARPATH_PATHWAYS = 50
_STARPATH_ST_SUFFIX = '_stpath_pred.h5ad'
_RNA_SETS = frozenset({'mmp_set', 'surv_set', 'slotspe', 'none'})


def _unique_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates('case_id', keep='first')


def fit_label_bins(
    df: pd.DataFrame,
    survival_time_col: str,
    censorship_col: str,
    n_label_bins: int,
    protocol: str,
) -> Optional[np.ndarray]:
    """Fit one set of edges that is subsequently reused by every split."""
    if n_label_bins <= 0:
        return None
    cases = _unique_cases(df)
    times = pd.to_numeric(cases[survival_time_col], errors='raise')
    if protocol == 'mmp_train_event_quantile':
        event_times = times[cases[censorship_col].astype(int).eq(0)]
        if len(event_times) < n_label_bins:
            raise ValueError('Too few observed events to fit survival quantiles')
        _, edges = pd.qcut(event_times, q=n_label_bins, retbins=True, labels=False)
    elif protocol == 'train_equal_width':
        minimum, maximum = float(times.min()), float(times.max())
        if not np.isfinite(minimum + maximum) or maximum <= minimum:
            raise ValueError('Cannot fit equal-width bins to constant/non-finite times')
        edges = np.linspace(minimum, maximum, n_label_bins + 1)
    else:
        raise ValueError(f'Unknown discretization protocol {protocol!r}')
    edges = np.asarray(edges, dtype=np.float64)
    if len(edges) != n_label_bins + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError(f'Invalid survival bin edges: {edges}')
    # Match MMP's open-ended benchmark bins and make endpoint handling stable.
    edges[0] = min(edges[0], -1e-6)
    edges[-1] = max(edges[-1], 1e6)
    return edges


def _read_h5_dense_x(
    path: str,
    expected_gene_seq: Optional[Sequence[str]] = None,
    expected_source_coords=None,
) -> np.ndarray:
    """Read STPath's dense matrix and verify its axes against the source WSI."""
    with h5py.File(path, 'r') as handle:
        node = handle['X']
        if isinstance(node, h5py.Dataset):
            values = node[:]
        else:
            raise TypeError(f'Sparse/group STPath X is unsupported in {path}')
        if expected_gene_seq is not None:
            var = handle['var']
            index_key = _decode(var.attrs.get('_index', '_index'))
            genes = tuple(
                _decode(value).strip() for value in var[index_key][:]
            )
            if genes != tuple(expected_gene_seq):
                raise ValueError(
                    f'ST gene columns/order differ from the indexed axis in {path}'
                )
        source_coords = None
        if expected_source_coords is not None:
            if 'obsm' not in handle or 'source_coords' not in handle['obsm']:
                raise ValueError(
                    f'ST source_coords are missing from obsm in {path}'
                )
            source_coords = np.asarray(handle['obsm']['source_coords'][:])
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f'Invalid ST matrix {values.shape} in {path}')
    if source_coords is not None:
        expected_coords = np.asarray(expected_source_coords)
        if (
            source_coords.ndim != 2
            or source_coords.shape[1:] != (2,)
            or source_coords.shape[0] != values.shape[0]
            or not np.isfinite(source_coords).all()
        ):
            raise ValueError(
                f'Invalid ST source_coords {source_coords.shape} for '
                f'X {values.shape} in {path}'
            )
        if expected_coords.shape != source_coords.shape:
            raise ValueError(
                f'ST/WSI source coordinate shape mismatch in {path}: '
                f'{source_coords.shape} vs {expected_coords.shape}'
            )
        if not np.array_equal(source_coords, expected_coords):
            raise ValueError(
                f'ST/WSI source_coords differ or are out of order in {path}'
            )
    return values


def _decode(value) -> str:
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def load_st_gene_sequence(st_dir: str) -> Tuple[str, ...]:
    """Return the exact column order stored in the STPath AnnData files."""
    if not os.path.isdir(st_dir):
        raise FileNotFoundError(f'Invalid STARPath ST directory: {st_dir}')
    candidates = sorted(
        entry.path for entry in os.scandir(st_dir)
        if entry.is_file() and entry.name.endswith(_STARPATH_ST_SUFFIX)
    )
    if not candidates:
        raise FileNotFoundError(f'No STPath h5ad files found in {st_dir}')
    with h5py.File(candidates[0], 'r') as handle:
        var = handle['var']
        index_key = _decode(var.attrs.get('_index', '_index'))
        genes = tuple(_decode(value).strip() for value in var[index_key][:])
        if len(genes) != int(handle['X'].shape[1]):
            raise ValueError('STPath var/X gene count mismatch')
    if not genes or any(not gene for gene in genes):
        raise ValueError('STPath gene sequence contains empty names')
    if len(set(genes)) != len(genes):
        raise ValueError('STPath gene sequence contains duplicate names')
    return genes


def _slide_key(value: str) -> str:
    """Normalise a slide filename without stripping its dotted UUID."""
    name = os.path.basename(str(value)).strip()
    for suffix in ('.svs', '.h5', '.pt'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name.lower()


class UnifiedSurvivalDataset(Dataset):
    """One patient per item with a model-specific, explicit data payload."""

    def __init__(
        self,
        *,
        df_histo: pd.DataFrame,
        df_gene: Optional[pd.DataFrame],
        data_source: Sequence[str],
        model_name: str,
        survival_time_col: str,
        censorship_col: str,
        n_label_bins: int = 4,
        label_bins: Optional[Sequence[float]] = None,
        discretization: str = 'mmp_train_event_quantile',
        bag_size: int = -1,
        is_training: bool = False,
        omics_layout: str = 'none',
        rna_normalization: str = 'none',
        rna_set: Optional[str] = None,
        signature_path: Optional[str] = None,
        titan_embeddings_path: Optional[str] = None,
        morphology_prototype_path: Optional[str] = None,
        st_dir: Optional[str] = None,
        starpath_bag_size: int = 512,
        sampling_seed: int = 1,
        split_name: Optional[str] = None,
    ) -> None:
        self.model_name = str(model_name).lower()
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.target_col = survival_time_col
        self.n_label_bins = int(n_label_bins)
        self.discretization = discretization
        self.bag_size = int(bag_size)
        self.is_training = bool(is_training)
        self.omics_layout = str(omics_layout).lower()
        self.rna_normalization = str(rna_normalization).lower()
        if rna_set is None:
            if self.model_name in {'abmil', 'transmil', 'titan'}:
                rna_set = 'none'
            elif self.model_name in {'mcat', 'mlp', 'snn', 's_mlp'}:
                rna_set = 'surv_set'
            elif self.model_name == 'slotspe':
                rna_set = 'slotspe'
            else:
                rna_set = 'mmp_set'
        self.rna_set = str(rna_set).strip().lower()
        if self.rna_set not in _RNA_SETS:
            raise ValueError(
                f'Unknown rna_set {rna_set!r}; expected one of '
                f'{sorted(_RNA_SETS)}'
            )
        if self.model_name == 'starpath' and self.rna_set not in {
            'mmp_set', 'surv_set'
        }:
            raise ValueError('STARPath rna_set must be mmp_set or surv_set')
        self.signature_path = signature_path
        self.morphology_prototype_path = morphology_prototype_path
        self.st_dir = st_dir
        self.starpath_bag_size = int(starpath_bag_size)
        self.sampling_seed = int(sampling_seed)
        self.split_name = str(
            split_name if split_name is not None
            else ('train' if self.is_training else 'eval')
        )
        if self.model_name == 'starpath' and (
            self.starpath_bag_size == 0 or self.starpath_bag_size < -1
        ):
            raise ValueError(
                'starpath_bag_size must be -1 (full slide) or a positive cap'
            )
        self.X = None
        self.y = None
        if self.model_name == 'starpath' and self.omics_layout != 'pathway':
            raise ValueError(
                "STARPath requires omics_layout='pathway' with 50 Hallmark groups"
            )
        if self.model_name == 'starpath' and self.rna_normalization not in {
            'standard', 'dual', 'starpath_dual'
        }:
            raise ValueError(
                'Joint STARPath requires a source-aware train-fold RNA recipe'
            )

        required = {'case_id', 'slide_id', survival_time_col, censorship_col}
        missing_columns = required.difference(df_histo.columns)
        if missing_columns:
            raise ValueError(f'Split is missing columns {sorted(missing_columns)}')
        data_df = df_histo.copy()
        data_df['case_id'] = data_df['case_id'].astype(str)
        data_df['slide_id'] = data_df['slide_id'].astype(str).str.removesuffix('.svs')
        data_df[survival_time_col] = pd.to_numeric(
            data_df[survival_time_col], errors='raise'
        )
        data_df[censorship_col] = pd.to_numeric(
            data_df[censorship_col], errors='raise'
        )
        if not np.isfinite(data_df[[survival_time_col, censorship_col]].to_numpy()).all():
            raise ValueError('Frozen split contains missing/non-finite survival labels')
        if not data_df[censorship_col].isin([0, 1]).all():
            raise ValueError('Censorship values must be 0/1')
        data_df[censorship_col] = data_df[censorship_col].astype(int)
        if (data_df[survival_time_col] < 0).any():
            raise ValueError('Survival times must be non-negative')
        consistency = data_df.groupby('case_id')[[survival_time_col, censorship_col]].nunique()
        if (consistency > 1).any().any():
            raise ValueError('A case has inconsistent survival labels across slides')

        self.requires_omics = self.omics_layout != 'none'
        self.missing_rna_cases = []
        if self.requires_omics:
            if df_gene is None or 'case_id' not in df_gene.columns:
                raise ValueError(f'{self.model_name} requires a case_id RNA table')
            gene = df_gene.copy()
            gene['case_id'] = gene['case_id'].astype(str)
            gene = gene.drop_duplicates('case_id', keep='first')
            rna_cases = set(gene['case_id'])
            histo_cases = list(dict.fromkeys(data_df['case_id'].tolist()))
            self.missing_rna_cases = [case for case in histo_cases if case not in rna_cases]
            if self.missing_rna_cases:
                raise ValueError(
                    f'[{self.model_name}] frozen split has {len(self.missing_rna_cases)} cases '
                    f'without configured RNA: {self.missing_rna_cases[:10]}; '
                    'the cohort must not change between models'
                )
            case_order = list(dict.fromkeys(data_df['case_id'].tolist()))
            gene = gene.set_index('case_id').loc[case_order]
            numeric = gene.apply(pd.to_numeric, errors='raise')
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f'{self.model_name} RNA contains non-finite values')
            self.omics_data = numeric.astype(
                np.float64 if self.model_name in {'mmp_trans', 'mmp_ot', 'survpath'} else np.float32
            )
            self.rna_gene_seq = tuple(map(str, self.omics_data.columns))
            if self.model_name == 'starpath':
                # Both joint-model views must start from the exact same raw RNA
                # table. Keep it immutable across repeated split transforms.
                self._starpath_raw_omics_data = self.omics_data.copy(deep=True)
                self.coarse_omics_data = self.omics_data.copy(deep=True)
                self.fine_omics_data = self.omics_data.copy(deep=True)
        else:
            self.omics_data = None
            self.rna_gene_seq = tuple()

        self.data_df = data_df.reset_index(drop=True)
        self.case_ids = list(dict.fromkeys(self.data_df['case_id'].tolist()))
        if not self.case_ids:
            raise ValueError('No cases remain after applying the data recipe')
        self.idx2sample_df = pd.DataFrame({'sample_id': self.case_ids})
        self.case_rows = {
            case: rows.reset_index(drop=True)
            for case, rows in self.data_df.groupby('case_id', sort=False)
        }

        if self.n_label_bins > 0:
            if label_bins is None:
                if not self.is_training:
                    raise ValueError('Validation/test must receive time bins fitted on training patients')
                label_bins = fit_label_bins(
                    self.data_df,
                    survival_time_col,
                    censorship_col,
                    self.n_label_bins,
                    discretization,
                )
            self.label_bins = np.asarray(label_bins, dtype=np.float64)
            if len(self.label_bins) != self.n_label_bins + 1:
                raise ValueError('label_bins length does not match n_label_bins')
        else:
            self.label_bins = None

        self._build_labels()
        self.omic_names = []
        self.omic_sizes = []
        self.pathway_names = tuple()
        if self.omics_layout in {'pathway', 'functional'}:
            self._build_gene_groups()

        self.feature_paths: Dict[str, str] = {}
        self.titan_embeddings: Dict[str, torch.Tensor] = {}
        self.st_paths: Dict[str, str] = {}
        self.st_gene_seq = tuple()
        self._validated_st_paths = set()
        self.morphology_centroids: Optional[torch.Tensor] = None
        self._morphology_summary_cache = {}
        needs_raw_wsi = self.model_name not in {'mlp', 'snn', 's_mlp', 'titan'}
        if needs_raw_wsi:
            self._index_wsi_features(data_source)
        if self.model_name == 'titan':
            self._index_titan_embeddings(titan_embeddings_path)
        if self.model_name == 'starpath':
            if st_dir is None or not os.path.isdir(st_dir):
                raise FileNotFoundError(f'Invalid STARPath ST directory: {st_dir}')
            self.st_gene_seq = load_st_gene_sequence(st_dir)
            for entry in os.scandir(st_dir):
                if not entry.is_file() or not entry.name.endswith(_STARPATH_ST_SUFFIX):
                    continue
                key = entry.name[:-len(_STARPATH_ST_SUFFIX)].lower()
                if key in self.st_paths:
                    raise ValueError(f'Duplicate STPath prediction for {key}')
                self.st_paths[key] = entry.path
            missing_st = [
                slide for slide in self.data_df['slide_id']
                if slide.lower() not in self.st_paths
            ]
            if missing_st:
                raise FileNotFoundError(
                    f'{len(missing_st)} split slides lack exact STPath files: '
                    f'{missing_st[:5]}'
                )
            # Keep the optional STARPath/TITAN dependency stack lazy so the
            # dataset recipes for other models remain independently usable.
            from mil_models.modal_starpath import load_morphology_centroids

            self.morphology_centroids = load_morphology_centroids(
                morphology_prototype_path
            )

    def _build_labels(self) -> None:
        times, censorships, labels = [], [], []
        for case in self.case_ids:
            row = self.case_rows[case].iloc[0]
            time = float(row[self.survival_time_col])
            censor = int(row[self.censorship_col])
            if self.label_bins is None:
                label = time
            else:
                label = pd.cut(
                    pd.Series([time]), self.label_bins, labels=False, include_lowest=True
                ).iloc[0]
                if pd.isna(label):
                    raise ValueError(f'Survival time {time} falls outside label bins')
                label = int(label)
            times.append(time)
            censorships.append(censor)
            labels.append(label)
        self.survival_time_labels = torch.tensor(times, dtype=torch.float32)
        self.censorship_labels = torch.tensor(censorships, dtype=torch.float32)
        label_dtype = torch.long if self.label_bins is not None else torch.float32
        self.disc_labels = torch.tensor(labels, dtype=label_dtype)

    def _build_gene_groups(self) -> None:
        if self.signature_path is None or not os.path.isfile(self.signature_path):
            raise FileNotFoundError(f'Invalid signature_path: {self.signature_path}')
        signatures = pd.read_csv(self.signature_path)
        signature_columns = tuple(str(column).strip() for column in signatures.columns)
        if self.model_name == 'starpath':
            if len(signature_columns) != _EXPECTED_STARPATH_PATHWAYS:
                raise ValueError(
                    'STARPath requires exactly 50 Hallmark signature columns in '
                    f'CSV order, got {len(signature_columns)}'
                )
            if any(not name for name in signature_columns):
                raise ValueError('STARPath Hallmark pathway names must be non-empty')
            if len(set(signature_columns)) != len(signature_columns):
                raise ValueError('STARPath Hallmark pathway names must be unique')
            self.pathway_names = signature_columns

        available = set(self.rna_gene_seq)
        for pathway_index, column in enumerate(signatures.columns):
            genes = sorted({
                str(gene).strip() for gene in signatures[column].dropna()
                if str(gene).strip() in available
            })
            if self.model_name == 'starpath' and not genes:
                raise ValueError(
                    'STARPath Hallmark pathway has no genes in the configured RNA '
                    f'table: {signature_columns[pathway_index]!r}'
                )
            if genes:
                self.omic_names.append(genes)
        if self.model_name == 'starpath' and (
            len(self.omic_names) != _EXPECTED_STARPATH_PATHWAYS
        ):
            raise ValueError(
                'STARPath must retain exactly 50 non-empty Hallmark gene vectors, '
                f'got {len(self.omic_names)}'
            )
        if self.omics_layout == 'functional' and len(self.omic_names) != 6:
            raise ValueError(f'Functional recipe requires six groups, got {len(self.omic_names)}')
        self.omic_sizes = [len(genes) for genes in self.omic_names]

    def _index_wsi_features(self, data_source: Sequence[str]) -> None:
        paths = {}
        for source in data_source:
            if not os.path.isdir(source):
                raise FileNotFoundError(f'WSI feature directory not found: {source}')
            for entry in os.scandir(source):
                if not entry.is_file() or not entry.name.lower().endswith(('.h5', '.pt')):
                    continue
                key = _slide_key(entry.name)
                if key in paths:
                    raise ValueError(f'Duplicate WSI feature for {key}')
                paths[key] = entry.path
        missing = [slide for slide in self.data_df['slide_id'] if slide.lower() not in paths]
        if missing:
            raise FileNotFoundError(
                f'{len(missing)} split slides lack WSI features: {missing[:5]}'
            )
        self.feature_paths = paths

    def _index_titan_embeddings(self, path: Optional[str]) -> None:
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError(f'TITAN embedding pickle not found: {path}')
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        embeddings = np.asarray(payload['embeddings'], dtype=np.float32)
        filenames = payload['filenames']
        if embeddings.shape != (len(filenames), 768):
            raise ValueError(f'Unexpected TITAN embedding shape {embeddings.shape}')
        mapping = {
            _slide_key(str(name)): torch.from_numpy(vector)
            for name, vector in zip(filenames, embeddings)
        }
        missing = [slide for slide in self.data_df['slide_id'] if slide.lower() not in mapping]
        if missing:
            raise KeyError(f'{len(missing)} split slides lack TITAN embeddings: {missing[:5]}')
        self.titan_embeddings = mapping

    def get_scaler(self):
        if not self.requires_omics or self.rna_normalization == 'none':
            return {'kind': 'none'}
        if self.model_name == 'starpath':
            values = self._starpath_raw_omics_data.to_numpy(dtype=np.float32)
            if self.rna_set == 'mmp_set':
                return {
                    'kind': 'starpath_mmp_standard',
                    'rna_set': self.rna_set,
                    'gene_seq': self.rna_gene_seq,
                    'scaler': StandardScaler().fit(values),
                }
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError('Training RNA contains no finite values')
            minimum, maximum = float(finite.min()), float(finite.max())
            value_range = maximum - minimum
            if not np.isfinite(value_range) or value_range <= 0:
                value_range = 1.0
            return {
                'kind': 'starpath_dual',
                'rna_set': self.rna_set,
                'gene_seq': self.rna_gene_seq,
                'coarse': {
                    'kind': 'standard',
                    'scaler': StandardScaler().fit(values),
                },
                'fine': {
                    'kind': 'global_minmax',
                    'minimum': minimum,
                    'range': value_range,
                },
            }
        values = self.omics_data.to_numpy()
        if self.rna_normalization == 'standard':
            return {'kind': 'standard', 'scaler': StandardScaler().fit(values)}
        if self.rna_normalization == 'global_minmax':
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                raise ValueError('Training RNA contains no finite values')
            minimum, maximum = float(finite.min()), float(finite.max())
            value_range = maximum - minimum
            if not np.isfinite(value_range) or value_range <= 0:
                value_range = 1.0
            return {
                'kind': 'global_minmax', 'minimum': minimum, 'range': value_range
            }
        raise ValueError(f'Unknown RNA normalization {self.rna_normalization!r}')

    def apply_scaler(self, scaler) -> None:
        if not self.requires_omics or scaler['kind'] == 'none':
            return
        if self.model_name == 'starpath':
            expected_kind = (
                'starpath_mmp_standard'
                if self.rna_set == 'mmp_set'
                else 'starpath_dual'
            )
            if scaler.get('kind') != expected_kind:
                raise ValueError(
                    f'Joint STARPath with {self.rna_set} requires a '
                    f'{expected_kind} train-fold scaler'
                )
            if scaler.get('rna_set') != self.rna_set:
                raise ValueError(
                    'STARPath RNA set differs from the training fold: '
                    f'{self.rna_set!r} vs {scaler.get("rna_set")!r}'
                )
            if tuple(scaler.get('gene_seq', ())) != self.rna_gene_seq:
                raise ValueError(
                    'STARPath RNA columns/order differ from the training fold'
                )
            raw = self._starpath_raw_omics_data
            values = raw.to_numpy(dtype=np.float32)
            if self.rna_set == 'mmp_set':
                transformed = scaler['scaler'].transform(values).astype(
                    np.float32, copy=False
                )
                standardized = pd.DataFrame(
                    transformed, index=raw.index, columns=raw.columns
                )
                # Preserve the joint-model input contract while giving both
                # stages the exact MMP train-fold, per-gene standardized view.
                self.coarse_omics_data = standardized
                self.fine_omics_data = standardized
                self.omics_data = standardized
                return
            coarse = scaler['coarse']['scaler'].transform(values).astype(
                np.float32, copy=False
            )
            fine_cfg = scaler['fine']
            finite = np.isfinite(values)
            zero_mask = finite & (values == 0)
            fine = np.zeros_like(values, dtype=np.float32)
            fine[finite] = (
                (values[finite] - fine_cfg['minimum']) / fine_cfg['range']
            ) * 2.0 - 1.0
            # Preserve RNA missing-by-zero semantics used by Fine STARPath.
            fine[zero_mask] = 0.0
            self.coarse_omics_data = pd.DataFrame(
                coarse, index=raw.index, columns=raw.columns
            )
            self.fine_omics_data = pd.DataFrame(
                fine, index=raw.index, columns=raw.columns
            )
            # Keep the historical attribute as the coarse view for code that
            # only inspects it; model input is the explicit mapping below.
            self.omics_data = self.coarse_omics_data
            return
        values = self.omics_data.to_numpy()
        columns, index = self.omics_data.columns, self.omics_data.index
        if scaler['kind'] == 'standard':
            transformed = scaler['scaler'].transform(values)
        elif scaler['kind'] == 'global_minmax':
            finite = np.isfinite(values)
            zero_mask = finite & (values == 0)
            transformed = np.zeros_like(values, dtype=np.float32)
            transformed[finite] = (
                (values[finite] - scaler['minimum']) / scaler['range']
            ) * 2.0 - 1.0
            transformed[zero_mask] = 0.0
        else:
            raise ValueError(f'Unsupported scaler {scaler}')
        self.omics_data = pd.DataFrame(transformed, index=index, columns=columns)

    def get_label_bins(self):
        return self.label_bins

    def get_sample_weights(self) -> torch.Tensor:
        if self.label_bins is None:
            raise ValueError('Weighted sampling requires discrete NLL labels')
        classes = [
            int(label) * 2 + int(censor)
            for label, censor in zip(self.disc_labels, self.censorship_labels)
        ]
        counts = Counter(classes)
        return torch.tensor([1.0 / counts[value] for value in classes], dtype=torch.double)

    def __len__(self) -> int:
        return len(self.case_ids)

    def _labels_for_index(self, idx: int) -> Dict[str, torch.Tensor]:
        labels = {
            'survival_time': self.survival_time_labels[idx].reshape(1),
            'censorship': self.censorship_labels[idx].reshape(1),
            'label': self.disc_labels[idx].reshape(1),
        }
        if self.label_bins is not None:
            labels['discrete_label'] = self.disc_labels[idx].reshape(1)
        return labels

    def _omics_for_case(self, case: str):
        if not self.requires_omics:
            return torch.empty(0, dtype=torch.float32)
        if self.model_name == 'starpath':
            return {
                'coarse_pathways': [
                    torch.tensor(
                        self.coarse_omics_data.loc[case, genes].to_numpy(),
                        dtype=torch.float32,
                    )
                    for genes in self.omic_names
                ],
                'fine_rna': torch.tensor(
                    self.fine_omics_data.loc[case].to_numpy(),
                    dtype=torch.float32,
                ),
            }
        if self.omics_layout == 'flat':
            return torch.tensor(self.omics_data.loc[case].to_numpy(), dtype=torch.float32)
        return [
            torch.tensor(self.omics_data.loc[case, genes].to_numpy(), dtype=torch.float32)
            for genes in self.omic_names
        ]

    @staticmethod
    def _load_wsi(path: str):
        if path.lower().endswith('.pt'):
            features = torch.as_tensor(torch.load(path, map_location='cpu')).float()
            return features.squeeze(0) if features.dim() == 3 else features, None, 512
        with h5py.File(path, 'r') as handle:
            features = torch.from_numpy(handle['features'][:]).float()
            coords = torch.from_numpy(handle['coords'][:]) if 'coords' in handle else None
            attrs = handle['coords'].attrs if 'coords' in handle else {}
            patch_size = int(attrs.get('patch_size_level0', attrs.get('patch_size', 512)))
        if features.dim() == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.dim() != 2:
            raise ValueError(f'WSI features must be 2D in {path}')
        if coords is not None and len(coords) != len(features):
            raise ValueError(f'WSI feature/coordinate mismatch in {path}')
        return features, coords, patch_size

    def _raw_patient_bag(self, rows: pd.DataFrame):
        features = [self._load_wsi(self.feature_paths[slide.lower()])[0]
                    for slide in rows['slide_id']]
        bag = torch.cat(features, dim=0)
        bag, _, mask = apply_sampling(self.bag_size, bag, [])
        return bag, mask

    def _starpath_sampling_rng(self, case_id: str, slide_id: str):
        """Return a stable, split-specific RNG for validation/test sampling."""
        identity = '\x1f'.join((
            str(self.sampling_seed), self.split_name, str(case_id), str(slide_id)
        ))
        digest = hashlib.blake2b(
            identity.encode('utf-8'), digest_size=8, person=b'STARPath'
        ).digest()
        return np.random.default_rng(int.from_bytes(digest, byteorder='little'))

    def _full_slide_morphology_route_summary(
        self,
        slide_id: str,
        features: torch.Tensor,
        st: torch.Tensor,
    ):
        """Compute one aligned atlas while retaining the small morphology cache."""
        if self.morphology_centroids is None:
            raise RuntimeError('STARPath morphology centroids were not initialized')
        cache_key = _slide_key(slide_id)
        morphology_cache = self._morphology_summary_cache

        from mil_models.modal_starpath import (
            summarize_full_slide_morphology_route_atlas,
        )

        summary = summarize_full_slide_morphology_route_atlas(
            features,
            st,
            self.morphology_centroids,
        )
        if cache_key not in morphology_cache:
            morphology_cache[cache_key] = summary[:3]
        # Do not persist [16,G_st] atlases in every DataLoader worker. The full
        # ST matrix is read for aligned patch sampling on every access anyway,
        # so caching atlases would add hundreds of MB per large cohort without
        # reducing HDF5 I/O.
        return morphology_cache[cache_key], summary[3:]

    def _starpath_payload(self, case_id: str, rows: pd.DataFrame):
        """Build C-route inputs from complete slides before synchronized sampling."""
        payload = {key: [] for key in (
            'slides', 'coords', 'st', 'morphology_tokens', 'morphology_occupancy',
            'morphology_valid', 'route_atlas', 'route_atlas_counts',
            'route_atlas_valid', 'slide_ids', 'patch_sizes',
        )}
        for slide_id in rows['slide_id']:
            features, coords, patch_size = self._load_wsi(self.feature_paths[slide_id.lower()])
            if coords is None:
                raise ValueError(f'STARPath requires coordinates for {slide_id}')
            st_path = self.st_paths[slide_id.lower()]
            expected_genes = None if st_path in self._validated_st_paths else self.st_gene_seq
            st = torch.from_numpy(_read_h5_dense_x(
                st_path, expected_genes,
                expected_source_coords=coords.detach().cpu().numpy(),
            ))
            self._validated_st_paths.add(st_path)
            if st.shape != (len(features), len(self.st_gene_seq)):
                raise ValueError(f'ST/WSI axes differ for {slide_id}: {tuple(st.shape)}')
            morphology, route = self._full_slide_morphology_route_summary(str(slide_id), features, st)
            rng = (self._starpath_sampling_rng(case_id, slide_id)
                   if not self.is_training and self.starpath_bag_size > 0 else None)
            features, coords, st = apply_aligned_sampling(
                self.starpath_bag_size, features, coords, st, rng=rng,
            )
            values = (features, coords, st, *morphology, *route, str(slide_id), int(patch_size))
            for key, value in zip(payload, values):
                payload[key].append(value)
        return payload

    def __getitem__(self, idx: int):
        case = self.case_ids[idx]
        rows = self.case_rows[case]
        out = self._labels_for_index(idx)
        out['omics'] = self._omics_for_case(case)
        if self.X is not None:
            out['img'] = self.X[idx]
            return out
        if self.model_name in {'mlp', 'snn', 's_mlp'}:
            out['img'] = torch.zeros(1, 1, dtype=torch.float32)
        elif self.model_name == 'titan':
            out['img'] = torch.stack([
                self.titan_embeddings[slide.lower()] for slide in rows['slide_id']
            ])
        elif self.model_name == 'starpath':
            out['img'] = self._starpath_payload(case, rows)
        else:
            out['img'], mask = self._raw_patient_bag(rows)
            if mask is not None:
                out['attn_mask'] = mask
        return out


def singleton_collate(batch):
    """Keep STARPath's slide payload intact while retaining a batch axis.

    ``default_collate`` cannot stack STARPath's variable per-slide tensors, but
    labels and RNA still need the same leading batch dimension used by every
    other survival loader (notably ``[1, 1]`` labels for ``NLLSurvLoss``).
    """
    if len(batch) != 1:
        raise ValueError('singleton_collate requires batch_size=1')
    item = batch[0]
    return {
        key: value if key == 'img' and isinstance(value, dict) else default_collate([value])
        for key, value in item.items()
    }


def titan_collate(batch):
    """Pad variable slide counts and emit a mask for patient-level averaging."""
    if not batch:
        raise ValueError('titan_collate received an empty batch')
    embeddings = [item['img'] for item in batch]
    if any(not torch.is_tensor(value) or value.ndim != 2 for value in embeddings):
        raise ValueError('Every TITAN item must contain a [slides,features] tensor')
    feature_dims = {int(value.shape[1]) for value in embeddings}
    if feature_dims != {768}:
        raise ValueError(f'TITAN feature dimension must be 768, got {feature_dims}')
    if any(value.shape[0] == 0 for value in embeddings):
        raise ValueError('Every TITAN patient must retain at least one slide')
    dtypes = {value.dtype for value in embeddings}
    devices = {value.device for value in embeddings}
    if len(dtypes) != 1 or len(devices) != 1:
        raise ValueError('TITAN embeddings in one batch must share dtype and device')

    max_slides = max(int(value.shape[0]) for value in embeddings)
    padded = embeddings[0].new_zeros((len(batch), max_slides, 768))
    slide_mask = torch.zeros(
        (len(batch), max_slides), dtype=torch.bool, device=embeddings[0].device
    )
    for row, value in enumerate(embeddings):
        slide_count = int(value.shape[0])
        padded[row, :slide_count] = value
        slide_mask[row, :slide_count] = True

    result = {
        key: default_collate([item[key] for item in batch])
        for key in batch[0]
        if key != 'img'
    }
    result['img'] = padded
    result['slide_mask'] = slide_mask
    return result


def cox_collate(batch):
    """Keep a logical Cox risk set as independent patient batches."""
    if len(batch) < 2:
        raise ValueError('Cox risk sets require at least two patients')
    return {
        'cox_patient_batches': [singleton_collate([item]) for item in batch]
    }


__all__ = [
    'UnifiedSurvivalDataset', 'fit_label_bins', 'load_st_gene_sequence',
    'singleton_collate', 'cox_collate', 'titan_collate',
]
