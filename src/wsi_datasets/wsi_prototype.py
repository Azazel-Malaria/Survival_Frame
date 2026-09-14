from __future__ import division, print_function

import os
from collections.abc import Sequence

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


_FEATURE_SUFFIXES = {'.h5', '.pt'}
_LEGACY_LEAF_TYPES = {'feats_h5': '.h5', 'feats_pt': '.pt'}


def _slide_key(value):
    """Return a case-insensitive slide stem without truncating dotted UUIDs."""
    name = os.path.basename(str(value)).strip()
    lower_name = name.lower()
    for suffix in ('.svs', '.h5', '.pt'):
        if lower_name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name.lower()


def _feature_records(source):
    """Index one flat feature directory and return its single container type."""
    source = os.path.abspath(os.fspath(source))
    if not os.path.isdir(source):
        raise FileNotFoundError(f'Prototype feature directory not found: {source}')

    records = []
    suffixes = set()
    for entry in sorted(os.scandir(source), key=lambda item: item.name.lower()):
        if not entry.is_file():
            continue
        suffix = os.path.splitext(entry.name)[1].lower()
        if suffix not in _FEATURE_SUFFIXES:
            continue
        suffixes.add(suffix)
        records.append(
            {
                'fpath': entry.path,
                'fname': entry.name,
                '_feature_key': _slide_key(entry.name),
            }
        )

    if not records:
        raise FileNotFoundError(
            f'No flat .h5/.pt prototype features found in {source}'
        )
    if len(suffixes) != 1:
        raise ValueError(
            f'Prototype feature directory mixes H5 and PT containers: {source}'
        )

    suffix = next(iter(suffixes))
    expected_suffix = _LEGACY_LEAF_TYPES.get(os.path.basename(source).lower())
    if expected_suffix is not None and suffix != expected_suffix:
        raise ValueError(
            f'Legacy directory {source} must contain only {expected_suffix} files'
        )
    return suffix, records


class WSIProtoDataset(Dataset):
    """One slide per item for train-fold morphology prototype clustering."""

    def __init__(
        self,
        df,
        data_source,
        sample_col='slide_id',
        slide_col='slide_id',
    ):
        if isinstance(data_source, (str, bytes, os.PathLike)) or not isinstance(
            data_source, Sequence
        ):
            raise TypeError('data_source must be a non-empty sequence of directories')
        if not data_source:
            raise ValueError('data_source must contain at least one directory')
        if not isinstance(df, dict) or 'histo' not in df:
            raise TypeError("Prototype split payload must contain a 'histo' dataframe")

        self.data_source = []
        all_records = []
        feature_suffix = None
        seen_sources = set()
        for source in data_source:
            source = os.path.realpath(os.fspath(source))
            if source in seen_sources:
                raise ValueError(f'Duplicate prototype feature source: {source}')
            seen_sources.add(source)
            current_suffix, records = _feature_records(source)
            if feature_suffix is None:
                feature_suffix = current_suffix
            elif current_suffix != feature_suffix:
                raise ValueError(
                    'All prototype feature sources must use one container type; '
                    f'found {feature_suffix} and {current_suffix}'
                )
            self.data_source.append(source)
            all_records.extend(records)

        self.feature_suffix = feature_suffix
        self.use_h5 = feature_suffix == '.h5'
        self.feats_df = pd.DataFrame.from_records(all_records)
        duplicate_features = self.feats_df[
            self.feats_df['_feature_key'].duplicated(keep=False)
        ]
        if not duplicate_features.empty:
            duplicate_keys = sorted(duplicate_features['_feature_key'].unique())
            raise ValueError(
                'Duplicate prototype features for slide keys '
                f'{duplicate_keys[:5]}'
            )

        if not isinstance(df['histo'], pd.DataFrame):
            raise TypeError("Prototype split 'histo' payload must be a dataframe")
        self.data_df = df['histo'].copy()
        if 'Unnamed: 0' in self.data_df.columns:
            raise ValueError("Prototype split must not contain an 'Unnamed: 0' column")
        missing_columns = {
            column for column in (sample_col, slide_col)
            if column not in self.data_df.columns
        }
        if missing_columns:
            raise ValueError(
                f'Prototype split is missing columns {sorted(missing_columns)}'
            )
        if self.data_df.empty:
            raise ValueError('Prototype split contains no slides')

        self.sample_col = sample_col
        self.slide_col = slide_col
        self.data_df[sample_col] = self.data_df[sample_col].astype(str)
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)
        self.data_df['_feature_key'] = self.data_df[slide_col].map(_slide_key)
        if self.data_df['_feature_key'].eq('').any():
            raise ValueError('Prototype split contains an empty slide ID')
        duplicate_split = self.data_df[
            self.data_df['_feature_key'].duplicated(keep=False)
        ]
        if not duplicate_split.empty:
            duplicate_keys = sorted(duplicate_split['_feature_key'].unique())
            raise ValueError(
                f'Prototype split contains duplicate slide IDs {duplicate_keys[:5]}'
            )

        split_keys = set(self.data_df['_feature_key'])
        feature_keys = set(self.feats_df['_feature_key'])
        missing_features = sorted(split_keys.difference(feature_keys))
        if missing_features:
            raise FileNotFoundError(
                f'{len(missing_features)} split slides lack prototype features: '
                f'{missing_features[:5]}'
            )

        self.data_df = self.data_df.merge(
            self.feats_df[['_feature_key', 'fpath']],
            how='left',
            on='_feature_key',
            validate='one_to_one',
        ).drop(columns=['_feature_key'])
        self.data_df = self.data_df[
            ['fpath'] + [column for column in self.data_df.columns if column != 'fpath']
        ]
        self.idx2sample_df = pd.DataFrame(
            {'sample_id': self.data_df[sample_col].astype(str).unique()}
        )
        self.data_df.index = self.data_df[sample_col].astype(str)
        self.data_df.index.name = 'sample_id'
        self.X = None
        self.y = None

    def __len__(self):
        return len(self.idx2sample_df)

    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        feat_paths = self.data_df.loc[self.get_sample_id(idx), 'fpath']
        if isinstance(feat_paths, str):
            return [feat_paths]
        if isinstance(feat_paths, pd.Series):
            return feat_paths.tolist()
        return list(feat_paths)

    def _load_features(self, path):
        if self.use_h5:
            with h5py.File(path, 'r') as handle:
                if 'features' not in handle:
                    raise KeyError(f"H5 prototype feature has no 'features' dataset: {path}")
                features = torch.from_numpy(handle['features'][:])
        else:
            features = torch.as_tensor(torch.load(path, map_location='cpu'))

        if features.ndim == 3:
            if features.shape[0] != 1:
                raise ValueError(
                    'Prototype features must be [patches,dim] or [1,patches,dim]; '
                    f'got {tuple(features.shape)} in {path}'
                )
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(
                f'Prototype features must be two-dimensional in {path}; '
                f'got {tuple(features.shape)}'
            )
        return features

    def __getitem__(self, idx):
        feature_tensors = [
            self._load_features(path) for path in self.get_feat_paths(idx)
        ]
        feature_dims = {int(features.shape[1]) for features in feature_tensors}
        if len(feature_dims) != 1:
            raise ValueError(
                f'Prototype feature dimensions disagree for sample {self.get_sample_id(idx)}'
            )
        return {
            'img': torch.cat(feature_tensors, dim=0),
            'coords': [],
        }
