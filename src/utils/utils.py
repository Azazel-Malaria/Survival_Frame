

import pdb
import math
import os
from os.path import join as j_
import pickle
import pandas as pd
import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import logging

from transformers import (get_constant_schedule_with_warmup, 
                         get_linear_schedule_with_warmup, 
                         get_cosine_schedule_with_warmup)

import re

def safe_list_to(data, device):
    """
    Moves data to device when data is either a torch.Tensor or a list/tuple/dict of tensors.
    e.g. [d.to(device) for d in data]

    Parameters
    ----------
    data: torch.tensor, tuple, list, dict
        The input data to put on a device.

    device: torch.device
        The device to move data do.


    Output
    ------
    data: torch.tensor, tuple, list, dict
        Data or each element of data on the device preserving the input structure e.g. if a list was provided then a list will be output.
    """

    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, tuple):
        return tuple(safe_list_to(value, device) for value in data)
    if isinstance(data, list):
        return [safe_list_to(value, device) for value in data]
    if isinstance(data, dict):
        return {key: safe_list_to(value, device) for key, value in data.items()}

    # Batch dictionaries also contain strings, paths, IDs, numeric metadata,
    # and optional values. They do not have a device and must pass through
    # unchanged while tensors nested alongside them are moved recursively.
    return data

def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"

def extract_patching_info(s):
    match = re.search(r"extracted_mag(\d+)x_patch(\d+)_fp", s)
    mag, patch_size = -1, -1
    if match:
        mag = int(match.group(1))
        patch_size = int(match.group(2))
        return mag, patch_size


def parse_model_name(model_name, ckpt=None, inference_prec=None):
    # 'extracted-vit_base_patch16_224.ibot.mgb100m20X_bs1024_cropadjust_opnorm_wd0.04_0012_fp16'

    # get inference precision
    if inference_prec is None:
        inference_prec = 'fp32'
        if model_name.endswith('_fp16'):
            inference_prec = 'fp16'
            model_name = model_name[:-len('_fp16')]

    model_name = model_name.replace('extracted-', '')
    parsed = model_name.split('.', maxsplit=2)
    enc = model_name
    algo = ''
    exp = ''
    if len(parsed) >= 3:
        enc = parsed[0]
        algo = parsed[1]
        exp = '.'.join(parsed[2:])

    # get ckpt
    if ckpt is None:
        exp_parsed = exp.split('.')
        ckpt = exp_parsed[-1].split('_')[-1]
        if ckpt.isnumeric():
            ckpt = int(ckpt)
            exp = '.'.join(exp_parsed[:-1]) + '_'.join(exp_parsed[-1].split('_')[:-1])
        else:
            ckpt = -1
    else:
        if str(ckpt).isnumeric():
            ckpt = int(ckpt)
        else:
            ckpt = -1
    return dict(pretrain_enc=enc, 
                pretrain_algo=algo, 
                pretrain_exp=exp, 
                pretrain_ckpt=ckpt,
                inference_prec=inference_prec)

def merge_dict(main_dict, new_dict):
    for k, v in new_dict.items():
        if k not in main_dict:
            main_dict[k] = []
        main_dict[k].append(v)
    return main_dict


def array2list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def summarize_reulsts(results_dict, ignore_keys = ['folds']):
    summary = {}
    for k, v in results_dict.items():
        if k in ignore_keys: continue
        summary[f"{k}_avg"] = np.mean(v)
        # summary[f"{k}_std"] = np.std(v)
    return summary


def seed_torch(seed=7):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_splits(args, fold_idx=None):
    """Read a fixed cohort and check separation before fitting any transform."""
    split_names = args.split_names.split(',')
    if not split_names or len(set(split_names)) != len(split_names):
        raise ValueError('Split names must be a nonempty unique list')
    gene = None
    if bool(getattr(args, 'requires_omics', False)):
        if not args.omics_path:
            raise ValueError('An explicit RNA path is required for this model')
        gene = _read_gene(args.omics_path, None)
    result = {name: {'histo': _read_histo(args.split_dir, name, fold_idx), 'gene': gene}
              for name in split_names}
    for i, name in enumerate(split_names):
        for other in split_names[i + 1:]:
            for key in ('case_id', 'slide_id', 'tissue_source_site'):
                left, right = result[name]['histo'], result[other]['histo']
                if key in left and key in right:
                    overlap = set(left[key].dropna()) & set(right[key].dropna())
                    if overlap:
                        raise ValueError(f'{key} overlap between {name}/{other}: {sorted(overlap)[:5]}')
    return result


def _read_histo(split_dir, split, fold_idx):
    if fold_idx is not None:
        split_path = j_(split_dir, f'{split}_{fold_idx}.csv')
    else:
        split_path = j_(split_dir, f'{split}.csv')

    if not os.path.isfile(split_path):
        fold_description = (
            '' if fold_idx is None else f' for fold {fold_idx}'
        )
        raise FileNotFoundError(
            f"Split CSV {split!r}{fold_description} does not exist: "
            f"{split_path}"
        )
    df = pd.read_csv(split_path)
    assert 'Unnamed: 0' not in df.columns
    return df

def _read_gene(omics_source, split):
    """Read patient-by-gene or gene-by-patient TCGA RNA as case-by-gene."""
    del split  # RNA files are shared across folds/splits.

    omics_source = os.fspath(omics_source)
    split_path = (
        j_(omics_source, 'rna_clean.csv')
        if os.path.isdir(omics_source)
        else omics_source
    )
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"{split_path} not found!")

    df = pd.read_csv(split_path)
    if df.empty:
        raise ValueError(f"RNA table is empty: {split_path}")

    tcga_pattern = r'^TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}'
    unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]

    # Patient-by-gene inputs use an explicit ID column or a CSV index column
    # populated by TCGA sample/case IDs.
    id_col = None
    if 'case_id' in df.columns:
        id_col = 'case_id'
    elif 'sample' in df.columns:
        id_col = 'sample'
    else:
        tcga_id_cols = [
            col for col in unnamed_cols
            if df[col].notna().all()
            and df[col].astype(str).str.match(tcga_pattern).all()
        ]
        if len(tcga_id_cols) == 1:
            id_col = tcga_id_cols[0]

    if id_col is not None:
        source_ids = df[id_col].astype(str)
        drop_columns = list(unnamed_cols)
        for metadata_col in ('sample', 'case_id'):
            if metadata_col not in drop_columns:
                drop_columns.append(metadata_col)
        df = df.drop(columns=drop_columns, errors='ignore').copy()
    else:
        # SlotSPE stores genes in rows and TCGA patients in columns. Preserve
        # its row order so that it becomes the feature-column order after the
        # transpose.
        patient_cols = [
            col for col in df.columns
            if bool(re.match(tcga_pattern, str(col)))
        ]
        non_patient_cols = [col for col in df.columns if col not in patient_cols]
        if not patient_cols or len(non_patient_cols) != 1:
            raise ValueError(
                f"Cannot determine RNA orientation in {split_path}; expected "
                "a patient ID column or one gene column followed by TCGA "
                "patient columns."
            )

        gene_col = non_patient_cols[0]
        if df[gene_col].isna().any():
            raise ValueError(f"Empty gene names found in {split_path}")
        gene_names = df[gene_col].astype(str).str.strip()
        if gene_names.eq('').any():
            raise ValueError(f"Empty gene names found in {split_path}")

        df = df.set_index(gene_col)[patient_cols].transpose(copy=True)
        # Transpose preserves the source row order as the output gene order.
        df.columns = gene_names.to_list()
        source_ids = pd.Series(df.index.astype(str), index=df.index)
        df = df.reset_index(drop=True)

    case_ids = source_ids.astype(str).str.extract(
        r'^(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})', expand=False
    )
    if case_ids.isna().any():
        raise ValueError(f"Malformed TCGA IDs found in {split_path}")

    df.insert(0, 'case_id', case_ids.to_numpy())

    # Prefer primary tumour RNA when multiple samples map to one case.
    sample_type = pd.to_numeric(source_ids.astype(str).str[13:15], errors='coerce')
    df['_sample_rank'] = np.select(
        [sample_type.eq(1).to_numpy(), sample_type.between(1, 9).to_numpy()],
        [0, 1],
        default=2
    )
    df['_row_order'] = np.arange(len(df))
    df = (df.sort_values(['case_id', '_sample_rank', '_row_order'], kind='stable')
            .drop_duplicates('case_id', keep='first')
            .drop(columns=['_sample_rank', '_row_order'])
            .reset_index(drop=True))

    non_numeric = [
        col for col in df.columns
        if col != 'case_id' and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numeric RNA feature columns in {split_path}: {non_numeric[:5]}"
        )

    return df



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs
    assert not (warmup_steps > 0 and warmup_epochs > 0), "Cannot have both warmup steps and epochs"
    accum_steps = args.accum_steps
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = warmup_epochs * math.ceil(len(dataloader) / accum_steps)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps)
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(math.ceil(len(dataloader) / accum_steps) * epochs),
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=math.ceil(len(dataloader) / accum_steps) * epochs,
        )
    return lr_scheduler


def get_optim(args, model=None, parameters=None):
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    if parameters is None:
        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(
            n, p) and p.requires_grad]
        parameters = [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ]

    if args.opt == "adamW":
        optimizer = optim.AdamW(parameters, lr=args.lr)
    elif args.opt == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
    elif args.opt == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=args.lr)
    else:
        raise NotImplementedError
    return optimizer
 

def print_network(net):
    num_params = 0
    num_params_train = 0

    logging.info(str(net))
    # print(str(net))
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    logging.info(f'Total number of parameters: {num_params}')
    logging.info(f'Total number of trainable parameters: {num_params_train}')

    # print('Total number of parameters: %d' % num_params)
    # print('Total number of trainable parameters: %d' % num_params_train)
