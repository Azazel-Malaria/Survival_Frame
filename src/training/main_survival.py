"""Train one fold with explicit model, data, loss and checkpoint contracts."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.trainer import train
from utils.file_utils import save_pkl
from utils.utils import read_splits, seed_torch
from utils.experiment_config import (
    canonical_cancer, default_batch_size, load_data_paths, resolve_data_recipe,
    resolve_protocol, split_directory, prototype_spec, validate_prototype, stable_hash,
)
from wsi_datasets.unified_survival import (
    UnifiedSurvivalDataset, singleton_collate, cox_collate, titan_collate,
)
from wsi_datasets.survival_sampling import EventAwareRiskSetBatchSampler

SINGLE_PATIENT_MODELS = {'starpath', 'abmil', 'transmil', 'mcat', 'survpath', 'slotspe'}
PROTO_MODELS = {'PANTHER', 'OT', 'H2T', 'ProtoCount'}


def configure_data_recipe(args):
    paths = load_data_paths(args.data_paths)
    if args.cancer_type is None:
        match = re.search(r'TCGA_([A-Za-z0-9]+)_', str(args.split_dir))
        if not match:
            raise ValueError('Pass --cancer_type or an explicit TCGA fold directory')
        args.cancer_type = match.group(1)
    args.cancer_type = canonical_cancer(args.cancer_type)
    if args.survival_endpoint is None:
        args.survival_endpoint = args.target_col.split('_')[0] if args.target_col else 'dss'
    protocol = resolve_protocol(args.survival_endpoint, args.split_mode,
                                args.early_stopping, args.checkpoint_selection, paths)
    if args.target_col is not None and args.target_col != protocol['target_col']:
        raise ValueError('target_col conflicts with survival_endpoint')
    args.target_col, args.censorship_col = protocol['target_col'], protocol['censorship_col']
    args.split_mode = protocol['split_mode']
    if args.split_names is not None and args.split_names != protocol['split_names']:
        raise ValueError('split_names conflicts with resolved split protocol')
    args.split_names = protocol['split_names']
    explicit_dir = Path(args.split_dir).resolve() if args.split_dir else None
    if explicit_dir:
        match = re.fullmatch(r'TCGA_[A-Za-z0-9]+_overall_survival_k=([0-4])', explicit_dir.name)
        if match is None:
            raise ValueError(f'Invalid canonical fold directory: {explicit_dir}')
        fold = int(match.group(1))
        if args.fold is not None and args.fold != fold:
            raise ValueError('--fold conflicts with split_dir')
    else:
        fold = 0 if args.fold is None else args.fold
    expected = split_directory(protocol, args.cancer_type, fold).resolve()
    if explicit_dir is not None and explicit_dir != expected:
        # Selecting best/early stopping changes the protocol explicitly, never the cohort.
        original = resolve_protocol(args.survival_endpoint, 'train_test', False, 'last', paths)
        if protocol['split_mode'] == 'train_val_test' and explicit_dir == split_directory(original, args.cancer_type, fold).resolve():
            print(f'Validation required: using {expected}')
        else:
            raise ValueError(f'split_dir does not belong to the resolved protocol: expected {expected}')
    args.split_dir, args.split_k, args.fold = str(expected), fold, fold
    for split in args.split_names.split(','):
        if not (expected / f'{split}.csv').is_file():
            raise FileNotFoundError(expected / f'{split}.csv')
    recipe = resolve_data_recipe(args.survival_model, args.cancer_type, args.rna_set, paths)
    for key in ('data_source', 'omics_path', 'signature_path', 'composition_path',
                'st_dir', 'titan_embeddings_path', 'starpath_titan_model_path'):
        if getattr(args, key) is None:
            setattr(args, key, recipe[key])
    args.rna_set = recipe['rna_set']
    args.requires_omics = args.rna_set != 'none'
    args.data_source = [str(Path(p).resolve()) for p in args.data_source.split(',')]
    if args.model_histo_type is None:
        args.model_histo_type = 'PANTHER' if args.survival_model in {'mmp_trans', 'mmp_ot', 'dimaf'} else 'MIL'
    if args.model_histo_config is None:
        args.model_histo_config = args.model_histo_type + '_default'
    if args.survival_model == 'mmp_ot':
        args.model_mm_type = 'coattn_mot'
    elif args.survival_model == 'survpath':
        args.model_mm_type = 'survpath'
    layouts = {'mcat': 'functional', 'mlp': 'flat', 'snn': 'flat', 's_mlp': 'flat',
               'slotspe': 'flat', 'abmil': 'none', 'transmil': 'none', 'titan': 'none'}
    layout = layouts.get(args.survival_model, 'pathway')
    if args.omics_layout not in {'auto', layout}:
        raise ValueError(f'{args.survival_model} requires omics_layout={layout}')
    args.omics_layout = args.omics_modality = layout
    normalization = ('none' if args.rna_set in {'none', 'slotspe'} else
                     'standard' if args.rna_set == 'mmp_set' else
                     'starpath_dual' if args.survival_model == 'starpath' else 'global_minmax')
    if args.rna_normalization not in {'auto', normalization}:
        raise ValueError(f'{args.survival_model}/{args.rna_set} requires normalization={normalization}')
    args.rna_normalization = normalization
    if args.model_histo_type in PROTO_MODELS or args.survival_model == 'starpath':
        args.load_proto = args.fix_proto = True
        if args.proto_path is None:
            args.proto_path = prototype_spec(protocol, args.cancer_type, fold, paths,
                                             n_proto=args.n_proto, in_dim=args.in_dim)['path']
        if len(args.data_source) != 1:
            raise ValueError('Prototype experiments require one explicitly indexed feature root')
        validate_prototype(args.proto_path, expected / 'train.csv', args.data_source[0],
                           args.n_proto, args.in_dim, protocol['endpoint'], protocol['split_mode'])
    args.result_group = protocol['result_group']
    return args


def normalize_survival_args(args):
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.survival_model, args.loss_fn)
    if args.batch_size <= 0 or args.accum_steps <= 0 or args.max_epochs <= 0:
        raise ValueError('batch_size, accum_steps and max_epochs must be positive')
    if args.loss_fn == 'cox' and args.batch_size < 2:
        raise ValueError('Cox training requires a patient risk-set batch_size >= 2')
    if args.loss_fn == 'nll' and args.survival_model in SINGLE_PATIENT_MODELS and args.batch_size != 1:
        raise ValueError(f'{args.survival_model} NLL uses one WSI patient per batch; use accum_steps for gradient accumulation')
    args.needs_discrete_labels = args.loss_fn == 'nll' or args.survival_model == 'slotspe'
    if args.survival_model == 'slotspe':
        if args.aux_nll_bins <= 0:
            raise ValueError('SlotSPE auxiliary NLL requires aux_nll_bins > 0')
        if args.loss_fn == 'nll' and args.n_label_bins != args.aux_nll_bins:
            raise ValueError('SlotSPE main and auxiliary NLL must use identical time bins')
    if args.needs_discrete_labels and args.n_label_bins <= 0:
        raise ValueError('Discrete survival heads require n_label_bins > 0')
    if not args.needs_discrete_labels:
        args.n_label_bins = 0
    if args.loss_fn == 'cox' and args.survival_model == 'slotspe':
        args.n_label_bins = args.aux_nll_bins
    if args.es_patience <= 0 or args.es_min_epochs <= 0:
        raise ValueError('Early-stopping patience and min epochs must be positive')
    if not 0 <= args.nll_alpha <= 1:
        raise ValueError('nll_alpha must be in [0,1]')
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.survival_model == 'starpath':
        if args.n_proto != 16:
            raise ValueError('STARPath C requires 16 morphology prototypes')
        args.pathway_type = 'hallmarks'
    return args


def build_unified_datasets(csv_splits, args, censorship_col):
    if args.weighted_sample and (args.loss_fn != 'nll' or args.n_label_bins != 4):
        raise ValueError('Weighted sampling requires four-bin NLL')
    loaders, scaler, bins = {}, None, None
    for name in ['train'] + [key for key in csv_splits if key != 'train']:
        split, training = csv_splits[name], name == 'train'
        dataset = UnifiedSurvivalDataset(
            df_histo=split['histo'], df_gene=split['gene'], data_source=args.data_source,
            model_name=args.survival_model, survival_time_col=args.target_col,
            censorship_col=censorship_col, n_label_bins=args.n_label_bins, label_bins=bins,
            discretization=args.discretization,
            bag_size=args.train_bag_size if training else args.val_bag_size,
            is_training=training, omics_layout=args.omics_layout,
            rna_normalization=args.rna_normalization, rna_set=args.rna_set,
            signature_path=args.signature_path, titan_embeddings_path=args.titan_embeddings_path,
            morphology_prototype_path=args.proto_path, st_dir=args.st_dir,
            starpath_bag_size=args.starpath_bag_size,
            sampling_seed=args.seed + 1_000_003 * args.split_k, split_name=f'fold={args.split_k}:{name}',
        )
        if training:
            scaler, bins = dataset.get_scaler(), dataset.get_label_bins()
        dataset.apply_scaler(scaler)
        singleton = args.survival_model in SINGLE_PATIENT_MODELS
        if training and args.loss_fn == 'cox':
            batch_sampler = EventAwareRiskSetBatchSampler(
                dataset, riskset_size=args.batch_size, shuffle=True,
                seed=args.seed + 1_000_003 * args.split_k,
            )
            collate = cox_collate if singleton else titan_collate if args.survival_model == 'titan' else None
            loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                num_workers=args.num_workers, collate_fn=collate)
            loader.deferred_cox = singleton
        else:
            sampler = (WeightedRandomSampler(dataset.get_sample_weights(), len(dataset), replacement=True)
                       if training and args.weighted_sample else None)
            collate = singleton_collate if singleton else titan_collate if args.survival_model == 'titan' else None
            loader = DataLoader(dataset, batch_size=1 if singleton else args.batch_size,
                                shuffle=training and sampler is None, sampler=sampler,
                                num_workers=args.num_workers, collate_fn=collate)
        loaders[name] = loader
        print(f'{name}: {len(dataset)} patients, {len(loader)} batches')
    if hasattr(args, 'results_dir') and args.results_dir:
        save_pkl(str(Path(args.results_dir) / 'data_transforms.pkl'),
                 {'scaler': scaler, 'label_bins': bins, 'rna_set': args.rna_set,
                  'rna_gene_seq': loaders['train'].dataset.rna_gene_seq})
    return loaders


def main(args):
    configure_data_recipe(args)
    normalize_survival_args(args)
    paths = load_data_paths(args.data_paths)
    if args.results_dir is None:
        configuration = {k: v for k, v in vars(args).items() if k not in {'fold', 'split_k', 'split_dir'}}
        run = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        args.results_dir = str(Path(paths['results_root']) / args.result_group / args.cancer_type /
                               args.survival_model.upper() / stable_hash(configuration) / run / f'fold_{args.split_k}')
    output = Path(args.results_dir).resolve()
    if (output / 'summary.csv').exists() and not args.overwrite:
        raise FileExistsError(f'Results already exist: {output}; select a new run directory')
    output.mkdir(parents=True, exist_ok=True)
    args.results_dir = str(output)
    args.embedding_cache_dir = args.embedding_cache_dir or str(output / 'embeddings')
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2, sort_keys=True) + '\n')
    print(f'Protocol: {args.result_group}; loss={args.loss_fn}; batch={args.batch_size}; checkpoint={args.checkpoint_selection}')
    seed_torch(args.seed)
    splits = read_splits(args)
    loaders = build_unified_datasets(splits, args, args.censorship_col)
    results, dumps = train(loaders, args)
    row = {'fold': args.split_k}
    row.update({f'{metric}_{split}': float(value) for split, metrics in results.items()
                for metric, value in metrics.items()})
    summary = pd.DataFrame([row])
    summary.to_csv(output / 'summary.csv', index=False)
    if args.fold_summary_path and Path(args.fold_summary_path).resolve() != output / 'summary.csv':
        destination = Path(args.fold_summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(destination, index=False)
    (output / 'summary.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    for split, payload in dumps.items():
        save_pkl(str(output / f'{split}_results.pkl'), payload)
    return row


def build_parser():
    # Generic training settings
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    ### optimizer settings ###
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='maximum number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='grad accumulation steps')
    parser.add_argument('--opt', type=str, default='adamW',
                        choices=['adamW', 'adam', 'sgd', 'RAdam'])
    parser.add_argument('--lr_scheduler', type=str,
                        choices=['cosine', 'linear', 'constant'], default='cosine')
    parser.add_argument('--warmup_steps', type=int,
                        default=-1, help='warmup iterations')
    parser.add_argument('--warmup_epochs', type=int,
                        default=1, help='warmup epochs')
    parser.add_argument('--batch_size', type=int, default=None)

    ### misc ###
    parser.add_argument('--print_every', default=100,
                        type=int, help='how often to print')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)

    ### Earlystopper args ###
    parser.add_argument('--early_stopping', type=int, choices=[0, 1],
                        default=0, help='enable early stopping')
    parser.add_argument('--es_min_epochs', type=int, default=3,
                        help='early stopping min epochs')
    parser.add_argument('--es_patience', type=int, default=5,
                        help='early stopping min patience')
    parser.add_argument('--es_metric', type=str, choices=['loss', 'c_index'], default='loss',
                        help='early stopping metric')

    ### model args ###
    parser.add_argument('--model_histo_type', type=str, choices=['H2T', 'OT', 'PANTHER', 'ProtoCount', 'MIL'],
                        default=None, help='type of histology model')
    parser.add_argument('--ot_eps', default=0.1, type=float,
                        help='Strength for entropic constraint regularization for OT')
    parser.add_argument('--model_histo_config', type=str,
                        default=None, help="name of model config file")
    parser.add_argument('--n_fc_layers', type=int)
    parser.add_argument('--em_iter', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--out_type', type=str, default='allcat')

    # Multimodal args ###
    parser.add_argument('--num_coattn_layers', default=1, type=int)
    parser.add_argument('--model_mm_type', default='coattn',
                        choices=['coattn', 'coattn_mot', 'survpath', 'histo', 'gene'],
                        help='Multimodal model type')
    parser.add_argument(
        '--survival_model', default='mmp_trans',
        choices=['mmp_trans', 'mmp_ot', 'survpath', 'abmil', 'transmil',
                 'mcat', 'mlp', 'snn', 's_mlp', 'titan', 'dimaf', 'starpath', 'slotspe'],
        help='Survival backbone',
    )
    parser.add_argument('--net_indiv', action='store_true', default=False)
    parser.add_argument('--append_embed', type=str, default='none',
                        choices=['none', 'modality', 'proto', 'mp', 'random'])
    parser.add_argument('--append_prob', action='store_true', default=False)
    parser.add_argument('--histo_agg', default='mean')
    parser.add_argument('--omics_dir', default='./data_csvs/rna')
    parser.add_argument('--omics_modality', default='pathway')
    parser.add_argument('--type_of_path', default='hallmarks')
    parser.add_argument('--omics_path', default=None,
                        help='Explicit RNA CSV or directory containing rna_clean.csv')
    parser.add_argument('--rna_set', default=None,
                        choices=['mmp_set', 'surv_set', 'slotspe', 'none'],
                        help='Unified RNA collection selected by the model recipe')
    parser.add_argument('--omics_layout', default='auto',
                        choices=['auto', 'none', 'flat', 'pathway', 'functional'])
    parser.add_argument('--rna_normalization', default='auto',
                        choices=['auto', 'standard', 'global_minmax',
                                 'starpath_dual', 'none'])
    parser.add_argument('--signature_path', default=None)
    parser.add_argument('--composition_path', default=None)
    parser.add_argument('--pathway_type', default='combine',
                        choices=['hallmarks', 'xena', 'combine'])
    parser.add_argument('--omic_projection_dim', type=int, default=64)

    # Prototype related
    parser.add_argument('--load_proto', action='store_true', default=False)
    parser.add_argument('--proto_path', type=str)
    parser.add_argument('--fix_proto', action='store_true', default=False)
    parser.add_argument('--n_proto', type=int, default=16)

    parser.add_argument('--in_dim', default=768, type=int,
                        help='dim of input features')
    parser.add_argument('--bag_size', type=int, default=-1)
    parser.add_argument('--train_bag_size', type=int, default=-1)
    parser.add_argument('--val_bag_size', type=int, default=-1)
    parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox'],
                        help='which loss function to use')
    parser.add_argument('--nll_alpha', type=float, default=0.5,
                        help='Balance between censored / uncensored loss')
    parser.add_argument('--discretization', default='mmp_train_event_quantile',
                        choices=['mmp_train_event_quantile', 'train_equal_width'])
    parser.add_argument('--weighted_sample', action='store_true', default=False,
                        help='Enable 4-bin x event/censored weighted training sampling')

    # Added-model data and auxiliary-loss settings.
    parser.add_argument('--titan_embeddings_path', default=None)
    parser.add_argument('--st_dir', default=None)
    parser.add_argument('--starpath_bag_size', type=int, default=512,
                        help='Per-slide TITAN patch cap; coarse morphology always uses the full slide')
    parser.add_argument('--starpath_slide_agg', default='mean',
                        choices=['mean', 'gated', 'set_transformer'])
    parser.add_argument('--starpath_titan_model_path', default=None)
    parser.add_argument('--starpath_pathway_dim', type=int, default=256)
    parser.add_argument('--starpath_compatibility_dim', type=int, default=256)
    parser.add_argument('--starpath_context_dim', type=int, default=256)
    parser.add_argument('--starpath_pathway_dropout', type=float, default=0.25)
    parser.add_argument('--starpath_uot_epsilon', type=float, default=0.07)
    parser.add_argument('--starpath_uot_tau_source', type=float, default=0.5)
    parser.add_argument('--starpath_uot_tau_target', type=float, default=0.5)
    parser.add_argument('--starpath_uot_iterations', type=int, default=50)
    parser.add_argument('--starpath_alpha_init', type=float, default=0.03)
    parser.add_argument('--starpath_alpha_max', type=float, default=0.10)
    parser.add_argument('--starpath_num_regions', type=int, default=12)
    parser.add_argument('--starpath_region_seed_k', type=int, default=8)
    parser.add_argument('--starpath_region_candidate_k', type=int, default=3)
    parser.add_argument('--starpath_region_temperature', type=float, default=0.2)
    parser.add_argument('--starpath_region_spatial_weight', type=float, default=1.0)
    parser.add_argument('--starpath_region_sigma_min', type=float, default=0.05)
    parser.add_argument('--starpath_uot_iters', type=int, default=40)
    parser.add_argument('--starpath_uot_reliability_beta', type=float, default=1.0)
    parser.add_argument('--starpath_occupancy_log_weight', type=float, default=0.25)
    parser.add_argument('--starpath_titan_inject_layers', default=None,
                        help=('Comma- or space-delimited zero-based TITAN block ids; '
                              'defaults to 2,4'))
    parser.add_argument(
        '--starpath_titan_trainable_layers',
        default=None,
        help=('Comma- or space-delimited zero-based TITAN block ids to train; '
              'defaults to 2,3,4,5; use none to freeze all'),
    )
    parser.add_argument('--starpath_memory_prior_weight', type=float, default=1.0)
    parser.add_argument('--starpath_inject_alpha_max', type=float, default=0.1)
    parser.add_argument('--starpath_atlas_temperature', type=float, default=1.0)
    parser.add_argument('--starpath_atlas_alpha_init', type=float, default=0.02)
    parser.add_argument('--starpath_atlas_alpha_max', type=float, default=0.20)
    parser.add_argument('--starpath_atlas_count_scale', type=float, default=16.0)
    parser.add_argument('--starpath_dynamic_alpha_init', type=float, default=0.05)
    parser.add_argument('--starpath_dynamic_alpha_max', type=float, default=0.10)
    parser.add_argument('--starpath_region_weight', type=float, default=0.05)
    parser.add_argument('--starpath_alignment_weight', type=float, default=0.05)
    parser.add_argument('--slotspe_reconstruction_weight', type=float, default=0.01)
    parser.add_argument('--slotspe_wsi_aux_weight', type=float, default=1.0)
    parser.add_argument('--slotspe_omics_aux_weight', type=float, default=1.0)
    parser.add_argument('--slotspe_static_kv', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--dimaf_disentanglement_weight', type=float, default=7.0)

    # experiment task / label args ###
    parser.add_argument('--exp_code', type=str, default=None,
                        help='experiment code for saving results')
    parser.add_argument('--fold_summary_path', default=None,
                        help='Deterministic one-row CSV used for cross-fold aggregation')
    parser.add_argument('--task', type=str, default='unspecified_survival_task')
    parser.add_argument('--survival_endpoint', type=str.lower,
                        choices=['dss', 'os'], default=None,
                        help='Survival endpoint; defaults to DSS')
    parser.add_argument('--target_col', type=str, default=None,
                        help='Legacy endpoint column alias; prefer --survival_endpoint')
    parser.add_argument('--n_label_bins', type=int, default=4,
                        help='number of bins for event time discretization')

    # dataset / split args ###
    parser.add_argument('--data_source', type=str, default=None,
                        help='manually specify the data source')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='manually specify the set of splits to use')
    parser.add_argument('--cancer_type', default=None,
                        help='Explicit TCGA cohort code for direct data recipes')
    parser.add_argument('--split_names', type=str, default=None,
                        help='delimited list for specifying names within each split')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing results')

    # logging args ###
    parser.add_argument('--results_dir', default=None,
                        help='results directory (default: ./results)')
    parser.add_argument('--tags', nargs='+', type=str, default=None,
                        help='tags for logging')


    parser.add_argument('--data_paths', default=None, help='JSON data roots configuration')
    parser.add_argument('--fold', type=int, choices=range(5), default=None)
    parser.add_argument('--split_mode', choices=['train_test', 'train_val_test'], default='train_test')
    parser.add_argument('--checkpoint_selection', choices=['last', 'best'], default='last')
    parser.add_argument('--checkpoint_metric', choices=['c_index', 'loss'], default='c_index')
    parser.add_argument('--aux_nll_bins', type=int, default=4)
    parser.add_argument('--embedding_cache_dir', default=None)
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
