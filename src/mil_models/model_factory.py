import os
from mil_models import (PANTHER, OT, H2T, ProtoCount)

from mil_models import (PANTHERConfig, OTConfig, ProtoCountConfig, H2TConfig)

from mil_models.model_multimodal import coattn, coattn_mot

import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from utils.file_utils import save_pkl, load_pkl
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_starpath_inject_layers(value, *, allow_empty=False):
    """Normalize one CLI/config value into sorted unique TITAN layer ids."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError('starpath_titan_inject_layers must contain integers, not bool')
    if isinstance(value, str):
        values = value.replace(',', ' ').split()
    elif isinstance(value, (int, np.integer)):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise TypeError(
                'starpath_titan_inject_layers must be an integer or integer list'
            ) from exc
    if not values:
        if allow_empty:
            return []
        raise ValueError('starpath_titan_inject_layers must not be empty')
    if any(isinstance(layer, (bool, np.bool_)) for layer in values):
        raise TypeError('starpath_titan_inject_layers must contain integers, not bool')
    try:
        layers = [int(layer) for layer in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            'starpath_titan_inject_layers must contain only integers'
        ) from exc
    if any(layer < 0 for layer in layers):
        raise ValueError('starpath_titan_inject_layers must be non-negative')
    return sorted(set(layers))


def _parse_starpath_trainable_layers(value):
    """Normalize an independent trainable-layer list (``none`` means frozen)."""
    if value is None:
        return []
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return []
    try:
        return _parse_starpath_inject_layers(value, allow_empty=True)
    except (TypeError, ValueError) as exc:
        message = str(exc).replace(
            'starpath_titan_inject_layers', 'starpath_titan_trainable_layers'
        )
        raise type(exc)(message) from exc


def create_embedding_model(args, mode='classification', config_dir=None):
    """
    Create classification or survival models
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
    config_path = os.path.join(config_dir, args.model_histo_config, 'config.json')
    assert os.path.exists(config_path), f"Config path {config_path} doesn't exist!"

    model_type = args.model_histo_type
    update_dict = {'in_dim': args.in_dim,
                   'out_size': args.n_proto,
                   'load_proto': args.load_proto,
                   'fix_proto': args.fix_proto,
                   'proto_path': args.proto_path}
    
    if mode == 'classification':
        update_dict.update({'n_classes': args.n_classes})
    elif mode == 'survival':
        if args.loss_fn == 'nll':
            update_dict.update({'n_classes': args.n_label_bins})
        elif args.loss_fn == 'cox':
            update_dict.update({'n_classes': 1})
        else:
            raise ValueError('Survival loss must be nll or cox')
    elif mode == 'emb': # Create just slide-representation model
        pass
    else:
        raise NotImplementedError(f"Not implemented for {mode}...")

    if model_type == 'PANTHER':
        update_dict.update({'out_type': args.out_type, 'tau': args.tau,
                            'ot_eps': args.ot_eps, 'em_iter': args.em_iter})
        config = PANTHERConfig.from_pretrained(config_path, update_dict=update_dict)
        model = PANTHER(config=config, mode=mode)
    elif model_type == 'OT':
        update_dict.update({'out_type': args.out_type, 'ot_eps': args.ot_eps})
        config = OTConfig.from_pretrained(config_path, update_dict=update_dict)
        model = OT(config=config, mode=mode)
    elif model_type == 'H2T':
        config = H2TConfig.from_pretrained(config_path, update_dict=update_dict)
        model = H2T(config=config, mode=mode)
    elif model_type == 'ProtoCount':
        config = ProtoCountConfig.from_pretrained(config_path, update_dict=update_dict)
        model = ProtoCount(config=config, mode=mode)
    else:
        raise NotImplementedError(f"Not implemented for {model_type}!")

    return model


def _create_unified_survival_model(args, omic_sizes, num_classes):
    """Lazily construct an added baseline without importing optional stacks."""
    name = str(args.survival_model).lower()
    from mil_models.survival_adapter import UnifiedSurvivalAdapter

    if name == 'abmil':
        from mil_models.model_ABMIL import ABMIL
        backbone = ABMIL(
            fusion=None, n_classes=num_classes, wsi_input_dim=args.feat_dim
        )
    elif name == 'transmil':
        from mil_models.model_TMIL import TMIL
        backbone = TMIL(
            fusion=None, n_classes=num_classes, wsi_input_dim=args.feat_dim
        )
        name = 'transmil'
    elif name == 'mcat':
        from mil_models.model_MCATPathways import MCATPathways
        if len(omic_sizes) != 6:
            raise ValueError(f'MCAT requires six functional groups, got {omic_sizes}')
        backbone = MCATPathways(
            omic_sizes=omic_sizes,
            n_classes=num_classes,
            fusion='concat',
            wsi_input_dim=args.feat_dim,
        )
    elif name == 'mlp':
        from mil_models.model_MLPOmics import MLPOmics
        backbone = MLPOmics(
            input_dim=args.omic_dim,
            n_classes=num_classes,
            projection_dim=getattr(args, 'omic_projection_dim', 64),
        )
    elif name == 'snn':
        from mil_models.model_SNNOmics import SNNOmics
        backbone = SNNOmics(
            omic_input_dim=args.omic_dim,
            n_classes=num_classes,
        )
    elif name == 's_mlp':
        import pandas as pd
        from mil_models.model_MaskedOmics import MaskedOmics

        composition = pd.read_csv(args.composition_path)
        if 'gene' not in composition.columns:
            raise ValueError('S-MLP composition CSV requires a gene column')
        composition['gene'] = composition['gene'].astype(str)
        composition = composition.set_index('gene')
        gene_seq = list(args.rna_gene_seq)
        missing = [gene for gene in gene_seq if gene not in composition.index]
        if missing:
            raise ValueError(
                f'S-MLP composition is missing {len(missing)} RNA genes; first={missing[:5]}'
            )
        composition = composition.loc[gene_seq]
        backbone = MaskedOmics(
            df_comp=composition,
            input_dim=len(gene_seq),
            num_classes=num_classes,
            device='cpu',
        )
    elif name == 'dimaf':
        from mil_models.model_dimaf import DIMAF
        backbone = DIMAF(
            rna_dims=omic_sizes,
            histo_dim=args.in_dim,
            num_classes=num_classes,
            num_proto_wsi=args.n_proto,
            disentanglement_weight=getattr(args, 'dimaf_disentanglement_weight', 7.0),
            d1_weight=0.5,
            d2_weight=0.5,
        )
    elif name == 'titan':
        from mil_models.model_titan import TITANSurvivalHead
        backbone = TITANSurvivalHead(input_dim=768, n_classes=num_classes)
    elif name == 'slotspe':
        from mil_models.model_slotspe import SlotSPE
        backbone = SlotSPE(
            input_dim=args.feat_dim,
            n_classes=num_classes,
            mode='survival',
            rna_gene_seq=args.rna_gene_seq,
            signature_path=args.signature_path,
            pathway_type=getattr(args, 'pathway_type', 'combine'),
            aux_nll_bins=getattr(args, 'aux_nll_bins', args.n_label_bins or 4),
            static_kv=getattr(args, 'slotspe_static_kv', True),
            lambda_recon_loss=getattr(args, 'slotspe_reconstruction_weight', 0.01),
            lambda_wsi_aux_nll=getattr(args, 'slotspe_wsi_aux_weight', 1.0),
            lambda_omics_aux_nll=getattr(args, 'slotspe_omics_aux_weight', 1.0),
        )
    elif name == 'starpath':
        from mil_models.modal_starpath import STARPath, STARPathPatientAdapter, build_slide_aggregator
        raw_inject_layers = getattr(args, 'starpath_titan_inject_layers', None)
        inject_layers = _parse_starpath_inject_layers(
            [2, 4] if raw_inject_layers is None else raw_inject_layers)
        raw_trainable_layers = getattr(args, 'starpath_titan_trainable_layers', None)
        trainable_layers = _parse_starpath_trainable_layers(
            [2, 3, 4, 5] if raw_trainable_layers is None else raw_trainable_layers)

        backbone = STARPath(
            input_dim=args.feat_dim,
            n_classes=num_classes,
            mode='survival',
            omic_sizes=omic_sizes,
            pathway_names=getattr(args, 'pathway_names', None),
            prototype_path=args.proto_path,
            coarse_pathway_dim=getattr(args, 'starpath_pathway_dim', 256),
            coarse_compatibility_dim=getattr(
                args, 'starpath_compatibility_dim', 256
            ),
            coarse_context_dim=getattr(args, 'starpath_context_dim', 256),
            coarse_pathway_dropout=getattr(
                args, 'starpath_pathway_dropout', 0.25
            ),
            coarse_uot_epsilon=getattr(args, 'starpath_uot_epsilon', 0.07),
            coarse_uot_tau_source=getattr(
                args, 'starpath_uot_tau_source', 0.5
            ),
            coarse_uot_tau_target=getattr(
                args, 'starpath_uot_tau_target', 0.5
            ),
            coarse_uot_iterations=getattr(
                args, 'starpath_uot_iterations', 50
            ),
            coarse_alpha_init=getattr(args, 'starpath_alpha_init', 0.03),
            coarse_alpha_max=getattr(args, 'starpath_alpha_max', 0.10),
            rna_gene_seq=args.rna_gene_seq,
            st_gene_seq=args.st_gene_seq,
            pathway_type=getattr(args, 'pathway_type', 'hallmarks'),
            pathway_signature_path=args.signature_path,
            model_path=getattr(args, 'starpath_titan_model_path', None),
            titan_trainable_layers=trainable_layers,
            region_num=getattr(args, 'starpath_num_regions', 12),
            region_seed_knn=getattr(args, 'starpath_region_seed_k', 8),
            region_candidate_topk=getattr(
                args, 'starpath_region_candidate_k', 3
            ),
            region_temperature=getattr(
                args, 'starpath_region_temperature', 0.2
            ),
            region_score_spatial_weight=getattr(
                args, 'starpath_region_spatial_weight', 1.0
            ),
            region_min_spatial_scale=getattr(
                args, 'starpath_region_sigma_min', 0.05
            ),
            transport_epsilon=getattr(args, 'starpath_uot_epsilon', 0.07),
            transport_tau_source=getattr(
                args, 'starpath_uot_tau_source', 0.5
            ),
            transport_tau_target=getattr(
                args, 'starpath_uot_tau_target', 0.5
            ),
            transport_sinkhorn_iters=getattr(
                args, 'starpath_uot_iters', 40
            ),
            transport_reliability_beta=getattr(
                args, 'starpath_uot_reliability_beta', 1.0
            ),
            transport_occupancy_log_weight=getattr(
                args, 'starpath_occupancy_log_weight', 0.25
            ),
            titan_inject_layers=inject_layers,
            region_assignment_log_weight=getattr(
                args, 'starpath_memory_prior_weight', 1.0
            ),
            inject_alpha_max=getattr(
                args, 'starpath_inject_alpha_max', 0.1
            ),
            atlas_temperature=getattr(
                args, 'starpath_atlas_temperature', 1.0
            ),
            atlas_alpha_init=getattr(
                args, 'starpath_atlas_alpha_init', 0.02
            ),
            atlas_alpha_max=getattr(
                args, 'starpath_atlas_alpha_max', 0.20
            ),
            atlas_count_scale=getattr(
                args, 'starpath_atlas_count_scale', 16.0
            ),
            dynamic_alpha_init=getattr(
                args, 'starpath_dynamic_alpha_init', 0.05
            ),
            dynamic_alpha_max=getattr(
                args, 'starpath_dynamic_alpha_max', 0.10
            ),
            loss_w_region=getattr(args, 'starpath_region_weight', 0.05),
            loss_w_align=getattr(args, 'starpath_alignment_weight', 0.05),
        )
        aggregator = build_slide_aggregator(
            getattr(args, 'starpath_slide_agg', 'mean'), input_dim=768
        )
        return STARPathPatientAdapter(backbone, aggregator)
    else:
        raise NotImplementedError(f'Unknown survival_model={name!r}')

    return UnifiedSurvivalAdapter(backbone, name, aux_nll_alpha=getattr(args, 'nll_alpha', 0.5))


def create_multimodal_survival_model(args, omic_sizes=()):
    if args.loss_fn == 'nll':
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        num_classes = 1
    else:
        raise ValueError('Survival loss must be nll or cox')

    survival_model = str(args.survival_model).lower()
    if survival_model not in {'mmp_trans', 'mmp_ot', 'survpath'}:
        return _create_unified_survival_model(args, omic_sizes, num_classes)

    if args.model_mm_type in ['coattn', 'gene', 'histo']:   # This enables self-attn/coattn within/across modalities
        #
        # ex 1: Coattn across both modalities - modality: 'both' args.model_mm_type: 'coattn' num_coattn_layers: 1
        # ex 2: Self-attn within a modality - modality: 'both' args.model_mm_type: 'histo' or 'gene'
        #
        model = coattn(omic_sizes=omic_sizes,
                       histo_in_dim=args.feat_dim,
                       path_proj_dim=256,
                       num_classes=num_classes,
                       num_coattn_layers=args.num_coattn_layers,
                       modality=args.model_mm_type,
                       histo_agg=args.histo_agg,                       
                       histo_model=args.model_histo_type,
                       append_embed=args.append_embed,
                       net_indiv=args.net_indiv,
                       )

    elif args.model_mm_type == 'survpath':
        model = coattn(omic_sizes=omic_sizes,
                       histo_in_dim=args.feat_dim,
                       path_proj_dim=256,
                       num_classes=num_classes,
                       num_coattn_layers=1,
                       modality=args.model_mm_type,
                       histo_agg='mean',                       
                       histo_model='mil',
                       append_embed=None,
                       net_indiv=False,
                       )

    elif args.model_mm_type == 'coattn_mot':
        model = coattn_mot(omic_sizes=omic_sizes,
                            histo_in_dim=args.feat_dim,
                            path_proj_dim=256,
                            num_classes=num_classes,
                            num_coattn_layers=args.num_coattn_layers,
                            modality=args.model_mm_type,
                            histo_agg=args.histo_agg,                       
                            histo_model=args.model_histo_type,
                            append_embed=args.append_embed,
                            net_indiv=args.net_indiv,
                            )

    else:
        raise ValueError(f'Unknown multimodal architecture {args.model_mm_type!r}')
    from mil_models.survival_adapter import UnifiedSurvivalAdapter
    return UnifiedSurvivalAdapter(model, survival_model)



def _ordered_dataset_sample_ids(dataset):
    """Patient order is part of the cache identity, never an intersection filter."""
    ids = dataset.idx2sample_df['sample_id'].astype(str).tolist()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Embedding cache requires nonempty, unique patient IDs')
    return ids


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _embedding_cache_metadata(datasets, args, mode):
    """Bind the cached features to their input cohort and effective encoder setup."""
    config_path = Path(__file__).resolve().parents[1] / 'configs' / args.model_histo_config / 'config.json'
    parameters = {name: getattr(args, name, None) for name in (
        'model_histo_type', 'model_histo_config', 'in_dim', 'n_proto', 'out_type',
        'tau', 'ot_eps', 'em_iter', 'load_proto', 'fix_proto', 'seed',
    )}
    metadata = {
        'schema': 2, 'mode': mode, 'parameters': parameters,
        'config_sha256': _file_sha256(config_path),
        'prototype_sha256': _file_sha256(args.proto_path) if args.load_proto else None,
        'data_sources': [str(Path(source).resolve()) for source in args.data_source],
        'splits': {},
    }
    for split, loader in datasets.items():
        dataset = loader.dataset
        split_metadata = {
            'sample_ids': _ordered_dataset_sample_ids(dataset),
            'bag_size': getattr(dataset, 'bag_size', -1),
            'sampling_seed': getattr(dataset, 'sampling_seed', None),
        }
        if hasattr(dataset, 'data_df'):
            split_metadata['rows_sha256'] = hashlib.sha256(
                dataset.data_df.to_json(orient='split', date_format='iso').encode()).hexdigest()
        features = getattr(dataset, 'feature_paths', {})
        split_metadata['feature_files'] = [
            [str(Path(path).resolve()), os.stat(path).st_size, os.stat(path).st_mtime_ns]
            for path in sorted(set(features.values()))
        ]
        for field in ('survival_time_labels', 'censorship_labels', 'disc_labels'):
            values = getattr(dataset, field, None)
            if values is not None:
                split_metadata[field] = torch.as_tensor(values).cpu().tolist()
        metadata['splits'][split] = split_metadata
    return metadata


def _embedding_cache_path(args, metadata):
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cache_dir = Path(args.embedding_cache_dir)
    return cache_dir / f'{args.model_histo_type}_{key}.pkl'


def _valid_embedding_payload(payload, metadata):
    if not isinstance(payload, dict) or payload.get('metadata') != metadata:
        return False
    embeddings = payload.get('embeddings', {})
    for split, spec in metadata['splits'].items():
        entry = embeddings.get(split)
        if not isinstance(entry, dict) or entry.get('sample_ids') != spec['sample_ids']:
            return False
        for field in ('X', 'y'):
            value = entry.get(field)
            if value is None or len(value) != len(spec['sample_ids']):
                return False
    return True


def prepare_emb(datasets, args, mode='classification'):
    """Compute official prototype representations with an input-bound cache."""
    metadata = _embedding_cache_metadata(datasets, args, mode)
    path = _embedding_cache_path(args, metadata)
    if path.is_file():
        payload = load_pkl(str(path))
        if not _valid_embedding_payload(payload, metadata):
            raise ValueError(f'Embedding cache metadata or row count is invalid: {path}')
        embeddings = payload['embeddings']
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        model = create_embedding_model(args, mode=mode).to(device)
        embeddings = {}
        # Preserve caller sampling state; cached representations use a fixed seed.
        numpy_state = np.random.get_state()
        try:
            with torch.random.fork_rng():
                torch.manual_seed(int(getattr(args, 'seed', 1)))
                np.random.seed(int(getattr(args, 'seed', 1)))
                for split, loader in datasets.items():
                    X, y = model.predict(loader, use_cuda=torch.cuda.is_available())
                    embeddings[split] = {
                        'X': X, 'y': y,
                        'sample_ids': metadata['splits'][split]['sample_ids'],
                    }
        finally:
            np.random.set_state(numpy_state)
        payload = {'metadata': metadata, 'embeddings': embeddings}
        if not _valid_embedding_payload(payload, metadata):
            raise ValueError('Prototype encoder returned an invalid patient row count')
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
            temporary_path = handle.name
        try:
            save_pkl(temporary_path, payload)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
    for split, loader in datasets.items():
        loader.dataset.X = embeddings[split]['X']
        loader.dataset.y = embeddings[split]['y']
    return datasets, str(path)
