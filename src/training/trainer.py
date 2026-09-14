import json
import os
from os.path import join as j_
import pdb
from collections.abc import Mapping
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from mil_models.tokenizer import PrototypeTokenizer
from mil_models import create_multimodal_survival_model, prepare_emb
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.checkpoint import CheckpointManager
from utils.utils import (AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROTO_MODELS = ['PANTHER', 'OT', 'H2T', 'ProtoCount']


def _bag_size(data):
    """Return a comparable patch/slide count for tensor or structured inputs."""
    if torch.is_tensor(data):
        if data.dim() >= 3:
            return int(data.shape[-2])
        if data.dim() == 2:
            return int(data.shape[0])
        return 1
    if isinstance(data, dict):
        slides = data.get('slides')
        if isinstance(slides, (list, tuple)):
            return int(sum(_bag_size(slide) for slide in slides))
        return _bag_size(slides) if slides is not None else 1
    if isinstance(data, (list, tuple)):
        return int(sum(_bag_size(value) for value in data))
    return 1


def _forward_survival(model, data, omics, batch, **kwargs):
    return model(data, omics, batch=batch, **kwargs)


def _prepare_survival_batch(batch):
    """Move one physical patient batch to the trainer device."""
    if not isinstance(batch, dict):
        raise TypeError(f'Expected a survival batch mapping, got {type(batch)!r}')
    batch = safe_list_to(batch, device)
    required = {'img', 'censorship', 'survival_time'}
    missing = sorted(required.difference(batch))
    if missing:
        raise KeyError(f'Survival batch is missing required keys: {missing}')
    return {
        'batch': batch,
        'data': batch['img'],
        'label': batch.get('label'),
        'discrete_label': batch.get('discrete_label'),
        'event_time': batch['survival_time'],
        'censorship': batch['censorship'],
        'attn_mask': batch.get('attn_mask'),
        'omics': batch.get('omics'),
    }


def _forward_prepared_survival(model, prepared, loss_fn, **kwargs):
    return _forward_survival(
        model,
        prepared['data'],
        prepared['omics'],
        prepared['batch'],
        attn_mask=prepared['attn_mask'],
        label=prepared['label'],
        discrete_label=prepared['discrete_label'],
        censorship=prepared['censorship'],
        survival_time=prepared['event_time'],
        loss_fn=loss_fn,
        **kwargs,
    )


def _capture_rng_state():
    state = {'cpu': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    torch.set_rng_state(state['cpu'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def _deferred_cox_backward(
    model,
    patient_batches,
    loss_fn,
    *,
    accum_steps,
    expected_riskset_size=None,
):
    """Backpropagate one Cox risk set using two memory-bounded passes.

    The first, gradient-free pass obtains every patient's log risk under a
    common logical risk set. The second pass restores each patient's RNG state
    and applies the Cox vector-Jacobian product to a replayed forward graph.
    Auxiliary losses are disabled in pass one and averaged exactly once over
    the replayed patients.
    """
    if not isinstance(loss_fn, CoxLoss):
        raise TypeError('Deferred risk-set training requires CoxLoss')
    if not isinstance(patient_batches, (list, tuple)):
        raise TypeError('cox_patient_batches must be a list or tuple')
    if len(patient_batches) < 2:
        raise ValueError('A Cox risk set must contain at least two patients')
    if expected_riskset_size is not None and expected_riskset_size < 2:
        raise ValueError('cox_riskset_size must be at least 2')

    rng_states = []
    detached_logits = []
    risk_scores = []
    event_time_values = []
    censorship_values = []
    bag_sizes = []

    with torch.no_grad():
        for patient_batch in patient_batches:
            # Keep the logical risk set on CPU and materialise only one
            # patient's structured WSI/pathway payload on the accelerator at
            # a time.
            bag_sizes.append(_bag_size(patient_batch['img']))
            prepared = _prepare_survival_batch(patient_batch)
            if prepared['event_time'].numel() != 1 or prepared['censorship'].numel() != 1:
                raise ValueError(
                    'Each deferred Cox physical batch must contain one patient'
                )
            rng_states.append(_capture_rng_state())
            out, _ = _forward_prepared_survival(
                model,
                prepared,
                loss_fn,
                defer_survival_loss=True,
                include_auxiliary_loss=False,
            )
            logit = out['logits']
            if logit.numel() != 1:
                raise ValueError(
                    'A Cox head must emit one log-risk scalar per patient'
                )
            detached_logits.append(logit.detach().reshape(1, 1))
            risk_scores.append(out['risk'].detach().reshape(1, 1))
            event_time_values.append(prepared['event_time'].detach().reshape(1, 1))
            censorship_values.append(prepared['censorship'].detach().reshape(1, 1))
            del out, prepared
    final_rng_state = _capture_rng_state()

    logical_logits = torch.cat(detached_logits, dim=0).requires_grad_(True)
    event_times = torch.cat(event_time_values, dim=0)
    censorships = torch.cat(censorship_values, dim=0)
    surv_loss = loss_fn(
        logits=logical_logits,
        times=event_times,
        censorships=censorships,
    )['loss']
    logit_gradients = torch.autograd.grad(
        surv_loss / float(accum_steps), logical_logits
    )[0].detach()

    auxiliary_sum = 0.0
    auxiliary_logs = {}
    try:
        for patient_idx, (patient_batch, rng_state) in enumerate(
            zip(patient_batches, rng_states)
        ):
            _restore_rng_state(rng_state)
            prepared = _prepare_survival_batch(patient_batch)
            out, replay_logs = _forward_prepared_survival(
                model,
                prepared,
                loss_fn,
                defer_survival_loss=True,
                include_auxiliary_loss=True,
            )
            replay_logit = out['logits'].reshape(1, 1)
            if replay_logit.shape != detached_logits[patient_idx].shape:
                raise RuntimeError('Cox replay changed the log-risk shape')
            if not torch.allclose(
                replay_logit.detach(),
                detached_logits[patient_idx],
                rtol=1e-5,
                atol=1e-6,
            ):
                max_delta = float(
                    (replay_logit.detach() - detached_logits[patient_idx])
                    .abs().max().item()
                )
                raise RuntimeError(
                    'Cox RNG replay did not reproduce the first-pass '
                    f'log risk (max absolute difference {max_delta:.3e})'
                )

            replay_objective = (
                replay_logit * logit_gradients[patient_idx:patient_idx + 1]
            ).sum()
            auxiliary_loss = out.get('auxiliary_loss')
            if auxiliary_loss is not None:
                replay_objective = replay_objective + (
                    auxiliary_loss
                    / float(len(patient_batches) * accum_steps)
                )
                auxiliary_sum += float(auxiliary_loss.detach().item())
            replay_objective.backward()

            for key, value in replay_logs.items():
                if key in {'loss', 'surv_loss'}:
                    continue
                auxiliary_logs[key] = auxiliary_logs.get(key, 0.0) + float(value)
            del replay_objective, out, prepared
    finally:
        # Replaying must not advance global randomness twice per optimizer
        # update. Preserve the stream position reached by the first pass.
        _restore_rng_state(final_rng_state)

    patient_count = len(patient_batches)
    auxiliary_mean = auxiliary_sum / float(patient_count)
    log_dict = {
        'surv_loss': float(surv_loss.detach().item()),
        'loss': float(surv_loss.detach().item()) + auxiliary_mean,
    }
    log_dict.update({
        key: value / float(patient_count)
        for key, value in auxiliary_logs.items()
    })
    return {
        'risk': torch.cat(risk_scores, dim=0),
        'censorship': censorships,
        'event_time': event_times,
        'log_dict': log_dict,
        'bag_size': sum(bag_sizes) / float(patient_count),
        'patient_count': patient_count,
    }


def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    early_stopping_enabled = bool(getattr(args, 'early_stopping', False))
    if (early_stopping_enabled or getattr(args, 'checkpoint_selection', 'last') == 'best') and (
        'val' not in datasets or datasets['val'] is None
    ):
        raise ValueError(
            'Early stopping or best selection requires an independent validation split. '
            'Include val in --split_names and provide val.csv; do not use the '
            'test split for model selection.'
        )
    if int(args.max_epochs) <= 0:
        raise ValueError('max_epochs must be a positive integer')

    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    if isinstance(loss_fn, CoxLoss):
        cox_batch_size = getattr(datasets['train'], 'batch_size', None)
        survival_model = str(getattr(args, 'survival_model', 'legacy')).lower()
        if (
            not getattr(datasets['train'], 'deferred_cox', False)
            and cox_batch_size is not None
            and cox_batch_size < 2
        ):
            raise ValueError(
                'Cox training requires batch_size > 1 to form a risk set'
            )

    args.feat_dim = args.in_dim # Patch feature dimension
    print('\nInit Model...', end=' ')

    # If prototype-based models, need to create slide-level embeddings
    if args.model_histo_type in PROTO_MODELS:
        datasets, _ = prepare_emb(datasets, args, mode='survival')
        new_in_dim = None
        for k, loader in datasets.items():
            assert loader.dataset.X is not None
            new_in_dim_curr = loader.dataset.X.shape[-1]
            if loader.dataset.X.shape[0] != len(loader.dataset):
                raise ValueError(
                    f'Prototype cache split {k!r} has {loader.dataset.X.shape[0]} '
                    f'patients, expected {len(loader.dataset)}'
                )
            if new_in_dim is None:
                new_in_dim = new_in_dim_curr
            else:
                assert new_in_dim == new_in_dim_curr

            # The original embedding is 1-D (long) feature vector
            # Reshape it to (n_proto, -1)
            tokenizer = PrototypeTokenizer(args.model_histo_type, args.out_type, args.n_proto)
            prob, mean, cov = tokenizer(loader.dataset.X)
            parts = [torch.as_tensor(prob).unsqueeze(dim=-1), torch.as_tensor(mean)]
            # Released DIMAF uses occupancy + mean; MMP also uses covariance.
            if args.survival_model != 'dimaf':
                parts.append(torch.as_tensor(cov))
            loader.dataset.X = torch.cat(parts, dim=-1)
        args.in_dim = datasets['train'].dataset.X.shape[-1]
    else:
        print(f"{args.model_histo_type} doesn't construct unsupervised slide-level embeddings!")

    ## Set the dimensionality for different inputs
    train_dataset = datasets['train'].dataset
    if hasattr(train_dataset, 'omics_data') and train_dataset.omics_data is not None:
        args.omic_dim = train_dataset.omics_data.shape[1]
    else:
        args.omic_dim = 0
    if hasattr(train_dataset, 'rna_gene_seq'):
        args.rna_gene_seq = tuple(train_dataset.rna_gene_seq)
    if hasattr(train_dataset, 'st_gene_seq'):
        args.st_gene_seq = tuple(train_dataset.st_gene_seq)
    if getattr(train_dataset, 'pathway_names', None) is not None:
        args.pathway_names = tuple(train_dataset.pathway_names)

    if getattr(train_dataset, 'omic_sizes', None) is not None:
        omic_sizes = train_dataset.omic_sizes
    else:
        omic_sizes = []

    model = create_multimodal_survival_model(args, omic_sizes=omic_sizes)
    model.to(device)

    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    checkpoints = CheckpointManager(
        args.results_dir, selection=args.checkpoint_selection, metric=args.checkpoint_metric,
        early_stopping=early_stopping_enabled, es_metric=args.es_metric,
        patience=args.es_patience, min_epochs=args.es_min_epochs, config=vars(args),
    )
    last_epoch = None
    with open(j_(args.results_dir, 'history.jsonl'), 'w', encoding='utf-8') as history:
        for epoch in range(args.max_epochs):
            last_epoch = epoch
            train_results = train_loop_survival(
                model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                print_every=args.print_every, accum_steps=args.accum_steps,
                deferred_cox=bool(getattr(datasets['train'], 'deferred_cox', False)),
                cox_riskset_size=args.batch_size,
            )
            record = {'epoch': epoch, 'train': train_results}
            stop = False
            if 'val' in datasets:
                val_results, _ = validate_survival(
                    model, datasets['val'], loss_fn, print_every=args.print_every,
                    verbose=True, require_observed_event=isinstance(loss_fn, CoxLoss),
                )
                record['val'] = val_results
                stop = checkpoints.step(epoch, model, val_results)
            history.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            history.flush()
            if stop:
                break
    checkpoint_record = checkpoints.finish(model, last_epoch)
    print('Checkpoint selection:', json.dumps(checkpoint_record, sort_keys=True))

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        if k == 'train':
            continue
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        # Attention dumps are supported for prototype-based MMP co-attention.
        # SurvPath patch bags vary in length, so their attention tensors cannot be stacked across cases.
        return_attn = False
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

    return results, dumps

## SURVIVAL
def train_loop_survival(
    model,
    loader,
    optimizer,
    lr_scheduler,
    loss_fn=None,
    print_every=50,
    accum_steps=32,
    *,
    deferred_cox=False,
    cox_riskset_size=64,
):
    if accum_steps < 1:
        raise ValueError('accum_steps must be a positive integer')
    if deferred_cox and not isinstance(loss_fn, CoxLoss):
        raise ValueError('deferred_cox can only be enabled with CoxLoss')
    if deferred_cox and cox_riskset_size < 2:
        raise ValueError('cox_riskset_size must be at least 2')
    model.train()
    optimizer.zero_grad()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    pending_backward_steps = 0

    for batch_idx, batch in enumerate(loader):
        is_deferred_batch = (
            isinstance(batch, dict) and 'cox_patient_batches' in batch
        )
        if is_deferred_batch:
            if not deferred_cox:
                raise ValueError(
                    'Received cox_patient_batches without deferred_cox enabled'
                )
            deferred_result = _deferred_cox_backward(
                model,
                batch['cox_patient_batches'],
                loss_fn,
                accum_steps=accum_steps,
                expected_riskset_size=cox_riskset_size,
            )
            risk = deferred_result['risk']
            censorship = deferred_result['censorship']
            event_time = deferred_result['event_time']
            log_dict = deferred_result['log_dict']
            batch_size = deferred_result['patient_count']
            bag_size = deferred_result['bag_size']
        else:
            if deferred_cox:
                raise ValueError(
                    'Cox loader must yield cox_patient_batches risk sets'
                )
            prepared = _prepare_survival_batch(batch)
            out, log_dict = _forward_prepared_survival(
                model, prepared, loss_fn
            )
            if out.get('loss') is None:
                continue

            (out['loss'] / float(accum_steps)).backward()
            risk = out['risk']
            censorship = prepared['censorship']
            event_time = prepared['event_time']
            batch_size = int(censorship.numel())
            bag_size = _bag_size(prepared['data'])

        pending_backward_steps += 1
        if pending_backward_steps == accum_steps:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pending_backward_steps = 0

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(risk.detach().cpu().reshape(-1, 1).numpy())
        all_censorships.append(censorship.detach().cpu().reshape(-1, 1).numpy())
        all_event_times.append(event_time.detach().cpu().reshape(-1, 1).numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=batch_size)

        bag_size_meter.update(bag_size, n=batch_size)

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    if pending_backward_steps:
        # Losses above use the configured accumulation denominator. Restore
        # the correct mean when the final accumulation window is shorter.
        gradient_scale = float(accum_steps) / float(pending_backward_steps)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # End of epoch survival-specific metrics to calculate / log
    if not all_risk_scores:
        raise RuntimeError('No trainable survival batches were produced')
    all_risk_scores = np.concatenate(all_risk_scores).reshape(-1)
    all_censorships = np.concatenate(all_censorships).reshape(-1)
    all_event_times = np.concatenate(all_event_times).reshape(-1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,
                      require_observed_event=False):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_log_risks = []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []

    for batch_idx, outer_batch in enumerate(loader):
        if isinstance(outer_batch, dict) and 'cox_patient_batches' in outer_batch:
            physical_batches = outer_batch['cox_patient_batches']
        else:
            physical_batches = [outer_batch]

        for batch in physical_batches:
            prepared = _prepare_survival_batch(batch)
            out, log_dict = _forward_prepared_survival(
                model, prepared, loss_fn, return_attn=return_attn
            )
            censorship = prepared['censorship']
            event_time = prepared['event_time']
            batch_size = int(censorship.numel())
            if return_attn:
                all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
                all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
                all_path_attn.append(out['path_attn'].detach().cpu().numpy())
            # End of iteration survival-specific metrics to calculate / log
            bag_size_meter.update(_bag_size(prepared['data']), n=batch_size)
            for key, val in log_dict.items():
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(val, n=batch_size)
            all_risk_scores.append(out['risk'].cpu().reshape(-1, 1).numpy())
            all_censorships.append(censorship.cpu().reshape(-1, 1).numpy())
            if isinstance(loss_fn, CoxLoss):
                all_log_risks.append(out['logits'].cpu().reshape(-1, 1).numpy())
            all_event_times.append(event_time.cpu().reshape(-1, 1).numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    

    # End of epoch survival-specific metrics to calculate / log
    if not all_risk_scores:
        raise RuntimeError('Validation loader produced no patients')
    all_risk_scores = np.concatenate(all_risk_scores).reshape(-1)
    all_censorships = np.concatenate(all_censorships).reshape(-1)
    all_event_times = np.concatenate(all_event_times).reshape(-1)
    if (
        require_observed_event
        and isinstance(loss_fn, CoxLoss)
        and not np.any(all_censorships == 0)
    ):
        raise ValueError(
            'Cannot use Cox validation loss for early stopping because the '
            'validation split contains no observed events (all patients are '
            'censored). Choose a validation fold with at least one event.'
        )
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)
        else:
            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        cohort_log_risks = torch.as_tensor(np.concatenate(all_log_risks))
        if cohort_log_risks.dim() == 1:
            cohort_log_risks = cohort_log_risks.unsqueeze(1)
        surv_loss_dict = loss_fn(
            logits=cohort_log_risks,
            times=torch.as_tensor(all_event_times).unsqueeze(1),
            censorships=torch.as_tensor(all_censorships).unsqueeze(1),
        )
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn
    return results, dumps
