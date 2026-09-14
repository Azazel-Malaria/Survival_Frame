"""Batch samplers for memory-constrained patient-level survival training."""

from __future__ import annotations

from typing import Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler


class EventAwareRiskSetBatchSampler(Sampler[List[int]]):
    """Construct fixed-size logical Cox batches with an informative event.

    Censorship follows this repository's convention: ``0`` means an observed
    event and ``1`` means censored. Every yielded set contains an observed
    event plus at least one *other* patient still at risk at that event time.
    Risk sets form a disjoint partition, so every patient occurs exactly once
    per epoch. If the cohort has too few eligible events to seed all sets, the
    sampler fails explicitly instead of silently oversampling those events.
    """

    def __init__(
        self,
        dataset,
        *,
        riskset_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.riskset_size = int(riskset_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.riskset_size < 2:
            raise ValueError('riskset_size must be at least 2')
        self._size = int(len(dataset))
        if self._size < 2:
            raise ValueError('Cox risk-set sampling requires at least two patients')
        if self.drop_last:
            raise ValueError(
                'drop_last=True is incompatible with one-pass disjoint risk sets'
            )
        self._group_sizes = self._make_group_sizes(
            self._size, self.riskset_size
        )

        times = torch.as_tensor(dataset.survival_time_labels).detach().cpu().numpy()
        censorships = (
            torch.as_tensor(dataset.censorship_labels).detach().cpu().numpy()
        )
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        censorships = np.asarray(censorships).reshape(-1)
        if len(times) != self._size or len(censorships) != self._size:
            raise ValueError('Dataset survival labels do not match dataset length')
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError('Survival times must be finite and non-negative')
        if not np.isin(censorships, (0, 1)).all():
            raise ValueError('Censorship labels must contain only 0/1')
        self._times = times

        observed = np.flatnonzero(censorships == 0)
        eligible = []
        for event_index in observed:
            comparable = np.flatnonzero(times >= times[event_index])
            if np.any(comparable != event_index):
                eligible.append(int(event_index))
        self._eligible_events = np.asarray(eligible, dtype=np.int64)
        if len(self._eligible_events) == 0:
            raise ValueError(
                'No observed event has another patient in its Cox risk set'
            )
        if len(self._eligible_events) < len(self._group_sizes):
            raise ValueError(
                f'{len(self._eligible_events)} eligible events cannot seed '
                f'{len(self._group_sizes)} disjoint Cox risk sets; increase '
                'riskset_size'
            )

    @staticmethod
    def _make_group_sizes(size: int, riskset_size: int) -> List[int]:
        """Partition ``size`` without producing a singleton remainder."""
        if size <= riskset_size:
            return [size]
        full_groups, remainder = divmod(size, riskset_size)
        sizes = [riskset_size] * full_groups
        if remainder == 0:
            return sizes
        if remainder >= 2:
            return sizes + [remainder]
        if riskset_size > 2:
            sizes[-1] -= 1
            sizes.append(2)
        else:
            # A cap of two cannot absorb an odd cohort without a singleton;
            # make the final logical set three patients instead.
            sizes[-1] += 1
        if sum(sizes) != size or min(sizes) < 2:
            raise RuntimeError(f'Invalid Cox risk-set partition: {sizes}')
        return sizes

    def _match_comparators(self, anchors: np.ndarray):
        """Return a disjoint at-risk comparator for every proposed anchor."""
        all_indices = np.arange(self._size, dtype=np.int64)
        remaining = all_indices[~np.isin(all_indices, anchors)]
        count = len(anchors)
        if len(remaining) < count:
            return None

        # Retain the patients with the largest available follow-up times, then
        # pair both sides in time order. This is the most permissive matching
        # for a fixed anchor set and detects infeasible proposals reliably.
        ordered_anchors = anchors[np.argsort(
            self._times[anchors], kind='stable'
        )]
        ordered_remaining = remaining[np.argsort(
            self._times[remaining], kind='stable'
        )]
        comparators = ordered_remaining[-count:]
        if np.any(self._times[comparators] < self._times[ordered_anchors]):
            return None
        return list(zip(map(int, ordered_anchors), map(int, comparators)))

    def _seed_pairs(self, rng: np.random.Generator):
        """Choose feasible, non-overlapping event/comparator seed pairs."""
        count = len(self._group_sizes)
        if self.shuffle:
            # Prefer a varying feasible event subset. Falling back to the
            # earliest events maximises matching feasibility.
            for _ in range(max(16, len(self._eligible_events))):
                proposal = rng.permutation(self._eligible_events)[:count]
                pairs = self._match_comparators(proposal)
                if pairs is not None:
                    rng.shuffle(pairs)
                    return pairs

        event_order = self._eligible_events[np.argsort(
            self._times[self._eligible_events], kind='stable'
        )]
        pairs = self._match_comparators(event_order[:count])
        if pairs is None:
            raise ValueError(
                'Eligible events cannot be assigned distinct at-risk '
                'comparators; increase riskset_size'
            )
        return pairs

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible sampling stream for a particular epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._group_sizes)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        all_indices = np.arange(self._size, dtype=np.int64)
        pairs = self._seed_pairs(rng)
        seeded = np.asarray([idx for pair in pairs for idx in pair], dtype=np.int64)
        remainder = all_indices[~np.isin(all_indices, seeded)]
        if self.shuffle:
            remainder = rng.permutation(remainder)

        cursor = 0
        for group_size, pair in zip(self._group_sizes, pairs):
            needed = group_size - 2
            selected = [pair[0], pair[1]]
            selected.extend(map(int, remainder[cursor:cursor + needed]))
            cursor += needed
            if self.shuffle:
                rng.shuffle(selected)
            yield selected

        if cursor != len(remainder):
            raise RuntimeError('Cox risk-set partition did not consume every patient')

        # Repeated DataLoader iterations vary naturally; set_epoch remains
        # available for controlled restarts.
        if self.shuffle:
            self.epoch += 1


__all__ = ['EventAwareRiskSetBatchSampler']
