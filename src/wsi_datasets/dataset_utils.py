import numpy as np
import torch

def apply_sampling(target_bag_size, all_features, all_coords):
    attn_mask = None
    if target_bag_size > 0:
        bag_size = all_features.size(0)
        attn_mask = torch.ones(bag_size)
        if bag_size < target_bag_size:
            sampled_features = torch.cat([all_features, torch.zeros(
                (target_bag_size - bag_size, all_features.shape[1]))], dim=0)
            attn_mask = torch.cat(
                [attn_mask, torch.zeros((target_bag_size - bag_size))])
            if len(all_coords) > 0:
                all_coords = np.concatenate(
                    [all_coords, np.zeros((target_bag_size - bag_size, 2))], axis=0)
        else:
            sampled_patch_ids = np.random.choice(
                np.arange(bag_size), target_bag_size, replace=False)
            sampled_features = all_features[sampled_patch_ids, :]
            attn_mask = attn_mask[:target_bag_size]
            if len(all_coords) > 0:
                all_coords = all_coords[sampled_patch_ids, :]
        all_features = sampled_features
    return all_features, all_coords, attn_mask


def apply_aligned_sampling(
    target_bag_size,
    *arrays,
    rng=None,
    indices=None,
):
    """Sample aligned patch-level arrays with one shared index set.

    Unlike :func:`apply_sampling`, this helper never pads short bags. It is
    used by STARPath, where patch features and coordinates must remain in exact
    one-to-one correspondence. ``-1`` retains the full bag; positive
    values cap it. A caller may provide either a local NumPy random generator
    or an explicit index vector, which makes evaluation sampling reproducible
    without mutating NumPy's process-global RNG state.
    """
    if not arrays:
        raise ValueError("apply_aligned_sampling requires at least one array")
    if isinstance(target_bag_size, bool):
        raise TypeError("target_bag_size must be an integer, not bool")
    try:
        target_bag_size = int(target_bag_size)
    except (TypeError, ValueError) as exc:
        raise TypeError("target_bag_size must be an integer") from exc
    if target_bag_size == 0 or target_bag_size < -1:
        raise ValueError("target_bag_size must be -1 or a positive integer")
    if rng is not None and indices is not None:
        raise ValueError("Pass rng or indices, not both")

    lengths = [len(array) for array in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Aligned arrays have different lengths: {lengths}")
    size = lengths[0]
    if size == 0:
        raise ValueError("Cannot sample an empty patch bag")
    if indices is None and (target_bag_size == -1 or size <= target_bag_size):
        return arrays
    if indices is None:
        chooser = np.random if rng is None else rng
        indices = np.sort(chooser.choice(size, target_bag_size, replace=False))
    else:
        indices = np.asarray(indices)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("indices must be a non-empty one-dimensional vector")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("indices must contain integers")
        indices = indices.astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("indices must not contain duplicates")
        if indices.min() < 0 or indices.max() >= size:
            raise IndexError(f"Sampling indices fall outside a bag of size {size}")
        if target_bag_size > 0 and len(indices) > target_bag_size:
            raise ValueError("indices exceed target_bag_size")

    sampled = []
    for array in arrays:
        if isinstance(array, torch.Tensor):
            sampled.append(array[torch.as_tensor(
                indices, dtype=torch.long, device=array.device
            )])
        else:
            sampled.append(np.asarray(array)[indices])
    return tuple(sampled)
