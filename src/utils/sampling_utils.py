import random
import torch
from torch.utils.data import Dataset, DataLoader



class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data. 
    
    Expects a list of tensors, each of shape [T_i, F]. 

    Returns a single time series tensor per item, without cropping.
    """

    def __init__(self, series_list):
        self.series_list = series_list  # list of tensors [T_i, F]

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, idx):
        return self.series_list[idx]

def sample_crop(x, L):
    """
    Return a contiguous subseries of length x from a time series tensor.

    Args:
        x: Time series tensor of shape [T, F]
        L: Desired crop length (1 <= L <= T)
    """

    T = x.size(0)
    start = random.randint(0, T - L)

    return x[start:start+L], start

def sample_overlapping_pair(x, L, min_overlap = 0.3):
    """
    Sample two length L contiguous crops from x with at least min_overlap fraction overlap.

    Args:
        x: Time series tensor of shape [T, F]
        L: Crop length
        min_overlap: Minimum required overlap as a fraction of L
    """

    T = x.size(0)
    a, s1 = sample_crop(x, L)
    min_ov = max(1, int(L * min_overlap))

    l = max(0, s1 - (L - min_ov))
    r = min(T - L, s1 + (L - min_ov))
    s2 = random.randint(l, r)
    p = x[s2:s2+L]

    return a, p

def sample_nonoverlap(x, L, forbidden_start):
    """
    Sample a length L contiguous crop from x that does not overlap the crop starting at forbidden_start

    Args:
        x: Time series tensor of shape [T, F].
        L: Crop length
        forbidden_start: Start index of the crop to avoid overlapping
    """

    T = x.size(0)
    valid = [s for s in range(T - L + 1) if s + L <= forbidden_start or s >= forbidden_start + L]
    if not valid: # fallback: farthest possible crop
        s = 0 if forbidden_start > (T - L) // 2 else T - L
    else:
        s = random.choice(valid)
    
    return x[s:s+L]

def jitter(x, var):
    """
    Add zero-mean Gaussian noise to x with variance var.

    Args:
        x: Time series tensor of shape [T, F].
        var: Noise variance.
    """

    std = var ** 0.5
    noise = torch.randn_like(x) * std
    
    return x + noise

def contrastive_collate(batch, min_len, max_len, num_neg = 4, min_var = 0, max_var = 0):
    """
    Build a batch of anchor, positive, mid, and per-sample negative crops using one shared crop length sampled for the batch.

    Args:
        batch: List of full time series tensors, each with shape [T_i, F]
        min_len: Minimum crop length allowed for the batch
        max_len: Maximum crop length allowed for the batch
        num_neg: Number of negatives to sample per anchor
        min_var: Minimum jitter variance allowed for the batch
        max_var: Maximum jitter variance allowed for the batch

    Returns:
        A dict containing anchor, positive, and mid tensors of shape [B, L, F], a negative tensor of shape [B, num_neg, L, F], the sampled batch length L, and the sampled jitter variance (if any).
    """

    lengths = [x.size(0) for x in batch]
    Lmax = min(max_len, max(lengths))
    Lmin = min_len
    
    if Lmax < Lmin:
        raise ValueError("No series in batch long enough for requested min_len.")
    if min_var < 0 or max_var < min_var:
        raise ValueError("Require 0 <= min_var <= max_var.")

    L = random.randint(Lmin, Lmax)
    jitter_var = random.uniform(min_var, max_var)

    batch = [x for x in batch if x.size(0) >= L]
    if len(batch) < 2:
        raise ValueError("Need at least 2 valid series for negatives.")

    anchors, positives, mids, negatives = [], [], [], []

    for i, x in enumerate(batch):
        anchor, anchor_start = sample_crop(x, L)
        anchor = jitter(anchor, jitter_var)
        _, positive = sample_overlapping_pair(x, L)
        positive = jitter(positive, jitter_var)
        mid = sample_nonoverlap(x, L, anchor_start)
        mid = jitter(mid, jitter_var)

        other_ids = [j for j in range(len(batch)) if j != i]
        negs = []
        for _ in range(num_neg):
            j = random.choice(other_ids)
            neg, _ = sample_crop(batch[j], L)
            neg = jitter(neg, jitter_var)
            negs.append(neg.transpose(0, 1))   # [F, L]

        anchors.append(anchor.transpose(0, 1))      # [F, L]
        positives.append(positive.transpose(0, 1))  # [F, L]
        mids.append(mid.transpose(0, 1))            # [F, L]
        negatives.append(torch.stack(negs))         # [K, F, L]

    return {
        "anchor": torch.stack(anchors),      # [B, F, L]
        "positive": torch.stack(positives),  # [B, F, L]
        "mid": torch.stack(mids),            # [B, F, L]
        "negative": torch.stack(negatives),  # [B, K, F, L]
        "length": L,
    }

"""
Example usage:

import functools
from torch.utils.data import DataLoader

# series_list: list of tensors, each shaped [T_i, F]
dataset = TimeSeriesDataset(series_list)

collate_fn = functools.partial(
    contrastive_collate,
    min_len = 128,
    max_len = 256,
    num_neg = 4
)

loader = DataLoader(
    dataset,
    batch_size = 32,
    shuffle = True,
    collate_fn = collate_fn,
    drop_last = True
)

for batch in loader:
    anchor = batch["anchor"]      # [B, L, F]
    positive = batch["positive"]  # [B, L, F]
    mid = batch["mid"]            # [B, L, F]
    negative = batch["negative"]  # [B, K, L, F]
    L = batch["length"]

    # pass to model / loss
"""