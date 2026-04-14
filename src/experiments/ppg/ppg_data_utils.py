import functools

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.sampling_utils import *



def load_data(
        file_path : str,
        clean : bool,
        seminoisy : bool,
        noisy : bool,
        dataset : str,
        return_labels : bool = True
    ):
    """
    Load PPG data from Drive. 
        - quality: high - [0,0,1] med - [0,1,0] low - [1,0,0]
        - afib: pos - [0,1]

    Arguments
    ---------
        - clean, seminoisy, noisy: bools indicating whether to include data of each quality level
        - dataset: 'train' or 'test'
        - return_labels: whether to return labels (if False, only returns signals)

    Returns
    -------
        - X: (N, L) ndarray of PPG signals
        - y: (N,) ndarray of binary labels indicating presence of afib if return_labels = True
    """

    with np.load(file_path + f'{dataset}.npz') as data:
        ppg_signal = data['signal']
        ppg_qual = data['qa_label']
        rhythm = data['rhythm']

    qual_mask = np.zeros(ppg_qual.shape[0], dtype = bool)
    if clean:
        qual_mask = (qual_mask | np.all(ppg_qual == np.array([0, 0, 1]), axis = 1))
    if seminoisy:
        qual_mask = (qual_mask | np.all(ppg_qual == np.array([0, 1, 0]), axis = 1))
    if noisy:
        qual_mask = (qual_mask | np.all(ppg_qual == np.array([1, 0, 0]), axis = 1))

    X, y = ppg_signal[qual_mask], rhythm[qual_mask]

    afib_label = np.array([0, 1])
    y = np.all(y == afib_label, axis = 1)

    if return_labels:
        return X, y

    return X

def make_dataloaders(
        X : np.ndarray,
        y : np.ndarray = None,
        batch_size : int = 256,
        val_ratio : float = 0.05,
        seed : int = 42,
        num_workers : int = 4,
        collate_fn_kwargs : dict = None
    ) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders from PPG signals and labels. If labels exist, they can be passed for use in stratified splitting.

    IMPORTANT: TimeSeriesDataset class takes signals as list of tensors of shape [T_i, F]

    Arguments
    ---------
        - X: (N, L) ndarray of PPG signals
        - y: (N,) ndarray of binary labels
        - batch_size: batch size for dataloaders
        - val_ratio: proportion of data to use for validation
        - seed: random seed for reproducibility
        - num_workers: number of worker processes for data loading
        - collate_fn_kwargs: dict of kwargs to pass to contrastive_collate function (if None, defaults will be used)

    Returns
    -------
        - dl_train: DataLoader for training data
        - dl_val: DataLoader for validation data
    """

    if y is not None:
        sss = StratifiedShuffleSplit(n_splits = 1, test_size = val_ratio, random_state = seed)
        train_idx, val_idx = next(sss.split(X, y))
    else:
        n = X.shape[0]
        indices = np.random.default_rng(seed).permutation(n)
        n_val = int(n * val_ratio)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

    sig_train = [torch.from_numpy(x).float() for x in X[train_idx]]
    sig_val = [torch.from_numpy(x).float() for x in X[val_idx]]
    ds_train = TimeSeriesDataset(sig_train)
    ds_val = TimeSeriesDataset(sig_val)

    if collate_fn_kwargs:
        min_len = collate_fn_kwargs.get('min_len', X.shape[1])
        max_len = collate_fn_kwargs.get('max_len', X.shape[1])
        num_neg = collate_fn_kwargs.get('num_neg', 5)
        min_overlap = collate_fn_kwargs.get('min_overlap', 0.3)
        min_var = collate_fn_kwargs.get('min_var', 0.5)
        max_var = collate_fn_kwargs.get('max_var', 2)

        collate_fn = functools.partial(
            contrastive_collate,
            min_len = min_len,
            max_len = max_len,
            num_neg = num_neg,
            min_overlap = min_overlap,
            min_var = min_var,
            max_var = max_var
        )

    dl = DataLoader(
        ds_train,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = num_workers,
        pin_memory = True,
        drop_last = True
    )

    if val_ratio:
        dl_val = DataLoader(
            ds_val,
            batch_size = batch_size,
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = num_workers,
            pin_memory = True,
            drop_last = True
        )

        return dl, dl_val

    return dl