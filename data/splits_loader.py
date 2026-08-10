from __future__ import annotations
from typing import Tuple
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DatasetSplits:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_split_pickles(ds_name: str, splits_dir: str) -> DatasetSplits:
    X_train = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_X_train.pkl"))
    y_train = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_y_train.pkl"))
    X_val = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_X_val.pkl"))
    y_val = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_y_val.pkl"))
    X_test = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_X_test.pkl"))
    y_test = pd.read_pickle(os.path.join(splits_dir, f"{ds_name}_y_test.pkl"))

    X_train = X_train.to_numpy(dtype=np.float32)
    X_val = X_val.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return DatasetSplits(
        name=ds_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
