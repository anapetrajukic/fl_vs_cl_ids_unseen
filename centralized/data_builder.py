from typing import Dict, List
import numpy as np
from sklearn.preprocessing import RobustScaler

from data.splits_loader import load_split_pickles


def _transform_clip(scaler: RobustScaler, X: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    return np.clip(Xs, clip_min, clip_max).astype(np.float32)


def build_central_data_with_sklearn_robust_scaler(
    *,
    seen_datasets: List[str],
    unseen_dataset: str,
    splits_dir: str,
    clip_min: float,
    clip_max: float,
) -> tuple[
    np.ndarray,                               # pooled benign train (scaled)
    Dict[str, tuple[np.ndarray, np.ndarray]], # seen_val: ds -> (X_val_scaled, y_val)
    Dict[str, tuple[np.ndarray, np.ndarray]], # seen_test: ds -> (X_test_scaled, y_test)
    tuple[np.ndarray, np.ndarray],            # unseen_val: (X_val_scaled, y_val)
    tuple[np.ndarray, np.ndarray],            # unseen_test: (X_test_scaled, y_test)
    RobustScaler,
]:
    seen_splits = {ds: load_split_pickles(ds, splits_dir) for ds in seen_datasets}
    unseen_sp = load_split_pickles(unseen_dataset, splits_dir) 

    # pool benign TRAIN only
    benign_list = []
    for ds in seen_datasets:
        sp = seen_splits[ds]
        X_b = sp.X_train[sp.y_train == 0]
        if len(X_b) == 0:
            raise RuntimeError(f"{ds}: no benign samples in TRAIN.")
        benign_list.append(X_b)

    X_train_b = np.concatenate(benign_list, axis=0)

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaler.fit(X_train_b)  # fit ONLY on pooled benign train (seen only)

    X_train_b_s = _transform_clip(scaler, X_train_b, clip_min, clip_max)

    seen_val: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seen_test: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ds in seen_datasets:
        sp = seen_splits[ds]
        seen_val[ds] = (_transform_clip(scaler, sp.X_val, clip_min, clip_max), sp.y_val)
        seen_test[ds] = (_transform_clip(scaler, sp.X_test, clip_min, clip_max), sp.y_test)

    unseen_val = (_transform_clip(scaler, unseen_sp.X_val, clip_min, clip_max), unseen_sp.y_val)
    unseen_test = (_transform_clip(scaler, unseen_sp.X_test, clip_min, clip_max), unseen_sp.y_test)

    return (
        X_train_b_s,
        seen_val,
        seen_test,
        unseen_val,
        unseen_test,
        scaler,
    )