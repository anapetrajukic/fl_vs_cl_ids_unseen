from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import random
import numpy as np
import optuna
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class HPOConfig:
    n_trials: int = 80
    timeout_sec: Optional[int] = None
    direction: str = "maximize"
    objective_mode: str = "mean"   # "mean" or "min"
    pruner: str = "median"         # "median" or "none"
    seed: int = 42
    log_all_tests_each_trial: bool = True
    log_every_k_trials: int = 1
    save_csv_each_trial: bool = True


def objective_reduce(vals: List[float], mode: str) -> float:
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "min":
        return float(np.min(vals))
    raise ValueError("objective_mode must be 'mean' or 'min'")


def make_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError("pruner must be 'median' or 'none'")
