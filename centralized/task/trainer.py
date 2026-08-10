from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import copy
import numpy as np
import torch
import torch.nn as nn

from workshop_dg_cl_vs_fl.centralized.task.vae_task import VaeTask


@dataclass(frozen=True)
class TrainConfig:
    max_rounds: int = 50
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    restore_best: bool = True


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        task: VaeTask,
        device: torch.device,
        cfg: TrainConfig,
    ):
        self.model = model
        self.task = task
        self.device = device
        self.cfg = cfg

        self.best_metric = -float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.no_improve = 0

    def maybe_early_stop(self, metric: float) -> bool:
        if metric > self.best_metric + float(self.cfg.early_stop_min_delta):
            self.best_metric = float(metric)
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.no_improve = 0
            return False

        self.no_improve += 1
        return (
            int(self.cfg.early_stop_patience) > 0
            and self.no_improve >= int(self.cfg.early_stop_patience)
        )

    def restore_best(self) -> None:
        #print("Restoring best model...")
        if bool(self.cfg.restore_best) and self.best_state is not None:
            self.model.load_state_dict(self.best_state)
