from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from typing import Any, Mapping, TypeAlias

StateDict: TypeAlias = Mapping[str, torch.Tensor] 

@dataclass(frozen=True)
class TaskOptimConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float | None = 2.0
    optimizer: str = "sgd"

    # SGD-only
    sgd_momentum: float = 0.0
    sgd_nesterov: bool = False


class BaseTask(ABC):
    def __init__(self, optim_cfg: TaskOptimConfig, n_train: int):
        self.optim_cfg = optim_cfg
        self.n_train = int(n_train)
        if self.n_train < 0:
            raise ValueError(f"n_train must be >= 0, got {self.n_train}")

    @abstractmethod
    def make_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        device: torch.device,
        round_idx: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        device: torch.device,
        split: str = "test",
    ) -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def get_weights(model: nn.Module) -> StateDict:
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    @staticmethod
    def set_weights(
        model: nn.Module,
        weights: StateDict,
        device: torch.device,
        strict: bool = True,
    ) -> None:
        weights = {k: v.to(device) for k, v in weights.items()}
        model.load_state_dict(weights, strict=strict)
