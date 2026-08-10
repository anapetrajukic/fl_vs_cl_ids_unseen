from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import copy

import torch
import torch.nn as nn

from federated.types.fl_types import ClientUpdate
from federated.server.strategy.base import BaseStrategy


@dataclass
class ServerConfig:
    rounds: int = 50
    eval_every: int = 1
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-4


class BaseServer:
    """
    early stopping based on an external metric thats passed in
    """

    def __init__(self, *, model: nn.Module, strategy: BaseStrategy, device: torch.device, cfg: ServerConfig):
        self.model = model.to(device)
        self.strategy = strategy
        self.device = device
        self.cfg = cfg

        self.best_metric: float = -float("inf")
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.no_improve: int = 0

    def client_payload(self) -> Dict[str, Any]:
        # If SCAFFOLD strategy-ensure c matches named_parameters() keys
        if hasattr(self.strategy, "_ensure_c"):
            params = {name: p.detach() for name, p in self.model.named_parameters()}
            self.strategy._ensure_c(params)  # type: ignore[attr-defined]

        if hasattr(self.strategy, "client_payload"):
            return self.strategy.client_payload()  
        return {}

    def aggregate(self, client_updates: Sequence[ClientUpdate]) -> None:
        if not client_updates:
            raise ValueError("Server: client_updates is empty.")

        server_state = self.model.state_dict()
        new_state = self.strategy.aggregate(
            server_state=server_state,
            client_updates=client_updates,
            device=self.device,
        )
        self.model.load_state_dict(new_state, strict=True)

    def maybe_early_stop(self, metric: float) -> bool:
        if metric > self.best_metric + float(self.cfg.early_stop_min_delta):
            self.best_metric = float(metric)
            self.best_state = copy.deepcopy(self.model.state_dict())
            self.no_improve = 0
            return False

        self.no_improve += 1
        return self.no_improve >= int(self.cfg.early_stop_patience)

    def restore_best(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state, strict=True)

    def reset(self) -> None:
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()  # type: ignore[attr-defined]
        self.best_metric = -float("inf")
        self.best_state = None
        self.no_improve = 0