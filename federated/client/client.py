from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import copy

import torch
import torch.nn as nn

from task.base import BaseTask
from federated.types.fl_types import ClientUpdate


@dataclass
class BaseClient:
    client_id: int
    device: torch.device
    task: BaseTask
    meta: Dict[str, Any] = field(default_factory=dict)

    # per-client state -used by SCAFFOLD
    scaffold_ci: Optional[Dict[str, torch.Tensor]] = None

    def fit(
        self,
        global_model: nn.Module,
        round_idx: int,
        *,
        server_payload: Optional[Dict[str, Any]] = None,
    ) -> ClientUpdate:
        """
        Arguments:
          global_model: global model from server
          round_idx: current FL round
          server_payload: optional dict passed from server to client
            - for SCAFFOLD, should include {"scaffold_c": Dict[str, Tensor]}
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        server_payload = server_payload or {}

        client_state = {
            "scaffold_ci": self.scaffold_ci,
        }

        train_meta = self.task.train(
            model=model,
            device=self.device,
            round_idx=round_idx,
            server_payload=server_payload,
            client_state=client_state,
        ) or {}

        if "scaffold_ci_new" in train_meta:
            self.scaffold_ci = train_meta.pop("scaffold_ci_new")

        return ClientUpdate(
            state_dict=self.task.get_weights(model),
            n_train=int(self.task.n_train),
            meta={**self.meta, **train_meta},
        )

    @torch.no_grad()
    def evaluate(self, model: nn.Module, split: str) -> Dict[str, float]:
        return self.task.evaluate(model=model, device=self.device, split=split)