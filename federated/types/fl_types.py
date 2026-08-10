from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TypeAlias
import torch


StateDict: TypeAlias = Mapping[str, torch.Tensor]  # model.state_dict() like mapping


# Client payload
@dataclass(frozen=True)
class ClientUpdate:
    """
    What a client sends to the server after local training.

    Required:
      - state_dict: model weights/buffers after local training

    Optional:
      - n_train: number of training samples
      - meta: strategy-specific extras (local_steps, tau, client_loss...)
    """
    state_dict: StateDict
    n_train: Optional[int] = None
    meta: dict[str, Any] = field(default_factory=dict)
