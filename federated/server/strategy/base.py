from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence
from federated.types.fl_types import StateDict, ClientUpdate
import torch


class BaseStrategy(ABC):
    """
    Base class for server aggregation strategies.
    Child must implement aggregate.
    """

    @abstractmethod
    def aggregate(
        self,
        server_state: StateDict,
        client_updates: Sequence[ClientUpdate],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Compute and return the new global model state_dict.
        Return a plain dict[str, Tensor] for model.load_state_dict(...).
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional: clear any internal state (momentum buffers, Adam moments,...)
        between runs/experiments.
        """
        pass

