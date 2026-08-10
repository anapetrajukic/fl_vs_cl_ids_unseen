from __future__ import annotations

# WATCH OUT!!! Aggregates everything that's in client_updates.state_dict! -> What about BatchNorm., DomainEmb

from typing import Sequence
import torch

from federated.server.strategy.base import BaseStrategy
from federated.server.utils import assert_same_parameter_keys
from federated.types.fl_types import StateDict, ClientUpdate


class FedAvgStrategy(BaseStrategy):
    """Standard Federated Averaging strategy"""
    def aggregate(
        self,
        server_state: StateDict,  # not used
        client_updates: Sequence[ClientUpdate],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        _ = server_state

        if not client_updates:
            raise ValueError("FedAvg: client_updates empty!")

        total_n = 0
        for u in client_updates:
            if u.n_train is None:
                raise ValueError("FedAvg: requires 'n_train' (number of training samples) from client!")
            n_i = int(u.n_train)
            if n_i <= 0:
                raise ValueError(f"FedAvg: Invalid n_train={n_i}. Expected a positive integer.")
            total_n += n_i

        
        ref_sd = client_updates[0].state_dict
        new_state: dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in ref_sd.items()}

        for u in client_updates:
            w = int(u.n_train) / float(total_n)
            sd_i = u.state_dict

            assert_same_parameter_keys(reference=new_state, candidate=sd_i, who="FedAvg")

            for k, v in sd_i.items():
                new_state[k] += v.detach().cpu() * w

        return {k: v.to(device) for k, v in new_state.items()}