from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence, Dict, Optional
import torch

from federated.server.strategy.base import BaseStrategy
from federated.server.utils import assert_same_parameter_keys
from federated.types.fl_types import StateDict, ClientUpdate


@dataclass
class ScaffoldConfig:
    pass


class ScaffoldStrategy(BaseStrategy):
    def __init__(self, cfg: Optional[ScaffoldConfig] = None):
        self.cfg = cfg or ScaffoldConfig()
        self._c: Optional[Dict[str, torch.Tensor]] = None  # CPU???

    def reset(self) -> None:
        self._c = None

    def _ensure_c(self, model_state: Dict[str, torch.Tensor]) -> None:
        if self._c is None:
            self._c = {k: torch.zeros_like(v.detach().cpu()) for k, v in model_state.items()}

    def client_payload(self) -> Dict[str, Any]:
        if self._c is None:
            raise RuntimeError("SCAFFOLD- server control variate not initialized yet.")
        return {"scaffold_c": self._c}

    def aggregate(
        self,
        server_state: StateDict,  # not used
        client_updates: Sequence[ClientUpdate],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        _ = server_state
        if not client_updates:
            raise ValueError("SCAFFOLD- client_updates empty!")

        # FedAvg on weights 
        total_n = 0
        for u in client_updates:
            if u.n_train is None:
                raise ValueError("SCAFFOLD- requires n_train")
            n_i = int(u.n_train)
            if n_i <= 0:
                raise ValueError("SCAFFOLD- invalid n_train")
            total_n += n_i

        ref_sd = client_updates[0].state_dict
        new_state_cpu = {k: torch.zeros_like(v.detach().cpu()) for k, v in ref_sd.items()}

        for u in client_updates:
            w = int(u.n_train) / float(total_n)
            sd_i = u.state_dict
            assert_same_parameter_keys(reference=new_state_cpu, candidate=sd_i, who="SCAFFOLD")

            for k, v in sd_i.items():
                new_state_cpu[k] += v.detach().cpu() * w

        # Update c with mean(delta_ci)
        # _c should already be initialized with BaseServer.client_payload() using named_parameters keys
        if self._c is None:
            raise RuntimeError("SCAFFOLD- server control variate not initialized (call server.client_payload() first).")
        assert self._c is not None

        mean_delta = {k: torch.zeros_like(v) for k, v in self._c.items()}

        for u in client_updates:
            d = u.meta.get("scaffold_delta_ci", None)
            if d is None:
                raise ValueError("SCAFFOLD- missing meta['scaffold_delta_ci'] from a client update.")
            assert_same_parameter_keys(reference=mean_delta, candidate=d, who="SCAFFOLD(delta_ci)")
            for k, v in d.items():
                mean_delta[k] += v.detach().cpu()

        m = float(len(client_updates))
        for k in self._c.keys():
            self._c[k] = self._c[k] + mean_delta[k] / m

        return {k: v.to(device) for k, v in new_state_cpu.items()}
