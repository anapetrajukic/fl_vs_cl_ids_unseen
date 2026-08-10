from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from federated.server.strategy.base import BaseStrategy
from federated.types.fl_types import ClientUpdate, StateDict


@dataclass
class FedAdamConfig:
    server_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    tau: float = 1e-8
    use_bias_correction: bool = False


class FedAdamStrategy(BaseStrategy):
    """
    FedAvg-style aggregation + adam update on the server

    Clients can remain unchanged (they train locally and return full state_dicts)
    This strategy computes client deltas relative to current server_state, aggregates
    them (weighted by n_train), treats the aggregate as a pseudo-gradient, and applies
    Adam moments on the server.

    USED annotation:
      client delta Δ_i = w_i - w_t
      aggregated delta Δ_t = sum_i p_i Δ_i
      pseudo-gradient g_t = -Δ_t
      server update: w_{t+1} = w_t - lr * Adam(g_t)
    """

    def __init__(self, cfg: Optional[FedAdamConfig] = None):
        self.cfg = cfg or FedAdamConfig()
        self._m: Optional[Dict[str, torch.Tensor]] = None
        self._v: Optional[Dict[str, torch.Tensor]] = None
        self._t: int = 0

    def reset(self) -> None:
        self._m = None
        self._v = None
        self._t = 0

    # HELPER functions
    @staticmethod
    def _is_float_tensor(x: torch.Tensor) -> bool:
        return torch.is_tensor(x) and torch.is_floating_point(x)

    def _init_buffers(self, server_state: StateDict, device: torch.device) -> None:
        self._m = {}
        self._v = {}

        for k, v in server_state.items():
            if self._is_float_tensor(v):
                z = torch.zeros_like(v, device=device)
                self._m[k] = z.clone()
                self._v[k] = z.clone()

    @staticmethod
    def _validate_client_update(cu: ClientUpdate) -> None:
        if not hasattr(cu, "state_dict") or cu.state_dict is None:
            raise ValueError("FedAdamStrategy expects ClientUpdate.state_dict to be present.")
        if not hasattr(cu, "n_train"):
            raise ValueError("FedAdamStrategy expects ClientUpdate.n_train for weighting.")

    @staticmethod
    def _client_weight(cu: ClientUpdate) -> float:
        w = float(cu.n_train)
        if w < 0:
            raise ValueError("ClientUpdate.n_train must be >= 0.")
        return w

    def _compute_client_delta(
        self,
        *,
        server_state: StateDict,
        client_state: StateDict,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute delta_i = client_state - server_state 
        """
        delta: Dict[str, torch.Tensor] = {}

        for k, sv in server_state.items():
            if not self._is_float_tensor(sv):
                continue

            if k not in client_state:
                raise KeyError(f"Client state_dict missing key: {k}")

            cv = client_state[k]
            if not torch.is_tensor(cv):
                raise TypeError(f"Client state_dict[{k}] is not a Tensor.")

            delta[k] = cv.detach().to(device=device, dtype=sv.dtype) - sv.detach().to(device=device)

        return delta

    ##################################### Main...
    def aggregate(
        self,
        server_state: StateDict,
        client_updates: Sequence[ClientUpdate],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if not client_updates:
            raise ValueError("FedAdamStrategy: client_updates is empty.")

        for cu in client_updates:
            self._validate_client_update(cu)

        weights = [self._client_weight(cu) for cu in client_updates]
        total_w = float(sum(weights))
        if total_w <= 0.0:
            raise ValueError("FedAdamStrategy: total client weight must be > 0.")

        # initialize moments lazily (based on actual model state shapes/dtypes)
        if self._m is None or self._v is None:
            self._init_buffers(server_state=server_state, device=device)

        assert self._m is not None and self._v is not None  # for type checkers

        new_state: dict[str, torch.Tensor] = {
            k: v.detach().clone().to(device=device) if torch.is_tensor(v) else v
            for k, v in server_state.items()
        }

        # 1) FedAvg weighted aggregate of client deltas
        agg_delta: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, device=device)
            for k, v in server_state.items()
            if self._is_float_tensor(v)
        }

        for cu, w in zip(client_updates, weights):
            alpha = float(w) / total_w
            d_i = self._compute_client_delta(
                server_state=server_state,
                client_state=cu.state_dict,  # type: ignore[arg-type]
                device=device,
            )
            for k in agg_delta.keys():
                agg_delta[k].add_(d_i[k], alpha=alpha)

        # 2) Server Adam on pseudo-gradient g = -Δ
        self._t += 1
        b1 = float(self.cfg.beta1)
        b2 = float(self.cfg.beta2)
        lr = float(self.cfg.server_lr)
        tau = float(self.cfg.tau)

        if not (0.0 <= b1 < 1.0):
            raise ValueError(f"beta1 must be in [0,1), got {b1}")
        if not (0.0 <= b2 < 1.0):
            raise ValueError(f"beta2 must be in [0,1), got {b2}")
        if lr <= 0.0:
            raise ValueError(f"server_lr must be > 0, got {lr}")
        if tau < 0.0:
            raise ValueError(f"tau must be >= 0, got {tau}")

        for k, p in new_state.items():
            if not self._is_float_tensor(p):
                # Keep non-floating tensors unchanged 
                continue

            g = -agg_delta[k]  # pseudo-gradient

            # m_t = beta1 * m_{t-1} + (1-beta1) * g_t
            self._m[k].mul_(b1).add_(g, alpha=(1.0 - b1))

            # v_t = beta2 * v_{t-1} + (1-beta2) * (g_t^2)
            self._v[k].mul_(b2).addcmul_(g, g, value=(1.0 - b2))

            m_t = self._m[k]
            v_t = self._v[k]

            if self.cfg.use_bias_correction:
                m_t = m_t / (1.0 - (b1 ** self._t))
                v_t = v_t / (1.0 - (b2 ** self._t))

            denom = v_t.sqrt().add(tau)

            # w <- w - lr * m / (sqrt(v) + tau)
            p.addcdiv_(m_t, denom, value=-lr)

        return new_state