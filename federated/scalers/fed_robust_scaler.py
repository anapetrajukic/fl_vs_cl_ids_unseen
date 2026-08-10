from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class ClientRobustStats:
    n: int
    q25: np.ndarray
    q50: np.ndarray
    q75: np.ndarray


class FederatedRobustScaler:
    """
    Privacy-preserving-ish robust scaler:
    each client computes per-feature quantiles on benign TRAIN only,
    server aggregates quantiles (weighted average by n), 
    uses center = q50, scale = (q75 - q25), clip after scaling
    """

    def __init__(
        self,
        eps: float = 1e-6,
        clip_min: float = -5.0,
        clip_max: float = 5.0,
    ):
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @staticmethod
    def compute_client_stats(X_benign: np.ndarray) -> ClientRobustStats:
        if X_benign.ndim != 2:
            raise ValueError("X_benign must be 2D")
        n = int(X_benign.shape[0])
        if n <= 0:
            raise ValueError("No benign samples provided for scaler.")

        q25 = np.quantile(X_benign, 0.25, axis=0)
        q50 = np.quantile(X_benign, 0.50, axis=0)
        q75 = np.quantile(X_benign, 0.75, axis=0)
        return ClientRobustStats(n=n, q25=q25, q50=q50, q75=q75)

    def fit_from_client_stats(self, stats_list: list[ClientRobustStats]):
        if not stats_list:
            raise ValueError("stats_list is empty")

        total_n = sum(s.n for s in stats_list)
        if total_n <= 0:
            raise ValueError("Total n is zero")

        def wavg(arrs):
            out = None
            for s in stats_list:
                w = s.n / total_n
                a = arrs(s)
                out = a * w if out is None else out + a * w
            return out

        q25 = wavg(lambda s: s.q25)
        q50 = wavg(lambda s: s.q50)
        q75 = wavg(lambda s: s.q75)

        center = q50
        scale = (q75 - q25)
        scale = np.where(np.abs(scale) < self.eps, 1.0, scale)

        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fit.")
        Xs = (X - self.center_) / self.scale_
        return np.clip(Xs, self.clip_min, self.clip_max).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler not fit.")
        return {
            "eps": self.eps,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "center_": self.center_,
            "scale_": self.scale_,
        }

    @classmethod
    def load_state_dict(cls, d: Dict[str, Any]) -> "FederatedRobustScaler":
        obj = cls(eps=float(d["eps"]), clip_min=float(d["clip_min"]), clip_max=float(d["clip_max"]))
        obj.center_ = np.array(d["center_"], dtype=np.float32)
        obj.scale_ = np.array(d["scale_"], dtype=np.float32)
        return obj
