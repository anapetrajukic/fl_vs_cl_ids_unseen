from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from models.vae import VAE, vae_loss, anomaly_scores
from task.base import BaseTask, TaskOptimConfig


@dataclass(frozen=True)
class VaeTaskConfig:
    epochs_per_round: int = 1
    batch_size: int = 512
    beta: float = 1.0
    beta_warmup_rounds: int = 0
    recon_type: str = "mse"
    score_type: str = "recon"


class VaeTask(BaseTask):

    def __init__(
        self,
        *,
        optim_cfg: TaskOptimConfig,
        task_cfg: VaeTaskConfig,
        X_train_benign_scaled: np.ndarray,
        seen_val: Dict[str, Tuple[np.ndarray, np.ndarray]],
        seen_test: Dict[str, Tuple[np.ndarray, np.ndarray]],
        unseen_val: Tuple[np.ndarray, np.ndarray],
        unseen_test: Tuple[np.ndarray, np.ndarray],
    ):
        if X_train_benign_scaled.ndim != 2:
            raise ValueError("X_train_benign_scaled must be 2D")

        super().__init__(optim_cfg=optim_cfg, n_train=int(len(X_train_benign_scaled)))

        self.cfg = task_cfg
        self.X_train_b = X_train_benign_scaled
        self.input_dim = int(self.X_train_b.shape[1])

        self.seen_val = {
            k: (X, np.asarray(y).reshape(-1))
            for k, (X, y) in seen_val.items()
        }
        self.seen_test = {
            k: (X, np.asarray(y).reshape(-1))
            for k, (X, y) in seen_test.items()
        }

        Xuv, yuv = unseen_val
        Xut, yut = unseen_test
        self.unseen_val = (Xuv, np.asarray(yuv).reshape(-1))
        self.unseen_test = (Xut, np.asarray(yut).reshape(-1))

    def make_model(
        self,
        *,
        hidden_dims: tuple[int, ...] = (256, 128),
        latent_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> nn.Module:
        return VAE(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            activation=activation,
        )

    def _make_optimizer(self, model: nn.Module):
        name = self.optim_cfg.optimizer.lower().strip()

        if name == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
            )
        if name == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
            )
        if name == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=self.optim_cfg.lr,
                weight_decay=self.optim_cfg.weight_decay,
            )
        if name == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.optim_cfg.lr,
                momentum=float(self.optim_cfg.sgd_momentum),
                nesterov=bool(self.optim_cfg.sgd_nesterov),
                weight_decay=self.optim_cfg.weight_decay,
            )

        raise ValueError(f"Unknown optimizer: {self.optim_cfg.optimizer}")

    def _iter_batches(self, X: np.ndarray, batch_size: int, shuffle: bool = True):
        n = len(X)
        idx = np.arange(n)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            j = idx[i : i + batch_size]
            yield X[j]

    def train(
        self,
        model: nn.Module,
        device: torch.device,
        round_idx: int,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        del kwargs

        model = model.to(device=device, dtype=torch.float32)
        model.train()

        opt = self._make_optimizer(model)

        if self.cfg.beta_warmup_rounds and (round_idx + 1) <= self.cfg.beta_warmup_rounds:
            beta_now = self.cfg.beta * float(round_idx + 1) / float(self.cfg.beta_warmup_rounds)
        else:
            beta_now = self.cfg.beta

        last_loss: float | None = None
        epoch_means: list[float] = []

        for _ in range(int(self.cfg.epochs_per_round)):
            batch_losses: list[float] = []

            for xb in self._iter_batches(self.X_train_b, int(self.cfg.batch_size), shuffle=True):
                x = torch.tensor(xb, dtype=torch.float32, device=device)

                opt.zero_grad(set_to_none=True)
                x_recon, mu_z, logvar = model(x)
                loss, recon, kl = vae_loss(
                    x,
                    x_recon,
                    mu_z,
                    logvar,
                    beta=float(beta_now),
                    recon_type=str(self.cfg.recon_type),
                )
                loss.backward()

                if self.optim_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(self.optim_cfg.grad_clip))

                opt.step()

                last_loss = float(loss.detach().cpu())
                batch_losses.append(last_loss)

            if batch_losses:
                epoch_means.append(float(np.mean(batch_losses)))

        return {
            "train_loss": last_loss if last_loss is not None else float("nan"),
            "train_loss_epoch_mean": float(np.mean(epoch_means)) if epoch_means else float("nan"),
            "beta_now": float(beta_now),
        }

    @torch.no_grad()
    def _eval_auroc(
        self,
        model: nn.Module,
        device: torch.device,
        X_np: np.ndarray,
        y_np: np.ndarray,
    ) -> float:
        model = model.to(device=device, dtype=torch.float32).eval()
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        s = anomaly_scores(model, X, score_type=str(self.cfg.score_type)).detach().cpu().numpy()
        y_np = np.asarray(y_np).reshape(-1)

        if (not np.all(np.isfinite(s))) or (len(np.unique(y_np)) < 2):
            return 0.5

        return float(roc_auc_score(y_np, s))

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        device: torch.device,
        split: str = "test",
    ) -> Dict[str, float]:
        if split == "val":
            vals = [
                self._eval_auroc(model, device, X, y)
                for X, y in self.seen_val.values()
            ]
            return {"auroc": float(np.mean(vals)) if vals else float("nan")}

        if split == "test":
            vals = [
                self._eval_auroc(model, device, X, y)
                for X, y in self.seen_test.values()
            ]
            return {"auroc": float(np.mean(vals)) if vals else float("nan")}

        if split == "unseen_val":
            X, y = self.unseen_val
            return {"auroc": self._eval_auroc(model, device, X, y)}

        if split == "unseen_test":
            X, y = self.unseen_test
            return {"auroc": self._eval_auroc(model, device, X, y)}

        raise ValueError("split must be one of: 'val', 'test', 'unseen_val', 'unseen_test'")

    @torch.no_grad()
    def evaluate_all(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        for ds, (Xv, yv) in self.seen_val.items():
            metrics[f"{ds}__val_auroc"] = self._eval_auroc(model, device, Xv, yv)

        for ds, (Xt, yt) in self.seen_test.items():
            metrics[f"{ds}__test_auroc"] = self._eval_auroc(model, device, Xt, yt)

        Xuv, yuv = self.unseen_val
        metrics["unseen__val_auroc"] = self._eval_auroc(model, device, Xuv, yuv)

        Xut, yut = self.unseen_test
        metrics["unseen__test_auroc"] = self._eval_auroc(model, device, Xut, yut)

        return metrics

    def seen_val_mean(self, metrics: Dict[str, float]) -> float:
        vals = [
            float(metrics[f"{ds}__val_auroc"])
            for ds in self.seen_val.keys()
            if f"{ds}__val_auroc" in metrics and np.isfinite(metrics[f"{ds}__val_auroc"])
        ]
        return float(np.mean(vals)) if vals else float("nan")