from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from task.base import BaseTask, TaskOptimConfig
from models.vae import VAE, vae_loss, anomaly_scores


@dataclass(frozen=True)
class VaeTaskConfig:
    local_epochs: int = 1
    batch_size: int = 512
    beta: float = 1.0
    beta_warmup_rounds: int = 0
    recon_type: str = "mse"
    score_type: str = "recon"
    prox_mu: float = 0.0  # 0.0 - no proximal penalty


class VaeTask(BaseTask):
    """
    Plain VAE anomaly detection task
    supports all implemented aggregation methods

    """

    def __init__(
        self,
        *,
        optim_cfg: TaskOptimConfig,
        task_cfg: VaeTaskConfig,
        X_train_benign_scaled: np.ndarray,
        X_val_scaled: np.ndarray,
        y_val: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
        aggregation_method: str,
        unseen_val: Tuple[np.ndarray, np.ndarray], 
        unseen_test: Tuple[np.ndarray, np.ndarray],   
    ):
        if X_train_benign_scaled.ndim != 2:
            raise ValueError("X_train_benign_scaled must be 2D.")

        super().__init__(optim_cfg=optim_cfg, n_train=int(len(X_train_benign_scaled)))

        self.cfg = task_cfg

        self.X_train_b = X_train_benign_scaled
        self.X_val = X_val_scaled
        self.y_val = np.asarray(y_val).reshape(-1)
        self.X_test = X_test_scaled
        self.y_test = np.asarray(y_test).reshape(-1)

        Xuv, yuv = unseen_val
        Xut, yut = unseen_test
        self.unseen_val = (Xuv, np.asarray(yuv).reshape(-1))
        self.unseen_test = (Xut, np.asarray(yut).reshape(-1))

        self.input_dim = int(self.X_train_b.shape[1])

        self.aggregation_method = str(aggregation_method).lower().strip()
        if self.aggregation_method not in {"fedavg", "fedadam", "fedprox", "scaffold"}:
            raise ValueError(
                f"Unknown aggregation_method='{aggregation_method}'. "
                "Expected one of: {'fedavg','fedadam','fedprox','scaffold'}"
            )

        if float(self.cfg.prox_mu) < 0.0:
            raise ValueError(f"prox_mu must be >= 0, got {self.cfg.prox_mu}")


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
        name = self.optim_cfg.optimizer.lower()
        if name == "adamw":
            return optim.AdamW(model.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
        if name == "adam":
            return optim.Adam(model.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
        if name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay)
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

    ##################################
    # One FL round of local training
    ##################################
    def train(
        self,
        model: nn.Module,
        device: torch.device,
        round_idx: int,
        *,
        server_payload: Optional[Dict[str, Any]] = None,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model = model.to(device)
        model.train()

        opt = self._make_optimizer(model)
        server_payload = server_payload or {}
        client_state = client_state or {}

        # beta warmup across rounds
        if self.cfg.beta_warmup_rounds and (round_idx + 1) <= self.cfg.beta_warmup_rounds:
            beta_now = self.cfg.beta * float(round_idx + 1) / float(self.cfg.beta_warmup_rounds)
        else:
            beta_now = self.cfg.beta

        ######### FedProx part #########
        use_fedprox = (self.aggregation_method == "fedprox") and (float(self.cfg.prox_mu) > 0.0)
        prox_mu = float(self.cfg.prox_mu)

        ref_params: Optional[dict[str, torch.Tensor]] = None
        if use_fedprox:
            ref_params = {name: p.detach().clone() for name, p in model.named_parameters()}

        ######### Scaffold part #########
        use_scaffold = (self.aggregation_method == "scaffold")

        c_cpu = None
        ci_cpu = None
        c_dev = None
        ci_dev = None

        w0 = None
        K_steps = 0

        if use_scaffold:
            c_cpu = server_payload.get("scaffold_c", None)
            if c_cpu is None:
                raise ValueError("SCAFFOLD requires server_payload['scaffold_c'].")

            ci_cpu = client_state.get("scaffold_ci", None)
            if ci_cpu is None:
                ci_cpu = {k: torch.zeros_like(p.detach().cpu()) for k, p in model.named_parameters()}

            for name, _p in model.named_parameters():
                if name not in c_cpu:
                    raise KeyError(f"SCAFFOLD- server c missing param: {name}")

            c_dev = {}
            ci_dev = {}
            for name, p in model.named_parameters():
                c_dev[name] = c_cpu[name].to(device=device, dtype=p.dtype)
                ci_dev[name] = ci_cpu[name].to(device=device, dtype=p.dtype)

            w0 = {k: p.detach().clone() for k, p in model.named_parameters()}

        last_loss: Optional[float] = None
        last_prox: Optional[float] = None

        for _ in range(self.cfg.local_epochs):
            for xb in self._iter_batches(self.X_train_b, self.cfg.batch_size, shuffle=True):
                x = torch.tensor(xb, dtype=torch.float32, device=device)

                opt.zero_grad(set_to_none=True)
                x_recon, mu_z, logvar = model(x)
                loss, recon, kl = vae_loss(
                    x, x_recon, mu_z, logvar, beta=beta_now, recon_type=self.cfg.recon_type
                )

                prox_term = None
                if use_fedprox:
                    assert ref_params is not None
                    prox_acc = torch.zeros((), device=device)
                    for name, p in model.named_parameters():
                        prox_acc = prox_acc + torch.sum((p - ref_params[name]) ** 2)
                    prox_term = 0.5 * prox_mu * prox_acc
                    loss = loss + prox_term

                loss.backward()

                if use_scaffold:
                    assert c_dev is not None and ci_dev is not None
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.grad.add_(c_dev[name] - ci_dev[name])

                if self.optim_cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), self.optim_cfg.grad_clip)

                opt.step()
                K_steps += 1

                last_loss = float(loss.detach().cpu())
                if prox_term is not None:
                    last_prox = float(prox_term.detach().cpu())

        out: Dict[str, Any] = {
            "train_loss": last_loss if last_loss is not None else float("nan"),
            "beta_now": float(beta_now),
        }

        if use_fedprox:
            out["prox_mu"] = float(prox_mu)
            out["prox_term_last"] = float(last_prox) if last_prox is not None else float("nan")

        if use_scaffold:
            assert w0 is not None
            assert ci_cpu is not None
            assert c_dev is not None

            if K_steps <= 0:
                raise RuntimeError("SCAFFOLD: K_steps is 0")
            lr = float(self.optim_cfg.lr)
            if lr <= 0:
                raise ValueError("SCAFFOLD: lr must be > 0")

            wK = {k: p.detach().clone() for k, p in model.named_parameters()}

            ci_new_cpu: Dict[str, torch.Tensor] = {}
            delta_ci_cpu: Dict[str, torch.Tensor] = {}

            for k in w0.keys():
                term = (w0[k] - wK[k]) / (K_steps * lr)
                ci_old_dev = ci_cpu[k].to(device=device, dtype=w0[k].dtype)
                ci_new_k = (ci_old_dev - c_dev[k] + term).detach().cpu()
                ci_new_cpu[k] = ci_new_k
                delta_ci_cpu[k] = (ci_new_k - ci_cpu[k]).detach().cpu()

            out["scaffold_steps"] = int(K_steps)
            out["scaffold_delta_ci"] = delta_ci_cpu
            out["scaffold_ci_new"] = ci_new_cpu

        return out

    #####################
    # Evaluation helpers
    #####################
    @torch.no_grad()
    def _eval_auroc(self, model: nn.Module, device: torch.device, X_np: np.ndarray, y_np: np.ndarray) -> float:
        model = model.to(device).eval()
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        s = anomaly_scores(model, X, score_type=self.cfg.score_type).detach().cpu().numpy()
        y_np = np.asarray(y_np).reshape(-1)

        if (not np.all(np.isfinite(s))) or (len(np.unique(y_np)) < 2):
            return 0.5
        return float(roc_auc_score(y_np, s))

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device, split: str) -> Dict[str, float]:
        if split == "val":
            au = self._eval_auroc(model, device, self.X_val, self.y_val)
            return {"auroc": au}
        if split == "test":
            au = self._eval_auroc(model, device, self.X_test, self.y_test)
            return {"auroc": au}
        raise ValueError("split must be 'val' or 'test'")

    @torch.no_grad()
    def evaluate_all(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        metrics["client__val_auroc"] = self._eval_auroc(model, device, self.X_val, self.y_val)
        metrics["client__test_auroc"] = self._eval_auroc(model, device, self.X_test, self.y_test)

        Xuv, yuv = self.unseen_val
        metrics["unseen__val_auroc"] = self._eval_auroc(model, device, Xuv, yuv)

        Xut, yut = self.unseen_test
        metrics["unseen__test_auroc"] = self._eval_auroc(model, device, Xut, yut)

        return metrics