from __future__ import annotations

from typing import Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1)
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        latent_dim: int = 32,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)

        act = make_activation(activation)

        # encoder: x -> h -> mu/logvar
        enc_layers = []
        last = self.input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(last, h), act, nn.Dropout(dropout)]
            last = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(last, self.latent_dim)
        self.fc_logvar = nn.Linear(last, self.latent_dim)

        # decoder: z -> x_recon
        dec_layers = []
        last = self.latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(last, h), act, nn.Dropout(dropout)]
            last = h
        dec_layers += [nn.Linear(last, self.input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    recon_type: Literal["mse", "huber"] = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if recon_type == "mse":
        recon = F.mse_loss(x_recon, x, reduction="mean")
    elif recon_type == "huber":
        recon = F.smooth_l1_loss(x_recon, x, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_type: {recon_type}")

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon + float(beta) * kl
    return total, recon, kl


@torch.no_grad()
def anomaly_scores(
    model: nn.Module,
    x: torch.Tensor,
    *,
    score_type: Literal["recon", "recon+kl"] = "recon",
) -> torch.Tensor:
    """
    Per-sample anomaly score.
    """
    model.eval()
    x_recon, mu, logvar = model(x)

    per_feat = F.mse_loss(x_recon, x, reduction="none")  # [B, D]
    recon_ps = per_feat.mean(dim=1)  # [B]

    if score_type == "recon":
        return recon_ps

    if score_type == "recon+kl":
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, Z]
        kl_ps = kl_per_dim.mean(dim=1)  # [B]
        return recon_ps + kl_ps

    raise ValueError(f"Unknown score_type: {score_type}")

