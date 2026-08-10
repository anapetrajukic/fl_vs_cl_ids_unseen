from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import torch.nn as nn

from federated.client.client import BaseClient
from federated.server.server import BaseServer
import torch
from models.vae import anomaly_scores
from pathlib import Path
import json
import re



######################################################################
# Per-round logging for seen/unseen trade-off plots - HELPER FUNCTIONS
######################################################################

"""
def _normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def append_round_metrics_txt(
    *,
    out_dir: str | Path,
    aggregation_method: str,
    unseen_dataset: str,
    round_idx: int,
    metrics: Dict[str, float],
    seen_datasets: List[str],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unseen_name = _normalize_name(unseen_dataset)
    method_name = _normalize_name(aggregation_method)
    out_path = out_dir / f"{unseen_name}_{method_name}.txt"

    row: Dict[str, Any] = {"round": int(round_idx)}

    seen_vals = []
    for ds in seen_datasets:
        k = f"{ds}__val_auroc"
        if k in metrics:
            row[f"{ds}__val"] = float(metrics[k])
            seen_vals.append(float(metrics[k]))

    if len(seen_vals) > 0:
        row["seen_val_mean"] = float(np.mean(seen_vals))
        row["seen_val_std"] = float(np.std(seen_vals))
        row["seen_val_min"] = float(np.min(seen_vals))
        row["seen_val_max"] = float(np.max(seen_vals))
        if len(seen_vals) == 2:
            row["seen_val_gap"] = float(abs(seen_vals[0] - seen_vals[1]))

    row["unseen_test"] = float(metrics["unseen__test_auroc"])

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@torch.no_grad()
def eval_auroc_arrays(
    *,
    model: torch.nn.Module,
    device: torch.device,
    X_np: np.ndarray,
    y_np: np.ndarray,
    score_type: str,
) -> float:
    model = model.to(device).eval()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    s = anomaly_scores(model, X, score_type=score_type).detach().cpu().numpy()
    y_np = np.asarray(y_np).reshape(-1)

    if (not np.all(np.isfinite(s))) or (len(np.unique(y_np)) < 2):
        return 0.5

    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_np, s))


def eval_all_datasets_end_of_seed(
    *,
    model: torch.nn.Module,
    clients: List[BaseClient],
    seen_datasets: List[str],
    unseen_dataset: str,
    unseen_val_scaled: Tuple[np.ndarray, np.ndarray],
    unseen_test_scaled: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    score_type: str, 
) -> Dict[str, float]:
    

    
    if len(seen_datasets) != len(clients):
        raise ValueError("seen_datasets and clients must align 1:1 in the same order.")

    metrics: Dict[str, float] = {}

    for ds_name, c in zip(seen_datasets, clients):
        metrics[f"{ds_name}__val_auroc"] = float(c.evaluate(model, "val")["auroc"])
        metrics[f"{ds_name}__test_auroc"] = float(c.evaluate(model, "test")["auroc"])

    Xuv, yuv = unseen_val_scaled
    Xut, yut = unseen_test_scaled

    metrics["unseen__val_auroc"] = float(
        eval_auroc_arrays(model=model, device=device, X_np=Xuv, y_np=yuv, score_type=score_type)
    )
    metrics["unseen__test_auroc"] = float(
        eval_auroc_arrays(model=model, device=device, X_np=Xut, y_np=yut, score_type=score_type)
    )

    metrics[f"{unseen_dataset}__val_auroc"] = metrics["unseen__val_auroc"]
    metrics[f"{unseen_dataset}__test_auroc"] = metrics["unseen__test_auroc"]

    return metrics
"""

class TrialPrunedWithState(optuna.TrialPruned):
    def __init__(self, state_dict: Dict[str, Any]):
        super().__init__()
        self.state_dict = state_dict


def run_federated_train(
    *,
    clients: Sequence[BaseClient],
    server: BaseServer,
    trial: Optional[optuna.Trial] = None,
    metric_key: str = "auroc",
    eval_split: str = "val",
    seen_datasets: List[str],
    unseen_dataset: str,
    unseen_val_scaled: Tuple[np.ndarray, np.ndarray],
    unseen_test_scaled: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
    score_type: str,
    aggregation_method: str,
    log_dir: str = ".",

) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    """
    Generic FL loop:
    each round- collect updates, aggregate
    every eval_every- evaluate clients on eval_split
    optional optuna pruning
    early stop on mean metric
    """
    cfg = server.cfg
    history: List[Dict[str, Any]] = []

    for rnd in range(cfg.rounds):
        payload = server.client_payload()
        updates = [c.fit(server.model, round_idx=rnd, server_payload=payload) for c in clients]
        server.aggregate(updates)

        ###########################################
        # LOGGING FOR seen/unseen trade-off plots
        ###########################################
        """
        logging_eval = eval_all_datasets_end_of_seed(
            model=server.model,
            clients=clients,
            seen_datasets=seen_datasets,
            unseen_dataset=unseen_dataset,
            unseen_val_scaled=unseen_val_scaled,
            unseen_test_scaled=unseen_test_scaled,
            device=device,
            score_type=score_type,
        )

        append_round_metrics_txt(
            out_dir=log_dir,
            aggregation_method=aggregation_method,
            unseen_dataset=unseen_dataset,
            round_idx=rnd + 1,
            metrics=logging_eval,
            seen_datasets=seen_datasets,
        )

        seen_vals = [float(logging_eval[f"{ds}__val_auroc"]) for ds in seen_datasets]
        print(
            f"round={rnd + 1} "
            f"seen_val_mean={np.mean(seen_vals):.6f} "
            f"unseen_test={float(logging_eval['unseen__test_auroc']):.6f}"
        )
        """

        if (rnd + 1) % cfg.eval_every == 0:
            vals = [float(c.evaluate(server.model, eval_split)[metric_key]) for c in clients]
            mean_val = float(np.mean(vals))

            history.append(
                {"round": rnd + 1, f"{eval_split}_{metric_key}s": vals, f"{eval_split}_{metric_key}_mean": mean_val}
            )

            if trial is not None:
                trial.report(mean_val, step=rnd + 1)
                if trial.should_prune():
                    raise TrialPrunedWithState(server.model.state_dict())

            if server.maybe_early_stop(mean_val):
                break

    ###########################################
    # LOGGING FOR seen/unseen trade-off plots
    ###########################################

    """
    logging_eval = eval_all_datasets_end_of_seed(
        model=server.model,
        clients=clients,
        seen_datasets=seen_datasets,
        unseen_dataset=unseen_dataset,
        unseen_val_scaled=unseen_val_scaled,
        unseen_test_scaled=unseen_test_scaled,
        device=device,
        score_type=score_type,
    )

    append_round_metrics_txt(
        out_dir=log_dir,
        aggregation_method=aggregation_method,
        unseen_dataset=unseen_dataset,
        round_idx=rnd + 1,
        metrics=logging_eval,
        seen_datasets=seen_datasets,
    )

    seen_vals = [float(logging_eval[f"{ds}__val_auroc"]) for ds in seen_datasets]
    print(
        f"round={rnd + 1} "
        f"seen_val_mean={np.mean(seen_vals):.6f} "
        f"unseen_test={float(logging_eval['unseen__test_auroc']):.6f}"
    )
    """
    server.restore_best()
    return server.model, history, float(server.best_metric)