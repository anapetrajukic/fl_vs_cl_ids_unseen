from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import re

import optuna
import numpy as np
import torch.nn as nn

from centralized.task.trainer import Trainer
from workshop_dg_cl_vs_fl.centralized.task.vae_task import VaeTask


class TrialPrunedWithState(optuna.TrialPruned):
    def __init__(self, state_dict: Dict[str, Any]):
        super().__init__()
        self.state_dict = state_dict


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _seen_val_mean_from_metrics(metrics: Dict[str, float]) -> float:
    vals = [
        float(v)
        for k, v in metrics.items()
        if k.endswith("__val_auroc") and not k.startswith("unseen__")
    ]
    if not vals:
        raise ValueError("No seen validation AUROC values found in metrics.")
    return float(np.mean(vals))


def run_train(
    *,
    task: VaeTask,
    trainer: Trainer,
    trial: Optional[optuna.Trial] = None,
    metric_key: str = "auroc",
    eval_split: str = "val",
    final_train: bool = False,  #logging only
    unseen_dataset_name="",  #logging only
    seed=42,
    out_dir: str = "./out/central_multiseed_hpo_new",
) -> Tuple[nn.Module, List[Dict[str, Any]], float]:

    history: List[Dict[str, Any]] = []

    log_path = None
    ### LOGGING ONLY!!

    #if final_train:
    #    log_dir = Path(out_dir) / "round_logs"
    #    log_dir.mkdir(parents=True, exist_ok=True)

    #    log_path = log_dir / f"centralized__{_safe_name(unseen_dataset_name)}__seed{seed}.txt"
    #    with log_path.open("w", encoding="utf-8") as f:
    #        f.write("round\tseen_val_mean\tunseen_test_auroc\n")

    for round_idx in range(int(trainer.cfg.max_rounds)):
        train_out = trainer.task.train(
            trainer.model,
            trainer.device,
            round_idx=round_idx,
        )

        metrics = task.evaluate_all(trainer.model, trainer.device)
        seen_val_mean = _seen_val_mean_from_metrics(metrics)
        unseen_test_auroc = float(metrics["unseen__test_auroc"])

        if eval_split == "val":
            metric_val = float(task.seen_val_mean(metrics))
        elif eval_split == "unseen_val":
            metric_val = float(metrics["unseen__val_auroc"])
        else:
            out = task.evaluate(trainer.model, trainer.device, split=eval_split)
            metric_val = float(out[metric_key])

        history_row: Dict[str, Any] = {
            "round": round_idx + 1,
            f"{eval_split}_{metric_key}": metric_val,
        }

        if isinstance(train_out, dict):
            for k, v in train_out.items():
                if isinstance(v, (int, float, bool, np.floating, np.integer)):
                    history_row[f"train__{k}"] = float(v)

        if final_train:
            history_row["seen_val_mean"] = seen_val_mean
            history_row["unseen_test_auroc"] = unseen_test_auroc
        #    print(log_path)
        #    with log_path.open("a", encoding="utf-8") as f:
        #        f.write(
        #            f"{round_idx + 1}\t{seen_val_mean:.6f}\t{unseen_test_auroc:.6f}\n"
        #        )
        #    print(f"{round_idx + 1}\t{seen_val_mean:.6f}\t{unseen_test_auroc:.6f}\n")

        history.append(history_row)

        if trial is not None:
            trial.report(metric_val, step=round_idx + 1)
            if trial.should_prune():
                raise TrialPrunedWithState(trainer.model.state_dict())

        if trainer.maybe_early_stop(metric_val):
            print("Early stop h0!")
            break

    trainer.restore_best()
    return trainer.model, history, float(trainer.best_metric)

