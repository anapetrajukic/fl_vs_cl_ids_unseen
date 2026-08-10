from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import optuna

from hpo.common import HPOConfig, set_seed
from centralized.cen_hpo.optuna_central_vae import run_hpo, _epochs_match_fed_steps
from centralized.data_builder import build_central_data_with_sklearn_robust_scaler
from workshop_dg_cl_vs_fl.centralized.task.vae_task import VaeTask, VaeTaskConfig
from centralized.task.trainer import Trainer, TrainConfig
from centralized.cen_hpo.train_loop import run_train
from task.base import TaskOptimConfig
from centralized.cen_hpo.optuna_central_vae import materialize_best_params


def hidden_dims_from_best_params(best_params: dict) -> tuple[int, ...]:
    n_layers = int(best_params["n_layers"])
    first = int(best_params["first_layer"])
    dims = [first]
    for i in range(1, n_layers):
        drop = float(best_params[f"layer_{i}_dropoff"])
        dims.append(max(16, int(dims[-1] * drop)))
    return tuple(dims)


def final_train_and_eval(
    *,
    seen_datasets: list[str],
    unseen_dataset: str,
    splits_dir: str,
    out_dir: str,
    device: torch.device,
    best_params: dict,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)

    clip_min = float(best_params["clip_min"])
    clip_max = float(best_params["clip_max"])

    X_train_b_s, seen_val, seen_test, unseen_val, unseen_test, scaler = (
        build_central_data_with_sklearn_robust_scaler(
            seen_datasets=seen_datasets,
            unseen_dataset=unseen_dataset,
            splits_dir=splits_dir,
            clip_min=clip_min,
            clip_max=clip_max,
        )
    )

    optim_cfg = TaskOptimConfig(
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        grad_clip=best_params["grad_clip"],
        optimizer=str(best_params["optimizer"]),
        sgd_momentum=float(best_params.get("sgd_momentum", 0.0)),
        sgd_nesterov=bool(best_params.get("sgd_nesterov", False)),
    )

    task_cfg = VaeTaskConfig(
        epochs_per_round=int(best_params["local_epochs"]),
        batch_size=int(best_params["batch_size"]),
        beta=float(best_params["beta"]),
        recon_type=str(best_params["recon_type"]),
        score_type=str(best_params["score_type"]),
    )

    train_cfg = TrainConfig(
        max_rounds=int(best_params["rounds"]),
        early_stop_patience=int(best_params.get("early_stop_patience", 10)),
        early_stop_min_delta=float(best_params.get("early_stop_min_delta", 1e-4)),
        restore_best=True,
    )

    task = VaeTask(
        optim_cfg=optim_cfg,
        task_cfg=task_cfg,
        X_train_benign_scaled=X_train_b_s,
        seen_val=seen_val,
        seen_test=seen_test,
        unseen_val=unseen_val,
        unseen_test=unseen_test,
    )

    model = task.make_model(
        hidden_dims=hidden_dims_from_best_params(best_params),
        latent_dim=int(best_params["latent_dim"]),
        dropout=float(best_params["dropout"]),
        activation=str(best_params["activation"]),
    )

    trainer = Trainer(
        model=model,
        task=task,
        device=device,
        cfg=train_cfg,
    )

    model, history, best_metric = run_train(
        task=task,
        trainer=trainer,
        trial=None,
        metric_key="auroc",
        eval_split="val",
        final_train= True,
        unseen_dataset_name = unseen_dataset,
        seed = seed,
    )

    metrics = task.evaluate_all(model, device)

    seen_val_aurocs = [float(metrics[f"{ds}__val_auroc"]) for ds in seen_datasets]
    seen_test_aurocs = [float(metrics[f"{ds}__test_auroc"]) for ds in seen_datasets]

    exp_dir = os.path.join(out_dir, "final_best")
    os.makedirs(exp_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(exp_dir, "central_model.pt"))

    scaler_state = {
        "type": type(scaler).__name__,
        "center_": None if getattr(scaler, "center_", None) is None else scaler.center_.tolist(),
        "scale_": None if getattr(scaler, "scale_", None) is None else scaler.scale_.tolist(),
        "quantile_range": (
            None if getattr(scaler, "quantile_range", None) is None else list(scaler.quantile_range)
        ),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
    }
    with open(os.path.join(exp_dir, "robust_scaler.json"), "w") as f:
        json.dump(scaler_state, f, indent=2)

    results = {
        "seed": seed,
        "seen_datasets": seen_datasets,
        "unseen_dataset": unseen_dataset,
        "best_params": best_params,
        "seen_val_mean": float(np.mean(seen_val_aurocs)),
        "seen_test_mean": float(np.mean(seen_test_aurocs)),
        "unseen__val_auroc": float(metrics["unseen__val_auroc"]),
        "unseen__test_auroc": float(metrics["unseen__test_auroc"]),
        "metrics": metrics,
        "history": history,
        "best_metric": float(best_metric),
        "no_leakage": {
            "objective": "single-objective: seen VAL aggregate",
            "scaler_fit": "pooled seen TRAIN benign only",
            "unseen_usage": {
                "unseen_val": "not optimized; logged only during HPO / reported here",
                "unseen_test": "held-out test only (reported, not optimized)",
            },
        },
        "compute_match": {
            "epochs_matched": int(
                _epochs_match_fed_steps(
                    seen_datasets=seen_datasets,
                    splits_dir=splits_dir,
                    batch_size=int(best_params["batch_size"]),
                    rounds=int(best_params["rounds"]),
                    local_epochs=int(best_params["local_epochs"]),
                )
            ),
            "rounds": int(best_params["rounds"]),
            "local_epochs": int(best_params["local_epochs"]),
        },
        "unseen_split": {
            "unseen_val_n": int(unseen_val[0].shape[0]),
            "unseen_test_n": int(unseen_test[0].shape[0]),
        },
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    SPLITS_DIR = "./data/splits"
    OUT_ROOT = "./out/centralized_multiseed"
    os.makedirs(OUT_ROOT, exist_ok=True)

    datasets = [ "NF-BoT-IoT-v3","NF-ToN-IoT-v3","NF-CICIDS2018-v3" ]

    HPO_SEEDS = [42,43,44,45] 
    EVAL_SEEDS = [46, 47, 48, 49, 50, 51, 52, 53]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results: Dict[str, Any] = {
        "datasets": datasets,
        "hpo_seeds": HPO_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "runs": {},
    }

    for unseen in datasets:
        seen = [d for d in datasets if d != unseen]
        fold_dir = os.path.join(OUT_ROOT, f"unseen_{unseen}")
        os.makedirs(fold_dir, exist_ok=True)

        hpo_cfg = HPOConfig(
            n_trials=80,
            timeout_sec=None,
            direction="maximize",
            objective_mode="mean",
            pruner="median",
            #pruner = "none",
            seed=123,
            log_all_tests_each_trial=False,
            log_every_k_trials=1,
            save_csv_each_trial=True,
        )

        study = run_hpo(
            hpo_seeds=HPO_SEEDS,
            seen_datasets=seen,
            unseen_dataset=unseen,
            splits_dir=SPLITS_DIR,
            out_dir=fold_dir,
            device=device,
            hpo_cfg=hpo_cfg,
        )

        chosen = study.best_trial
        best_params = materialize_best_params(chosen)

        chosen_meta = {
            "trial_number": int(chosen.number),
            "value": float(chosen.value) if chosen.value is not None else None,
            "user_attrs": chosen.user_attrs,
            "selection_rule": {
                "objective": "maximize seen validation objective",
            },
        }
        with open(os.path.join(fold_dir, "best_trial.json"), "w") as f:
            json.dump(chosen_meta, f, indent=2)

        ## with open(os.path.join(fold_dir, "chosen_pareto_trial.json"), "w") as f:
        ##     json.dump(chosen_meta, f, indent=2)

        with open(os.path.join(fold_dir, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=2)

        eval_results: Dict[str, Any] = {}
        for seed in EVAL_SEEDS:
            out_dir_seed = os.path.join(fold_dir, f"eval_seed_{seed}")
            os.makedirs(out_dir_seed, exist_ok=True)
            eval_results[str(seed)] = final_train_and_eval(
                seen_datasets=seen,
                unseen_dataset=unseen,
                splits_dir=SPLITS_DIR,
                out_dir=out_dir_seed,
                device=device,
                best_params=best_params,
                seed=seed,
            )

        all_results["runs"][unseen] = {
            "seen_datasets": seen,
            "unseen_dataset": unseen,
            "best_trial_number": int(chosen.number),
            "best_trial_value": float(chosen.value) if chosen.value is not None else None,
            "best_params": best_params,
            "eval": eval_results,
        }

        with open(os.path.join(fold_dir, "summary.json"), "w") as f:
            json.dump(all_results["runs"][unseen], f, indent=2)

    with open(os.path.join(OUT_ROOT, "summary_all_folds.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("Saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()