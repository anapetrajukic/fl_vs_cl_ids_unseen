from __future__ import annotations

from typing import List, Dict, Tuple
import os
import json
from math import ceil

import numpy as np
import optuna
import torch

from hpo.common import HPOConfig, set_seed, objective_reduce, make_pruner

from centralized.data_builder import build_central_data_with_sklearn_robust_scaler
from workshop_dg_cl_vs_fl.centralized.task.vae_task import VaeTask, VaeTaskConfig
from centralized.task.trainer import Trainer, TrainConfig
from centralized.cen_hpo.train_loop import run_train  
from task.base import TaskOptimConfig
from data.splits_loader import load_split_pickles


def materialize_best_params(best_trial: optuna.trial.FrozenTrial) -> dict:
    params = dict(best_trial.params)
    attrs = dict(best_trial.user_attrs)

    merged = dict(params)

    for k in [
        "clip_min",
        "clip_max",
        "early_stop_patience",
        "rounds",
        "optimizer",
        "weight_decay",
        "grad_clip",
        "lr",
        "sgd_momentum",
        "sgd_nesterov",
        "beta",
        "beta_warmup_rounds",
        "recon_type",
        "score_type",
        "latent_dim",
        "dropout",
        "activation",
        "hidden_dims",
        "n_layers",
        "first_layer",
        "local_epochs",
        "batch_size",
    ]:
        if k not in merged and k in attrs:
            merged[k] = attrs[k]

    if "clip_min" not in merged or "clip_max" not in merged:
        if "clip_abs" in merged:
            merged["clip_min"] = -float(merged["clip_abs"])
            merged["clip_max"] = float(merged["clip_abs"])

    return merged

def _epochs_match_fed_steps(
    *,
    seen_datasets: list[str],
    splits_dir: str,
    batch_size: int,
    rounds: int,
    local_epochs: int,
) -> int:
    """
    FOR NOW DIAGNOSTIC ONLY!!
    Match centralized total optimizer steps to FL total client optimizer steps
    FL total steps:
        rounds * local_epochs * sum_i ceil(n_i / batch_size)

    Central steps per epoch:
        ceil(sum_i n_i / batch_size)

    So:
        epochs ~= fed_steps / central_steps_per_epoch
    """
    n_i = []
    for ds in seen_datasets:
        sp = load_split_pickles(ds, splits_dir)
        n_b = int(np.sum(sp.y_train == 0))
        if n_b <= 0:
            raise ValueError(f"{ds}: no benign samples in TRAIN.")
        n_i.append(n_b)

    fed_steps = int(rounds) * int(local_epochs) * int(sum(ceil(x / batch_size) for x in n_i))
    N = int(sum(n_i))
    steps_per_epoch = int(ceil(N / batch_size))
    epochs = int(ceil(fed_steps / steps_per_epoch))
    return max(1, epochs)


def objective_factory(
    *,
    seen_datasets: List[str],
    unseen_dataset: str,
    splits_dir: str,
    device: torch.device,
    hpo_cfg: HPOConfig,
    hpo_seeds: List[int],
    prune_first_k_seeds: int = 1,
):

    def objective(trial: optuna.Trial) -> float:
        #################################
        ### Unseen used only for logging
        ### Sample hyperparams once
        #################################

        n_layers = trial.suggest_int("n_layers", 1, 4)
        first = trial.suggest_int("first_layer", 64, 1024)
        dims = [first]
        for i in range(1, n_layers):
            drop = trial.suggest_float(f"layer_{i}_dropoff", 0.5, 1.0)
            dims.append(max(16, int(dims[-1] * drop)))
        hidden_dims = tuple(dims)

        latent_dim = trial.suggest_int("latent_dim", 8, 128)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        activation = trial.suggest_categorical("activation", ["relu", "leakyrelu", "elu", "gelu"])

        weight_decay = trial.suggest_categorical(
            "weight_decay", [0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4]
        )
        grad_clip = trial.suggest_categorical("grad_clip", [None, 1.0, 2.0, 5.0])

        optimizer = "sgd"
        lr = trial.suggest_float("lr", 3e-4, 2e-2, log=True)
        sgd_momentum = trial.suggest_float("sgd_momentum", 0.0, 0.95)
        sgd_nesterov = False
        if sgd_momentum > 0.0:
            sgd_nesterov = trial.suggest_categorical("sgd_nesterov", [False, True])

        beta = trial.suggest_float("beta", 0.1, 2.0, log=True)
        beta_warmup_rounds = trial.suggest_int("beta_warmup_rounds", 0, 15)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        local_epochs = trial.suggest_int("local_epochs", 1, 8)

        recon_type = trial.suggest_categorical("recon_type", ["mse", "huber"])
        score_type = trial.suggest_categorical("score_type", ["recon", "recon+kl"])

        clip_abs = trial.suggest_categorical("clip_abs", [3.0, 4.0, 5.0, 6.0, 8.0])
        clip_min = -clip_abs
        clip_max = clip_abs

        rounds = 120
        max_rounds = int(rounds)
        epochs_per_round = int(local_epochs)

        #early_stop_patience = max(2, int(round(10 / local_epochs)))
        early_stop_patience = 8
        
        #####################################
        ######## diagnostic only
        #####################################

        epochs_matched = _epochs_match_fed_steps(
            seen_datasets=seen_datasets,
            splits_dir=splits_dir,
            batch_size=int(batch_size),
            rounds=int(rounds),
            local_epochs=int(local_epochs),
        )

        trial.set_user_attr("rounds", int(rounds))
        trial.set_user_attr("local_epochs", int(local_epochs))
        trial.set_user_attr("epochs_matched", int(epochs_matched))
        trial.set_user_attr("max_rounds", int(max_rounds))
        trial.set_user_attr("epochs_per_round", int(epochs_per_round))
        trial.set_user_attr("optimizer", str(optimizer))
        trial.set_user_attr("lr", float(lr))
        trial.set_user_attr("batch_size", int(batch_size))

        trial.set_user_attr("n_layers", int(n_layers))
        trial.set_user_attr("first_layer", int(first))
        for i in range(1, n_layers):
            trial.set_user_attr(f"layer_{i}_dropoff", float(dims[i] / dims[i - 1]))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("beta_warmup_rounds", int(beta_warmup_rounds))
        trial.set_user_attr("weight_decay", float(weight_decay))
        trial.set_user_attr("grad_clip", grad_clip)
        trial.set_user_attr("sgd_momentum", float(sgd_momentum))
        trial.set_user_attr("sgd_nesterov", bool(sgd_nesterov))
        trial.set_user_attr("recon_type", str(recon_type))
        trial.set_user_attr("score_type", str(score_type))
        trial.set_user_attr("clip_abs", float(clip_abs))
        trial.set_user_attr("clip_min", float(clip_min))
        trial.set_user_attr("clip_max", float(clip_max))
        trial.set_user_attr("early_stop_patience", int(early_stop_patience))
        trial.set_user_attr("latent_dim", int(latent_dim))
        trial.set_user_attr("dropout", float(dropout))
        trial.set_user_attr("activation", str(activation))
        trial.set_user_attr("hidden_dims", list(hidden_dims))
  

        seed_seen_objs: List[float] = []


        metric_sums: Dict[str, float] = {}
        n_metric = 0

        
        for seed_idx, seed in enumerate(hpo_seeds):
            set_seed(seed)

            X_train_b_s, seen_val, seen_test, unseen_val, unseen_test, _scaler = (
                build_central_data_with_sklearn_robust_scaler(
                    seen_datasets=seen_datasets,
                    unseen_dataset=unseen_dataset,
                    splits_dir=splits_dir,
                    clip_min=float(clip_min),
                    clip_max=float(clip_max),
                )
            )

            optim_cfg = TaskOptimConfig(
                lr=float(lr),
                weight_decay=float(weight_decay),
                grad_clip=grad_clip,
                optimizer=str(optimizer),
                sgd_momentum=float(sgd_momentum),
                sgd_nesterov=bool(sgd_nesterov),
            )

            task_cfg = VaeTaskConfig(
                epochs_per_round=int(epochs_per_round),
                batch_size=int(batch_size),
                beta=float(beta),
                beta_warmup_rounds=int(beta_warmup_rounds),
                recon_type=str(recon_type),
                score_type=str(score_type),
            )

            train_cfg = TrainConfig(
                max_rounds=int(max_rounds),
                early_stop_patience=1,  # one 'round', with ___ epochs per round, rounds are only for logging
                early_stop_min_delta=1,
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
                hidden_dims=hidden_dims,
                latent_dim=int(latent_dim),
                dropout=float(dropout),
                activation=str(activation),
            )

            trainer = Trainer(
                model=model,
                task=task,
                device=device,
                cfg=train_cfg,
            )

            use_trial = trial if (seed_idx < prune_first_k_seeds) else None

            model, history, best_metric = run_train(
                task=task,
                trainer=trainer,
                trial=use_trial,
                metric_key="auroc",
                eval_split="val",
            )

            metrics = task.evaluate_all(model, device)

            # Objective: seen VAL aggregate
            seen_val_aurocs = [float(metrics[f"{ds}__val_auroc"]) for ds in seen_datasets]
            seed_seen_objs.append(float(objective_reduce(seen_val_aurocs, mode=hpo_cfg.objective_mode)))


            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            n_metric += 1

            # trainer/run stats
            metric_sums["train__best_metric"] = metric_sums.get("train__best_metric", 0.0) + float(best_metric)
            metric_sums["train__history_len"] = metric_sums.get("train__history_len", 0.0) + float(len(history))

            trial.set_user_attr("unseen_val_n", int(unseen_val[0].shape[0]))
            trial.set_user_attr("unseen_test_n", int(unseen_test[0].shape[0]))

        seen_mean = float(np.mean(seed_seen_objs))

        trial.set_user_attr("hpo_seeds", list(hpo_seeds))
        trial.set_user_attr("seed_seen_obj_mean", seen_mean)
        trial.set_user_attr("seed_seen_obj_std", float(np.std(seed_seen_objs)))


        if n_metric > 0:
            metric_means = {k: metric_sums[k] / float(n_metric) for k in metric_sums.keys()}
            for k, v in metric_means.items():
                trial.set_user_attr(k, float(v))

            seen_test_vals = [metric_means.get(f"{ds}__test_auroc", float("nan")) for ds in seen_datasets]
            if all(np.isfinite(seen_test_vals)):
                trial.set_user_attr("seen_test_mean", float(np.mean(seen_test_vals)))
                trial.set_user_attr("seen_test_min", float(np.min(seen_test_vals)))

        return seen_mean

    return objective


def run_hpo(
    *,
    seen_datasets: List[str],
    unseen_dataset: str,
    splits_dir: str,
    out_dir: str,
    device: torch.device,
    hpo_cfg: HPOConfig,
    hpo_seeds: List[int],
    prune_first_k_seeds: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=hpo_cfg.seed)
    pruner = make_pruner(hpo_cfg.pruner)
    storage = f"sqlite:///{os.path.join(out_dir, 'study.db')}"

    seeds_tag = "-".join(map(str, hpo_seeds))
    study_name = (
        f"vae_hpo__sampler_{hpo_cfg.seed}"
        f"__unseen_{unseen_dataset}__seen_{'_'.join(seen_datasets)}"
        f"__hpseeds_{seeds_tag}__space_v4"
    )

    study = optuna.create_study(
        direction = 'maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    obj = objective_factory(
        seen_datasets=seen_datasets,
        unseen_dataset=unseen_dataset,
        splits_dir=splits_dir,
        device=device,
        hpo_cfg=hpo_cfg,
        hpo_seeds=hpo_seeds,
        prune_first_k_seeds=prune_first_k_seeds,
    )

    def _save_every_trial(study_: optuna.Study, trial_: optuna.trial.FrozenTrial):
        df = study_.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    callbacks = [_save_every_trial] if hpo_cfg.save_csv_each_trial else None

    study.optimize(
        obj,
        n_trials=hpo_cfg.n_trials,
        timeout=hpo_cfg.timeout_sec,
        show_progress_bar=True,
        callbacks=callbacks,
        catch=(ValueError,),
    )


    best_out = {
        "number": study.best_trial.number,
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
    }
    with open(os.path.join(out_dir, "best_trial.json"), "w") as f:
        json.dump(best_out, f, indent=2)


    best_params_full = materialize_best_params(study.best_trial)

    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params_full, f, indent=2)

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    return study