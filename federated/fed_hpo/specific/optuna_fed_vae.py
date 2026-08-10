from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import os
import json

import numpy as np
import optuna
import torch

from data.splits_loader import load_split_pickles  
from workshop_dg_cl_vs_fl.federated.scalers.fed_robust_scaler import FederatedRobustScaler
from task.base import TaskOptimConfig
from federated.task.vae_task import VaeTask, VaeTaskConfig
from models.vae import anomaly_scores

from federated.client.client import BaseClient
from federated.server.server import BaseServer, ServerConfig
from federated.server.strategy.FedAvgStrategy import FedAvgStrategy
from federated.server.strategy.FedAdamStrategy import FedAdamStrategy, FedAdamConfig
from federated.server.strategy.ScaffoldStrategy import ScaffoldStrategy

from hpo.common import HPOConfig, set_seed, objective_reduce, make_pruner
from federated.fed_hpo.fed_loop import run_federated_train
from sklearn.metrics import roc_auc_score

from pathlib import Path
import json

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

    return float(roc_auc_score(y_np, s))


def build_clients_and_scaler(
    *,
    seen_datasets: List[str],
    unseen_dataset: str,                       
    splits_dir: str,
    device: torch.device,
    optim_cfg: TaskOptimConfig,
    task_cfg: VaeTaskConfig,
    clip_min: float,
    clip_max: float,
    aggregation_method: str,
) -> Tuple[
    List[BaseClient],
    FederatedRobustScaler,
    Dict[str, Any],
    int,
    Tuple[np.ndarray, np.ndarray],              # unseen_val (scaled)
    Tuple[np.ndarray, np.ndarray],              # unseen_test (scaled)
]:
    
    #Loads: seen datasets (for clients), unseen dataset (for evaluation only)

    #Fits FederatedRobustScaler using ONLY seen TRAIN benign.

    #Returns: clients, fed_scaler, scaler_state, input_dim, unseen_val_scaled, unseen_test_scaled
    
    seen_splits = {ds: load_split_pickles(ds, splits_dir) for ds in seen_datasets}
    unseen_sp = load_split_pickles(unseen_dataset, splits_dir)  # NEW

    input_dim = seen_splits[seen_datasets[0]].X_train.shape[1]
    for ds in seen_datasets[1:]:
        if seen_splits[ds].X_train.shape[1] != input_dim:
            raise RuntimeError("Feature dimension mismatch across seen datasets.")
    if unseen_sp.X_train.shape[1] != input_dim:
        raise RuntimeError("Feature dimension mismatch: unseen dataset differs from seen datasets.")
    
    #basic fed scaler
    fed_scaler = FederatedRobustScaler(eps=1e-6, clip_min=clip_min, clip_max=clip_max)

    stats = []
    for ds in seen_datasets:
        sp = seen_splits[ds]
        X_benign = sp.X_train[sp.y_train == 0]
        if len(X_benign) == 0:
            raise RuntimeError(f"{ds}: no benign samples in TRAIN.")
        stats.append(fed_scaler.compute_client_stats(X_benign))
    fed_scaler.fit_from_client_stats(stats)

    # Build unseen val/test (scaled) for global eval
    unseen_val_scaled = (fed_scaler.transform(unseen_sp.X_val), np.asarray(unseen_sp.y_val).reshape(-1))
    unseen_test_scaled = (fed_scaler.transform(unseen_sp.X_test), np.asarray(unseen_sp.y_test).reshape(-1))

    clients: List[BaseClient] = []
    for i, ds in enumerate(seen_datasets):
        sp = seen_splits[ds]

        X_train_b = sp.X_train[sp.y_train == 0]
        X_train_b_s = fed_scaler.transform(X_train_b)

        X_val_s = fed_scaler.transform(sp.X_val)
        X_test_s = fed_scaler.transform(sp.X_test)

        task = VaeTask(
            optim_cfg=optim_cfg,
            task_cfg=task_cfg,
            X_train_benign_scaled=X_train_b_s,
            X_val_scaled=X_val_s,
            y_val=sp.y_val,
            X_test_scaled=X_test_s,
            y_test=sp.y_test,
            aggregation_method=aggregation_method,
            unseen_val=unseen_val_scaled,
            unseen_test=unseen_test_scaled,
        )

        clients.append(BaseClient(client_id=i, device=device, task=task, meta={"dataset": ds}))

    return clients, fed_scaler, fed_scaler.state_dict(), int(input_dim), unseen_val_scaled, unseen_test_scaled



####################################################
# End of seed eval (seen val/test + unseen val/test)
####################################################
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



def objective_factory(
    *,
    aggregation_method: str,
    hpo_seeds: List[int],
    seen_datasets: List[str],
    unseen_dataset: str,
    splits_dir: str,
    device: torch.device,
    hpo_cfg: HPOConfig,
):
    aggregation_method_norm = str(aggregation_method).lower().strip()

    def objective(trial: optuna.Trial) -> float:
        ####################################################
        # Sample ONE hyperparam configuration θ
        #####################################################

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

        # Methodspecific
        server_lr = server_beta1 = server_beta2 = server_tau = None
        server_bias_correction = None
        prox_mu = None

        if aggregation_method_norm == "fedadam":
            server_lr = trial.suggest_float("server_lr", 1e-3, 2e-2, log=True)
            server_tau = trial.suggest_float("server_tau", 1e-4, 5e-3, log=True)
            server_beta1 = trial.suggest_float("server_beta1", 0.85, 0.95)
            server_beta2 = trial.suggest_float("server_beta2", 0.97, 0.995)
            server_bias_correction = True

        elif aggregation_method_norm == "fedprox":
            prox_mu = trial.suggest_float("prox_mu", 1e-4, 1e-1, log=True)

        elif aggregation_method_norm in {"fedavg", "scaffold"}:
            pass
        else:
            raise ValueError(f"Unknown aggregation_method='{aggregation_method_norm}'")

        #####################################################
        # Fixed / derived configuration
        #####################################################
        rounds = 120
        early_stop_patience = max(2, int(round(10 / local_epochs)))

        optim_cfg = TaskOptimConfig(
            lr=float(lr),
            weight_decay=float(weight_decay),
            grad_clip=grad_clip,
            optimizer=str(optimizer),
            sgd_momentum=float(sgd_momentum),
            sgd_nesterov=bool(sgd_nesterov),
        )

        if aggregation_method_norm == "fedprox":
            task_cfg = VaeTaskConfig(
                local_epochs=int(local_epochs),
                batch_size=int(batch_size),
                beta=float(beta),
                beta_warmup_rounds=int(beta_warmup_rounds),
                recon_type=str(recon_type),
                score_type=str(score_type),
                prox_mu=float(prox_mu),
            )
        else:
            task_cfg = VaeTaskConfig(
                local_epochs=int(local_epochs),
                batch_size=int(batch_size),
                beta=float(beta),
                beta_warmup_rounds=int(beta_warmup_rounds),
                recon_type=str(recon_type),
                score_type=str(score_type),
            )

        server_cfg = ServerConfig(
            rounds=int(rounds),
            eval_every=1,
            early_stop_patience=int(early_stop_patience),
            early_stop_min_delta=4e-4,
        )

        trial.set_user_attr("aggregation_method", aggregation_method_norm)
        trial.set_user_attr("hpo_seeds", list(hpo_seeds))

        # architecture
        trial.set_user_attr("n_layers", int(n_layers))
        trial.set_user_attr("first_layer", int(first))
        trial.set_user_attr("hidden_dims", list(hidden_dims))
        for i in range(1, n_layers):
            trial.set_user_attr(f"layer_{i}_dropoff", float(dims[i] / dims[i - 1]))

        # model/task/optim
        trial.set_user_attr("latent_dim", int(latent_dim))
        trial.set_user_attr("dropout", float(dropout))
        trial.set_user_attr("activation", str(activation))
        trial.set_user_attr("weight_decay", float(weight_decay))
        trial.set_user_attr("grad_clip", grad_clip)
        trial.set_user_attr("optimizer", str(optimizer))
        trial.set_user_attr("lr", float(lr))
        trial.set_user_attr("sgd_momentum", float(sgd_momentum))
        trial.set_user_attr("sgd_nesterov", bool(sgd_nesterov))
        trial.set_user_attr("beta", float(beta))
        trial.set_user_attr("beta_warmup_rounds", int(beta_warmup_rounds))
        trial.set_user_attr("batch_size", int(batch_size))
        trial.set_user_attr("local_epochs", int(local_epochs))
        trial.set_user_attr("recon_type", str(recon_type))
        trial.set_user_attr("score_type", str(score_type))

        # clipping
        trial.set_user_attr("clip_abs", float(clip_abs))
        trial.set_user_attr("clip_min", float(clip_min))
        trial.set_user_attr("clip_max", float(clip_max))

        # fixed / derived values
        trial.set_user_attr("rounds", int(rounds))
        trial.set_user_attr("early_stop_patience", int(early_stop_patience))

        # method-specific
        if aggregation_method_norm == "fedadam":
            trial.set_user_attr("server_lr", float(server_lr))
            trial.set_user_attr("server_tau", float(server_tau))
            trial.set_user_attr("server_beta1", float(server_beta1))
            trial.set_user_attr("server_beta2", float(server_beta2))
            trial.set_user_attr("server_bias_correction", bool(server_bias_correction))
        elif aggregation_method_norm == "fedprox":
            trial.set_user_attr("prox_mu", float(prox_mu))

        ####################################################
        # Evaluate θ across seeds
        ####################################################

        seed_seen_objs: List[float] = []
        seed_unseen_val_objs: List[float] = []

        metric_sums: Dict[str, float] = {}
        n_metric = 0

        for seed_idx, seed in enumerate(hpo_seeds):
            set_seed(seed)

            clients, fed_scaler, _scaler_state, _input_dim, unseen_val_s, unseen_test_s = build_clients_and_scaler(
                aggregation_method=aggregation_method_norm,
                seen_datasets=seen_datasets,
                unseen_dataset=unseen_dataset,
                splits_dir=splits_dir,
                device=device,
                optim_cfg=optim_cfg,
                task_cfg=task_cfg,
                clip_min=float(clip_min),
                clip_max=float(clip_max),
            )

            model = clients[0].task.make_model(
                hidden_dims=hidden_dims,
                latent_dim=int(latent_dim),
                dropout=float(dropout),
                activation=str(activation),
            )

            if aggregation_method_norm == "fedavg":
                strategy = FedAvgStrategy()
            elif aggregation_method_norm == "fedprox":
                strategy = FedAvgStrategy()  # FedProx is client side penalty
            elif aggregation_method_norm == "scaffold":
                strategy = ScaffoldStrategy()
            elif aggregation_method_norm == "fedadam":
                assert (
                    server_lr is not None
                    and server_tau is not None
                    and server_beta1 is not None
                    and server_beta2 is not None
                )
                strategy = FedAdamStrategy(
                    cfg=FedAdamConfig(
                        server_lr=float(server_lr),
                        beta1=float(server_beta1),
                        beta2=float(server_beta2),
                        tau=float(server_tau),
                        use_bias_correction=bool(server_bias_correction),
                    )
                )
            else:
                raise ValueError("unreachable")

            server = BaseServer(model=model, strategy=strategy, device=device, cfg=server_cfg)

            use_trial = trial if seed_idx == 0 else None

            model_trained, history, best_mean_val = run_federated_train(
                clients=clients,
                server=server,
                trial=use_trial,
                metric_key="auroc",
                eval_split="val",
                seen_datasets=seen_datasets,
                unseen_dataset=unseen_dataset,
                unseen_val_scaled=unseen_val_s,
                unseen_test_scaled=unseen_test_s,
                device=device,
                score_type=task_cfg.score_type,
                aggregation_method=aggregation_method_norm,
            )

            #  Objective-> seen VAL aggregate 
            seen_val_aurocs = [float(c.evaluate(model_trained, "val")["auroc"]) for c in clients]
            seen_obj_seed = float(objective_reduce(seen_val_aurocs, mode=hpo_cfg.objective_mode))
            seed_seen_objs.append(seen_obj_seed)

            # Unseen VAL AUROC (logged only)
            Xuv, yuv = unseen_val_s
            unseen_val_obj_seed = float(
                eval_auroc_arrays(
                    model=model_trained,
                    device=device,
                    X_np=Xuv,
                    y_np=yuv,
                    score_type=task_cfg.score_type,
                )
            )
            seed_unseen_val_objs.append(unseen_val_obj_seed)

            seed_metrics = eval_all_datasets_end_of_seed(
                model=model_trained,
                clients=clients,
                seen_datasets=seen_datasets,
                unseen_dataset=unseen_dataset,
                unseen_val_scaled=unseen_val_s,
                unseen_test_scaled=unseen_test_s,
                device=device,
                score_type=task_cfg.score_type,
            )

            for k, v in seed_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            n_metric += 1

        seen_mean = float(np.mean(seed_seen_objs))
        seen_std = float(np.std(seed_seen_objs))
        unseen_val_mean = float(np.mean(seed_unseen_val_objs))
        unseen_val_std = float(np.std(seed_unseen_val_objs))

        trial.set_user_attr("seed_seen_obj_mean", seen_mean)
        trial.set_user_attr("seed_seen_obj_std", seen_std)
        trial.set_user_attr("seed_unseen_val_obj_mean", unseen_val_mean)
        trial.set_user_attr("seed_unseen_val_obj_std", unseen_val_std)

        if n_metric > 0:
            metric_means = {k: metric_sums[k] / float(n_metric) for k in metric_sums.keys()}
            for k, v in metric_means.items():
                trial.set_user_attr(k, float(v))

            seen_test_vals = [metric_means.get(f"{ds}__test_auroc", float("nan")) for ds in seen_datasets]
            if all(np.isfinite(seen_test_vals)):
                trial.set_user_attr("seen_test_mean", float(np.mean(seen_test_vals)))
                trial.set_user_attr("seen_test_min", float(np.min(seen_test_vals)))

            trial.set_user_attr(
                "unseen_test_auroc",
                float(metric_means.get("unseen__test_auroc", float("nan")))
            )

        return seen_mean

    return objective


def _materialize_best_params(best_trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """
    Merge sampled params + logged user_attrs into one complete config
    that final_train_and_eval can use
    """
    params = dict(best_trial.params)
    attrs = dict(best_trial.user_attrs)

    merged: Dict[str, Any] = dict(params)

    # common fields maybe needed later even if not sampled
    fallback_keys = [
        "optimizer",
        "clip_abs",
        "clip_min",
        "clip_max",
        "rounds",
        "early_stop_patience",
        "weight_decay",
        "grad_clip",
        "lr",
        "sgd_momentum",
        "sgd_nesterov",
        "beta",
        "beta_warmup_rounds",
        "batch_size",
        "local_epochs",
        "recon_type",
        "score_type",
        "latent_dim",
        "dropout",
        "activation",
        "n_layers",
        "first_layer",
    ]
    for k in fallback_keys:
        if k not in merged and k in attrs:
            merged[k] = attrs[k]

    if "clip_min" not in merged or "clip_max" not in merged:
        if "clip_abs" in merged:
            merged["clip_min"] = -float(merged["clip_abs"])
            merged["clip_max"] = float(merged["clip_abs"])

    if "early_stop_patience" not in merged and "local_epochs" in merged:
        merged["early_stop_patience"] = max(2, int(round(10 / int(merged["local_epochs"]))))

    # method-specific
    for k in ["prox_mu", "server_lr", "server_tau", "server_beta1", "server_beta2", "server_bias_correction"]:
        if k not in merged and k in attrs:
            merged[k] = attrs[k]

    # layer dropoffs fallback
    n_layers = int(merged["n_layers"])
    for i in range(1, n_layers):
        key = f"layer_{i}_dropoff"
        if key not in merged and key in attrs:
            merged[key] = attrs[key]

    return merged


def run_hpo(
    *,
    aggregation_method: str,
    hpo_seeds: List[int],
    seen_datasets: List[str],
    unseen_dataset: str,
    splits_dir: str,
    out_dir: str,
    device: torch.device,
    hpo_cfg: HPOConfig,
):
    os.makedirs(out_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=hpo_cfg.seed)
    pruner = make_pruner(hpo_cfg.pruner)

    storage = f"sqlite:///{os.path.join(out_dir, 'study.db')}"
    seeds_tag = "-".join(map(str, hpo_seeds))

    study_name = (
        f"fed_vae_hpo_multiobj__{aggregation_method}"
        f"__unseen_{unseen_dataset}__seen_{'_'.join(seen_datasets)}"
        f"__hpseeds_{seeds_tag}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    obj = objective_factory(
        aggregation_method=aggregation_method,
        hpo_seeds=hpo_seeds,
        seen_datasets=seen_datasets,
        unseen_dataset=unseen_dataset,
        splits_dir=splits_dir,
        device=device,
        hpo_cfg=hpo_cfg,
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

    best_trial = study.best_trial
    best_params_full = _materialize_best_params(best_trial)

    best_out = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "best_params_full": best_params_full,
    }

    with open(os.path.join(out_dir, "best_trial.json"), "w") as f:
        json.dump(best_out, f, indent=2)

    # Save FULL config, not just study.best_params
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best_params_full, f, indent=2)

    df = study.trials_dataframe(attrs=("number", "values", "params", "user_attrs", "state"))
    df.to_csv(os.path.join(out_dir, "trials.csv"), index=False)

    return study
