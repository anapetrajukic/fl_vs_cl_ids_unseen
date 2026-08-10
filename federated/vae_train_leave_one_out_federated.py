from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Literal, Tuple

import numpy as np
import torch
import optuna

from hpo.common import HPOConfig, set_seed

from federated.fed_hpo.specific.optuna_fed_vae import run_hpo, build_clients_and_scaler

from task.base import TaskOptimConfig
from federated.task.vae_task import VaeTaskConfig
from models.vae import anomaly_scores

from federated.server.server import BaseServer, ServerConfig
from federated.server.strategy.FedAvgStrategy import FedAvgStrategy
from federated.server.strategy.FedAdamStrategy import FedAdamStrategy, FedAdamConfig
from federated.server.strategy.ScaffoldStrategy import ScaffoldStrategy


AggregationMethod = Literal["scaffold", "fedavg", "fedadam", "fedprox"]


def hidden_dims_from_best_params(best_params: dict) -> tuple[int, ...]:
    n_layers = int(best_params["n_layers"])
    first = int(best_params["first_layer"])
    dims = [first]
    for i in range(1, n_layers):
        drop = float(best_params[f"layer_{i}_dropoff"])
        dims.append(max(16, int(dims[-1] * drop)))
    return tuple(dims)


def _build_server_strategy(
    *,
    aggregation_method: AggregationMethod,
    best_params: dict,
):
    if aggregation_method == "fedavg":
        return FedAvgStrategy()

    if aggregation_method == "scaffold":
        return ScaffoldStrategy()

    if aggregation_method == "fedprox":
        return FedAvgStrategy()

    if aggregation_method == "fedadam":
        server_lr = float(best_params.get("server_lr", 1e-3))
        server_tau = float(best_params.get("server_tau", 1e-3))
        server_beta1 = float(best_params.get("server_beta1", 0.9))
        server_beta2 = float(best_params.get("server_beta2", 0.99))
        server_bias_correction = bool(best_params.get("server_bias_correction", False))

        return FedAdamStrategy(
            cfg=FedAdamConfig(
                server_lr=server_lr,
                beta1=server_beta1,
                beta2=server_beta2,
                tau=server_tau,
                use_bias_correction=server_bias_correction,
            )
        )

    raise ValueError(f"Unknown aggregation_method='{aggregation_method}'")


@torch.no_grad()
def _eval_auroc_arrays(model, device, X_np, y_np, score_type: str) -> float:
    model = model.to(device).eval()
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    s = anomaly_scores(model, X, score_type=score_type).detach().cpu().numpy()
    y_np = np.asarray(y_np).reshape(-1)
    if (not np.all(np.isfinite(s))) or (len(np.unique(y_np)) < 2):
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_np, s))


def final_train_and_eval(
    *,
    aggregation_method: AggregationMethod,
    seen_datasets: list[str],
    unseen_dataset: str,
    splits_dir: str,
    out_dir: str,
    device: torch.device,
    best_params: dict,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)

    optim_cfg = TaskOptimConfig(
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
        grad_clip=best_params["grad_clip"],
        optimizer=str(best_params["optimizer"]),
        sgd_momentum=float(best_params.get("sgd_momentum", 0.0)),
        sgd_nesterov=bool(best_params.get("sgd_nesterov", False)),
    )

    if aggregation_method == "fedprox":
        task_cfg = VaeTaskConfig(
            local_epochs=int(best_params["local_epochs"]),
            batch_size=int(best_params["batch_size"]),
            beta=float(best_params["beta"]),
            beta_warmup_rounds=int(best_params["beta_warmup_rounds"]),
            recon_type=str(best_params["recon_type"]),
            score_type=str(best_params["score_type"]),
            prox_mu=float(best_params.get("prox_mu", 1e-3)),
        )
    else:
        task_cfg = VaeTaskConfig(
            local_epochs=int(best_params["local_epochs"]),
            batch_size=int(best_params["batch_size"]),
            beta=float(best_params["beta"]),
            beta_warmup_rounds=int(best_params["beta_warmup_rounds"]),
            recon_type=str(best_params["recon_type"]),
            score_type=str(best_params["score_type"]),
        )

    clients, fed_scaler, scaler_state, _input_dim, unseen_val_s, unseen_test_s = build_clients_and_scaler(
        seen_datasets=seen_datasets,
        unseen_dataset=unseen_dataset,
        splits_dir=splits_dir,
        device=device,
        optim_cfg=optim_cfg,
        task_cfg=task_cfg,
        clip_min=float(best_params["clip_min"]),
        clip_max=float(best_params["clip_max"]),
        aggregation_method=aggregation_method,
    )

    model = clients[0].task.make_model(
        hidden_dims=hidden_dims_from_best_params(best_params),
        latent_dim=int(best_params["latent_dim"]),
        dropout=float(best_params["dropout"]),
        activation=str(best_params["activation"]),
    )

    server_cfg = ServerConfig(
        rounds=int(best_params["rounds"]),
        eval_every=1,
        early_stop_patience=int(best_params["early_stop_patience"]),
        early_stop_min_delta=1e-4,
    )

    strategy = _build_server_strategy(aggregation_method=aggregation_method, best_params=best_params)
    server = BaseServer(model=model, strategy=strategy, device=device, cfg=server_cfg)

    history: List[Dict[str, Any]] = []
    for rnd in range(server_cfg.rounds):
        payload = server.client_payload()
        updates = [c.fit(server.model, round_idx=rnd, server_payload=payload) for c in clients]
        server.aggregate(updates)

        vals = [float(c.evaluate(server.model, "val")["auroc"]) for c in clients]
        mean_val = float(np.mean(vals))
        history.append({"round": rnd + 1, "val_aurocs": vals, "val_mean": mean_val})

        if server.maybe_early_stop(mean_val):
            break

    server.restore_best()
    model = server.model

    per_seen: Dict[str, Dict[str, float]] = {}
    for ds_name, c in zip(seen_datasets, clients):
        per_seen[ds_name] = {
            "val_auroc": float(c.evaluate(model, "val")["auroc"]),
            "test_auroc": float(c.evaluate(model, "test")["auroc"]),
        }

    seen_val_mean = float(np.mean([per_seen[d]["val_auroc"] for d in seen_datasets]))
    seen_test_mean = float(np.mean([per_seen[d]["test_auroc"] for d in seen_datasets]))

    Xuv, yuv = unseen_val_s
    Xut, yut = unseen_test_s
    unseen_val_auroc = _eval_auroc_arrays(model, device, Xuv, yuv, score_type=task_cfg.score_type)
    unseen_test_auroc = _eval_auroc_arrays(model, device, Xut, yut, score_type=task_cfg.score_type)


    exp_dir = os.path.join(out_dir, "final_best")
    os.makedirs(exp_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(exp_dir, "global_model.pt"))
    #with open(os.path.join(exp_dir, "fed_scaler.json"), "w") as f:
    #    json.dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in scaler_state.items()}, f, indent=2)

    scaler_state = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in fed_scaler.state_dict().items()
    }
    with open(os.path.join(exp_dir, "robust_scaler.json"), "w") as f:
        json.dump(scaler_state, f, indent=2)

    results = {
        "seed": seed,
        "aggregation_method": aggregation_method,
        "seen_datasets": seen_datasets,
        "unseen_dataset": unseen_dataset,
        "best_params": best_params,
        "per_seen": per_seen,
        "seen_val_mean": seen_val_mean,
        "seen_test_mean": seen_test_mean,
        "unseen__val_auroc": float(unseen_val_auroc),
        "unseen__test_auroc": float(unseen_test_auroc),
        "history": history,
        "no_leakage": {
            "objective": "single-objective: seen VAL aggregate",
            "scaler_fit": "seen TRAIN benign only",
            "unseen_usage": {
                "unseen_val": "not optimized; reported only",
                "unseen_test": "held-out test only (reported, not optimized)",
            },
        },
        "unseen_split": {
            "unseen_val_n": int(Xuv.shape[0]),
            "unseen_test_n": int(Xut.shape[0]),
        },
    }

    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    SPLITS_DIR = "./data/splits"
    OUT_ROOT = "./out/federated_multiseed"
    os.makedirs(OUT_ROOT, exist_ok=True)

    datasets = ["NF-BoT-IoT-v3","NF-CICIDS2018-v3", "NF-ToN-IoT-v3"]
    METHODS: List[AggregationMethod] = ["fedadam","fedavg", "scaffold","fedprox"]

    HPO_SEEDS = [42,43,44,45]  
    EVAL_SEEDS = [46, 47, 48, 49, 50, 51, 52, 53]

    #HPO_SEEDS = [45]
    #EVAL_SEEDS = [45]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results: Dict[str, Any] = {
        "methods": METHODS,
        "datasets": datasets,
        "hpo_seeds": HPO_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "runs": {},
    }

    for method in METHODS:
        all_results["runs"][method] = {}
        method_dir = os.path.join(OUT_ROOT, method)
        os.makedirs(method_dir, exist_ok=True)

        for unseen in datasets:
            seen = [d for d in datasets if d != unseen]
            fold_dir = os.path.join(method_dir, f"unseen_{unseen}")
            os.makedirs(fold_dir, exist_ok=True)

            
            hpo_cfg = HPOConfig(
                n_trials=80,
                timeout_sec=None,
                objective_mode="mean",
                pruner="median",
                seed=123,
                log_all_tests_each_trial=False,
                log_every_k_trials=1,
                save_csv_each_trial=True,
            )

            study = run_hpo(
                aggregation_method=method,
                hpo_seeds=HPO_SEEDS,
                seen_datasets=seen,
                unseen_dataset=unseen,
                splits_dir=SPLITS_DIR,
                out_dir=fold_dir,
                device=device,
                hpo_cfg=hpo_cfg,
            )

            chosen = study.best_trial

            with open(os.path.join(fold_dir, "best_params.json"), "r") as f:
                best_params = json.load(f)


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

            eval_results: Dict[str, Any] = {}
            for seed in EVAL_SEEDS:
                out_dir_seed = os.path.join(fold_dir, f"eval_seed_{seed}")
                os.makedirs(out_dir_seed, exist_ok=True)
                eval_results[str(seed)] = final_train_and_eval(
                    aggregation_method=method,
                    seen_datasets=seen,
                    unseen_dataset=unseen,
                    splits_dir=SPLITS_DIR,
                    out_dir=out_dir_seed,
                    device=device,
                    best_params=best_params,
                    seed=seed,
                )

            all_results["runs"][method][unseen] = {
                "seen_datasets": seen,
                "unseen_dataset": unseen,
                "best_trial_number": int(chosen.number),
                "best_trial_value": float(chosen.value) if chosen.value is not None else None,
                "best_params": best_params,
                "eval": eval_results,
            }

            with open(os.path.join(fold_dir, "summary.json"), "w") as f:
                json.dump(all_results["runs"][method][unseen], f, indent=2)

        with open(os.path.join(method_dir, "summary.json"), "w") as f:
            json.dump(all_results["runs"][method], f, indent=2)

    with open(os.path.join(OUT_ROOT, "summary_all_methods.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("Saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()