import os
import csv
import yaml
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# PARCHE CRÍTICO: Forzamos Localhost antes de que Flower levante Ray
os.environ["RAY_NODE_IP_ADDRESS"] = "127.0.0.1"

import numpy as np
import torch
import torch.nn as nn
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.dataset import load_global_test_data
from src.metrics import calculate_classification_metrics, calculate_patient_level_metrics
from src.model import get_resnet18_2d
from src.server import SimpleMeanFedProx

def aggregate_metrics(metrics):
    """Agrega las métricas de los clientes haciendo una media simple."""
    if not metrics:
        return {}
    
    acc = sum([m["accuracy"] for _, m in metrics]) / len(metrics)
    f1 = sum([m["f1_score"] for _, m in metrics]) / len(metrics)
    sens = sum([m["sensitivity"] for _, m in metrics]) / len(metrics)
    spec = sum([m["specificity"] for _, m in metrics]) / len(metrics)

    return {"accuracy": acc, "f1_score": f1, "sensitivity": sens, "specificity": spec}

def _resolve_config_path(run_config):
    """Resuelve el YAML tanto para run_sim.py como para flwr run."""
    raw_path = os.environ.get("FL_CONFIG_PATH") or run_config.get(
        "config_path",
        "configs/experiment_fedavg.yaml",
    )
    path = Path(raw_path)
    candidates = [path if path.is_absolute() else Path.cwd() / path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / "configs" / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No se encuentra el archivo de configuración: {raw_path}")

def _select_server_eval_device(config):
    """Usa CPU por defecto para no competir con los clientes por VRAM."""
    requested = config.get("server_evaluation", {}).get("device", "cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _build_metrics_logger(config):
    """Prepara el CSV donde se guardan las métricas globales por ronda."""
    output_dir = Path(config.get("results", {}).get("output_dir", "results"))
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = config.get("experiment_name", "experiment")
    strategy_name = config["federated_learning"]["strategy"]
    proximal_mu = config["federated_learning"].get("proximal_mu", 0.0)
    if strategy_name == "FedProx":
        mu_label = str(proximal_mu).replace(".", "p")
        strategy_variant = f"FedProx_mu_{mu_label}"
    else:
        strategy_variant = strategy_name

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = raw_dir / f"{experiment_name}_{strategy_variant}_{run_id}_metrics.csv"

    def write_row(server_round, loss, metrics):
        row = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "experiment_name": experiment_name,
            "strategy": strategy_name,
            "strategy_variant": strategy_variant,
            "proximal_mu": proximal_mu,
            "server_round": server_round,
            "loss": loss,
            **metrics,
        }

        file_exists = log_path.exists()
        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    return write_row, log_path

def _build_global_evaluate_fn(config):
    """Crea la evaluación centralizada sobre global_test.csv."""
    device = _select_server_eval_device(config)
    testloader, test_df = load_global_test_data(config, return_dataframe=True)
    model = get_resnet18_2d(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    write_metrics_row, log_path = _build_metrics_logger(config)
    print(f"[Servidor] Métricas globales: {log_path}")

    def evaluate(server_round, parameters, eval_config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        total_loss = 0.0
        all_preds, all_labels, all_scores = [], [], []

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                total_loss += criterion(outputs, labels).item()

                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                all_scores.extend(probabilities[:, 1].cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                del images, labels, outputs, probabilities

        avg_loss = total_loss / len(testloader.dataset)
        labels_np = np.asarray(all_labels)
        preds_np = np.asarray(all_preds)
        scores_np = np.asarray(all_scores)

        metrics = {
            f"global_{name}": value
            for name, value in calculate_classification_metrics(labels_np, preds_np).items()
        }
        metrics.update({
            f"patient_global_{name}": value
            for name, value in calculate_patient_level_metrics(
                test_df,
                labels_np,
                y_score=scores_np,
            ).items()
        })

        for modality in sorted(test_df["modality"].unique()):
            mask = test_df["modality"].to_numpy() == modality
            modality_metrics = calculate_classification_metrics(labels_np[mask], preds_np[mask])
            metrics.update({
                f"modality_{modality}_{name}": value
                for name, value in modality_metrics.items()
            })
            patient_modality_metrics = calculate_patient_level_metrics(
                test_df.loc[mask],
                labels_np[mask],
                y_score=scores_np[mask],
            )
            metrics.update({
                f"patient_modality_{modality}_{name}": value
                for name, value in patient_modality_metrics.items()
            })

        for hospital_id in sorted(test_df["hospital_id"].unique()):
            mask = test_df["hospital_id"].to_numpy() == hospital_id
            hospital_metrics = calculate_classification_metrics(labels_np[mask], preds_np[mask])
            metrics.update({
                f"hospital_{hospital_id}_{name}": value
                for name, value in hospital_metrics.items()
            })
            patient_hospital_metrics = calculate_patient_level_metrics(
                test_df.loc[mask],
                labels_np[mask],
                y_score=scores_np[mask],
            )
            metrics.update({
                f"patient_hospital_{hospital_id}_{name}": value
                for name, value in patient_hospital_metrics.items()
            })

        print(
            f"[Servidor] Ronda {server_round}: "
            f"global_accuracy={metrics['global_accuracy']:.4f}, "
            f"global_f1={metrics['global_f1_score']:.4f}, "
            f"patient_global_f1={metrics['patient_global_f1_score']:.4f}, "
            f"loss={avg_loss:.4f}"
        )
        write_metrics_row(server_round, avg_loss, metrics)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, metrics

    return evaluate

def server_fn(context: Context) -> ServerAppComponents:
    """Función que Flower llama para preparar el servidor."""

    config_path = _resolve_config_path(context.run_config)
    print(f"\n--- Cargando configuración: {config_path} ---\n")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    strategy_name = config['federated_learning']['strategy']
    num_clients = config['data']['total_clients']
    num_rounds = config['federated_learning']['num_rounds']

    # 2. Configurar la Estrategia (FedProx o FedAvg vía tu clase server.py)
    # Usamos los valores del YAML para alimentar tu SimpleMeanFedProx
    mu_value = config['federated_learning'].get('proximal_mu', 0.1) if strategy_name == "FedProx" else 0.0

    evaluate_fn = None
    if config.get("server_evaluation", {}).get("enabled", True):
        evaluate_fn = _build_global_evaluate_fn(config)

    strategy = SimpleMeanFedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        proximal_mu=mu_value,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )

    # 3. Configurar el número de rondas
    server_config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=server_config)

# Instanciamos la ServerApp para flwr run
app = ServerApp(server_fn=server_fn)
