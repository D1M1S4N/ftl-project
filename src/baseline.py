import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import MSLesSegDataset, load_client_data, load_global_test_data
from src.metrics import calculate_classification_metrics, calculate_patient_level_metrics
from src.model import get_resnet18_2d


METRIC_COLUMNS = ["accuracy", "f1_score", "sensitivity", "specificity"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def make_loader(dataframe, config, shuffle):
    base_path = config["data"].get("base_path", ".")
    dataset = MSLesSegDataset(
        dataframe=dataframe,
        base_path=base_path,
        transform=build_transform(),
    )
    return DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=shuffle)


def load_all_training_data():
    frames = []
    for client_id in range(1, 5):
        csv_path = Path("data") / f"hospital_{client_id}_train.csv"
        frames.append(pd.read_csv(csv_path))
    return pd.concat(frames, ignore_index=True)


def load_baseline_data(config, mode, target_client=None, modality=None):
    if mode == "centralized":
        df = load_all_training_data()
        return "centralized_all", df, make_loader(df, config, shuffle=True)

    if mode == "modality":
        if modality not in {"FLAIR", "T1w"}:
            raise ValueError("--modality debe ser FLAIR o T1w cuando mode=modality.")
        df = load_all_training_data()
        df = df[df["modality"] == modality].reset_index(drop=True)
        return f"local_only_{modality}", df, make_loader(df, config, shuffle=True)

    if mode == "local":
        if target_client is None:
            raise ValueError("--client es obligatorio cuando mode=local.")
        trainloader = load_client_data(target_client, config, split="train")
        df = pd.read_csv(Path("data") / f"hospital_{target_client}_train.csv")
        train_ids = set(trainloader.dataset.dataframe["patient_id"])
        df = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
        modality_label = df["modality"].iloc[0]
        return f"local_only_hospital_{target_client}_{modality_label}", df, trainloader

    raise ValueError(f"Modo de baseline no soportado: {mode}")


def evaluate_model(model, config, device):
    testloader, test_df = load_global_test_data(config, return_dataframe=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")

    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs, dim=1)

            total_loss += criterion(outputs, labels).item()
            all_scores.extend(probabilities[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del images, labels, outputs, probabilities

    labels_np = np.asarray(all_labels)
    preds_np = np.asarray(all_preds)
    scores_np = np.asarray(all_scores)

    metrics = {
        "loss": total_loss / len(testloader.dataset),
        **{
            f"global_{name}": value
            for name, value in calculate_classification_metrics(labels_np, preds_np).items()
        },
        **{
            f"patient_global_{name}": value
            for name, value in calculate_patient_level_metrics(
                test_df,
                labels_np,
                y_score=scores_np,
            ).items()
        },
    }

    for modality in sorted(test_df["modality"].unique()):
        mask = test_df["modality"].to_numpy() == modality
        slice_metrics = calculate_classification_metrics(labels_np[mask], preds_np[mask])
        patient_metrics = calculate_patient_level_metrics(
            test_df.loc[mask],
            labels_np[mask],
            y_score=scores_np[mask],
        )
        metrics.update({
            f"modality_{modality}_{name}": value
            for name, value in slice_metrics.items()
        })
        metrics.update({
            f"patient_modality_{modality}_{name}": value
            for name, value in patient_metrics.items()
        })

    for hospital_id in sorted(test_df["hospital_id"].unique()):
        mask = test_df["hospital_id"].to_numpy() == hospital_id
        slice_metrics = calculate_classification_metrics(labels_np[mask], preds_np[mask])
        patient_metrics = calculate_patient_level_metrics(
            test_df.loc[mask],
            labels_np[mask],
            y_score=scores_np[mask],
        )
        metrics.update({
            f"hospital_{hospital_id}_{name}": value
            for name, value in slice_metrics.items()
        })
        metrics.update({
            f"patient_hospital_{hospital_id}_{name}": value
            for name, value in patient_metrics.items()
        })

    return metrics


def write_results(rows, output_dir, run_id):
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"baselines_{run_id}_metrics.csv"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def train_baseline(config, mode="centralized", target_client=None, modality=None, epochs=None):
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_name, train_df, trainloader = load_baseline_data(
        config,
        mode=mode,
        target_client=target_client,
        modality=modality,
    )

    if epochs is None:
        epochs = (
            config["client_training"]["local_epochs"]
            * config["federated_learning"].get("num_rounds", 10)
        )

    print(f"--- Iniciando baseline: {baseline_name} en {device} ---")
    print(f"Pacientes train: {train_df['patient_id'].nunique()} | Slices train: {len(train_df)}")

    model = get_resnet18_2d(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["client_training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()

    start_time = datetime.now()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del images, labels, outputs, loss

        print(f"Epoca {epoch + 1}/{epochs} - Loss: {total_loss / len(trainloader):.4f}")

    elapsed_seconds = (datetime.now() - start_time).total_seconds()
    metrics = evaluate_model(model, config, device)

    row = {
        "run_id": start_time.strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline_name,
        "mode": mode,
        "target_client": target_client if target_client is not None else "",
        "modality": modality or "",
        "epochs": epochs,
        "learning_rate": config["client_training"]["learning_rate"],
        "batch_size": config["data"]["batch_size"],
        "train_patients": train_df["patient_id"].nunique(),
        "train_slices": len(train_df),
        "elapsed_seconds": elapsed_seconds,
        **metrics,
    }

    print("\n--- Resultados finales sobre global_test.csv ---")
    for key in ("global_accuracy", "global_f1_score", "patient_global_accuracy", "patient_global_f1_score"):
        print(f"{key}: {row[key]:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def build_baseline_specs(args):
    if args.all_baselines:
        return [
            {"mode": "centralized"},
            {"mode": "modality", "modality": "FLAIR"},
            {"mode": "modality", "modality": "T1w"},
            {"mode": "local", "target_client": 1},
            {"mode": "local", "target_client": 2},
            {"mode": "local", "target_client": 3},
            {"mode": "local", "target_client": 4},
        ]

    return [{
        "mode": args.mode,
        "target_client": args.client,
        "modality": args.modality,
    }]


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena baselines centralizados y local-only.")
    parser.add_argument("--config", type=str, default="configs/experiment_fedavg.yaml")
    parser.add_argument("--mode", type=str, choices=["centralized", "local", "modality"], default="centralized")
    parser.add_argument("--client", type=int, default=None, help="ID del hospital si mode=local (1-4).")
    parser.add_argument("--modality", type=str, choices=["FLAIR", "T1w"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--all-baselines", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    for spec in build_baseline_specs(args):
        rows.append(train_baseline(config, epochs=args.epochs, **spec))

    output_path = write_results(rows, args.output_dir, run_id)
    print(f"\nResultados guardados en {output_path}")
