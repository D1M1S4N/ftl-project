import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


METRIC_COLUMNS = ["accuracy", "f1_score", "sensitivity", "specificity"]


def ensure_dirs(output_dir):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    raw_dir = output_dir / "raw"
    for directory in (tables_dir, figures_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir, raw_dir


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_configs(config_dir):
    rows = []
    for config_path in sorted(config_dir.glob("experiment_*.yaml")):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        rows.append({
            "experiment_name": config.get("experiment_name", config_path.stem),
            "config_file": str(config_path),
            "strategy": config["federated_learning"]["strategy"],
            "num_rounds": config["federated_learning"]["num_rounds"],
            "local_epochs": config["client_training"]["local_epochs"],
            "batch_size": config["data"]["batch_size"],
            "learning_rate": config["client_training"]["learning_rate"],
            "optimizer": config["client_training"]["optimizer"],
            "aggregation_method": config["federated_learning"].get("aggregation_method", ""),
            "proximal_mu": config["federated_learning"].get("proximal_mu", 0.0),
            "model": config["model"]["architecture"],
            "in_channels": config["model"]["in_channels"],
            "num_classes": config["model"]["num_classes"],
        })
    config_df = pd.DataFrame(rows)
    if config_df.empty:
        return config_df

    config_df["canonical_rank"] = 0
    config_df = (
        config_df
        .sort_values(["strategy", "proximal_mu", "canonical_rank", "config_file"])
        .drop_duplicates(subset=["strategy", "proximal_mu"], keep="first")
        .drop(columns=["canonical_rank"])
        .sort_values("config_file")
    )
    return config_df


def build_methodology_diagram(figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axis("off")

    hospital_boxes = [
        ("Hospital 1\nFLAIR", 0.08, 0.72, "#3B82F6"),
        ("Hospital 2\nFLAIR", 0.08, 0.46, "#3B82F6"),
        ("Hospital 3\nT1w", 0.08, 0.20, "#10B981"),
        ("Hospital 4\nT1w", 0.08, -0.06, "#10B981"),
    ]

    for label, x, y, color in hospital_boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.22, 0.16, facecolor=color, alpha=0.18,
                                   edgecolor=color, linewidth=2))
        ax.text(x + 0.11, y + 0.08, label, ha="center", va="center",
                fontsize=12, weight="bold", color="#111827")
        ax.annotate("", xy=(0.55, 0.43), xytext=(x + 0.22, y + 0.08),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.8))

    ax.add_patch(plt.Rectangle((0.55, 0.31), 0.26, 0.24, facecolor="#F59E0B",
                               alpha=0.18, edgecolor="#F59E0B", linewidth=2))
    ax.text(0.68, 0.43, "Servidor federado\nFedAvg / FedProx",
            ha="center", va="center", fontsize=12, weight="bold", color="#111827")

    ax.annotate("", xy=(0.68, 0.18), xytext=(0.68, 0.31),
                arrowprops=dict(arrowstyle="->", color="#374151", lw=1.8))
    ax.add_patch(plt.Rectangle((0.50, -0.05), 0.36, 0.23, facecolor="#6B7280",
                               alpha=0.12, edgecolor="#6B7280", linewidth=2))
    ax.text(0.68, 0.065, "Evaluacion global\nEM vs control\nFLAIR + T1w",
            ha="center", va="center", fontsize=12, weight="bold", color="#111827")

    ax.text(0.08, 0.94, "Entrenamiento local por hospital",
            fontsize=13, weight="bold", color="#111827")
    ax.text(0.50, 0.94, "Agregacion de pesos",
            fontsize=13, weight="bold", color="#111827")
    ax.text(0.50, 0.24, "Test centralizado sin solape de pacientes",
            fontsize=11, color="#374151")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1.02)

    save_figure(fig, figures_dir / "methodology_diagram.png")


def build_dataset_assets(data_dir, tables_dir, figures_dir):
    summary = pd.read_csv(data_dir / "hospital_summary.csv")
    summary.to_csv(tables_dir / "dataset_distribution.csv", index=False)

    assignment = pd.read_csv(data_dir / "patient_assignment.csv")
    patient_summary = (
        assignment
        .groupby(["hospital_id", "modality", "label"], as_index=False)
        .size()
        .rename(columns={"size": "patients"})
    )
    patient_summary.to_csv(tables_dir / "patient_distribution.csv", index=False)

    labels = summary["hospital_id"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, summary["control_slices"], label="Control", color="#64748B")
    ax.bar(labels, summary["ms_slices"], bottom=summary["control_slices"],
           label="EM", color="#DC2626")
    for idx, row in summary.iterrows():
        ax.text(idx, row["slices"] + 80, row["modality"], ha="center", fontsize=10)
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Slices")
    ax.set_title("Distribucion de slices por hospital")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "dataset_partition.png")

    modality = (
        summary
        .groupby("modality", as_index=False)[["control_slices", "ms_slices"]]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(modality))
    width = 0.34
    ax.bar(x - width / 2, modality["control_slices"], width,
           label="Control", color="#64748B")
    ax.bar(x + width / 2, modality["ms_slices"], width,
           label="EM", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels(modality["modality"])
    ax.set_ylabel("Slices")
    ax.set_title("Distribucion por modalidad")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, figures_dir / "class_distribution_by_modality.png")


def read_metric_files(raw_dir):
    files = sorted(raw_dir.glob("*_metrics.csv"))
    frames = []
    for path in files:
        if path.name.startswith("baselines_"):
            continue
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if not df.empty and "server_round" in df.columns:
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    metrics["server_round"] = pd.to_numeric(metrics["server_round"], errors="coerce")
    metrics = metrics.dropna(subset=["server_round"])
    metrics["server_round"] = metrics["server_round"].astype(int)
    metrics = normalize_metric_columns(metrics)
    return metrics


def read_baseline_files(raw_dir):
    files = sorted(raw_dir.glob("baselines_*_metrics.csv"))
    frames = []
    for path in files:
        if path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        if not df.empty and "baseline" in df.columns:
            df["source_file"] = str(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_metric_columns(metrics):
    if "strategy_variant" not in metrics.columns:
        metrics["strategy_variant"] = metrics["strategy"]
    else:
        metrics["strategy_variant"] = metrics["strategy_variant"].fillna(metrics["strategy"])

    if "proximal_mu" not in metrics.columns:
        metrics["proximal_mu"] = np.where(metrics["strategy"] == "FedProx", np.nan, 0.0)
    else:
        metrics["proximal_mu"] = pd.to_numeric(metrics["proximal_mu"], errors="coerce")
        metrics.loc[metrics["strategy"] != "FedProx", "proximal_mu"] = 0.0

    return metrics


def latest_run_per_variant(metrics):
    rows = []
    for strategy_variant, group in metrics.groupby("strategy_variant"):
        latest_run_id = group.sort_values(["timestamp", "server_round"])["run_id"].iloc[-1]
        rows.append(group[group["run_id"] == latest_run_id])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_experiment_tables(metrics, tables_dir):
    latest = latest_run_per_variant(metrics)
    if latest.empty:
        return latest

    final_rows = (
        latest
        .sort_values(["strategy_variant", "server_round"])
        .groupby("strategy_variant", as_index=False)
        .tail(1)
    )

    summary_cols = [
        "experiment_name",
        "strategy",
        "strategy_variant",
        "proximal_mu",
        "server_round",
        "loss",
        "global_accuracy",
        "global_f1_score",
        "global_sensitivity",
        "global_specificity",
        "patient_global_accuracy",
        "patient_global_f1_score",
        "patient_global_sensitivity",
        "patient_global_specificity",
    ]
    existing_summary_cols = [col for col in summary_cols if col in final_rows.columns]
    final_rows[existing_summary_cols].to_csv(
        tables_dir / "experiment_summary.csv",
        index=False,
    )

    modality_rows = []
    for _, row in final_rows.iterrows():
        for modality in ("FLAIR", "T1w"):
            entry = {
                "experiment_name": row["experiment_name"],
                "strategy": row["strategy"],
                "strategy_variant": row["strategy_variant"],
                "proximal_mu": row["proximal_mu"],
                "modality": modality,
            }
            for metric in METRIC_COLUMNS:
                column = f"modality_{modality}_{metric}"
                if column in row:
                    entry[metric] = row[column]
            modality_rows.append(entry)
    pd.DataFrame(modality_rows).to_csv(
        tables_dir / "per_modality_metrics.csv",
        index=False,
    )

    patient_modality_rows = []
    for _, row in final_rows.iterrows():
        for modality in ("FLAIR", "T1w"):
            entry = {
                "experiment_name": row["experiment_name"],
                "strategy": row["strategy"],
                "strategy_variant": row["strategy_variant"],
                "proximal_mu": row["proximal_mu"],
                "modality": modality,
            }
            for metric in METRIC_COLUMNS:
                column = f"patient_modality_{modality}_{metric}"
                entry[metric] = row[column] if column in row else np.nan
            patient_modality_rows.append(entry)
    pd.DataFrame(patient_modality_rows).to_csv(
        tables_dir / "per_modality_patient_metrics.csv",
        index=False,
    )

    hospital_rows = []
    for _, row in final_rows.iterrows():
        for hospital_id in (1, 2, 3, 4):
            entry = {
                "experiment_name": row["experiment_name"],
                "strategy": row["strategy"],
                "strategy_variant": row["strategy_variant"],
                "proximal_mu": row["proximal_mu"],
                "hospital_id": hospital_id,
            }
            for metric in METRIC_COLUMNS:
                column = f"hospital_{hospital_id}_{metric}"
                if column in row:
                    entry[metric] = row[column]
            hospital_rows.append(entry)
    pd.DataFrame(hospital_rows).to_csv(
        tables_dir / "per_hospital_metrics.csv",
        index=False,
    )

    patient_hospital_rows = []
    for _, row in final_rows.iterrows():
        for hospital_id in (1, 2, 3, 4):
            entry = {
                "experiment_name": row["experiment_name"],
                "strategy": row["strategy"],
                "strategy_variant": row["strategy_variant"],
                "proximal_mu": row["proximal_mu"],
                "hospital_id": hospital_id,
            }
            for metric in METRIC_COLUMNS:
                column = f"patient_hospital_{hospital_id}_{metric}"
                entry[metric] = row[column] if column in row else np.nan
            patient_hospital_rows.append(entry)
    pd.DataFrame(patient_hospital_rows).to_csv(
        tables_dir / "per_hospital_patient_metrics.csv",
        index=False,
    )

    return latest


def latest_baseline_rows(baselines):
    if baselines.empty:
        return baselines
    if "timestamp" not in baselines.columns:
        return baselines
    rows = []
    for baseline, group in baselines.groupby("baseline"):
        rows.append(group.sort_values("timestamp").tail(1))
    return pd.concat(rows, ignore_index=True)


def build_baseline_tables(baselines, federated_latest, tables_dir):
    latest_baselines = latest_baseline_rows(baselines)
    if latest_baselines.empty:
        return

    baseline_cols = [
        "baseline",
        "mode",
        "target_client",
        "modality",
        "epochs",
        "train_patients",
        "train_slices",
        "loss",
        "global_accuracy",
        "global_f1_score",
        "global_sensitivity",
        "global_specificity",
        "patient_global_accuracy",
        "patient_global_f1_score",
        "patient_global_sensitivity",
        "patient_global_specificity",
    ]
    existing_baseline_cols = [col for col in baseline_cols if col in latest_baselines.columns]
    latest_baselines[existing_baseline_cols].to_csv(
        tables_dir / "baseline_summary.csv",
        index=False,
    )

    modality_rows = []
    for _, row in latest_baselines.iterrows():
        for modality in ("FLAIR", "T1w"):
            entry = {
                "baseline": row["baseline"],
                "mode": row["mode"],
                "train_modality": row.get("modality", ""),
                "eval_modality": modality,
            }
            for metric in METRIC_COLUMNS:
                column = f"patient_modality_{modality}_{metric}"
                if column in row:
                    entry[metric] = row[column]
            modality_rows.append(entry)
    pd.DataFrame(modality_rows).to_csv(
        tables_dir / "baseline_patient_modality_metrics.csv",
        index=False,
    )

    comparison_rows = []
    for _, row in latest_baselines.iterrows():
        comparison_rows.append({
            "method": row["baseline"],
            "family": "baseline",
            "round_or_epochs": row.get("epochs", ""),
            "loss": row.get("loss", np.nan),
            "global_accuracy": row.get("global_accuracy", np.nan),
            "global_f1_score": row.get("global_f1_score", np.nan),
            "patient_global_accuracy": row.get("patient_global_accuracy", np.nan),
            "patient_global_f1_score": row.get("patient_global_f1_score", np.nan),
        })

    if not federated_latest.empty:
        final_rows = (
            federated_latest
            .sort_values(["strategy_variant", "server_round"])
            .groupby("strategy_variant", as_index=False)
            .tail(1)
        )
        for _, row in final_rows.iterrows():
            comparison_rows.append({
                "method": row["strategy_variant"],
                "family": "federated",
                "round_or_epochs": row.get("server_round", ""),
                "loss": row.get("loss", np.nan),
                "global_accuracy": row.get("global_accuracy", np.nan),
                "global_f1_score": row.get("global_f1_score", np.nan),
                "patient_global_accuracy": row.get("patient_global_accuracy", np.nan),
                "patient_global_f1_score": row.get("patient_global_f1_score", np.nan),
            })

    pd.DataFrame(comparison_rows).to_csv(
        tables_dir / "method_comparison.csv",
        index=False,
    )


def plot_metric_curve(metrics, metric_column, ylabel, title, output_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for strategy_variant, group in metrics.groupby("strategy_variant"):
        group = group.sort_values("server_round")
        ax.plot(group["server_round"], group[metric_column],
                marker="o", linewidth=2, label=strategy_variant)
    ax.set_xlabel("Ronda federada")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def build_experiment_figures(metrics, figures_dir):
    latest = latest_run_per_variant(metrics)
    if latest.empty:
        return

    if "global_accuracy" in latest:
        plot_metric_curve(
            latest,
            "global_accuracy",
            "Accuracy",
            "Accuracy global por ronda",
            figures_dir / "fedavg_vs_fedprox_accuracy.png",
        )

    if "global_f1_score" in latest:
        plot_metric_curve(
            latest,
            "global_f1_score",
            "F1-score",
            "F1 global por ronda",
            figures_dir / "fedavg_vs_fedprox_f1.png",
        )

    if "patient_global_f1_score" in latest:
        plot_metric_curve(
            latest,
            "patient_global_f1_score",
            "F1-score por paciente",
            "F1 global por paciente y ronda",
            figures_dir / "fedavg_vs_fedprox_patient_f1.png",
        )

    final_rows = (
        latest
        .sort_values(["strategy_variant", "server_round"])
        .groupby("strategy_variant", as_index=False)
        .tail(1)
    )

    modality_records = []
    for _, row in final_rows.iterrows():
        for modality in ("FLAIR", "T1w"):
            column = f"modality_{modality}_f1_score"
            if column in row:
                modality_records.append({
                    "strategy_variant": row["strategy_variant"],
                    "modality": modality,
                    "f1_score": row[column],
                })

    if modality_records:
        modality_df = pd.DataFrame(modality_records)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        strategies = list(modality_df["strategy_variant"].unique())
        modalities = ["FLAIR", "T1w"]
        x = np.arange(len(modalities))
        width = 0.8 / max(len(strategies), 1)
        for idx, strategy_variant in enumerate(strategies):
            subset = modality_df[modality_df["strategy_variant"] == strategy_variant]
            values = [
                subset.loc[subset["modality"] == modality, "f1_score"].iloc[0]
                if not subset.loc[subset["modality"] == modality].empty else np.nan
                for modality in modalities
            ]
            ax.bar(x - 0.4 + width / 2 + idx * width, values, width, label=strategy_variant)
        ax.set_xticks(x)
        ax.set_xticklabels(modalities)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-score")
        ax.set_title("Comparacion final por modalidad")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        save_figure(fig, figures_dir / "modality_comparison.png")


def build_assets(project_root, output_dir):
    tables_dir, figures_dir, raw_dir = ensure_dirs(output_dir)

    config_table = load_configs(project_root / "configs")
    if not config_table.empty:
        config_table.to_csv(tables_dir / "experiment_config.csv", index=False)

    build_methodology_diagram(figures_dir)
    build_dataset_assets(project_root / "data", tables_dir, figures_dir)

    metrics = read_metric_files(raw_dir)
    baselines = read_baseline_files(raw_dir)
    federated_latest = pd.DataFrame()

    if not metrics.empty:
        federated_latest = build_experiment_tables(metrics, tables_dir)
        build_experiment_figures(federated_latest, figures_dir)

    if not baselines.empty:
        build_baseline_tables(baselines, federated_latest, tables_dir)

    if metrics.empty and baselines.empty:
        print("No hay metricas de experimentos en results/raw. Se generaron solo activos de metodologia/datos.")
        return

    print(f"Activos generados en {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Genera tablas y figuras para el informe LaTeX.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_assets(args.project_root.resolve(), args.output_dir.resolve())
