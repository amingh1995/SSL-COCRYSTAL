import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import optuna
import torch
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# -----------------------------
# 1. Metric Calculation
# -----------------------------
def calculate_classification_metrics_precision(targets, predictions):
    """Calculate binary classification metrics including precision."""
    predictions_binary = (predictions >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions_binary).ravel()
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    bacc = (tpr + tnr) / 2
    recall = recall_score(targets, predictions_binary)
    precision = precision_score(targets, predictions_binary)
    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, predictions_binary)
    return bacc, recall, auc, tpr, tnr, f1, precision


    
# -----------------------------
# 2. Reporting Function
# -----------------------------

def report_ensemble_metrics_precision(preds, labels, title="Overall"):
    """Print and return metrics as dict."""
    bacc, recall, auc, tpr, tnr, f1, precision = calculate_classification_metrics_precision(labels, preds)
    print(f"\n📊 Metrics Report ({title})")
    print(f"Balanced Accuracy : {bacc:.3f}")
    print(f"Recall (TPR)      : {recall:.3f}")
    print(f"Precision         : {precision:.3f}")
    print(f"AUC               : {auc:.3f}")
    print(f"True Neg Rate     : {tnr:.3f}")
    print(f"F1 Score          : {f1:.3f}")

    return {
        "Model": title.split(" [")[0],
        "Split": title.split(" [")[1][:-1],
        "BACC": bacc,
        "Recall": recall,
        "Precision": precision,
        "AUC": auc,
        "TNR": tnr,
        "F1": f1
    }



def plot_radar_metrics(df_metrics, models=None, metrics=None, split="test"):
    """
    Plot radar chart comparing selected models across selected metrics using actual metric values (no normalization).
    Fixed scale: 0.5–1.0 with clear gray increments every 0.1.
    """

    # --- Filter by split ---
    df = df_metrics[df_metrics["Split"] == split]

    # --- Select models and metrics ---
    if models:
        df = df[df["Model"].isin(models)]
    if metrics:
        df = df[["Model"] + metrics]
    else:
        metrics = [col for col in df.columns if col not in ["Model", "Split"] and np.issubdtype(df[col].dtype, np.number)]
        df = df[["Model"] + metrics]

    # --- Radar setup ---
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    # --- Fixed scale 0.5–1.0 ---
    lower, upper = 0.5, 1.0
    ax.set_ylim(lower, upper)

    # --- Add light gray concentric circles ---
    yticks = np.arange(lower, upper + 0.001, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks], color="gray", fontsize=9)
    ax.yaxis.grid(True, color="lightgray", linestyle="--", linewidth=0.7)
    ax.xaxis.grid(True, color="lightgray", linestyle="--", linewidth=0.7)

    # --- Plot each model ---
    for _, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Model"])
        ax.fill(angles, values, alpha=0.15)

    # --- Styling ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_title(f"Radar Comparison ({split} split)", size=14, fontweight='bold', pad=25)
    ax.legend(bbox_to_anchor=(1.25, 1.05), loc='upper left', frameon=False)

    plt.tight_layout()
    plt.show()


def plot_radar_as_bars(df_metrics, models=None, metrics=None, split="test", value_fmt="{:.3f}", figsize=(9, 6)):
    """
    Plot grouped bar chart comparing selected models across selected metrics.
    Matches the style of the radar plot (harmonious colors, strong contrast, annotated values).

    Args:
        df_metrics (pd.DataFrame): DataFrame with columns ['Model', 'Split', metric columns...]
        models (list): Models to include.
        metrics (list): Metrics to include (e.g. ['BACC', 'AUC', 'Precision', 'F1']).
        split (str): Which split to visualize ('test' or 'ext').
        value_fmt (str): Format for numeric annotation labels.
        figsize (tuple): Figure size.
    """

    # --- Filter data ---
    df = df_metrics[df_metrics["Split"] == split]
    if models:
        df = df[df["Model"].isin(models)]

    # Choose metrics if not provided
    if metrics is None:
        metrics = [col for col in df.columns if col not in ["Model", "Split"] and np.issubdtype(df[col].dtype, np.number)]

    # --- Prepare for grouped bar plotting ---
    df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

    # --- 🎨 Refined harmonious color palette (dark–light variants) ---
    custom_palette = [
        "#1f77b4", "#6baed6",   # blues
        "#2ca02c", "#98df8a",   # greens
        "#9467bd", "#c5b0d5",   # purples
        "#17becf", "#9edae5",   # teals
        "#8c564b", "#c49c94",   # neutrals
    ]
    unique_models = df["Model"].unique()
    palette = custom_palette[:len(unique_models)]

    # --- Plot ---
    sns.set(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melted,
        x="Metric",
        y="Value",
        hue="Model",
        palette=palette,
        width=0.7
    )

    # --- Annotate values on bars ---
    for container in ax.containers:
        ax.bar_label(container, fmt=value_fmt, fontsize=9, padding=3)

    # --- Formatting ---
    ax.set_title(f"Model Performance Comparison ({split} split)", fontsize=15, weight="bold", pad=15)
    ax.set_ylabel("Score", fontsize=12, labelpad=8)
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0.5, 1.0)  # same scale as radar for consistency
    ax.yaxis.grid(True, color="lightgray", linestyle="--", linewidth=0.7)
    ax.xaxis.grid(False)
    plt.xticks(fontsize=11, weight="bold")
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()


