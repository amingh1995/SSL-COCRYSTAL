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

