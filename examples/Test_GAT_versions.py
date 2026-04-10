import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning') 
RDLogger.DisableLog('rdApp.*')



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.data_processing_utils import create_graph_dataloader, allowable_features
from models.models_utils import create_best_model_HP
from src.train_evaluate_utils import evaluate_model_on_all_splits
from src.metrics_utils import report_ensemble_metrics_precision
from src.plot_utils import plot_radar_metrics, plot_radar_as_bars


# =============================
# Config
# =============================
DEVICE = "cpu"  # enforce CPU for reproducibility
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

# data and model path STRUCTURE
DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")
PRETRAIN_DIR = os.path.join(ROOT_DIR, "data", "Hyperparam_tuning")  # 🔴 adjust if needed
SAVE_DIR = os.path.join(ROOT_DIR, "results")  # 🔴 create this folder if not exists

os.makedirs(SAVE_DIR, exist_ok=True)


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = 'cpu'  # Override to CPU for consistency (remove if you want to use GPU/MPS)
BATCH_SIZE = 32


def main():
    # =============================
    # ====== Load Data ============
    # =============================

    # 🔴 Make sure these files exist inside DATA_DIR
    df_remaining = pd.read_csv(os.path.join(DATA_DIR, "df_train_val.csv"))
    df_test = pd.read_csv(os.path.join(DATA_DIR, "df_test_dft_full.csv"))
    df_ext = pd.read_csv(os.path.join(DATA_DIR, "test_external.csv"))

    # Stratified split
    df_train, df_val = train_test_split(
        df_remaining,
        test_size=0.15,
        stratify=df_remaining["labels"],
        random_state=104,
    )

    # =============================
    # ====== DataLoaders ==========
    # =============================

    def get_loaders(df):
        return create_graph_dataloader(
            df["smiles1"],
            df["smiles2"],
            df["labels"],
            batch_size=BATCH_SIZE,
        )

    loaders = {
        "train": get_loaders(df_train),
        "val": get_loaders(df_val),
        "test": get_loaders(df_test),
        "ext": get_loaders(df_ext),
    }

    labels_dict = {
        "train": df_train["labels"].values,
        "val": df_val["labels"].values,
        "test": df_test["labels"].values,
        "ext": df_ext["labels"].values,
    }

    # =============================
    # ====== Model Setup ==========
    # =============================

    Best_params_GAT = {
        "num_convs": 6,
        "feat_emb_dim": 300,
        "drop_ratio_conv": 0.15,
        "attention_heads": 3,
        "fc_layer_sizes": [400, 250],
        "problem_type": "binary_classification",
        "num_classes": 2,
    }

    model = create_best_model_HP(
        params=Best_params_GAT,
        in_features_size=1,
        Data_Type_name="GAT",
        batch_size=BATCH_SIZE,
        problem_type="binary_classification",
        allowable_features=allowable_features,
        train_loader=loaders["train"],
        adj_weight_ratio=2,
        device=DEVICE,
        mode_proj="default",
        mode="pair",
    )

    # =============================
    # ====== Load Weights =========
    # =============================

    # 🔴 Make sure these files exist inside MODEL_DIR

    WEIGHTS = {
        "GAT": "GAT_Scratch.pth",
        "GAT-Pretrained": "GAT_contrastive_finetuned.pth",
        "GAT-SSL": "GAT_SSL.pth",
    }


    # =============================
    # ====== Evaluation ===========
    # =============================

    results = {}

    for name, weight_file in WEIGHTS.items():
        weight_path = os.path.join(MODEL_DIR, weight_file)

        # 🔴 DEBUG TIP: uncomment if file not found
        # print("Checking:", weight_path)

        metrics, preds = evaluate_model_on_all_splits(
            weight_path=weight_path,
            base_model_created=model,
            dataloaders_dict=loaders,
            Data_Type_name="GAT",
            device=DEVICE,
        )
        results[name] = preds

    # =============================
    # ====== Metrics ==============
    # =============================

    metrics_list = []

    for model_name, preds_dict in results.items():
        for split in ["test", "ext"]:
            preds = np.array(preds_dict[split])
            labels = labels_dict[split]

            metrics = report_ensemble_metrics_precision(
                preds,
                labels,
                title=f"{model_name} [{split}]",
            )
            metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)

    print("\n📊 Summary Metrics:")
    print(df_metrics.round(3))

    # =============================
    # ====== Visualization ========
    # =============================

    models = ["GAT", "GAT-Pretrained", "GAT-SSL"]
    metrics_names = ["BACC", "Recall", "TNR", "AUC", "Precision", "F1"]

    plot_radar_metrics(df_metrics, models=models, metrics=metrics_names, split="test")
    plot_radar_as_bars(df_metrics, models=models, metrics=metrics_names, split="test")


# =============================
# ====== ENTRY POINT ==========
# =============================

if __name__ == "__main__":
    main()