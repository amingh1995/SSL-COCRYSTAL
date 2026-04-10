
"""
Example inference script for cocrystal prediction using trained GAT models.

This script:
- Takes two SMILES strings
- Builds graph representation
- Loads trained models
- Outputs prediction probabilities

Designed for reproducibility 
"""


# =============================
# Imports
# =============================
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import RDLogger

# Silence RDKit warnings
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# =============================
# Project imports
# =============================
from src.data_processing_utils import create_graph_dataloader, allowable_features
from models.models_utils import create_best_model_HP

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"



# =============================
# ====== LOAD DATA ============
# =============================

# 🔴 Files must be in Git_data/datasets/
df_remaining = pd.read_csv(os.path.join(DATA_DIR, "df_train_val.csv"))
df_test_pairs = pd.read_csv(os.path.join(DATA_DIR, "df_test_dft_full.csv"))
df_ext = pd.read_csv(os.path.join(DATA_DIR, "test_external.csv"))


# =============================
# ====== SPLIT DATA ===========
# =============================

df_train, df_val = train_test_split(
    df_remaining,
    test_size=0.15,
    stratify=df_remaining["labels"],
    random_state=104
)


# =============================
# ====== GRAPH LOADERS ========
# =============================

def build_loader(df):
    return create_graph_dataloader(
        df["smiles1"],
        df["smiles2"],
        df["labels"],
        batch_size=batch_size
    )

graph_train_loader = build_loader(df_train)
graph_val_loader = build_loader(df_val)
graph_test_loader = build_loader(df_test_pairs)
graph_test_loader_EXT = build_loader(df_ext)





# =============================
# ====== MODEL SETUP ==========
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


def build_model():
    model = create_best_model_HP(
        params=Best_params_GAT,
        in_features_size=1,
        Data_Type_name="GAT",
        batch_size=1,
        problem_type="binary_classification",
        allowable_features=allowable_features,
        train_loader=graph_train_loader,  # not needed for inference
        adj_weight_ratio=2,
        device=DEVICE,
        mode_proj="default",
        mode="pair",
    )
    return model


# =============================
# ====== LOAD MODELS ==========
# =============================

MODEL_PATHS = {
    "GAT_Scratch": "GAT_Scratch.pth",
    "GAT_Pretrained_FT": "GAT_contrastive_finetuned.pth",
    "GAT_SSL": "GAT_SSL.pth",
}


def load_models():
    models = {}

    for name, file in MODEL_PATHS.items():
        path = os.path.join(MODEL_DIR, file)

        model = build_model()
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        models[name] = model

    return models


# =============================
# ====== PREDICTION ===========
# =============================

def predict_pair(smiles1, smiles2, models):
    """
    Predict cocrystal formation probability for a pair of SMILES
    """

    # Create a single-sample dataloader
    loader = create_graph_dataloader(
        [smiles1],
        [smiles2],
        [0],  # dummy label
        batch_size=1
    )

    batch = next(iter(loader))

    results = {}

    if isinstance(models, dict):
        for name, model in models.items():
            with torch.no_grad():
                output = model(batch.to(DEVICE))

                # Handle tuple outputs
                if isinstance(output, tuple):
                    output = output[0]

                prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
                pred = int(prob > 0.5)

            results[name] = {
                "probability": float(prob),
                "prediction": pred
            }
    else:
        # Single model
        name = "GAT"
        model = models
        with torch.no_grad():
            output = model(batch.to(DEVICE))

            # Handle tuple outputs
            if isinstance(output, tuple):
                output = output[0]

            prob = torch.sigmoid(output).cpu().numpy().flatten()[0]
            pred = int(prob > 0.5)

        results[name] = {
            "probability": float(prob),
            "prediction": pred
        }

    return results


# =============================
# ====== EXAMPLE RUN ==========
# =============================

if __name__ == "__main__":

    # 🔬 Example SMILES (you can change these)
    smiles_aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    smiles_p_aminophenol = "NC1=CC=C(O)C=C1"          # p-aminophenol
    smiles_cip = "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O"
    smiles_urea = "C(N)=O"
    smiles_caffeine = "Cn1cnc2c1c(=O)n(c(=O)n2C)C"  # Caffeine
    smiles1 = smiles_cip
    smiles2 = smiles_urea
    print("\n🧪 Input Pair:")
    print("SMILES 1:", smiles1)
    print("SMILES 2:", smiles2)

    # Load models
    models = load_models()
    models= models["GAT_SSL"]  # 🔴 choose specific model if desired, or keep all for comparison

    # Predict
    results = predict_pair(smiles1=smiles1, smiles2=smiles2, models=models)

    print("\n📊 Prediction Results:")
    for model_name, res in results.items():
        print(f"\n🔹 {model_name}")
        print(f"   Probability: {res['probability']:.4f}")
        print(f"   Prediction : {res['prediction']} (1 = cocrystal, 0 = no)")













