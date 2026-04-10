

# =============================
# ====== IMPORTS ==============
# =============================

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
from src.train_evaluate_utils import train_val_model_history, validate_model
# =============================
# Config
# =============================
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")

# data and model path STRUCTURE
DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")
PRETRAIN_DIR = os.path.join(ROOT_DIR, "data", "Hyperparam_tuning")  # 🔴 adjust if needed
SAVE_DIR = os.path.join(ROOT_DIR, "results")  # 🔴 create this folder if not exists

os.makedirs(SAVE_DIR, exist_ok=True)




device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"

batch_size = 32
num_epochs = 100
patience = 7
adj_w = 1


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

params_gat = None  # using best config internally

problem_type = "binary_classification"


# =============================
# ====== LOAD PRETRAINED ======
# =============================

# 🔴 put pretrained models inside Git_data/models OR Hyperparam_tuning
PRETRAIN_MODEL_PATH = os.path.join(MODEL_DIR, "GAT_Conv1DHist_0.0051.pth")  
# 🔴 CHANGE filename if needed

pretrained_state = torch.load(PRETRAIN_MODEL_PATH, map_location=device)


# Build pretrain encoder
pretrain_encoder = create_best_model_HP(
    params=params_gat,
    in_features_size=1,
    Data_Type_name="GAT",
    batch_size=batch_size,
    problem_type=problem_type,
    allowable_features=allowable_features,
    mode="pretrain",
    mode_proj="mlp"
)

# Remove unused layers
if hasattr(pretrain_encoder, "projector"):
    del pretrain_encoder.projector
if hasattr(pretrain_encoder, "loss_fn"):
    del pretrain_encoder.loss_fn

# Load only encoder weights
filtered_state = {
    k: v for k, v in pretrained_state.items()
    if not k.startswith(("projector", "loss_fn"))
}
pretrain_encoder.load_state_dict(filtered_state, strict=False)


# =============================
# ====== BUILD FINAL MODEL ====
# =============================

gat_model = create_best_model_HP(
    params=params_gat,
    in_features_size=1,
    Data_Type_name="GAT",
    batch_size=batch_size,
    problem_type=problem_type,
    allowable_features=allowable_features,
    train_loader=graph_train_loader,
    adj_weight_ratio=adj_w,
    device=device,
    mode_proj="default",
    mode="pair"
)

# Inject pretrained encoder
gat_model.encoder.load_state_dict(pretrain_encoder.state_dict())


# =============================
# ====== TRAIN MODEL ==========
# =============================

MODEL_NAME = "GAT_finetuned.pth"
model_save_path = os.path.join(SAVE_DIR, MODEL_NAME)

best_model_state, best_model, df_train_hist, df_val_hist, df_test_hist, df_ext_hist = train_val_model_history(
    gat_model,
    train_mm_loader=None,
    train_graph_loader=graph_train_loader,
    val_mm_loader=None,
    val_graph_loader=graph_val_loader,
    test_mm_loader=None,
    test_graph_loader=graph_test_loader,
    test_mm_loader_EXT=None,
    graph_test_loader_EXT=graph_test_loader_EXT,
    model_file_path=model_save_path,
    criterion=gat_model.loss_fn,
    optimizer=torch.optim.Adam(gat_model.parameters(), lr=3e-4, weight_decay=1e-4),
    device=device,
    problem_type=problem_type,
    Data_Type_name="GAT",
    patience=patience,
    num_epochs=num_epochs
)


# =============================
# ====== EVALUATION ===========
# =============================

print("\n🧪 Evaluating trained model...")

state_dict = torch.load(model_save_path, map_location="cpu")
gat_model.load_state_dict(state_dict)

gat_model.eval()

avg_test_loss, test_metrics, _ = validate_model(
    gat_model,
    None,
    graph_test_loader,
    criterion=gat_model.loss_fn,
    device=device,
    Data_Type_name="GAT"
)

print("\n📊 Test Metrics:")
print(test_metrics)


avg_test_loss_ext, test_metrics_ext, _ = validate_model(
    gat_model,
    None,
    graph_test_loader_EXT,
    criterion=gat_model.loss_fn,
    device=device,
    Data_Type_name="GAT"
)

print("\n📊 External Test Metrics:")
print(test_metrics_ext)