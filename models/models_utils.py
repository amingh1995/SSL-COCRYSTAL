

from torch_geometric.nn import GINEConv, GATv2Conv, global_add_pool
import torch.nn.functional as F

import torch.nn as nn
import pickle
import torch

import torch
from torch import nn
from torch_geometric.data import Data




def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_label_ratio(data_loader):
    label_counts = {'0': 0, '1': 0}
    total_entries = 0

    for batch in data_loader:
        labels = batch['label']
        label_counts['0'] += (labels == 0).sum().item()
        label_counts['1'] += (labels == 1).sum().item()
        total_entries += labels.shape[0]

    ratio_1_to_0 = label_counts['1'] / label_counts['0'] if label_counts['0'] != 0 else float('inf')
    return ratio_1_to_0, total_entries

def graph_label_ratio(graph_loader):
    label_counts = {'0': 0, '1': 0}
    total_entries = 0

    for batch in graph_loader:
        labels = batch.y
        label_counts['0'] += (labels == 0).sum().item()
        label_counts['1'] += (labels == 1).sum().item()
        total_entries += labels.shape[0]

    ratio_1_to_0 = label_counts['1'] / label_counts['0'] if label_counts['0'] != 0 else float('inf')
    return ratio_1_to_0, total_entries




# === GAT Encoder ===
class GATEncoder(nn.Module):
    def __init__(self, params, allowable_features):
        super().__init__()
        self.num_layers = params['num_convs']
        self.feat_emb_dim = params['feat_emb_dim']
        self.edge_feat_dim = self.feat_emb_dim
        self.hidden_dim = self.feat_emb_dim
        self.drop_ratio = params['drop_ratio_conv']
        self.attention_heads = params['attention_heads']

        self.embeddings = nn.ModuleList([
            nn.Embedding(len(allowable_features[key]), self.feat_emb_dim)
            for key in [
                'possible_atomic_num_list', 'possible_hybridization_list', 'possible_Aromatic',
                'possible_degree_list', 'possible_formal_charge_list', 'possible_numH_list',
                'possible_implicit_valence_list', 'possible_chirality_list'
            ]
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight.data)

        self.edge_embedding = nn.Linear(4, self.edge_feat_dim)
        self.convs = nn.ModuleList()
        in_channels = self.feat_emb_dim
        for _ in range(self.num_layers):
            self.convs.append(GATv2Conv(in_channels, self.hidden_dim, heads=self.attention_heads, edge_dim=self.edge_feat_dim))
            in_channels = self.hidden_dim * self.attention_heads

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index.long(), data.edge_attr
        edge_attr = self.edge_embedding(edge_attr.float())
        x = sum(emb(x[:, i].long()) for i, emb in enumerate(self.embeddings))
        h_list = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_ratio, training=self.training)
            h_list.append(global_add_pool(x, data.batch))
        return torch.cat(h_list, dim=1)


# === Full GAT Model ===
class GAT(nn.Module):
    def __init__(self, params, allowable_features):
        super().__init__()
        self.encoder = GATEncoder(params, allowable_features)
        fc_input_dim = params['num_convs'] * params['feat_emb_dim'] * params['attention_heads']

        layers = []
        for size in params['fc_layer_sizes']:
            layers.append(nn.Linear(fc_input_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=params['drop_ratio_conv']))
            fc_input_dim = size
        self.fc_layers = nn.Sequential(*layers)

        problem_type = params.get('problem_type', 'binary_classification')
        if problem_type == 'regression':
            self.output_layer = nn.Linear(fc_input_dim, 1)
            self.output_activation = None
        elif problem_type == 'binary_classification':
            self.output_layer = nn.Linear(fc_input_dim, 1)
            self.output_activation = nn.Sigmoid()
        elif problem_type == 'multiclass_classification':
            self.output_layer = nn.Linear(fc_input_dim, params['num_classes'])
            self.output_activation = None
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    def forward(self, data):
        x = self.encoder(data)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x if self.output_activation is None else self.output_activation(x)

    def extract_features(self, data):
        return self.fc_layers(self.encoder(data))




# === CCNet Encoder ===
class CCNetEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_dim = params['L']
        fusion_dim = params.get('fusion_dim', self.input_dim)
        dropout = params.get('dropout', 0.4)
        use_layernorm = params.get('use_layernorm', True)
        fc_layer_sizes = params.get('fc_layer_sizes', [256, 128, 64])

        layers = []
        in_dim = self.input_dim
        for h in fc_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.squeeze(1)
        return self.encoder(x)


# === PointNet Encoder ===
class PointNetEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        num_features = params['num_features']
        num_convs = params['num_convs']
        conv_channels = params['conv_channels']
        dropout = params['dropout']

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        in_channels = num_features
        for c in conv_channels:
            self.convs.append(nn.Conv1d(in_channels, c, kernel_size=1))
            self.bns.append(nn.BatchNorm1d(c))
            self.relus.append(nn.ReLU())
            in_channels = c
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x shape: [B, points, features]
        x = x.permute(0,2,1)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x = relu(bn(conv(x)))
        x = self.adaptive_pool(x).view(x.size(0), -1)
        return x


# === Freq Encoder ===
# class FreqEncoder(nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         input_dim = params['input_dim']
#         fc_layer_sizes = params['fc_layer_sizes']
#         dropout = params['dropout']
#         layers = []
#         in_dim = input_dim
#         for h in fc_layer_sizes:
#             layers.extend([
#                 nn.Linear(in_dim, h),
#                 nn.BatchNorm1d(h),
#                 nn.ReLU(),
#                 nn.Dropout(dropout)
#             ])
#             in_dim = h
#         self.encoder = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         return self.encoder(x)



# === Freq Encoder ===
class MLPEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim = params['input_dim']
        fc_layer_sizes = params['fc_layer_sizes']
        dropout = params['dropout']
        layers = []
        in_dim = input_dim
        for h in fc_layer_sizes:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)


# === 1D Convolutional Encoder for 80-length Sequences ===
class Conv1DHistogramEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        # params: { 'input_length':80, 'conv_channels':[32,64,128], 'kernel_size':5, 'pool_kernel':2, 'dropout':0.2 }
        L = params.get('input_length', 80)
        channels = params.get('conv_channels', [32, 64, 128])
        k = params.get('kernel_size', 5)
        p = params.get('pool_kernel', 2)
        dropout = params.get('dropout', 0.2)
        # build conv layers
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(p))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv_net = nn.Sequential(*layers)
        # compute flattened dimension
        L_out = L // (p**len(channels))
        self.flatten_dim = in_ch * L_out
    def forward(self, x):
        # x: [B, L]
        x = x.unsqueeze(1)  # [B,1,L]
        x = self.conv_net(x)
        B, C, L2 = x.shape
        return x.view(B, -1)





"""++++++++++++++++===========+===========++++++++++===========++++++++++++++++++++++++++++"""

        
class PointNet(nn.Module):
    def __init__(self, params):
        super(PointNet, self).__init__()
        num_features = params['num_features']
        num_convs = params['num_convs']
        conv_channels = params['conv_channels']
        fc_layer_sizes = params['fc_layer_sizes']
        dropout = params['dropout']

        fc_layer_sizes = (2 * conv_channels[-1],) + tuple(fc_layer_sizes[1:])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()

        in_channels = num_features
        for i in range(num_convs):
            self.convs.append(nn.Conv1d(in_channels, conv_channels[i], kernel_size=1))
            self.bns.append(nn.BatchNorm1d(conv_channels[i]))
            self.relus.append(nn.ReLU())
            in_channels = conv_channels[i]

        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)

        fc_layers = []
        in_features = fc_layer_sizes[0]
        for out_features in fc_layer_sizes[1:]:
            fc_layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_features = out_features

        self.feature_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x1 = relu(bn(conv(x1)))
            x2 = relu(bn(conv(x2)))
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)
        x = torch.cat([x1, x2], dim=1).view(x1.size(0), -1)
        x = self.feature_layers(x)
        return self.sigmoid(self.output_layer(x))

    def extract_features(self, x1, x2):
        x1, x2 = x1.permute(0, 2, 1), x2.permute(0, 2, 1)
        for conv, bn, relu in zip(self.convs, self.bns, self.relus):
            x1 = relu(bn(conv(x1)))
            x2 = relu(bn(conv(x2)))
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)
        x = torch.cat([x1, x2], dim=1).view(x1.size(0), -1)
        return self.feature_layers(x)





class CCNetMLP_Enhanced(nn.Module):
    def __init__(self, params):
        super(CCNetMLP_Enhanced, self).__init__()
        L = params['L']
        fc_layer_sizes = params['fc_layer_sizes']
        fusion_dim = params['L']
        dropout = params['dropout']
        use_layernorm = params.get('use_layernorm', True)

        self.proj1 = nn.Linear(L, fusion_dim)
        self.proj2 = nn.Linear(L, fusion_dim)

        in_dim = 2 * L + fusion_dim
        layers = []
        for h in fc_layer_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        x1, x2 = x1.squeeze(1), x2.squeeze(1)
        f1 = self.proj1(x1)
        f2 = self.proj2(x2)
        fusion = f1 * f2
        features = torch.cat([x1, x2, fusion], dim=1)
        x = self.feature_layers(features)
        logits = self.output_layer(x)  # shape: [batch, 1]
        return self.sigmoid(logits)  # return raw logits for BCEWithLogitsLoss
        
    def extract_features(self, x1, x2):
        f1 = self.proj1(x1)
        f2 = self.proj2(x2)
        fusion = f1 * f2
        features = torch.cat([x1, x2, fusion], dim=1)  # <== FIXED HERE
        return self.feature_layers(features)



class MLP_freq(nn.Module):
    def __init__(self, params):
        super(MLP_freq, self).__init__()
        input_dim = params['input_dim']
        fc_layer_sizes = params['fc_layer_sizes']
        dropout = params['dropout']

        self.feature_layers = nn.Sequential()
        layers = []
        for out_features in fc_layer_sizes:
            layers.extend([
                nn.Linear(input_dim, out_features),
                nn.BatchNorm1d(out_features),
                nn.SELU(),
                nn.Dropout(dropout)
            ])
            input_dim = out_features

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature_layers(x)
        return self.sigmoid(self.output_layer(x))

    def extract_features(self, x):
        x = x.view(x.size(0), -1)
        return self.feature_layers(x)


class MLP_Mordred_Freq(nn.Module):
    def __init__(self, params):
        super(MLP_Mordred_Freq, self).__init__()
        input_dim = params['input_dim']  # Mordred1 + Mordred2 + freq_1_2
        fc_layer_sizes = params['fc_layer_sizes']
        dropout = params['dropout']

        layers = []
        for out_dim in fc_layer_sizes:
            layers.extend([
                nn.Linear(input_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = out_dim

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dim, 1)

    def forward(self, Mordred1, Mordred2, freq_1,freq_2):
        x = torch.cat([Mordred1, freq_1, Mordred2,freq_2], dim=1)
        x = self.feature_layers(x)
        return self.output_layer(x)




class MLP_Model(nn.Module):
    def __init__(self, params):
        super(MLP_Model, self).__init__()
        self.params = params  # ✅ So you can access input_dim externally
        self.input_dim = params['input_dim']
        fc_layer_sizes = params['fc_layer_sizes']
        dropout = params['dropout']

        layers = []
        in_dim = self.input_dim  # ✅ Fix variable name
        for out_features in fc_layer_sizes:
            layers.extend([
                nn.Linear(in_dim, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_features

        self.feature_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1).view(x1.size(0), -1)
        x = self.feature_layers(x)
        return self.sigmoid(self.output_layer(x))

    def extract_features(self, x1, x2):
        x = torch.cat([x1, x2], dim=1).view(x1.size(0), -1)
        return self.feature_layers(x)
    



def infer_feature_dim(model, sample_batch, device='cpu'):
    """
    Infers the output feature dimension of model.extract_features(...)
    Handles both single-input (x) and dual-input (x1, x2) models.

    Args:
        model: PyTorch model with extract_features method
        sample_batch: Tensor or tuple/list of tensors (inputs)
        device: 'cpu' or 'cuda'

    Returns:
        int: Feature vector dimension (second axis of output)
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if isinstance(sample_batch, (tuple, list)):
            sample_batch = [x.to(device) for x in sample_batch]
            feats = model.extract_features(*sample_batch)
        else:
            feats = model.extract_features(sample_batch.to(device))

    return feats.shape[1]


class MultimodalFusionModel(nn.Module):
    def __init__(self,
                 model1_name: str,
                 model1: nn.Module,
                 dim1: int,
                 model2_name: str,
                 model2: nn.Module,
                 dim2: int,
                 params: dict,
                 fusion_type: str = 'gated'):
        super().__init__()
        self.model1_name, self.model2_name = model1_name, model2_name
        self.model1, self.model2 = model1, model2
        self.dim1, self.dim2 = dim1, dim2
        self.fusion_type = fusion_type
        self.fusion_dim = params.get('fusion_dim', 128)
        self.dropout = params.get('dropout', 0.3)
        self.mlp_layers = params.get('mlp_layers', [64, 32])

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(dim1)
        self.bn2 = nn.BatchNorm1d(dim2)

        # Projections
        if fusion_type in ('hadamard', 'gated'):
            self.proj1 = nn.Linear(dim1, self.fusion_dim)
            self.proj2 = nn.Linear(dim2, self.fusion_dim)
        if fusion_type == 'gated':
            self.gate_proj1 = nn.Linear(dim1, self.fusion_dim)
            self.gate_proj2 = nn.Linear(dim2, self.fusion_dim)

        # MLP head
        head_in = dim1 + dim2 if fusion_type == 'concat' else dim1 + dim2 + self.fusion_dim
        layers = []
        for h in self.mlp_layers:
            layers += [nn.Linear(head_in, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(self.dropout)]
            head_in = h
        layers.append(nn.Linear(head_in, 1))
        self.mlp = nn.Sequential(*layers)

    def extract_single_feature(self, model_name, model, multimodal_batch, graph_batch):
        if model_name == 'GAT':
            return model.extract_features(graph_batch)
        elif model_name == 'PC':
            return model.extract_features(multimodal_batch['PC1'], multimodal_batch['PC2'])
        elif model_name == 'CCNet':
            return model.extract_features(multimodal_batch['PC1_hist'], multimodal_batch['PC2_hist'])
        elif model_name == 'Mordred':
            return model.extract_features(multimodal_batch['Mordred1'], multimodal_batch['Mordred2'])
        elif model_name == 'freq':
            return model.extract_features(multimodal_batch['freq_1_2'])
        elif model_name == 'Mordred_freq':
            return model.extract_features(multimodal_batch['Mordred1'], multimodal_batch['Mordred2'], multimodal_batch['freq_1_2'])

        else:
            raise ValueError(f"Unknown modality: {model_name}")

    def forward(self, multimodal_batch, graph_batch):
        x1 = self.bn1(self.extract_single_feature(self.model1_name, self.model1, multimodal_batch, graph_batch))
        x2 = self.bn2(self.extract_single_feature(self.model2_name, self.model2, multimodal_batch, graph_batch))

        if self.fusion_type == 'concat':
            fused = torch.cat([x1, x2], dim=1)
        else:
            f1, f2 = self.proj1(x1), self.proj2(x2)
            if self.fusion_type == 'hadamard':
                interaction = f1 * f2
            else:  # gated
                gate = torch.sigmoid(self.gate_proj1(x1) + self.gate_proj2(x2))
                interaction = gate * f1 + (1 - gate) * f2
            fused = torch.cat([x1, x2, interaction], dim=1)

        return self.mlp(fused).squeeze(-1)

    def predict_proba(self, multimodal_batch, graph_batch):
        return torch.sigmoid(self.forward(multimodal_batch, graph_batch))

    def extract_features(self, multimodal_batch, graph_batch):
        x1 = self.bn1(self.extract_single_feature(self.model1_name, self.model1, multimodal_batch, graph_batch))
        x2 = self.bn2(self.extract_single_feature(self.model2_name, self.model2, multimodal_batch, graph_batch))
        if self.fusion_type == 'concat':
            return torch.cat([x1, x2], dim=1)
        f1, f2 = self.proj1(x1), self.proj2(x2)
        if self.fusion_type == 'hadamard':
            return torch.cat([x1, x2, f1 * f2], dim=1)
        gate = torch.sigmoid(self.gate_proj1(x1) + self.gate_proj2(x2))
        return torch.cat([x1, x2, gate * f1 + (1 - gate) * f2], dim=1)


def compute_pos_weight(loader, adj_weight_ratio, device='cpu'):
    for batch in loader:
        if hasattr(batch, 'y'):
            labels = batch.y
        elif isinstance(batch, dict) and 'label' in batch:
            labels = batch['label']
        else:
            raise ValueError("Could not find labels in batch")
        break

    if hasattr(batch, 'y'):
        ratio_1_to_0, _ = graph_label_ratio(loader)
    else:
        ratio_1_to_0, _ = calculate_label_ratio(loader)

    ratio_0_to_1 = 1.0 / ratio_1_to_0
    return torch.tensor([ratio_0_to_1 * adj_weight_ratio], device=device)




def build_fusion_model(
    model1_name, model1, example_batch1,
    model2_name, model2, example_batch2,
    params,
    problem_type='binary_classification',
    train_loader=None,
    adj_weight_ratio=1.0,
    device='cpu'
):
    """
    Factory that infers feature dims and returns ready-to-train fusion model with loss function.
    """
    # === Infer dimensions
    dim1 = infer_feature_dim(model1, example_batch1, device=device)
    dim2 = infer_feature_dim(model2, example_batch2, device=device)

    fusion_model = MultimodalFusionModel(
        model1_name=model1_name,
        model1=model1,
        dim1=dim1,
        model2_name=model2_name,
        model2=model2,
        dim2=dim2,
        params=params,
        fusion_type=params.get('fusion_type', 'gated')
    ).to(device)
    
    # === Compute pos_weight if needed
    if problem_type == "binary_classification":
        if train_loader is None:
            raise ValueError("train_loader is required for pos_weight")
        pos_weight = compute_pos_weight(train_loader, adj_weight_ratio, device)
    else:
        pos_weight = None

    # if problem_type == "binary_classification":
    #     if train_loader is None:
    #         raise ValueError("train_loader is required for classification with pos_weight")
    #     if model1_name in ['GAT', 'graph_GAT'] or model2_name in ['GAT', 'graph_GAT']:
    #         ratio_1_to_0, _ = graph_label_ratio(train_loader)
    #     else:
    #         ratio_1_to_0, _ = calculate_label_ratio(train_loader)
    #     ratio_0_to_1 = 1.0 / ratio_1_to_0
    #     pos_weight = torch.tensor([ratio_0_to_1 * adj_weight_ratio], device=device)
    # else:
    #     pos_weight = None

    # === Attach loss function
    if problem_type == "regression":
        fusion_model.loss_fn = nn.MSELoss()
    elif problem_type == "binary_classification":
        fusion_model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    print(f"✅ Fusion Model created: {model1_name} + {model2_name} with fusion '{params.get('fusion_type', 'gated')}'")
    print(f"   ➕ Feature dims: {dim1} and {dim2}, pos_weight = {pos_weight}")
    return fusion_model





def off_diagonal(x):
    # Returns a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m, "Input must be a square matrix"
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, device, batch_size, embed_size, lambd=0.005):
        super(BarlowTwinsLoss, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lambd = lambd

        # BatchNorm without learnable affine params (per Barlow Twins paper)
        self.bn = nn.BatchNorm1d(self.embed_size, affine=False).to(self.device)

    def forward(self, z1, z2):
        """
        z1: torch.Tensor of shape (batch_size, embed_size)
        z2: torch.Tensor of shape (batch_size, embed_size)
        """
        # Normalize embeddings
        z1_norm = self.bn(z1)
        z2_norm = self.bn(z2)

        # Compute cross-correlation matrix
        c = (z1_norm.T @ z2_norm) / self.batch_size  # shape: (embed_size, embed_size)

        # On-diagonal loss (should be close to 1)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # Off-diagonal loss (should be close to 0)
        off_diag = off_diagonal(c).pow_(2).sum()

        # Total loss
        loss = on_diag + self.lambd * off_diag
        return loss

# === Shared Projection Layer for Contrastive Learning ===
class SharedProjector(nn.Module):
    def __init__(self, input_dim, proj_dim=128, mode='default', shared_proj_HP=None):
        super().__init__()
        self.mode = mode
        shared_proj_HP = shared_proj_HP or [256, 128]

        if mode == 'default':
            self.proj = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim)
            )
        elif mode == 'mlp':
            layers = []
            in_dim = input_dim
            for h in shared_proj_HP:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, proj_dim))
            self.proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported projector mode: {mode}")

    def forward(self, x):
        return self.proj(x)

# === create_best_model_HP (supports contrastive pretraining) ===
def create_best_model_HP_old(params, in_features_size, Data_Type_name, batch_size, problem_type,
                         allowable_features=None, train_loader=None, adj_weight_ratio=1.0,
                         device='cpu', mode_proj= 'default', mode='pair', proj_dim=128, lambd=0.005):
    """
    Factory for contrastive encoders (mode='pretrain') or full models (mode='pair').
    """
    params = params or {}

    # === Default hyperparameters ===
    defaults = {
        'PC': {'num_features': 4, 'num_convs': 5,
               'conv_channels': [128, 128, 128, 256, 320], 'dropout': 0.25},
        'freq': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'freq_old': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'Mordred': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},
        'Mordred_freq': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},

        'GAT': {'num_convs': 6, 'feat_emb_dim': 300, 'drop_ratio_conv': 0.15,
                'attention_heads': 3, 'fc_layer_sizes': [400, 250],
                'problem_type': problem_type, 'num_classes': 2},
        'CCNetMLP': {'L': in_features_size*1, 'fc_layer_sizes': [256, 128, 64, 32],
                     'fusion_dim': in_features_size, 'dropout': 0.4, 'use_layernorm': True},
        'Conv1DHist': {'input_length': 80, 'conv_channels': [32, 64, 128],
                       'kernel_size': 5, 'pool_kernel': 2, 'dropout': 0.2}
    }

    merged = {**defaults.get(Data_Type_name, {}), **params}

    # === Select Encoder or Full Model ===
    if Data_Type_name == 'PC':
        model = PointNetEncoder(merged) if mode == 'pretrain' else PointNet(merged)
        encoder_output_dim = merged['conv_channels'][-1]
    elif Data_Type_name == 'freq_old':
        model = FreqEncoder(merged) if mode == 'pretrain' else MLP_freq(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Mordred':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Mordred_freq':
        model =  MLP_Mordred_Freq(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    
    elif Data_Type_name == 'freq':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name in ['GAT', 'graph_GAT']:
        model = GATEncoder(merged, allowable_features) if mode == 'pretrain' else GAT(merged, allowable_features)
        encoder_output_dim = merged['num_convs'] * merged['feat_emb_dim'] * merged['attention_heads']
    elif Data_Type_name == 'CCNetMLP':
        model = CCNetEncoder(merged) if mode == 'pretrain' else CCNetMLP_Enhanced(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Conv1DHist':
        if mode != 'pretrain':
            raise ValueError("Conv1DHistogramEncoder is only available in pretrain mode.")
        model = Conv1DHistogramEncoder(merged)
        # Compute output dimension: last conv output channels × reduced L
        n_layers = len(merged['conv_channels'])
        L_out = merged['input_length'] // (merged['pool_kernel'] ** n_layers)
        encoder_output_dim = merged['conv_channels'][-1] * L_out
    else:
        raise ValueError(f"Unsupported Data_Type_name: {Data_Type_name}")

    # === Attach Shared Projection Head if Pretraining ===
    if mode == 'pretrain':
        #model.projector = SharedProjector(encoder_output_dim, proj_dim=proj_dim)     # Default projection (2-layer)        
        model.projector = SharedProjector(input_dim=encoder_output_dim, proj_dim=proj_dim, mode=mode_proj, shared_proj_HP=[400, 250])         # MLP-based projection

        model.loss_fn = BarlowTwinsLoss(device=device, batch_size=batch_size, embed_size=proj_dim, lambd=lambd)
    # === Downstream Loss Functions ===
    elif mode == 'pair':
        if problem_type == 'binary_classification':
            if train_loader is None:
                raise ValueError("train_loader is required for classification tasks")
            if Data_Type_name in ['GAT', 'graph_GAT']:
                ratio_1_to_0, _ = graph_label_ratio(train_loader)
            else:
                ratio_1_to_0, _ = calculate_label_ratio(train_loader)
            pos_weight = torch.tensor([1.0 / ratio_1_to_0 * adj_weight_ratio], device=device)
            model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif problem_type == 'regression':
            model.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    return model


# === create_best_model_HP (supports contrastive pretraining) ===
def create_best_model_HP(params, in_features_size, Data_Type_name, batch_size, problem_type,
                         allowable_features=None, train_loader=None, adj_weight_ratio=1.0,
                         device='cpu', mode_proj= 'default', mode='pair', proj_dim=128, lambd=0.005):
    """
    Factory for contrastive encoders (mode='pretrain') or full models (mode='pair').
    """
    params = params or {}

    # === Default hyperparameters ===
    defaults = {
        'PC': {'num_features': 4, 'num_convs': 5,
               'conv_channels': [128, 128, 128, 256, 320], 'dropout': 0.25},
        'freq': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'freq_old': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'Mordred': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},
        'Mordred_freq': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},

        'GAT': {'num_convs': 6, 'feat_emb_dim': 300, 'drop_ratio_conv': 0.15,
                'attention_heads': 3, 'fc_layer_sizes': [400, 250],
                'problem_type': problem_type, 'num_classes': 2},
        'CCNetMLP': {'L': in_features_size*1, 'fc_layer_sizes': [256, 128, 64, 32],
                     'fusion_dim': in_features_size, 'dropout': 0.4, 'use_layernorm': True},
        'Conv1DHist': {'input_length': 80, 'conv_channels': [32, 64, 128],
                       'kernel_size': 5, 'pool_kernel': 2, 'dropout': 0.2}
    }

    merged = {**defaults.get(Data_Type_name, {}), **params}

    # === Select Encoder or Full Model ===
    if Data_Type_name == 'PC':
        model = PointNetEncoder(merged) if mode == 'pretrain' else PointNet(merged)
        encoder_output_dim = merged['conv_channels'][-1]
    elif Data_Type_name == 'freq_old':
        model = FreqEncoder(merged) if mode == 'pretrain' else MLP_freq(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Mordred':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Mordred_freq':
        model =  MLP_Mordred_Freq(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    
    elif Data_Type_name == 'freq':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name in ['GAT', 'graph_GAT']:
        model = GATEncoder(merged, allowable_features) if mode == 'pretrain' else GAT(merged, allowable_features)
        encoder_output_dim = merged['num_convs'] * merged['feat_emb_dim'] * merged['attention_heads']
    elif Data_Type_name == 'CCNetMLP':
        model = CCNetEncoder(merged) if mode == 'pretrain' else CCNetMLP_Enhanced(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Conv1DHist':
        if mode != 'pretrain':
            raise ValueError("Conv1DHistogramEncoder is only available in pretrain mode.")
        model = Conv1DHistogramEncoder(merged)
        # Compute output dimension: last conv output channels × reduced L
        n_layers = len(merged['conv_channels'])
        L_out = merged['input_length'] // (merged['pool_kernel'] ** n_layers)
        encoder_output_dim = merged['conv_channels'][-1] * L_out
    else:
        raise ValueError(f"Unsupported Data_Type_name: {Data_Type_name}")

    # === Attach Shared Projection Head if Pretraining ===
    if mode == 'pretrain':
        #model.projector = SharedProjector(encoder_output_dim, proj_dim=proj_dim)     # Default projection (2-layer)        
        model.projector = SharedProjector(input_dim=encoder_output_dim, proj_dim=proj_dim, mode=mode_proj, shared_proj_HP=[400, 250])         # MLP-based projection

        model.loss_fn = BarlowTwinsLoss(device=device, batch_size=batch_size, embed_size=proj_dim, lambd=lambd)
    # === Downstream Loss Functions ===
    elif mode == 'pair':
        if problem_type == 'binary_classification':
            if train_loader is None:
                raise ValueError("train_loader is required for classification tasks")
            if Data_Type_name in ['GAT', 'graph_GAT']:
                ratio_1_to_0, _ = graph_label_ratio(train_loader)
            else:
                ratio_1_to_0, _ = calculate_label_ratio(train_loader)
            pos_weight = torch.tensor([1.0 / ratio_1_to_0 * adj_weight_ratio], device=device)
            # pos_weight = torch.tensor([1.0])

            model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif problem_type == 'regression':
            model.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    return model

# === create_best_model_HP (supports contrastive pretraining) ===
def create_best_Teacher_HP(params, in_features_size, Data_Type_name, batch_size, problem_type,
                         allowable_features=None, ratio_1_to_0=None, adj_weight_ratio=1.0,
                         device='cpu', mode_proj= 'default', mode='pair', proj_dim=128, lambd=0.005):
    """
    Factory for contrastive encoders (mode='pretrain') or full models (mode='pair').
    """
    params = params or {}

    # === Default hyperparameters ===
    defaults = {
        'PC': {'num_features': 4, 'num_convs': 5,
               'conv_channels': [128, 128, 128, 256, 320], 'dropout': 0.25},
        'freq': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'freq_old': {'input_dim': in_features_size* 1, 'fc_layer_sizes': [50], 'dropout': 0.15},
        'Mordred': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},
        'Mordred_freq': {'input_dim': in_features_size * 1, 'fc_layer_sizes': [256, 128, 64, 32], 'dropout': 0.4},

        'GAT': {'num_convs': 6, 'feat_emb_dim': 300, 'drop_ratio_conv': 0.15,
                'attention_heads': 3, 'fc_layer_sizes': [400, 250],
                'problem_type': problem_type, 'num_classes': 2},
        'CCNetMLP': {'L': in_features_size*1, 'fc_layer_sizes': [256, 128, 64, 32],
                     'fusion_dim': in_features_size, 'dropout': 0.4, 'use_layernorm': True},
        'Conv1DHist': {'input_length': 80, 'conv_channels': [32, 64, 128],
                       'kernel_size': 5, 'pool_kernel': 2, 'dropout': 0.2}
    }

    merged = {**defaults.get(Data_Type_name, {}), **params}

    # === Select Encoder or Full Model ===
    if Data_Type_name == 'PC':
        model = PointNetEncoder(merged) if mode == 'pretrain' else PointNet(merged)
        encoder_output_dim = merged['conv_channels'][-1]
    
    elif Data_Type_name == 'Mordred':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Mordred_freq':
        model =  MLP_Mordred_Freq(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    
    elif Data_Type_name == 'freq':
        model = MLPEncoder(merged) if mode == 'pretrain' else MLP_Model(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name in ['GAT', 'graph_GAT']:
        model = GATEncoder(merged, allowable_features) if mode == 'pretrain' else GAT(merged, allowable_features)
        encoder_output_dim = merged['num_convs'] * merged['feat_emb_dim'] * merged['attention_heads']
    elif Data_Type_name == 'CCNetMLP':
        model = CCNetEncoder(merged) if mode == 'pretrain' else CCNetMLP_Enhanced(merged)
        encoder_output_dim = merged['fc_layer_sizes'][-1]
    elif Data_Type_name == 'Conv1DHist':
        if mode != 'pretrain':
            raise ValueError("Conv1DHistogramEncoder is only available in pretrain mode.")
        model = Conv1DHistogramEncoder(merged)
        # Compute output dimension: last conv output channels × reduced L
        n_layers = len(merged['conv_channels'])
        L_out = merged['input_length'] // (merged['pool_kernel'] ** n_layers)
        encoder_output_dim = merged['conv_channels'][-1] * L_out
    else:
        raise ValueError(f"Unsupported Data_Type_name: {Data_Type_name}")

    # === Attach Shared Projection Head if Pretraining ===
    if mode == 'pretrain':
        #model.projector = SharedProjector(encoder_output_dim, proj_dim=proj_dim)     # Default projection (2-layer)        
        model.projector = SharedProjector(input_dim=encoder_output_dim, proj_dim=proj_dim, mode=mode_proj, shared_proj_HP=[400, 250])         # MLP-based projection

        model.loss_fn = BarlowTwinsLoss(device=device, batch_size=batch_size, embed_size=proj_dim, lambd=lambd)
    # === Downstream Loss Functions ===
    elif mode == 'pair':
        if problem_type == 'binary_classification':
            
            pos_weight = torch.tensor([1.0 / ratio_1_to_0 * adj_weight_ratio], device=device)
            model.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        elif problem_type == 'regression':
            model.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

    return model

