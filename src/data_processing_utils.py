
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


from torch.utils.data import Dataset, Subset


import torch
from torch import nn


from rdkit import Chem
import pandas as pd


from torch_geometric.nn import GINEConv, GATv2Conv, global_add_pool
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pickle
import torch
from tqdm import tqdm
from torch.utils.data import random_split

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, Dataset, DataLoader

def load_source_dict_data(base_path_PC):
    
    save_path_PC_original= os.path.join(base_path_PC, 'PC_original_dict.pkl') 

    # Load the dictionary
    with open(save_path_PC_original, 'rb') as f:
        PC_dft_dict = pickle.load(f)
    
    

    # === Reload example ===
    with open(os.path.join(base_path_PC, "MultiModal_dict_CC_mord_Freq.pkl"), "rb") as f:
        loaded_CC = pickle.load(f)
    
    with open(os.path.join(base_path_PC, "MultiModal_dict_CC_EXT_mord_Freq.pkl"), "rb") as f:
        loaded_CC_EXT = pickle.load(f)
    
    with open(os.path.join(base_path_PC, "MultiModal_dict_unique_molecules_mord_Freq.pkl"), "rb") as f:
        loaded_unique = pickle.load(f)
    


    return PC_dft_dict,loaded_CC, loaded_CC_EXT, loaded_unique
    




def calculate_label_ratio_from_df(df, label_column='labels'):
    """
    Calculate the ratio of class 1 to class 0 in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        label_column (str): Name of the column containing binary labels (0 and 1).

    Returns:
        ratio_1_to_0 (float): Ratio of count(label==1) / count(label==0)
        total_entries (int): Total number of labeled samples
    """
    label_counts = df[label_column].value_counts().to_dict()

    count_0 = label_counts.get(0, 0)
    count_1 = label_counts.get(1, 0)

    total_entries = count_0 + count_1
    ratio_1_to_0 = count_1 / count_0 if count_0 != 0 else float('inf')

    return ratio_1_to_0, total_entries










# ====================== NORMALIZATION ======================
def Normalize_PC_train_test(train, test, features):
    scaler = MinMaxScaler()
    train_2D = np.reshape(train, (-1, features))
    scaler.fit(train_2D)
    train_norm_2D = scaler.transform(train_2D)
    train_norm = np.reshape(train_norm_2D, train.shape)
    test_2D = np.reshape(test, (-1, features))
    test_norm_2D = scaler.transform(test_2D)
    test_norm = np.reshape(test_norm_2D, test.shape)
    return train_norm, test_norm




def data_split_normalize_mord_Freq(data_zip, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=104):
    (
        _, _, data_PC1, data_PC2,
        smiles_1, smiles_2,
        ID1_smiles_1, ID2_smiles_2,
        df_tabular_features, freq_1_2_tabular_features, labels,
        PC1_hist, PC2_hist, hist_centers,
        mordred1, mordred2,
        freq_1, freq_2,  # newly unpacked
        freq_1_df,freq_2_df
    ) = data_zip

    labels = np.array(labels)

    indices = np.arange(len(labels))
    idx_temp, idx_test, _, labels_test = train_test_split(
        indices, labels, test_size=test_ratio, stratify=labels, random_state=random_state)
    val_ratio_adj = val_ratio / (val_ratio + train_ratio)
    idx_train, idx_val, labels_train, labels_val = train_test_split(
        idx_temp, labels[idx_temp], test_size=val_ratio_adj, stratify=labels[idx_temp], random_state=random_state)

    def idx(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.iloc[idx_train], x.iloc[idx_val], x.iloc[idx_test]
        elif isinstance(x, list):
            x = np.array(x)
            return x[idx_train], x[idx_val], x[idx_test]
        else:
            return x[idx_train], x[idx_val], x[idx_test]

    # === SPLIT ===
    data_PC1_train, data_PC1_val, data_PC1_test = idx(data_PC1)
    data_PC2_train, data_PC2_val, data_PC2_test = idx(data_PC2)
    freq_1_2_train, freq_1_2_val, freq_1_2_test = idx(freq_1_2_tabular_features)
    mordred1_train, mordred1_val, mordred1_test = idx(mordred1)
    mordred2_train, mordred2_val, mordred2_test = idx(mordred2)
    smiles_1_train, smiles_1_val, smiles_1_test = idx(smiles_1)
    smiles_2_train, smiles_2_val, smiles_2_test = idx(smiles_2)
    ID1_smiles_1_train, ID1_smiles_1_val, ID1_smiles_1_test = idx(ID1_smiles_1)
    ID2_smiles_2_train, ID2_smiles_2_val, ID2_smiles_2_test = idx(ID2_smiles_2)
    PC1_hist_train, PC1_hist_val, PC1_hist_test = idx(PC1_hist)
    PC2_hist_train, PC2_hist_val, PC2_hist_test = idx(PC2_hist)

    freq_1_train, freq_1_val, freq_1_test = idx(freq_1)
    freq_2_train, freq_2_val, freq_2_test = idx(freq_2)

    # === NORMALIZE POINT CLOUD ===
    features = 4
    data_PC1_train_norm, data_PC1_val_norm = Normalize_PC_train_test(data_PC1_train, data_PC1_val, features)
    data_PC1_train_norm, data_PC1_test_norm = Normalize_PC_train_test(data_PC1_train, data_PC1_test, features)
    data_PC2_train_norm, data_PC2_val_norm = Normalize_PC_train_test(data_PC2_train, data_PC2_val, features)
    data_PC2_train_norm, data_PC2_test_norm = Normalize_PC_train_test(data_PC2_train, data_PC2_test, features)

    # === NORMALIZE TABULAR ===
    scaler_freq = MinMaxScaler()
    freq_1_2_train_norm = scaler_freq.fit_transform(freq_1_2_train)
    freq_1_2_val_norm = scaler_freq.transform(freq_1_2_val)
    freq_1_2_test_norm = scaler_freq.transform(freq_1_2_test)

    # 1) Shared scaler for freq_1 and freq_2  ⬇⬇⬇
    combined_freq_train = np.vstack([freq_1_train, freq_2_train])
    scaler_freq_pair = MinMaxScaler()
    scaler_freq_pair.fit(combined_freq_train)
    
    freq_1_train_norm = scaler_freq_pair.transform(freq_1_train)
    freq_1_val_norm   = scaler_freq_pair.transform(freq_1_val)
    freq_1_test_norm  = scaler_freq_pair.transform(freq_1_test)
    
    freq_2_train_norm = scaler_freq_pair.transform(freq_2_train)
    freq_2_val_norm   = scaler_freq_pair.transform(freq_2_val)
    freq_2_test_norm  = scaler_freq_pair.transform(freq_2_test)

    # === Normalize mordreds ===
    combined_train = np.vstack([mordred1_train, mordred2_train])
    scaler_mordred = MinMaxScaler()
    scaler_mordred.fit(combined_train)

    mordred1_train_norm = scaler_mordred.transform(mordred1_train)
    mordred1_val_norm = scaler_mordred.transform(mordred1_val)
    mordred1_test_norm = scaler_mordred.transform(mordred1_test)

    mordred2_train_norm = scaler_mordred.transform(mordred2_train)
    mordred2_val_norm = scaler_mordred.transform(mordred2_val)
    mordred2_test_norm = scaler_mordred.transform(mordred2_test)

    # === BUILD DICT ===
    def build_dict(pc1, pc2, freq, mord1, mord2, smiles1, smiles2, id1, id2, lbl, pc1_hist, pc2_hist,
                   f1, f2):
        return {
            'data_PC1': pc1,
            'data_PC2': pc2,
            'freq_1_2': freq,
            'freq_1': f1,
            'freq_2': f2,
            'Mordred1': mord1,
            'Mordred2': mord2,
            'smiles_1': smiles1,
            'smiles_2': smiles2,
            'ID1_smiles_1': id1,
            'ID2_smiles_2': id2,
            'labels': lbl,
            'df_tabular_features': df_tabular_features,
            'PC1_hist': pc1_hist,
            'PC2_hist': pc2_hist,
            'hist_centers': hist_centers
        }

    data_dict = {
        'train': build_dict(data_PC1_train, data_PC2_train, freq_1_2_train, mordred1_train, mordred2_train,
                            smiles_1_train, smiles_2_train, ID1_smiles_1_train, ID2_smiles_2_train, labels_train,
                            PC1_hist_train, PC2_hist_train,
                            freq_1_train, freq_2_train),
        'val': build_dict(data_PC1_val, data_PC2_val, freq_1_2_val, mordred1_val, mordred2_val,
                          smiles_1_val, smiles_2_val, ID1_smiles_1_val, ID2_smiles_2_val, labels_val,
                          PC1_hist_val, PC2_hist_val,
                          freq_1_val, freq_2_val),
        'test': build_dict(data_PC1_test, data_PC2_test, freq_1_2_test, mordred1_test, mordred2_test,
                           smiles_1_test, smiles_2_test, ID1_smiles_1_test, ID2_smiles_2_test, labels_test,
                           PC1_hist_test, PC2_hist_test,
                           freq_1_test, freq_2_test)
    }

    norm_data_dict = {
        'train': build_dict(data_PC1_train_norm, data_PC2_train_norm, freq_1_2_train_norm,
                            mordred1_train_norm, mordred2_train_norm,
                            smiles_1_train, smiles_2_train, ID1_smiles_1_train, ID2_smiles_2_train, labels_train,
                            PC1_hist_train, PC2_hist_train,
                            freq_1_train_norm, freq_2_train_norm),
        'val': build_dict(data_PC1_val_norm, data_PC2_val_norm, freq_1_2_val_norm,
                          mordred1_val_norm, mordred2_val_norm,
                          smiles_1_val, smiles_2_val, ID1_smiles_1_val, ID2_smiles_2_val, labels_val,
                          PC1_hist_val, PC2_hist_val,
                          freq_1_val_norm, freq_2_val_norm),
        'test': build_dict(data_PC1_test_norm, data_PC2_test_norm, freq_1_2_test_norm,
                           mordred1_test_norm, mordred2_test_norm,
                           smiles_1_test, smiles_2_test, ID1_smiles_1_test, ID2_smiles_2_test, labels_test,
                           PC1_hist_test, PC2_hist_test,
                           freq_1_test_norm, freq_2_test_norm)
    }

    return data_dict, norm_data_dict

# ====================== STRATIFIED SPLIT ======================




def data_split_normalize_ext_mord_Freq(data_zip_source, data_zip_ext, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=104):

    (
        _, _, data_PC1, data_PC2,
        smiles_1, smiles_2,
        ID1_smiles_1, ID2_smiles_2,
        df_tabular_features, freq_1_2_tabular_features, labels,
        PC1_hist, PC2_hist, hist_centers,
        mordred1, mordred2,
        freq_1, freq_2,
        freq_1_df, freq_2_df
    ) = data_zip_source

    (
        _, _, data_PC1_ext, data_PC2_ext,
        smiles_1_ext, smiles_2_ext,
        ID1_smiles_1_ext, ID2_smiles_2_ext,
        df_tabular_features_ext, freq_1_2_tabular_features_ext, labels_ext,
        PC1_hist_ext, PC2_hist_ext, hist_centers_ext,
        mordred1_ext, mordred2_ext,
        freq_1_ext, freq_2_ext,
        freq_1_df_ext, freq_2_df_ext
    ) = data_zip_ext

    # Use ext data as test set
    data_PC1_test, data_PC2_test = data_PC1_ext, data_PC2_ext
    smiles_1_test, smiles_2_test = smiles_1_ext, smiles_2_ext
    ID1_smiles_1_test, ID2_smiles_2_test = ID1_smiles_1_ext, ID2_smiles_2_ext
    freq_1_2_test = freq_1_2_tabular_features_ext
    labels_test = labels_ext
    mordred1_test, mordred2_test = mordred1_ext, mordred2_ext
    freq_1_test, freq_2_test = freq_1_ext, freq_2_ext
    freq_1_df_test, freq_2_df_test = freq_1_df_ext, freq_2_df_ext

    # Split source data for training
    def split_data(x):
        if isinstance(x, list): x = np.array(x)
        return x[:int(len(x) * train_ratio)]

    data_PC1_train = split_data(data_PC1)
    data_PC2_train = split_data(data_PC2)
    smiles_1_train = split_data(smiles_1)
    smiles_2_train = split_data(smiles_2)
    ID1_smiles_1_train = split_data(ID1_smiles_1)
    ID2_smiles_2_train = split_data(ID2_smiles_2)
    freq_1_2_train = split_data(freq_1_2_tabular_features)
    labels_train = split_data(labels)
    mordred1_train = split_data(mordred1)
    mordred2_train = split_data(mordred2)
    freq_1_train = split_data(freq_1)
    freq_2_train = split_data(freq_2)
    freq_1_df_train = split_data(freq_1_df)
    freq_2_df_train = split_data(freq_2_df)

    # === Normalize point cloud ===
    features = 4
    data_PC1_train_norm, data_PC1_test_norm = Normalize_PC_train_test(data_PC1_train, data_PC1_test, features)
    data_PC2_train_norm, data_PC2_test_norm = Normalize_PC_train_test(data_PC2_train, data_PC2_test, features)

    # === Normalize tabular ===
    scaler_freq = MinMaxScaler()
    freq_1_2_train_norm = scaler_freq.fit_transform(freq_1_2_train)
    freq_1_2_test_norm = scaler_freq.transform(freq_1_2_test)

    # === Normalize Mordred with shared scaler ===
    scaler_mord = MinMaxScaler()
    combined_train_mord = np.vstack([mordred1_train, mordred2_train])
    scaler_mord.fit(combined_train_mord)

    mordred1_train_norm = scaler_mord.transform(mordred1_train)
    mordred2_train_norm = scaler_mord.transform(mordred2_train)
    mordred1_test_norm = scaler_mord.transform(mordred1_test)
    mordred2_test_norm = scaler_mord.transform(mordred2_test)

    # === Normalize Frequency 1 & 2 with shared scaler ===
    scaler_freq_pair = MinMaxScaler()
    combined_train_freq = np.vstack([freq_1_train, freq_2_train])
    scaler_freq_pair.fit(combined_train_freq)

    freq_1_train_norm = scaler_freq_pair.transform(freq_1_train)
    freq_2_train_norm = scaler_freq_pair.transform(freq_2_train)
    freq_1_test_norm = scaler_freq_pair.transform(freq_1_test)
    freq_2_test_norm = scaler_freq_pair.transform(freq_2_test)

    # === Final dicts ===
    def build_dict(pc1, pc2, freq, mord1, mord2, smiles1, smiles2, id1, id2, lbl,
                   tabular, hist1, hist2, center, f1, f2, f1_df, f2_df):
        return {
            'data_PC1': pc1,
            'data_PC2': pc2,
            'freq_1_2': freq,
            'Mordred1': mord1,
            'Mordred2': mord2,
            'smiles_1': smiles1,
            'smiles_2': smiles2,
            'ID1_smiles_1': id1,
            'ID2_smiles_2': id2,
            'labels': lbl,
            'df_tabular_features': tabular,
            'PC1_hist': hist1,
            'PC2_hist': hist2,
            'hist_centers': center,
            'freq_1': f1,
            'freq_2': f2,
            'freq_1_df': f1_df,
            'freq_2_df': f2_df
        }

    data_dict = {
        'train': build_dict(data_PC1_train, data_PC2_train, freq_1_2_train,
                            mordred1_train, mordred2_train,
                            smiles_1_train, smiles_2_train, ID1_smiles_1_train, ID2_smiles_2_train,
                            labels_train, df_tabular_features, PC1_hist_ext, PC2_hist_ext, hist_centers_ext,
                            freq_1_train, freq_2_train, freq_1_df_train, freq_2_df_train),

        'test': build_dict(data_PC1_test, data_PC2_test, freq_1_2_test,
                           mordred1_test, mordred2_test,
                           smiles_1_test, smiles_2_test, ID1_smiles_1_test, ID2_smiles_2_test,
                           labels_test, df_tabular_features_ext, PC1_hist_ext, PC2_hist_ext, hist_centers_ext,
                           freq_1_test, freq_2_test, freq_1_df_test, freq_2_df_test)
    }

    norm_data_dict = {
        'train': build_dict(data_PC1_train_norm, data_PC2_train_norm, freq_1_2_train_norm,
                            mordred1_train_norm, mordred2_train_norm,
                            smiles_1_train, smiles_2_train, ID1_smiles_1_train, ID2_smiles_2_train,
                            labels_train, df_tabular_features, PC1_hist_ext, PC2_hist_ext, hist_centers_ext,
                            freq_1_train_norm, freq_2_train_norm, freq_1_df_train, freq_2_df_train),

        'test': build_dict(data_PC1_test_norm, data_PC2_test_norm, freq_1_2_test_norm,
                           mordred1_test_norm, mordred2_test_norm,
                           smiles_1_test, smiles_2_test, ID1_smiles_1_test, ID2_smiles_2_test,
                           labels_test, df_tabular_features_ext, PC1_hist_ext, PC2_hist_ext, hist_centers_ext,
                           freq_1_test_norm, freq_2_test_norm, freq_1_df_test, freq_2_df_test)
    }

    return data_dict, norm_data_dict




def data_split_normalize_mord_Unique_Freq(data_zip_unique, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10, random_state=104):
    (
        PC_array, smiles_list, ID_list,
        df_tabular_features,
        freq_tabular_features,  # original freq dataframe
        PC_hist_smooth, hist_centers,
        mordred_array,
        freq_tabular_features_array,  # numpy array (float64)
        freq_tabular_features_df      # cleaned freq dataframe (no _1/_2 suffix)
    ) = data_zip_unique

    # === Generate indices ===
    total_indices = np.arange(len(ID_list))

    # Split train+val vs test
    idx_temp, idx_test = train_test_split(
        total_indices, test_size=test_ratio, random_state=random_state, shuffle=True
    )

    # Split train vs val
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    idx_train, idx_val = train_test_split(
        idx_temp, test_size=val_ratio_adj, random_state=random_state, shuffle=True
    )

    # === Helper Function to Index Arrays ===
    def split(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.iloc[idx_train], x.iloc[idx_val], x.iloc[idx_test]
        elif isinstance(x, list):
            x = np.array(x)
        return x[idx_train], x[idx_val], x[idx_test]

    # === Split All ===
    PC_train, PC_val, PC_test = split(PC_array)
    freq_train, freq_val, freq_test = split(freq_tabular_features_array)
    mordred_train, mordred_val, mordred_test = split(mordred_array)
    smiles_train, smiles_val, smiles_test = split(smiles_list)
    ID_train, ID_val, ID_test = split(ID_list)
    PC_hist_train, PC_hist_val, PC_hist_test = split(PC_hist_smooth)

    # === Normalize Point Clouds ===
    def Normalize_PC(train, val_or_test, features=4):
        train = train.copy()
        val_or_test = val_or_test.copy()
        for i in range(features):
            mean = train[:, :, i].mean()
            std = train[:, :, i].std()
            train[:, :, i] = (train[:, :, i] - mean) / std
            val_or_test[:, :, i] = (val_or_test[:, :, i] - mean) / std
        return train, val_or_test

    PC_train_norm, PC_val_norm = Normalize_PC(PC_train, PC_val)
    PC_train_norm, PC_test_norm = Normalize_PC(PC_train, PC_test)

    # === Normalize Freq and Mordred ===
    scaler_freq = MinMaxScaler()
    freq_train_norm = scaler_freq.fit_transform(freq_train)
    freq_val_norm = scaler_freq.transform(freq_val)
    freq_test_norm = scaler_freq.transform(freq_test)

    scaler_mordred = MinMaxScaler()
    mordred_train_norm = scaler_mordred.fit_transform(mordred_train)
    mordred_val_norm = scaler_mordred.transform(mordred_val)
    mordred_test_norm = scaler_mordred.transform(mordred_test)

    # === Final Dictionary Builder ===
    def build_dict(pc, freq, mordred, smiles, ids, pc_hist):
        return {
            'data_PC': pc,
            'freq': freq,
            'Mordred': mordred,
            'smiles': smiles,
            'ID': ids,
            'df_tabular_features': df_tabular_features,
            'PC_hist': pc_hist,
            'hist_centers': hist_centers
        }

    # === Unnormalized Data ===
    data_dict = {
        'train': build_dict(PC_train, freq_train, mordred_train, smiles_train, ID_train, PC_hist_train),
        'val': build_dict(PC_val, freq_val, mordred_val, smiles_val, ID_val, PC_hist_val),
        'test': build_dict(PC_test, freq_test, mordred_test, smiles_test, ID_test, PC_hist_test)
    }

    # === Normalized Data ===
    norm_data_dict = {
        'train': build_dict(PC_train_norm, freq_train_norm, mordred_train_norm, smiles_train, ID_train, PC_hist_train),
        'val': build_dict(PC_val_norm, freq_val_norm, mordred_val_norm, smiles_val, ID_val, PC_hist_val),
        'test': build_dict(PC_test_norm, freq_test_norm, mordred_test_norm, smiles_test, ID_test, PC_hist_test)
    }

    return data_dict, norm_data_dict


def normalize_and_build_mordred_dicts(
    smiles1_train, smiles2_train, labels_train, mordred1_train, mordred2_train,
    smiles1_val, smiles2_val, labels_val, mordred1_val, mordred2_val,
    smiles1_test, smiles2_test, labels_test, mordred1_test, mordred2_test,
    smiles1_EXT_test, smiles2_EXT_test, labels_EXT_test, mordred1_EXT_test, mordred2_EXT_test
):
    # === Use the same scaler for both Mordred1 and Mordred2 ===
    scaler = MinMaxScaler()
    
    # Fit on combined mordred1 and mordred2 training data
    combined_train = np.vstack([mordred1_train, mordred2_train])
    scaler.fit(combined_train)

    # Transform all sets
    mordred1_train_norm = scaler.transform(mordred1_train)
    mordred2_train_norm = scaler.transform(mordred2_train)

    mordred1_val_norm = scaler.transform(mordred1_val)
    mordred2_val_norm = scaler.transform(mordred2_val)

    mordred1_test_norm = scaler.transform(mordred1_test)
    mordred2_test_norm = scaler.transform(mordred2_test)

    mordred1_EXT_test_norm = scaler.transform(mordred1_EXT_test)
    mordred2_EXT_test_norm = scaler.transform(mordred2_EXT_test)

    # === Build data dict ===
    def build_dict(sm1, sm2, lbls, m1, m2):
        return {
            "smiles_1": list(sm1),
            "smiles_2": list(sm2),
            "labels": list(lbls),
            "Mordred1": m1,
            "Mordred2": m2
        }

    data_dicts = {
        "train": build_dict(smiles1_train, smiles2_train, labels_train, mordred1_train_norm, mordred2_train_norm),
        "val": build_dict(smiles1_val, smiles2_val, labels_val, mordred1_val_norm, mordred2_val_norm),
        "test": build_dict(smiles1_test, smiles2_test, labels_test, mordred1_test_norm, mordred2_test_norm),
        "EXT_test": build_dict(smiles1_EXT_test, smiles2_EXT_test, labels_EXT_test, mordred1_EXT_test_norm, mordred2_EXT_test_norm)
    }

    return data_dicts




# MultiModal_dict_CC_mord = load_CC_data_modalities_PC_dft_mord(multi_modal_CC_data_dict_mord, PC_dft_dict)
# MultiModal_dict_CC_EXT_mord = load_CC_data_modalities_PC_dft_mord(multi_modal_EXT_CC_data_dict_mord, PC_dft_dict)


# data_dict, norm_data_dict = data_split_normalize_ext_mord(MultiModal_dict_CC_mord, MultiModal_dict_CC_EXT_mord, train_ratio=0.8, val_ratio = 0.1, test_ratio = 0.1, Random_state= 104)

def create_datasets_ext_mord_Freq(data_zip_source, data_zip_ext, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=104):
    data_dict, norm_data_dict = data_split_normalize_ext_mord_Freq(
        data_zip_source, data_zip_ext,
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state
    )
    train_dict, val_dict, test_dict = norm_data_dict['train'], norm_data_dict['test'], norm_data_dict['test']
    train_dataset = MultiModal_Dataset_mord_Freq(train_dict)
    val_dataset = MultiModal_Dataset_mord_Freq(val_dict)
    test_dataset = MultiModal_Dataset_mord_Freq(test_dict)
    return train_dataset, val_dataset, test_dataset


def create_datasets_mord_Freq(data_zip, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=104):
    data_dict, norm_data_dict = data_split_normalize_mord_Freq(
        data_zip, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, random_state=random_state
    )
    train_dict, val_dict, test_dict = norm_data_dict['train'], norm_data_dict['val'], norm_data_dict['test']
    train_dataset = MultiModal_Dataset_mord_Freq(train_dict)
    val_dataset = MultiModal_Dataset_mord_Freq(val_dict)
    test_dataset = MultiModal_Dataset_mord_Freq(test_dict)
    return train_dataset, val_dataset, test_dataset


def create_datasets_mord_Unique_Freq(data_zip_unique, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=104):
    data_dict, norm_data_dict = data_split_normalize_mord_Unique_Freq(
        data_zip_unique, train_ratio=train_ratio,
        val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state
    )
    train_dict, val_dict, test_dict = norm_data_dict['train'], norm_data_dict['val'], norm_data_dict['test']
    train_dataset = MultiModal_Dataset_mord_Unique_Freq(train_dict)
    val_dataset = MultiModal_Dataset_mord_Unique_Freq(val_dict)
    test_dataset = MultiModal_Dataset_mord_Unique_Freq(test_dict)
    return train_dataset, val_dataset, test_dataset




def create_Mordred_datasets_ext_mord(norm_data_dict ):

    train_dict, val_dict, test_dict , EXT_test_dict = norm_data_dict['train'], norm_data_dict['val'], norm_data_dict['test'], norm_data_dict['EXT_test']
    train_dataset = Dataset_mord(train_dict)
    val_dataset = Dataset_mord(val_dict)
    test_dataset = Dataset_mord(test_dict)
    EXT_test_dataset = Dataset_mord(EXT_test_dict)

    return train_dataset, val_dataset, test_dataset, EXT_test_dataset



class MultiModal_Dataset(Dataset):
    def __init__(self, data_dict):
        self.data_PC1 = torch.tensor(data_dict['data_PC1'], dtype=torch.float32)
        self.data_PC2 = torch.tensor(data_dict['data_PC2'], dtype=torch.float32)
        self.freq_1_2 = torch.tensor(data_dict['freq_1_2'], dtype=torch.float32)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.float32)
        # self.df_tabular_features = torch.tensor(data_dict['df_tabular_features'].astype(np.float32).values,dtype=torch.float32)
        # self.df_tabular_features =data_dict['df_tabular_features']

        self.PC1_hist = torch.tensor(data_dict['PC1_hist'], dtype=torch.float32)
        self.PC2_hist = torch.tensor(data_dict['PC2_hist'], dtype=torch.float32)
        self.hist_centers = torch.tensor(data_dict['hist_centers'], dtype=torch.float32)  # shared across all

        self.smiles_1 = pd.Series(data_dict['smiles_1']).reset_index(drop=True)
        self.smiles_2 = pd.Series(data_dict['smiles_2']).reset_index(drop=True)
        self.ID1_smiles_1 = pd.Series(data_dict['ID1_smiles_1']).reset_index(drop=True)
        self.ID2_smiles_2 = pd.Series(data_dict['ID2_smiles_2']).reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        PC1 = self.data_PC1[idx]
        PC2 = self.data_PC2[idx]
        freq_1_2 = self.freq_1_2[idx]
        # df_tabular = self.df_tabular_features[idx]
        PC1_hist = self.PC1_hist[idx]
        PC2_hist = self.PC2_hist[idx]
        label = self.labels[idx]

        smiles_1 = self.smiles_1.iloc[idx]
        smiles_2 = self.smiles_2.iloc[idx]
        ID1_smiles_1 = self.ID1_smiles_1.iloc[idx]
        ID2_smiles_2 = self.ID2_smiles_2.iloc[idx]

        return {
            "PC1": PC1,
            "PC2": PC2,
            "freq_1_2": freq_1_2,
            "PC1_hist": PC1_hist,
            "PC2_hist": PC2_hist,
            "hist_centers": self.hist_centers,  # shared for all
            "smiles_1": smiles_1,
            "smiles_2": smiles_2,
            "ID1": ID1_smiles_1,
            "ID2": ID2_smiles_2,
            "label": label
        }

class MultiModal_Dataset_mord_Freq(Dataset):
    def __init__(self, data_dict):
        self.data_PC1 = torch.tensor(data_dict['data_PC1'], dtype=torch.float32)
        self.data_PC2 = torch.tensor(data_dict['data_PC2'], dtype=torch.float32)
        # self.freq_1_2 = torch.tensor(data_dict['freq_1_2'], dtype=torch.float32)
        self.freq_1_2 =torch.tensor(data_dict['freq_1'], dtype=torch.float32)

        self.mordred1 = torch.tensor(data_dict['Mordred1'], dtype=torch.float32)
        self.mordred2 = torch.tensor(data_dict['Mordred2'], dtype=torch.float32)
        self.freq_1 = torch.tensor(data_dict['freq_1'], dtype=torch.float32)
        self.freq_2 = torch.tensor(data_dict['freq_2'], dtype=torch.float32)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.float32)

        self.PC1_hist = torch.tensor(data_dict['PC1_hist'], dtype=torch.float32)
        self.PC2_hist = torch.tensor(data_dict['PC2_hist'], dtype=torch.float32)
        self.hist_centers = torch.tensor(data_dict['hist_centers'], dtype=torch.float32)  # shared across all

        self.smiles_1 = pd.Series(data_dict['smiles_1']).reset_index(drop=True)
        self.smiles_2 = pd.Series(data_dict['smiles_2']).reset_index(drop=True)
        self.ID1_smiles_1 = pd.Series(data_dict['ID1_smiles_1']).reset_index(drop=True)
        self.ID2_smiles_2 = pd.Series(data_dict['ID2_smiles_2']).reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "PC1": self.data_PC1[idx],
            "PC2": self.data_PC2[idx],
            "freq_1_2": self.freq_1_2[idx],
            "Mordred1": self.mordred1[idx],
            "Mordred2": self.mordred2[idx],
            "freq_1":self.freq_1[idx],
            "freq_2":self.freq_2[idx],
            "PC1_hist": self.PC1_hist[idx],
            "PC2_hist": self.PC2_hist[idx],
            "hist_centers": self.hist_centers,  # shared for all
            "smiles_1": self.smiles_1.iloc[idx],
            "smiles_2": self.smiles_2.iloc[idx],
            "ID1": self.ID1_smiles_1.iloc[idx],
            "ID2": self.ID2_smiles_2.iloc[idx],
            "label": self.labels[idx]
        }
    


class MultiModal_Dataset_mord_Unique_Freq(Dataset):
    def __init__(self, data_dict):
        self.data_PC1 = torch.tensor(data_dict['data_PC'], dtype=torch.float32)
        self.freq = torch.tensor(data_dict['freq'], dtype=torch.float32)
        self.mordred = torch.tensor(data_dict['Mordred'], dtype=torch.float32)

        self.PC1_hist = torch.tensor(data_dict['PC_hist'], dtype=torch.float32)
        self.hist_centers = torch.tensor(data_dict['hist_centers'], dtype=torch.float32)

        self.smiles = pd.Series(data_dict['smiles']).reset_index(drop=True)
        self.ID = pd.Series(data_dict['ID']).reset_index(drop=True)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            "PC": self.data_PC1[idx],
            "freq": self.freq[idx],
            "Mordred": self.mordred[idx],
            "PC_hist": self.PC1_hist[idx],
            "hist_centers": self.hist_centers,  # shared for all samples
            "smiles": self.smiles.iloc[idx],
            "ID": self.ID.iloc[idx]
        }


class Dataset_mord(Dataset):
    def __init__(self, data_dict):

        self.mordred1 = torch.tensor(data_dict['Mordred1'], dtype=torch.float32)
        self.mordred2 = torch.tensor(data_dict['Mordred2'], dtype=torch.float32)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.float32)

        self.smiles_1 = pd.Series(data_dict['smiles_1']).reset_index(drop=True)
        self.smiles_2 = pd.Series(data_dict['smiles_2']).reset_index(drop=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "Mordred1": self.mordred1[idx],
            "Mordred2": self.mordred2[idx],
            "smiles_1": self.smiles_1.iloc[idx],
            "smiles_2": self.smiles_2.iloc[idx],
            "label": self.labels[idx]
        }









""" GRAPH DATASET CLASSES """



# ===============================================
# Feature Definitions
# ===============================================
    
allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)),  # Atomic numbers 1 to 118
        'possible_hybridization_list': [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.SP2D,  # Added missing hybridization
            Chem.rdchem.HybridizationType.UNSPECIFIED
        ],
        'possible_Aromatic': [False, True],
        'possible_degree_list': list(range(11)),  # Max degree 10
        'possible_formal_charge_list': list(range(-5, 6)),  # Formal charge range -5 to 5
        'possible_numH_list': list(range(9)),  # Max 8 Hs on an atom
        'possible_implicit_valence_list': list(range(9)),  # Max implicit valence 8
        'possible_chirality_list': [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
            Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,  # Added
            Chem.rdchem.ChiralType.CHI_SQUAREPLANAR           # Added
        ],
        'possible_bonds': [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ],
        'possible_isAromaticBond': [False, True],
        'possible_isInRing': [False, True],
        'possible_bond_dirs': [
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT
        ],
    }



def atom_feature(atom, allowable_features):
    try:
        return [
            allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum()),
            allowable_features['possible_hybridization_list'].index(atom.GetHybridization()),
            allowable_features['possible_Aromatic'].index(atom.GetIsAromatic()),
            allowable_features['possible_degree_list'].index(atom.GetDegree()),
            allowable_features['possible_formal_charge_list'].index(atom.GetFormalCharge()),
            allowable_features['possible_numH_list'].index(atom.GetTotalNumHs()),
            allowable_features['possible_implicit_valence_list'].index(atom.GetImplicitValence()),
            allowable_features['possible_chirality_list'].index(atom.GetChiralTag())
        ]
    except ValueError as e:
        print(f"Error extracting atom features: {e} for atom: {atom.GetSymbol()}")
        return None  # Gracefully handle errors by returning None

def bond_feature(bond, allowable_features):
    try:
        return [
            allowable_features['possible_bonds'].index(bond.GetBondType()),
            allowable_features['possible_isAromaticBond'].index(bond.GetIsAromatic()),
            allowable_features['possible_isInRing'].index(bond.IsInRing()),
            allowable_features['possible_bond_dirs'].index(bond.GetBondDir())
        ]
    except ValueError as e:
        print(f"Error extracting bond features: {e} for bond between atoms: {bond.GetBeginAtomIdx()}-{bond.GetEndAtomIdx()}")
        return None  # Gracefully handle errors by returning None



# Canonicalize SMILES
def cano(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None


# Canonicalize and create a sorted pair key
def canonicalize_and_pair(row):
    smi1 = cano(row['smiles1'])
    smi2 = cano(row['smiles2'])
    if smi1 is None or smi2 is None:
        return pd.Series([None, None, None])
    return pd.Series([smi1, smi2, tuple(sorted([smi1, smi2]))])





def split_smiles(smiles, labels, train_size, val_size, random_state):
    """
    Stratified split of SMILES and labels into train, validation, and test sets.
    
    Ensures that each split maintains the original class distribution.
    
    Parameters:
        smiles (list or array-like): List of SMILES strings.
        labels (list or array-like): Corresponding labels.
        train_size (float): Fraction of the data to use for training.
        val_size (float): Fraction of the data to use for validation (out of total data).
        random_state (int): Random seed for reproducibility.
        
    Returns:
        smiles_train, smiles_val, smiles_test, y_train, y_val, y_test
    """
    # First split: train and (val + test)
    smiles_train, smiles_val_test, y_train, y_val_test = train_test_split(
        smiles,
        labels,
        test_size=1 - train_size,
        stratify=labels,
        random_state=random_state
    )

    # Second split: validation and test (from val_test set)
    relative_val_size = val_size / (1 - train_size)
    smiles_val, smiles_test, y_val, y_test = train_test_split(
        smiles_val_test,
        y_val_test,
        test_size=1 - relative_val_size,
        stratify=y_val_test,
        random_state=random_state
    )

    return smiles_train, smiles_val, smiles_test, y_train, y_val, y_test



def smi_to_pyg(smi, y):
    """Convert SMILES to PyTorch Geometric graph."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)  # This ensures that all implicit hydrogens are added

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
    edge_indices = list(zip(*atom_pairs))
    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = [atom_feature(atom, allowable_features) for atom in mol.GetAtoms()]
    bond_features = [bond_feature(bond, allowable_features) for bond in bonds]
    
    data_obj = Data(
        x=torch.FloatTensor(atom_features),
        edge_index=torch.LongTensor(edge_indices),
        edge_attr=torch.FloatTensor(bond_features),
        y=torch.FloatTensor([y]),
        mol=mol,
        smiles=smi
    )
    return data_obj
    

# ========================== Step 2: Convert SMILES to Graphs ==========================

def combine_Graph(Graph_list):
    """
    Merge a Graph with multiple subgraphs
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object
    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr
    y = Graph_list[0].y

    combined_Graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y = y)

    return combined_Graph






# -------------------------
# 1. Dataset Class
# -------------------------
class MyDataset_graph (Dataset):
    """Custom dataset to handle SMILES and labels."""
    def __init__(self, smiles, labels):
        mols = [smi_to_pyg(smi, y) for smi, y in tqdm(zip(smiles, labels), total=len(smiles))]
        self.X = [m for m in mols if m]  # Filter out None values

    def __getitem__(self, idx):
        return self.X[idx]
    
    def __len__(self):
        return len(self.X)





def create_graph_dataloader(smiles1, smiles2, labels, batch_size):
    """
    Split SMILES, convert them to graphs, and create PyTorch DataLoaders.
    """
    # Split SMILES
    
    
    smiles1_dataset = MyDataset_graph(smiles1, labels)
    smiles2_dataset = MyDataset_graph(smiles2, labels)
    
    combined_dataset = [combine_Graph([smiles1_dataset[i], smiles2_dataset[i]]) for i in range(len(smiles1_dataset))]
    
    
    dataset_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    
    print(" DataLoader size:", len(dataset_loader))
    
    return dataset_loader


def create_graph_dataloader_train_aug(smiles1, smiles2, labels, batch_size):
    """
    Split SMILES, convert them to graphs, and create PyTorch DataLoaders.
    """
    # Split SMILES
    
    
    smiles1_dataset = MyDataset_graph(smiles1, labels)
    smiles2_dataset = MyDataset_graph(smiles2, labels)
    
    combined_dataset = [combine_Graph([smiles1_dataset[i], smiles2_dataset[i]]) for i in range(len(smiles1_dataset))]
    
    
    dataset_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(" DataLoader size:", len(dataset_loader))
    
    return dataset_loader



def create_Unique_graph_dataloader(smiles1, labels, batch_size):
    """
    Split SMILES, convert them to graphs, and create PyTorch DataLoaders.
    """
    # Split SMILES
    
    
    smiles1_dataset = MyDataset_graph(smiles1, labels)
    
    
    
    dataset_loader = DataLoader(smiles1_dataset, batch_size=batch_size, shuffle=False)
    
    print(" DataLoader size:", len(dataset_loader))
    
    return dataset_loader

def graph_loader_to_dict_list(loader):
    """
    Converts a PyTorch Geometric DataLoader to a list of dictionaries, 
    each representing one graph with numpy arrays.
    
    Each dictionary has:
        - 'x': node features
        - 'edge_index': edge indices
        - 'edge_attr': edge features
        - 'y': label
    """
    all_graphs = []

    for batch in loader:
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            edge_mask = mask[batch.edge_index[0]]  # keep edges from nodes in the current graph

            graph_dict = {
                'x': batch.x[mask].detach().cpu().numpy(),
                'edge_index': batch.edge_index[:, edge_mask].detach().cpu().numpy(),
                'edge_attr': batch.edge_attr[edge_mask].detach().cpu().numpy(),
                'y': batch.y[i].detach().cpu().numpy()
            }
            all_graphs.append(graph_dict)
    
    return all_graphs



def graph_label_ratio(loader):
    label_counts = {'0': 0, '1': 0}  # Dictionary to store counts of labels
    total_entries = 0  # Total number of entries in the DataLoader
    
    for batch in loader:
        
        
        label_i = batch.y
        #labels = batch.y.numpy()  # Assuming labels are stored in .y attribute and converted to numpy array
        label_counts['0'] += (label_i == 0).sum()  # Count occurrences of label 0
        label_counts['1'] += (label_i == 1).sum()  # Count occurrences of label 1
        total_entries += label_i.shape[0]  # Update total entries count
    
    # Calculate the ratio of 1 to 0 in the labels
    ratio_1_to_0 = label_counts['1'] / label_counts['0'] if label_counts['0'] != 0 else float('inf')
    
    return ratio_1_to_0, total_entries



def get_true_labels(dataloader):
    labels = []
    for batch in dataloader:
        y = batch["label"] if isinstance(batch, dict) else batch[-1]
        labels.append(y)
    return torch.cat(labels).cpu().numpy()
# === Load and Split Features ===
def load_and_split_mordred(name, save_folder_mordred):
    save_folder =save_folder_mordred 

    data = np.load(os.path.join(save_folder, f'mordred_full_{name}.npy'))
    half = data.shape[1] // 2
    mordred1 = data[:, :half]
    mordred2 = data[:, half:]
    return mordred1, mordred2

def get_smiles_and_labels(dataset):
    return dataset.smiles_1.tolist(), dataset.smiles_2.tolist(), dataset.labels.tolist()



