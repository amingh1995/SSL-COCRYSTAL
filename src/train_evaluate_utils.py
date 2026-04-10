#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 22:08:14 2025

@author: amin
"""

import numpy as np
from tqdm import tqdm, trange
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score, precision_score, roc_curve, confusion_matrix
import pandas as pd



def apply_random_rotation(point_cloud):
    # Generate random angles for rotation around x, y, and z axes
    theta_x = np.random.uniform(low=0.0, high=2*np.pi)
    theta_y = np.random.uniform(low=0.0, high=2*np.pi)
    theta_z = np.random.uniform(low=0.0, high=2*np.pi)
    
    # Compute the rotation matrices around each axis
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(theta_x), -np.sin(theta_x)],
                                  [0, np.sin(theta_x), np.cos(theta_x)]])
    
    rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                                  [0, 1, 0],
                                  [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    rotation_matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                                  [np.sin(theta_z), np.cos(theta_z), 0],
                                  [0, 0, 1]])
    
    # Apply the rotations to the x, y, and z coordinates of the point cloud
    point_cloud_xyz = point_cloud[:, :3]
    rotated_xyz = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, np.dot(rotation_matrix_x, point_cloud_xyz.T))).T
    
    # Concatenate the rotated x, y, and z coordinates with the unchanged additional feature values
    rotated_point_cloud = np.concatenate([rotated_xyz, point_cloud[:, 3:]], axis=1)
    
    return rotated_point_cloud


def augment_rot_asymmetric(point_cloud_dataset, labels, pos_aug=2, neg_aug=2):
    # Move to CPU and convert to NumPy
    if isinstance(point_cloud_dataset, torch.Tensor):
        point_cloud_dataset = point_cloud_dataset.cpu().float().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Separate positive and negative samples
    pos_mask = labels == 1
    neg_mask = labels == 0

    PC_pos = point_cloud_dataset[pos_mask]
    PC_neg = point_cloud_dataset[neg_mask]

    labels_pos = labels[pos_mask]
    labels_neg = labels[neg_mask]

    # Function to apply N augmentations
    def apply_multiple_rotations(pc_set, num_aug):
        augmented = [pc_set]
        for _ in range(num_aug - 1):  # already includes original
            augmented_i = np.zeros_like(pc_set)
            for i in range(pc_set.shape[0]):
                augmented_i[i] = apply_random_rotation(pc_set[i])
            augmented.append(augmented_i)
        return np.concatenate(augmented, axis=0)

    # Augment each class
    PC_pos_aug = apply_multiple_rotations(PC_pos, pos_aug)
    PC_neg_aug = apply_multiple_rotations(PC_neg, neg_aug)

    # Concatenate
    PC_all = np.concatenate([PC_pos_aug, PC_neg_aug], axis=0)
    labels_all = np.concatenate([labels_pos] * pos_aug + [labels_neg] * neg_aug)

    # Convert back to torch
    PC_all = torch.tensor(PC_all, dtype=torch.float32)
    labels_all = torch.tensor(labels_all, dtype=torch.long)

    return PC_all, labels_all



def calculate_classification_metrics(targets, predictions):
    """Calculate binary classification metrics."""
    predictions_binary = (predictions >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions_binary).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    tnr = tn / (tn + fp)  # True Negative Rate
    bacc = (tpr + tnr) / 2
    recall = recall_score(targets, predictions_binary)
    auc = roc_auc_score(targets, predictions)
    f1 = f1_score(targets, predictions_binary)
    return bacc, recall, auc, tpr, tnr, f1







def train_epoch(model, multimodal_loader, graph_loader, criterion, optimizer, device, Data_Type_name):
    model.train()
    total_loss = 0.0
    predictions = []
    targets = []


    # pick correct iterator
    data_iter = (
        zip(multimodal_loader, graph_loader) if Data_Type_name == 'multimodal'
        else graph_loader if Data_Type_name == 'GAT'
        else multimodal_loader
    )

    # wrap with tqdm for progress bar
    desc =  "Training"
    data_iter = tqdm(data_iter, desc=desc, leave=False)

    for batch in data_iter:
        if Data_Type_name == 'multimodal':
            multimodal_batch, graph_batch = batch
            multimodal_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in multimodal_batch.items()
            }
            graph_batch = graph_batch.to(device)
            target_batch = multimodal_batch['label'].to(device).float()
            outputs = model(multimodal_batch, graph_batch)
    
            # Ensure shape is correct for loss computation
            if outputs.dim() == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(-1)

        elif Data_Type_name == 'Mordred':
            target_batch = batch['label'].to(device).float()
            # outputs = model(batch['df_tabular'].to(device))
            
            Mordred1 = batch['Mordred1'].to(device)
            Mordred2 = batch['Mordred2'].to(device)
            target_batch = batch['label'].to(device).float()
            outputs = model(Mordred1, Mordred2)
            

        elif Data_Type_name == 'Mordred_freq':
            target_batch = batch['label'].to(device).float()
        
            Mordred1 = batch['Mordred1'].to(device)
            Mordred2 = batch['Mordred2'].to(device)
            freq_1 = batch['freq_1'].to(device)
            freq_2 = batch['freq_2'].to(device)

            outputs = model(Mordred1, Mordred2, freq_1,freq_2)
    
        
        elif Data_Type_name == 'freq_hist':
            target_batch = batch['label'].to(device).float()
        
            freq_input = batch['freq_1_2'].to(device)
            PC1_hist = batch['PC1_hist'].to(device)
            PC2_hist = batch['PC2_hist'].to(device)
            outputs = model( freq_input, PC1_hist, PC2_hist)

        elif Data_Type_name == 'GAT':
            batch = batch.to(device)
            target_batch = batch.y.to(device).float()
            outputs = model(batch)

        elif Data_Type_name == 'PC':
            PC1_aug, labels_aug = augment_rot_asymmetric(batch['PC1'], batch['label'], pos_aug=1, neg_aug=1)
            PC2_aug, _ = augment_rot_asymmetric(batch['PC2'], batch['label'], pos_aug=1, neg_aug=1)
            PC1_aug = PC1_aug.to(device).float()
            PC2_aug = PC2_aug.to(device).float()
            target_batch = labels_aug.to(device).float()
            outputs = model(PC1_aug, PC2_aug)

        elif Data_Type_name == 'HistFusion':
            PC1_hist = batch['PC1_hist'].to(device)
            PC2_hist = batch['PC2_hist'].to(device)
            target_batch = batch['label'].to(device).float()
            logits, outputs = model(PC1_hist, PC2_hist)
        
            # Ensure shape is correct for loss computation
            if outputs.dim() == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(-1)


        elif Data_Type_name == 'freq':

            freq_1 = batch['freq_1'].to(device)
            freq_2 = batch['freq_2'].to(device)
            target_batch = batch['label'].to(device).float()
            outputs = model(freq_1, freq_2)

        elif Data_Type_name == 'freq_old':
            freq_input = batch['freq_1_2'].to(device)
            target_batch = batch['label'].to(device).float()
            outputs = model(freq_input)

        elif Data_Type_name == 'CCNetMLP':
            PC1_hist = batch['PC1_hist'].to(device)
            PC2_hist = batch['PC2_hist'].to(device)
            target_batch = batch['label'].to(device).float()
            outputs = model(PC1_hist, PC2_hist)
            # Sanity fix for undesired shape [32, 1, 1]
    
            # Fix output shape if necessary
            if outputs.dim() == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(-1)

        else:
            raise ValueError(f"Unsupported Data_Type_name: {Data_Type_name}")
        
        # print("outputs:", outputs)
        # print("target:", target_batch)
        # print("outputs shape:", outputs.shape)
        # print("target shape:", target_batch.shape)
        target_batch = target_batch.view(-1, 1)  # shape [32, 1]

        loss = criterion(outputs, target_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: grad mean = {param.grad.mean().item():.6f}")

        predictions.extend(outputs.detach().cpu().numpy().flatten())
        total_loss += loss.item() * len(target_batch)
        targets.extend(target_batch.cpu().numpy())

    avg_loss = total_loss / len(targets)
    return avg_loss, predictions, targets


def validate_model(model, multimodal_loader, graph_loader, criterion, device, Data_Type_name):
    model.eval()
    total_loss = 0.0
    val_predictions, val_targets = [], []

    data_iter = (
        zip(multimodal_loader, graph_loader) if Data_Type_name == 'multimodal'
        else graph_loader if Data_Type_name == 'GAT'
        else multimodal_loader
    )

    with torch.no_grad():
        for batch in data_iter:
            if Data_Type_name == 'multimodal':
                multimodal_batch, graph_batch = batch
                multimodal_batch = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in multimodal_batch.items()
                }
                graph_batch = graph_batch.to(device)
                target_batch = multimodal_batch['label'].to(device).float()
                outputs = model(multimodal_batch, graph_batch)
        
                # Ensure shape is correct for loss computation
                if outputs.dim() == 3 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(-1)
                
            elif Data_Type_name == 'Mordred':
                target_batch = batch['label'].to(device).float()
                # outputs = model(batch['df_tabular'].to(device))
                
    
                Mordred1 = batch['Mordred1'].to(device)
                Mordred2 = batch['Mordred2'].to(device)
                target_batch = batch['label'].to(device).float()
                outputs = model(Mordred1, Mordred2)
            


            elif Data_Type_name == 'Mordred_freq':
                target_batch = batch['label'].to(device).float()
            
                Mordred1 = batch['Mordred1'].to(device)
                Mordred2 = batch['Mordred2'].to(device)
                freq_1 = batch['freq_1'].to(device)
                freq_2 = batch['freq_2'].to(device)

                outputs = model(Mordred1, Mordred2, freq_1,freq_2)
        
            
        
            elif Data_Type_name == 'freq_hist':
                target_batch = batch['label'].to(device).float()
            
                freq_input = batch['freq_1_2'].to(device)
                PC1_hist = batch['PC1_hist'].to(device)
                PC2_hist = batch['PC2_hist'].to(device)
                outputs = model( freq_input, PC1_hist, PC2_hist)
    
                
            elif Data_Type_name == 'GAT':
                batch = batch.to(device)
                target_batch = batch.y.to(device).float()
                outputs = model(batch)

            elif Data_Type_name == 'PC':
                PC1 = batch['PC1'].to(device).float()
                PC2 = batch['PC2'].to(device).float()
                target_batch = batch['label'].to(device).float()
                outputs = model(PC1, PC2)
                
            elif Data_Type_name == 'freq_old':
                freq_input = batch['freq_1_2'].to(device)
                target_batch = batch['label'].to(device).float()
                outputs = model(freq_input)
                
            elif Data_Type_name == 'freq':
    
                freq_1 = batch['freq_1'].to(device)
                freq_2 = batch['freq_2'].to(device)
                target_batch = batch['label'].to(device).float()
                outputs = model(freq_1, freq_2)


            elif Data_Type_name == 'CCNetMLP':
                PC1_hist = batch['PC1_hist'].to(device)
                PC2_hist = batch['PC2_hist'].to(device)
                target_batch = batch['label'].to(device).float()
                outputs = model(PC1_hist, PC2_hist)
                
                # Fix output shape if necessary
                if outputs.dim() == 3 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(-1)
            elif Data_Type_name == 'HistFusion':
                PC1_hist = batch['PC1_hist'].to(device)
                PC2_hist = batch['PC2_hist'].to(device)
                target_batch = batch['label'].to(device).float()
                logits, outputs = model(PC1_hist, PC2_hist)
            
                # Ensure shape is correct for loss computation
                if outputs.dim() == 3 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported Data_Type_name: {Data_Type_name}")

            val_predictions.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(target_batch.cpu().numpy().flatten())
            total_loss += criterion(outputs, target_batch.view(-1, 1)).item() * target_batch.size(0)

    avg_val_loss = total_loss / len(val_targets)
    metrics = calculate_classification_metrics(np.array(val_targets), np.array(val_predictions))
    return avg_val_loss, metrics, val_predictions




def train_val_model(model,
                    train_mm_loader, train_graph_loader,
                    val_mm_loader,   val_graph_loader,
                    test_mm_loader,  test_graph_loader,
                    test_mm_loader_EXT, graph_test_loader_EXT,
                    model_file_path,
                    criterion, optimizer,
                    device, problem_type,
                    Data_Type_name,
                    patience, num_epochs,
                    lambda_var=1):
    """
    Trains `model` and selects the best checkpoint by maximizing:
        val_score = val_bacc - lambda_var * Var([val_tpr, val_tnr])
    """

    best_val_score = -float('inf')
    best_model_state = None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"\n================= EPOCH: {epoch+1}/{num_epochs} ======================")

        # --- TRAIN ---
        avg_train_loss, _, _ = train_epoch(
            model, train_mm_loader, train_graph_loader,
            criterion, optimizer, device, Data_Type_name
        )

        # --- VALIDATE ON TRAIN (optional logging) ---
        _, train_metrics, _ = validate_model(
            model, train_mm_loader, train_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- VALIDATE ON VAL SPLIT ---
        avg_val_loss, val_metrics, _ = validate_model(
            model, val_mm_loader, val_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- EVALUATE ON EXT TEST (for logging) ---
        avg_test_loss_EXT, test_metrics_EXT, _ = validate_model(
            model, test_mm_loader_EXT, graph_test_loader_EXT,
            criterion, device, Data_Type_name
        )

        # --- LOGGING ---
        if problem_type == 'regression':
            r2, rmse, mae, mse = val_metrics
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Validation R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        elif problem_type == 'binary_classification':
            train_bacc, train_recall, train_auc, train_tpr, train_tnr, train_f1 = train_metrics
            val_bacc,   val_recall,   val_auc,   val_tpr,   val_tnr,   val_f1   = val_metrics
            ext_bacc,   ext_recall,   ext_auc,   ext_tpr,   ext_tnr,   ext_f1   = test_metrics_EXT

            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Train → BACC: {train_bacc:.4f}, TPR: {train_tpr:.4f}, TNR: {train_tnr:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
            print(f"Valid → BACC: {val_bacc:.4f}, TPR: {val_tpr:.4f}, TNR: {val_tnr:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
            print(f"EXT test → BACC: {ext_bacc:.4f}, TPR: {ext_tpr:.4f}, TNR: {ext_tnr:.4f}, AUC: {ext_auc:.4f}, F1: {ext_f1:.4f}")

            # compute composite validation score
            var_t = np.var([val_tpr, val_tnr])
            val_score = val_bacc - lambda_var * var_t
            print(f"Composite Val Score: {val_score:.4f}  (BACC {val_bacc:.4f}  –  λ·Var {var_t:.6f})")

        elif problem_type == 'multiclass_classification':
            bacc      = val_metrics['overall']['macro_recall']
            recall_PM = val_metrics['per_class']['0.0']['recall']
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Validation BACC: {bacc:.4f}, Minor Recall: {recall_PM:.4f}")

            # For multiclass you could set val_score = bacc if desired
            # var_t = 0.0; val_score = bacc

        # --- EARLY STOPPING / CHECKPOINT on val_score ---
        if problem_type == 'binary_classification':
            current_score = val_score
        elif problem_type == 'multiclass_classification':
            current_score = bacc
        else:  # regression
            # fallback: neg val_loss
            current_score = -avg_val_loss

        if current_score > best_val_score:
            print(f"✅ Best score improved from {best_val_score:.4f} → {current_score:.4f}. Saving model.")
            best_val_score    = current_score
            best_model_state  = model.state_dict()
            best_model        = model
            torch.save(best_model_state, model_file_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    # --- FINAL EVALUATION on TEST SET ---
    print("\n🧪 Evaluating on test data...")
    avg_test_loss, test_metrics, Test_preds = validate_model(
        best_model, test_mm_loader, test_graph_loader,
        criterion, device, Data_Type_name
    )
    print("📊 Test Metrics:", test_metrics)

    # --- FINAL EVALUATION on EXTERNAL TEST SET ---
    print("\n🧪 Evaluating on EXTernal test data...")
    avg_test_loss_EXT, test_metrics_EXT, Test_preds_EXT = validate_model(
        best_model, test_mm_loader_EXT, graph_test_loader_EXT,
        criterion, device, Data_Type_name
    )
    print("📊 EXT Test Metrics:", test_metrics_EXT)

    return best_model_state, best_model






def train_val_model_history(model,
                    train_mm_loader, train_graph_loader,
                    val_mm_loader,   val_graph_loader,
                    test_mm_loader,  test_graph_loader,
                    test_mm_loader_EXT, graph_test_loader_EXT,
                    model_file_path,
                    criterion, optimizer,
                    device, problem_type,
                    Data_Type_name,
                    patience, num_epochs,
                    lambda_var=1):
    """
    Trains `model` and selects the best checkpoint by maximizing:
        val_score = val_bacc - lambda_var * Var([val_tpr, val_tnr])

    Also logs metrics per epoch into DataFrames (train/val/test/ext_test).
    """

    best_val_score = -float('inf')
    best_model_state = None
    early_stop_counter = 0
    best_model = model

    # --- storage dicts for metrics ---
    metrics_history = {
        "train": {"epoch": [], "bacc": [], "tpr": [], "tnr": [], "auc": []},
        "val":   {"epoch": [], "bacc": [], "tpr": [], "tnr": [], "auc": []},
        "test":  {"epoch": [], "bacc": [], "tpr": [], "tnr": [], "auc": []},
        "ext":   {"epoch": [], "bacc": [], "tpr": [], "tnr": [], "auc": []}
    }

    for epoch in range(num_epochs):
        print(f"\n================= EPOCH: {epoch+1}/{num_epochs} ======================")

        # --- TRAIN ---
        avg_train_loss, _, _ = train_epoch(
            model, train_mm_loader, train_graph_loader,
            criterion, optimizer, device, Data_Type_name
        )

        # --- VALIDATE ON TRAIN ---
        _, train_metrics, _ = validate_model(
            model, train_mm_loader, train_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- VALIDATE ON VAL SPLIT ---
        avg_val_loss, val_metrics, _ = validate_model(
            model, val_mm_loader, val_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- EVALUATE ON TEST SPLIT ---
        _, test_metrics, _ = validate_model(
            model, test_mm_loader, test_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- EVALUATE ON EXT TEST SPLIT ---
        _, ext_metrics, _ = validate_model(
            model, test_mm_loader_EXT, graph_test_loader_EXT,
            criterion, device, Data_Type_name
        )

        if problem_type == 'binary_classification':
            # unpack
            train_bacc, _, train_auc, train_tpr, train_tnr, _ = train_metrics
            val_bacc,   _, val_auc,   val_tpr,   val_tnr,   _ = val_metrics
            test_bacc,  _, test_auc,  test_tpr,  test_tnr,  _ = test_metrics
            ext_bacc,   _, ext_auc,   ext_tpr,   ext_tnr,   _ = ext_metrics

            # --- log metrics ---
            metrics_history["train"]["epoch"].append(epoch+1)
            metrics_history["train"]["bacc"].append(train_bacc)
            metrics_history["train"]["tpr"].append(train_tpr)
            metrics_history["train"]["tnr"].append(train_tnr)
            metrics_history["train"]["auc"].append(train_auc)

            metrics_history["val"]["epoch"].append(epoch+1)
            metrics_history["val"]["bacc"].append(val_bacc)
            metrics_history["val"]["tpr"].append(val_tpr)
            metrics_history["val"]["tnr"].append(val_tnr)
            metrics_history["val"]["auc"].append(val_auc)

            metrics_history["test"]["epoch"].append(epoch+1)
            metrics_history["test"]["bacc"].append(test_bacc)
            metrics_history["test"]["tpr"].append(test_tpr)
            metrics_history["test"]["tnr"].append(test_tnr)
            metrics_history["test"]["auc"].append(test_auc)

            metrics_history["ext"]["epoch"].append(epoch+1)
            metrics_history["ext"]["bacc"].append(ext_bacc)
            metrics_history["ext"]["tpr"].append(ext_tpr)
            metrics_history["ext"]["tnr"].append(ext_tnr)
            metrics_history["ext"]["auc"].append(ext_auc)

            # print logging
            print(f"Train → BACC: {train_bacc:.4f}, TPR: {train_tpr:.4f}, TNR: {train_tnr:.4f}, AUC: {train_auc:.4f}")
            print(f"Valid → BACC: {val_bacc:.4f}, TPR: {val_tpr:.4f}, TNR: {val_tnr:.4f}, AUC: {val_auc:.4f}")
            print(f"Test  → BACC: {test_bacc:.4f}, TPR: {test_tpr:.4f}, TNR: {test_tnr:.4f}, AUC: {test_auc:.4f}")
            print(f"EXT   → BACC: {ext_bacc:.4f}, TPR: {ext_tpr:.4f}, TNR: {ext_tnr:.4f}, AUC: {ext_auc:.4f}")

            # composite validation score
            var_t = np.var([val_tpr, val_tnr])
            val_score = val_bacc - lambda_var * var_t
            print(f"Composite Val Score: {val_score:.4f}")

        # --- EARLY STOPPING / CHECKPOINT ---
        current_score = (
            val_score if problem_type == 'binary_classification'
            else -avg_val_loss
        )

        if current_score > best_val_score:
            print(f"✅ Best score improved from {best_val_score:.4f} → {current_score:.4f}. Saving model.")
            best_val_score    = current_score
            best_model_state  = model.state_dict()
            best_model        = model
            torch.save(best_model_state, model_file_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    # --- FINAL: convert to DataFrames and save ---
    df_train = pd.DataFrame(metrics_history["train"])
    df_val   = pd.DataFrame(metrics_history["val"])
    df_test  = pd.DataFrame(metrics_history["test"])
    df_ext   = pd.DataFrame(metrics_history["ext"])

    # save with same name as model
    base_name = model_file_path.split(".")[0]
    df_train.to_csv(f"{base_name}_train_metrics.csv", index=False)
    df_val.to_csv(f"{base_name}_val_metrics.csv", index=False)
    df_test.to_csv(f"{base_name}_test_metrics.csv", index=False)
    df_ext.to_csv(f"{base_name}_ext_metrics.csv", index=False)

    print(f"✅ Metrics saved with prefix: {base_name}_*.csv")

    return best_model_state, best_model, df_train, df_val, df_test, df_ext




def train_val_model_PU(model,
                    train_mm_loader, train_graph_loader,
                    val_mm_loader,   val_graph_loader,
                    test_mm_loader,  test_graph_loader,
                    test_mm_loader_EXT, graph_test_loader_EXT,
                    model_file_path,
                    criterion, optimizer,
                    device, problem_type,
                    Data_Type_name,
                    patience, num_epochs,
                    lambda_var=1):
    """
    Trains `model` and selects the best checkpoint by maximizing:
        val_score = val_bacc - lambda_var * Var([val_tpr, val_tnr])
    """

    best_val_score = -float('inf')
    best_model_state = None
    early_stop_counter = 0

    # NEW: Store all metrics history
    metrics_history = {
        "train": [],
        "val": [],
        "test": [],
        "ext_test": []
    }


# for epoch in trange(num_epochs, desc="Training epochs"):
#     train_loss = train_one_epoch(...)
#     val_loss = evaluate(...)
    for epoch in range(num_epochs):

        # --- TRAIN ---
        avg_train_loss, _, _ = train_epoch(
            model, train_mm_loader, train_graph_loader,
            criterion, optimizer, device, Data_Type_name
        )

        # --- VALIDATE ON TRAIN ---
        _, train_metrics, _ = validate_model(
            model, train_mm_loader, train_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- VALIDATE ON VAL SPLIT ---
        avg_val_loss, val_metrics, _ = validate_model(
            model, val_mm_loader, val_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- EVALUATE ON EXT TEST ---
        avg_test_loss, test_metrics, _ = validate_model(
            model, test_mm_loader, test_graph_loader,
            criterion, device, Data_Type_name
        )

        # --- EVALUATE ON EXT TEST ---
        avg_test_loss_EXT, test_metrics_EXT, _ = validate_model(
            model, test_mm_loader_EXT, graph_test_loader_EXT,
            criterion, device, Data_Type_name
        )

        # === NEW: Append metrics to history ===
        metrics_history["train"].append({
            "epoch": epoch + 1,
            "loss": avg_train_loss,
            "metrics": train_metrics
        })
        metrics_history["val"].append({
            "epoch": epoch + 1,
            "loss": avg_val_loss,
            "metrics": val_metrics
        })
        metrics_history["test"].append({
            "epoch": epoch + 1,
            "loss": avg_test_loss,
            "metrics": test_metrics
        })
        metrics_history["ext_test"].append({
            "epoch": epoch + 1,
            "loss": avg_test_loss_EXT,
            "metrics": test_metrics_EXT
        })

        # --- LOGGING ---
        if problem_type == 'regression':
            r2, rmse, mae, mse = val_metrics
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Validation R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        elif problem_type == 'binary_classification':
            train_bacc, train_recall, train_auc, train_tpr, train_tnr, train_f1 = train_metrics
            val_bacc,   val_recall,   val_auc,   val_tpr,   val_tnr,   val_f1   = val_metrics
            test_bacc,   test_recall,   test_auc,   test_tpr,   test_tnr,   test_f1   = test_metrics
            ext_bacc,   ext_recall,   ext_auc,   ext_tpr,   ext_tnr,   ext_f1   = test_metrics_EXT

            print(f"==========================   Epoch: {epoch} ======================== ")

            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Train → BACC: {train_bacc:.4f}, TPR: {train_tpr:.4f}, TNR: {train_tnr:.4f}, AUC: {train_auc:.4f}, F1: {train_f1:.4f}")
            print(f"Valid → BACC: {val_bacc:.4f}, TPR: {val_tpr:.4f}, TNR: {val_tnr:.4f}, AUC: {val_auc:.4f}, F1: {val_f1:.4f}")
            print(f"Test → BACC: {test_bacc:.4f}, TPR: {test_tpr:.4f}, TNR: {test_tnr:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}")

            print(f"EXT test → BACC: {ext_bacc:.4f}, TPR: {ext_tpr:.4f}, TNR: {ext_tnr:.4f}, AUC: {ext_auc:.4f}, F1: {ext_f1:.4f}")

            # compute composite validation score
            var_t = np.var([val_tpr, val_tnr])
            val_score = val_bacc - lambda_var * var_t
            print(f"Composite Val Score: {val_score:.4f}  (BACC {val_bacc:.4f}  –  λ·Var {var_t:.6f})")


            print(f"==================================================================== ")


        elif problem_type == 'multiclass_classification':
            bacc      = val_metrics['overall']['macro_recall']
            recall_PM = val_metrics['per_class']['0.0']['recall']
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Validation BACC: {bacc:.4f}, Minor Recall: {recall_PM:.4f}")

        # --- EARLY STOPPING / CHECKPOINT ---
        if problem_type == 'binary_classification':
            current_score = val_score
        elif problem_type == 'multiclass_classification':
            current_score = bacc
        else:
            current_score = -avg_val_loss

        if current_score > best_val_score:
            print(f"✅ Best score improved from {best_val_score:.4f} → {current_score:.4f}. Saving model.")
            best_val_score    = current_score
            best_model_state  = model.state_dict()
            best_model        = model
            torch.save(best_model_state, model_file_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    # --- FINAL EVALUATION on TEST SET ---
    print("\n🧪 Evaluating on test data...")
    avg_test_loss, test_metrics, Test_preds = validate_model(
        best_model, test_mm_loader, test_graph_loader,
        criterion, device, Data_Type_name
    )
    print("📊 Test Metrics:", test_metrics)

    # --- FINAL EVALUATION on EXTERNAL TEST SET ---
    print("\n🧪 Evaluating on EXTernal test data...")
    avg_test_loss_EXT, test_metrics_EXT, Test_preds_EXT = validate_model(
        best_model, test_mm_loader_EXT, graph_test_loader_EXT,
        criterion, device, Data_Type_name
    )
    print("📊 EXT Test Metrics:", test_metrics_EXT)

    return best_model_state, best_model, metrics_history


def train_epoch_Unique(model1, model2, loader1, loader2,
                       optimizer1, optimizer2, device, Data_Type_name_2):
    model1.train()
    model2.train()
    total_loss = 0.0

    for batch1, batch2 in zip(loader1, loader2):
        # Prepare input1: if dict, cat all tensors except label
        if isinstance(batch1, dict):
            parts = []
            for k, v in batch1.items():
                if k != 'label' and torch.is_tensor(v):
                    parts.append(v.to(device))
            input1 = torch.cat(parts, dim=1)
        else:
            input1 = batch1.to(device)

        # Prepare input2 based on modality
        if Data_Type_name_2 == 'Mordred':
            input2 = batch2['Mordred'].to(device)
        elif Data_Type_name_2 == 'Mordred_freq':
            m1 = batch2['Mordred1'].to(device)
            m2 = batch2['Mordred2'].to(device)
            f  = batch2['freq_1_2'].to(device)
            input2 = torch.cat([m1, m2, f], dim=1)
        elif Data_Type_name_2 == 'PC':
            input2 = batch2['PC1'].to(device).float()
        elif Data_Type_name_2 == 'freq':
            input2 = batch2['freq_1_2'].to(device)
        elif Data_Type_name_2 in ('CCNetMLP','Conv1DHist'):
            input2 = batch2['PC_hist'].to(device)
        else:
            raise ValueError(f"Unsupported Data_Type_name_2: {Data_Type_name_2}")

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # Forward through projector
        out1 = model1.projector(model1(input1))
        out2 = model2.projector(model2(input2))
        loss = model1.loss_fn(out1, out2)

        loss.backward()
        optimizer1.step()
        optimizer2.step()

        total_loss += loss.item() * input1.size(0)

    return total_loss / len(loader1.dataset)


def validate_model_Unique(model1, model2, loader1, loader2,
                          device, Data_Type_name_2):
    model1.eval()
    model2.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch1, batch2 in zip(loader1, loader2):
            if isinstance(batch1, dict):
                parts = [v.to(device) for k,v in batch1.items() if k!='label' and torch.is_tensor(v)]
                input1 = torch.cat(parts, dim=1)
            else:
                input1 = batch1.to(device)

            if Data_Type_name_2 == 'Mordred':
                input2 = batch2['Mordred'].to(device)
            elif Data_Type_name_2 == 'Mordred_freq':
                m1 = batch2['Mordred1'].to(device)
                m2 = batch2['Mordred2'].to(device)
                f  = batch2['freq_1_2'].to(device)
                input2 = torch.cat([m1, m2, f], dim=1)
            elif Data_Type_name_2 == 'PC':
                input2 = batch2['PC1'].to(device).float()
            elif Data_Type_name_2 == 'freq':
                input2 = batch2['freq_1_2'].to(device)
            elif Data_Type_name_2 in ('CCNetMLP','Conv1DHist'):
                input2 = batch2['PC_hist'].to(device)
            else:
                raise ValueError(f"Unsupported Data_Type_name_2: {Data_Type_name_2}")

            out1 = model1.projector(model1(input1))
            out2 = model2.projector(model2(input2))
            loss = model1.loss_fn(out1, out2)
            total_loss += loss.item() * input1.size(0)

    return total_loss / len(loader1.dataset)


def train_val_model_Unique(model1, model2,
                           train_loader1, train_loader2,
                           val_loader1,   val_loader2,
                           test_loader1, test_loader2,
                           model_path,
                           optimizer1, optimizer2,
                           device, num_epochs, patience,
                           Data_Type_name_2):
    best_val_loss = float('inf')
    best_state1, best_state2 = None, None
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
        train_loss = train_epoch_Unique(
            model1, model2,
            train_loader1, train_loader2,
            optimizer1, optimizer2,
            device, Data_Type_name_2
        )
        val_loss = validate_model_Unique(
            model1, model2,
            val_loader1, val_loader2,
            device, Data_Type_name_2
        )
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state1 = model1.state_dict()
            best_state2 = model2.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    torch.save({'model1': best_state1, 'model2': best_state2}, model_path)
    print(f"✅ Best model saved to {model_path}")
    
    
    
    
    # Final evaluation
    print("\n🧪 Evaluating on test data...")
    test_loss = validate_model_Unique(
        model1, model2,
        test_loader1, test_loader2,
        device, Data_Type_name_2
    )
    print("📊 Test Metrics:")
    print(test_loss)

    return model1, best_state1, model2, best_state2














def compute_embedding_norm(embeddings):
    
    """
    Compute the average L2 norm of a batch of embeddings.

    Given:
        embeddings: Tensor of shape [batch_size, dim]
    
    Formula:
        For each embedding vector z_i ∈ ℝ^d:
            ||z_i||₂ = sqrt(sum(z_i^2))
        Then average over batch:
            mean_norm = (1/N) * Σᵢ ||z_i||₂

    Purpose:
        - Monitor feature scale
        - Detect embedding collapse (norm ≈ 0) or explosion (very large norm)
    
    Returns:
        Scalar mean L2 norm of the batch
    """
    norms = torch.norm(embeddings, dim=1)  # Compute ||z_i||₂ for each embedding
    return norms.mean().item()             # Average over batch

def off_diagonal(x):
    """
    Extract off-diagonal elements from a square matrix x ∈ ℝ^{d×d}.
    
    Used in Barlow Twins loss:
        - The goal is to reduce redundancy across dimensions of the learned representation.
        - Off-diagonal values in the correlation matrix should approach 0.

    Returns:
        A flattened 1D tensor of all elements in x where i ≠ j.
    """
    n, m = x.shape
    assert n == m, "Matrix must be square"
    
    # Trick to exclude diagonal: flatten the matrix, offset reshape, then remove diagonal entries
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def compute_cross_correlation(z1, z2):
    """
    Compute the Barlow Twins cross-correlation matrix between two batches of embeddings.

    Given:
        z1, z2: Tensors of shape [batch_size, embed_dim]
                Must correspond to two "views" or modalities of the same samples.

    Step 1 - Standardization:
        For each dimension j:
            μ_j = mean(z[:, j])       # mean of j-th feature
            σ_j = std(z[:, j])        # std of j-th feature
            z[:, j] = (z[:, j] - μ_j) / σ_j

    Step 2 - Cross-correlation matrix C ∈ ℝ^{d×d}:
        C_ij = (1/N) * Σₙ z1[n,i] * z2[n,j]
        Ideal:
            - Diagonal elements C_ii ≈ 1
            - Off-diagonal elements C_ij ≈ 0 for i ≠ j

    Step 3 - Loss terms:
        Diagonal loss (invariance):
            on_diag = Σ_i (C_ii - 1)^2
        Off-diagonal loss (redundancy reduction):
            off_diag = Σ_{i≠j} (C_ij)^2

    Returns:
        on_diag: scalar value, penalizes deviation from identity diagonal
        off_diag: scalar value, penalizes non-zero off-diagonal terms
    """
    batch_size = z1.size(0)

    # === Step 1: Standardize across batch ===
    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)

    # === Step 2: Cross-correlation matrix ===
    c = (z1_norm.T @ z2_norm) / batch_size  # C ∈ ℝ^{d×d}

    # === Step 3: Loss components ===
    on_diag = torch.diagonal(c).add_(-1).pow(2).sum().item()  # Σ_i (C_ii - 1)^2
    off_diag = off_diagonal(c).pow(2).sum().item()            # Σ_{i≠j} (C_ij)^2

    return on_diag, off_diag


def train_epoch_Unique(model1, model2, loader1, loader2,
                       optimizer1, optimizer2, device, Data_Type_name_2):
    model1.train()
    model2.train()
    total_loss = 0.0
    norms_gnn, diag_losses, off_diag_losses = [], [], []

    for batch1, batch2 in zip(loader1, loader2):
        if isinstance(batch1, dict):
            parts = [v.to(device) for k, v in batch1.items() if k != 'label' and torch.is_tensor(v)]
            input1 = torch.cat(parts, dim=1)
        else:
            input1 = batch1.to(device)

        if Data_Type_name_2 == 'Mordred':
            input2 = batch2['Mordred'].to(device)
        elif Data_Type_name_2 == 'Mordred_freq':
            m1 = batch2['Mordred1'].to(device)
            m2 = batch2['Mordred2'].to(device)
            f  = batch2['freq_1_2'].to(device)
            input2 = torch.cat([m1, m2, f], dim=1)
        elif Data_Type_name_2 == 'PC':
            input2 = batch2['PC'].to(device).float()
        elif Data_Type_name_2 == 'freq':
            input2 = batch2['freq_1_2'].to(device)
        elif Data_Type_name_2 in ('CCNetMLP','Conv1DHist'):
            input2 = batch2['PC_hist'].to(device)
        else:
            raise ValueError(f"Unsupported Data_Type_name_2: {Data_Type_name_2}")

        optimizer1.zero_grad()
        optimizer2.zero_grad()

        h1 = model1(input1)
        h2 = model2(input2)
        z1 = model1.projector(h1)
        z2 = model2.projector(h2)

        loss = model1.loss_fn(z1, z2)
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        total_loss += loss.item() * input1.size(0)

        # Track alignment metrics
        norms_gnn.append(compute_embedding_norm(h1))
        diag_loss, off_diag_loss = compute_cross_correlation(z1, z2)
        diag_losses.append(diag_loss)
        off_diag_losses.append(off_diag_loss)

    return (
        total_loss / len(loader1.dataset),
        sum(norms_gnn) / len(norms_gnn),
        sum(diag_losses) / len(diag_losses),
        sum(off_diag_losses) / len(off_diag_losses)
    )


def validate_model_Unique(model1, model2, loader1, loader2,
                          device, Data_Type_name_2):
    model1.eval()
    model2.eval()
    total_loss = 0.0
    norms_gnn, diag_losses, off_diag_losses = [], [], []

    with torch.no_grad():
        for batch1, batch2 in zip(loader1, loader2):
            if isinstance(batch1, dict):
                parts = [v.to(device) for k,v in batch1.items() if k != 'label' and torch.is_tensor(v)]
                input1 = torch.cat(parts, dim=1)
            else:
                input1 = batch1.to(device)

            if Data_Type_name_2 == 'Mordred':
                input2 = batch2['Mordred'].to(device)
            elif Data_Type_name_2 == 'Mordred_freq':
                m1 = batch2['Mordred1'].to(device)
                m2 = batch2['Mordred2'].to(device)
                f  = batch2['freq_1_2'].to(device)
                input2 = torch.cat([m1, m2, f], dim=1)
            elif Data_Type_name_2 == 'PC':
                input2 = batch2['PC'].to(device).float()
            elif Data_Type_name_2 == 'freq':
                input2 = batch2['freq_1_2'].to(device)
            elif Data_Type_name_2 in ('CCNetMLP','Conv1DHist'):
                input2 = batch2['PC_hist'].to(device)
            else:
                raise ValueError(f"Unsupported Data_Type_name_2: {Data_Type_name_2}")

            h1 = model1(input1)
            h2 = model2(input2)
            z1 = model1.projector(h1)
            z2 = model2.projector(h2)

            loss = model1.loss_fn(z1, z2)
            total_loss += loss.item() * input1.size(0)

            # Alignment metrics
            norms_gnn.append(compute_embedding_norm(h1))
            diag_loss, off_diag_loss = compute_cross_correlation(z1, z2)
            diag_losses.append(diag_loss)
            off_diag_losses.append(off_diag_loss)

    return (
        total_loss / len(loader1.dataset),
        sum(norms_gnn) / len(norms_gnn),
        sum(diag_losses) / len(diag_losses),
        sum(off_diag_losses) / len(off_diag_losses)
    )


def train_val_model_Unique(model1, model2,
                           train_loader1, train_loader2,
                           val_loader1,   val_loader2,
                           test_loader1, test_loader2,
                           model_path,
                           optimizer1, optimizer2,
                           device, num_epochs, patience,
                           Data_Type_name_2):
    best_val_loss = float('inf')
    best_state1, best_state2 = None, None
    early_stop_counter = 0


    # Initialize monitoring dict
    history = {
        'train_loss': [], 'val_loss': [],
        'train_diag': [], 'val_diag': [],
        'train_offdiag': [], 'val_offdiag': [],
        'train_l2norm': [], 'val_l2norm': [],
    }



    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")

        train_loss, train_norm, train_diag, train_off = train_epoch_Unique(
            model1, model2, train_loader1, train_loader2,
            optimizer1, optimizer2, device, Data_Type_name_2
        )
        val_loss, val_norm, val_diag, val_off = validate_model_Unique(
            model1, model2, val_loader1, val_loader2,
            device, Data_Type_name_2
        )

        print(f"Train Loss: {train_loss:.4f} | L2 Norm: {train_norm:.4f} | Diag: {train_diag:.4f} | Off-Diag: {train_off:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | L2 Norm: {val_norm:.4f} | Diag: {val_diag:.4f} | Off-Diag: {val_off:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state1 = model1.state_dict()
            best_state2 = model2.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break
        
        
        
        # Log values
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_diag'].append(train_diag)
        history['val_diag'].append(val_diag)
        history['train_offdiag'].append(train_off)
        history['val_offdiag'].append(val_off)
        history['train_l2norm'].append(train_norm)
        history['val_l2norm'].append(val_norm)



    # torch.save({'model1': best_state1, 'model2': best_state2}, model_path)
    torch.save(best_state1, model_path)

    print(f"✅ Best model saved to {model_path}")

    # Final Test
    print("\n🧪 Evaluating on test data...")
    test_loss, test_norm, test_diag, test_off = validate_model_Unique(
        model1, model2, test_loader1, test_loader2, device, Data_Type_name_2
    )

    print(f"Test Loss: {test_loss:.4f} | L2 Norm: {test_norm:.4f} | Diag: {test_diag:.4f} | Off-Diag: {test_off:.4f}")
    return model1, best_state1, model2, best_state2, history


















def evaluate_model_on_all_splits(
    weight_path,
    base_model_created,
    dataloaders_dict,
    Data_Type_name='freq',
    device='cpu'
):
    """
    Evaluate a trained model on multiple splits using appropriate dataloaders.

    Args:
        weight_path (str): Path to the saved .pth weight file.
        base_model_created (nn.Module): Model instance (created but not yet loaded).
        dataloaders_dict (dict): Dict with keys like 'train', 'val', 'test', 'ext' → DataLoaders.
        Data_Type_name (str): One of 'freq', 'CCNetMLP', 'Mordred', 'GAT', etc.
        device (str): 'cpu' or 'cuda'.

    Returns:
        tuple: (metrics_dict, predictions_dict) with split names as keys.
    """

    # Load weights and move model to device
    state_dict = torch.load(weight_path, map_location=device)
    base_model_created.load_state_dict(state_dict)
    model = base_model_created.to(device)
    model.eval()

    all_metrics = {}
    all_preds = {}

    for split_name, loader in dataloaders_dict.items():
        print(f"\n🧪 Evaluating on {split_name.upper()} data...")

        # Select correct loader type
        multimodal_loader = None
        graph_loader = None

        if Data_Type_name == 'GAT':
            graph_loader = loader
        else:
            multimodal_loader = loader

        # Skip if loader is None (safety)
        if loader is None:
            print(f"⚠️ Skipping {split_name.upper()} — No loader found.")
            continue

        # Validate
        avg_loss, metrics, preds = validate_model(
            model=model,
            multimodal_loader=multimodal_loader,
            graph_loader=graph_loader,
            criterion=model.loss_fn,
            device=device,
            Data_Type_name=Data_Type_name
        )

        print(f"📊 {split_name.upper()} Metrics:")
        print(metrics)

        all_metrics[split_name] = metrics
        all_preds[split_name] = preds

    return all_metrics, all_preds

