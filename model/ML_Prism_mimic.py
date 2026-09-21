import os
# DEVE essere fatto prima di importare torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, matthews_corrcoef, brier_score_loss, classification_report, precision_score, recall_score, f1_score, accuracy_score

import shap
import glob

from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split
import torch.nn.functional as F

from imblearn.over_sampling import SMOTE 

from typing import List



"""LSTM"""

# in questo caso (sepsis_onset_index=0) si prende la metà della serie.
"""def pad_or_truncate_sequence(seq, target_length, sepsis_onset_index, pred_length=1, padding_value=0.0):
    # Se onset == 0, usa metà sequenza per default
    if sepsis_onset_index == 0:
        sepsis_onset_index = len(seq) // 2

    # Escludi l'istante stesso dell'onset
    sepsis_onset_index = max(0, sepsis_onset_index - pred_length)

    # Inizio della finestra: contiamo indietro target_length elementi
    start_index = max(0, sepsis_onset_index - target_length)
    end_index = sepsis_onset_index

    truncated_seq = seq[start_index:end_index]

    # Padding se troppo corta
    if len(truncated_seq) < target_length:
        pad_len = target_length - len(truncated_seq)
        padding = torch.full((pad_len, *seq.shape[1:]), padding_value)
        truncated_seq = torch.cat([padding, truncated_seq], dim=0)

    # Troncamento finale se troppo lunga (per sicurezza)
    if len(truncated_seq) > target_length:
        truncated_seq = truncated_seq[-target_length:]

    # Check finale
    if len(truncated_seq) != target_length:
        print("🚨 ERRORE: lunghezza finale anomala")
        print(f"  onset={sepsis_onset_index}, start={start_index}, end={end_index}")
        print(f"  truncated shape: {truncated_seq.shape}")

    return truncated_seq"""

# in questo caso (sepsis_onset_index=0) anzichè prendere la metà della serie, si prende la fine.
def pad_or_truncate_sequence(seq, target_length, sepsis_onset_index, patient_id = 1, pred_length=1, padding_value=0.0):
    # Se onset == 0, usa la fine della sequenza e conta all'indietro
    if sepsis_onset_index == 0:
        sepsis_onset_index = len(seq)  # prendi la fine della sequenza

    # Escludi l'istante stesso dell'onset
    sepsis_onset_index = max(0, sepsis_onset_index - pred_length)

    # Inizio della finestra: contiamo indietro target_length elementi
    start_index = max(0, sepsis_onset_index - target_length)
    end_index = sepsis_onset_index

    truncated_seq = seq[start_index:end_index]

    # if patient_id == 1:
    #     print(f"Sepsis onset index: {sepsis_onset_index} - Start index: {start_index} - End index: {end_index}")
    #     print(f"\n Original truncated seq: {truncated_seq}")

    # Padding se troppo corta
    if len(truncated_seq) < target_length:
        pad_len = target_length - len(truncated_seq)
        padding = torch.full((pad_len, *seq.shape[1:]), padding_value)
        truncated_seq = torch.cat([padding, truncated_seq], dim=0)

    # Troncamento finale (per sicurezza)
    if len(truncated_seq) > target_length:
        truncated_seq = truncated_seq[-target_length:]

    # Check finale
    if len(truncated_seq) != target_length:
        print("🚨 ERRORE: lunghezza finale anomala")
        print(f"  onset={sepsis_onset_index}, start={start_index}, end={end_index}")
        print(f"  truncated shape: {truncated_seq.shape}")

    return truncated_seq

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, bidirectional=True, dropout=0.3, pooling="mean"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.pooling = pooling
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Classificatore finale con dropout
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 1)
        )

    def forward(self, x):
        # LSTM
        output, (h_n, c_n) = self.lstm(x)  
        # output: [B, T, H*directions]

        if self.pooling == "last":
            # Ultimo timestep
            out = output[:, -1, :]   # [B, H*directions]
        elif self.pooling == "mean":
            # Media su tutta la sequenza
            out = output.mean(dim=1)  # [B, H*directions]
        elif self.pooling == "max":
            # Max pooling
            out, _ = output.max(dim=1)  # [B, H*directions]
        else:
            raise ValueError("Pooling non supportato: scegli tra ['last', 'mean', 'max']")

        # Classificazione
        out = self.fc(out)        # [B, 1]
        return out.view(-1)    # [B]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # se usi multi-GPU

    # Per forzare determinismo nei layer di PyTorch (es. convoluzioni, dropout, etc.)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""FEATURE IMPORTANCE ANALYSIS"""

def permutation_feature_importance(
    model,
    X,
    y,
    metric_fn=matthews_corrcoef,  # <--- MCC di default, puoi cambiarlo
    n_repeats=5,
    baseline_score=None,
    feature_names=None,
    verbose=True
    ):

    model.eval()
    device = next(model.parameters()).device
    X = X.clone().to(device)
    y = y.clone().to(device)

    # --- baseline ---
    with torch.no_grad():
        base_probs = torch.sigmoid(model(X))
        base_preds = (base_probs > 0.5).long()

    mask = y != -100
    y_true = y[mask].cpu().numpy()
    y_pred = base_preds[mask].cpu().numpy()

    if baseline_score is None:
        baseline_score = metric_fn(y_true, y_pred)

    if verbose:
        print(f"\nBaseline {metric_fn.__name__}: {baseline_score:.4f}")

    # --- importanze ---
    importances = []
    std_devs = []

    for feature_idx in tqdm(range(X.shape[2]), desc="Permutation importance"):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.clone()
            perm = torch.randperm(X.shape[0])
            X_perm[:, :, feature_idx] = X[perm, :, feature_idx]

            with torch.no_grad():
                probs = torch.sigmoid(model(X_perm))
                preds = (probs > 0.5).long()

                # Applica lo stesso mask
                preds_masked = preds[mask].cpu().numpy()
                score = metric_fn(y_true, preds_masked)
                scores.append(score)

        diffs = baseline_score - np.array(scores)
        importances.append(np.mean(diffs))
        std_devs.append(np.std(diffs))

        if verbose:
            fname = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
            print(f"{fname}: Δ = {np.mean(diffs):.4f} ± {np.std(diffs):.4f}")

    # --- dataframe finale ---
    df_importance = pd.DataFrame({
        "Feature": feature_names if feature_names else [f"f{i}" for i in range(X.shape[2])],
        "Importance": importances,
        "StdDev": std_devs
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_importance

"""MIMIC"""
def load_and_prepare_data(folder_path: str):
    # === Caricamento dati ===
    df_train_X = pd.read_csv(os.path.join(folder_path,'train', 'X_train.tsv'), sep='\t')
    df_val_X   = pd.read_csv(os.path.join(folder_path,'val', 'X_val.tsv'), sep='\t')
    df_test_X  = pd.read_csv(os.path.join(folder_path,'test', 'X_test.tsv'), sep='\t')

    df_train_y = pd.read_csv(os.path.join(folder_path, 'train', 'y_train.tsv'), sep='\t', index_col=0)
    df_val_y   = pd.read_csv(os.path.join(folder_path, 'val', 'y_val.tsv'), sep='\t', index_col=0)
    df_test_y  = pd.read_csv(os.path.join(folder_path, 'test', 'y_test.tsv'), sep='\t', index_col=0)

    # === Riallina gli indici ===
    df_val_X = df_val_X.copy()
    df_val_X['Index'] = df_val_X['Index'] + df_train_X['Index'].iloc[-1] + 1

    df_test_X = df_test_X.copy()
    df_test_X['Index'] = df_test_X['Index'] + df_val_X['Index'].iloc[-1] + 1

    # === Concatenazione globale ===
    all_df = pd.concat([df_train_X, df_val_X, df_test_X])
    all_labels = pd.concat([df_train_y, df_val_y, df_test_y]).rename(columns={'0':'sepsis'})
    all_labels['Index'] = np.arange(0, len(all_df['Index'].unique()), 1, dtype=int)
    all_labels = all_labels[['Index', 'sepsis']]

    # === Funzione per aggiungere timestep ===
    def add_timestep(df, labels):
        group_list = []
        for i, group in df.groupby(by='Index'):
            group = group.copy()
            group['timestep'] = np.arange(-49, 0, 1, dtype=int)  # aggiunge timestep
            group_list.append(group)
        df_timestep = pd.concat(group_list, axis=0)
        df_timestep_labels = df_timestep.merge(labels, on='Index', how='left')
        return df_timestep_labels

    # === Dataset con timestep ===
    all_df_timestep_labels   = add_timestep(all_df, all_labels)
    train_df_timestep_labels = add_timestep(df_train_X, df_train_y.rename(columns={'0':'sepsis'}).reset_index().rename(columns={'index':'Index'}))
    val_df_timestep_labels   = add_timestep(df_val_X, df_val_y.rename(columns={'0':'sepsis'}).reset_index().rename(columns={'index':'Index'}))
    test_df_timestep_labels  = add_timestep(df_test_X, df_test_y.rename(columns={'0':'sepsis'}).reset_index().rename(columns={'index':'Index'}))

    return all_df_timestep_labels, train_df_timestep_labels, val_df_timestep_labels, test_df_timestep_labels

def debug_patient(patient_id, group, processed_seq, sepsis_onset_index, target_len):
    print(f"\n=== Patient {patient_id} ===")
    print(f"Total timesteps: {len(group)}")
    print(f"Sepsis onset index: {sepsis_onset_index}")
    print(f"Target length: {target_len}")
    print(f"Processed shape: {processed_seq.shape}")

    # Mostra prime e ultime righe originali
    print("\nOriginal first 5 timesteps:")
    print(group.drop(columns=["Index", "sepsis", "timestep"])[22:46])

    print("\nOriginal last 5 timesteps:")
    print(group.drop(columns=["Index", "sepsis", "timestep"]).tail())

    # Mostra sequenza processata
    print("\nProcessed sequence (first 5 rows):")
    print(processed_seq[:5])
    print("\nProcessed sequence (last 5 rows):")
    print(processed_seq[-5:])

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        alpha : peso per la classe positiva (0.25 è un valore tipico)
        gamma : focusing parameter (2 è un valore comune)
        reduction : 'mean', 'sum', o 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits : output del modello (non passati attraverso sigmoid)
        targets : label binarie (0/1)
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Calcola il focal loss
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train_and_evaluate_kfold_val_earlystop(
    data_by_patient,
    model_class,
    device,
    n_splits=5,
    batch_size=64,
    hidden_dim=4,
    num_layers=4,
    dropout=0.3,
    pooling="mean",
    lr=5e-3,
    max_epochs=50,
    val_ratio=0.2,
    patience=5,
    loss_fn=None,
    method="undersample",
    weight_decay_value=1e-4):
    """
    Cross-validation con validation set interno, early stopping e metriche complete.
    Il bilanciamento viene applicato solo sul sottoinsieme di training interno.
    """

    set_deterministic(42)

    patient_ids = list(data_by_patient.keys())
    patient_labels = [data_by_patient[pid]["y"].item() for pid in patient_ids]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")

        # --- Split esterno train/test ---
        train_ids = [patient_ids[i] for i in train_idx]
        test_ids = [patient_ids[i] for i in test_idx]

        X_train_full = torch.stack([data_by_patient[pid]["x"] for pid in train_ids])
        y_train_full = torch.stack([data_by_patient[pid]["y"] for pid in train_ids])
        X_test = torch.stack([data_by_patient[pid]["x"] for pid in test_ids])
        y_test = torch.stack([data_by_patient[pid]["y"] for pid in test_ids])

        # --- Split interno train/val (per early stopping) ---
        full_dataset = TensorDataset(X_train_full, y_train_full)
        val_size = int(val_ratio * len(full_dataset))
        train_size = len(full_dataset) - val_size

        split_gen = torch.Generator().manual_seed(42 + fold)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_gen)

        # --- Ricostruzione train dict per bilanciamento ---
        train_data_by_patient = {
            i: {"x": x, "y": y} for i, (x, y) in enumerate(train_dataset)
        }

        print("Applying balancing to training subset only...")
        train_data_by_patient = balance_patients(train_data_by_patient, method=method, random_state=42 + fold)

        # --- DataLoader ---
        loader_gen = torch.Generator().manual_seed(42 + fold)

        train_loader = DataLoader(
            TensorDataset(
                torch.stack([v["x"] for v in train_data_by_patient.values()]),
                torch.stack([v["y"] for v in train_data_by_patient.values()])
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_gen
        )

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        print(f"Train size (after balancing): {len(train_loader.dataset)}")
        print(f"Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

        """# --- CALCOLO DEL PESO DELLA CLASSE (CLASS WEIGHTING) ---
        # 1. Calcola il conteggio delle etichette (solo sul training set interno)
        y_train_counts = y_train_full[train_dataset.indices].float().sum(dim=0)
        n_samples = len(train_dataset)
        
        # 2. Calcola i pesi inversi della frequenza (per bilanciare l'imbalance)
        # Peso per la classe 0 (Non-sepsi)
        weight_0 = n_samples / (2 * (n_samples - y_train_counts))
        # Peso per la classe 1 (Sepsi)
        weight_1 = n_samples / (2 * y_train_counts) 
        
        # Il BCEWithLogitsLoss in PyTorch richiede solo il peso per la classe positiva (pos_weight)
        # Calcoliamo pos_weight come rapporto tra il peso della classe 0 e il peso della classe 1.
        pos_weight = weight_1 / weight_0"""

        pos_weight=torch.Tensor([1.0])
        
        print(f"Classe Sepsi (1) Weight (pos_weight): {pos_weight.item():.2f}")

        # Modello
        model = model_class(
            input_dim=X_train_full.shape[2],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            pooling=pooling
        ).to(device)

        # --- APPLICAZIONE DEL PESO ALLA LOSS FUNCTION ---
        # Usiamo BCEWithLogitsLoss con pos_weight
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device)
        ) 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)

        # Early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y.float())
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            train_loss /= len(train_loader)

            print(f"Epoch {epoch+1:02d}/{max_epochs} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch+1}")
                    break

        # Ripristina best model
        model.load_state_dict(best_state)
        model.eval()

        # Test finale
        with torch.no_grad():
            logits = model(X_test.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            y_pred = (probs > 0.5).astype(int)
            y_true = y_test.cpu().numpy()

        # Metriche globali
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, probs)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        brier = brier_score_loss(y_true, probs)
        cm = confusion_matrix(y_true, y_pred)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        # Metriche per classe
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        class0_metrics = class_report["0"]
        class1_metrics = class_report["1"]

        print(f"\n📊 Fold {fold+1} results:")
        print(f"ACC={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | Brier={brier:.3f}")
        print(f"Non-sepsis (0): P={class0_metrics['precision']:.3f}, R={class0_metrics['recall']:.3f}, F1={class0_metrics['f1-score']:.3f}")
        print(f"Sepsis (1):     P={class1_metrics['precision']:.3f}, R={class1_metrics['recall']:.3f}, F1={class1_metrics['f1-score']:.3f}")
        print(f"Confusion matrix:\n {cm}")

        fold_results.append({
            "acc": acc, "auc": auc, "f1": f1, "mcc": mcc, "brier": brier,
            "prec": prec, "rec": rec, "cm": cm,
            "class0": class0_metrics,
            "class1": class1_metrics
        })

    # Metriche medie tra i fold
    mean_metrics = {k: np.mean([f[k] for f in fold_results]) for k in ["acc", "auc", "prec", "rec", "f1", "mcc", "brier"]}
    mean_class0 = {m: np.mean([f["class0"][m] for f in fold_results]) for m in ["precision", "recall", "f1-score"]}
    mean_class1 = {m: np.mean([f["class1"][m] for f in fold_results]) for m in ["precision", "recall", "f1-score"]}

    print("\n================= RISULTATI MEDI =================")
    print(f"ACC={mean_metrics['acc']:.3f} | AUC={mean_metrics['auc']:.3f} | MCC={mean_metrics['mcc']:.3f} | Brier={mean_metrics['brier']:.3f}")
    print(f"\nClasse 0 (non sepsis): P={mean_class0['precision']:.3f}, R={mean_class0['recall']:.3f}, F1={mean_class0['f1-score']:.3f}")
    print(f"Classe 1 (sepsis):     P={mean_class1['precision']:.3f}, R={mean_class1['recall']:.3f}, F1={mean_class1['f1-score']:.3f}")

    # Confusion matrix finale
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    final_cm = confusion_matrix(all_y_true, all_y_pred)
    print(f"Final confusion matrix:\n{final_cm}")

    return {
        "folds": fold_results,
        "mean_global": mean_metrics,
        "mean_class0": mean_class0,
        "mean_class1": mean_class1
    }


"""BALANCING"""

def prepare_mimic_data(
    mimic_path='./datasets/MIMIC/carry_forward/mean', 
    target_len=24, 
    pred_length=1, 
    pop_patient=False, 
    n=1, 
    random_state=42, 
    verbose=True,
    features_selected=None):

    set_seed(42)

    # Assumendo che queste funzioni e la loro implementazione esistano
    mimic_df, train_df, val_df, test_df = load_and_prepare_data(mimic_path)

    mimic_df = mimic_df[features_selected]

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if pop_patient:
        # --- Modifica: Selezione Bilanciata per il Pop-out ---
        rng = np.random.default_rng(random_state)
        unique_ids = mimic_df['Index'].unique()
        
        # 1. Trova l'etichetta 'sepsis' (y) per ogni paziente (è la stessa per tutte le timestep)
        # Usiamo groupby e take(0) per ottenere la prima (e unica) etichetta per ogni ID
        patient_labels = mimic_df.groupby('Index')['sepsis'].first()
        
        # 2. Suddividi gli ID in base all'etichetta
        septic_ids = patient_labels[patient_labels == 1].index.tolist()
        non_septic_ids = patient_labels[patient_labels == 0].index.tolist()

        if n > len(unique_ids):
            raise ValueError(f"n={n} è maggiore del numero di pazienti disponibili ({len(unique_ids)}).")

        # 3. Calcola il numero di pazienti da estrarre per ogni classe
        # Cerchiamo di avere n/2 per classe, ma rispettando i limiti di ciascuna
        n_septic_max = len(septic_ids)
        n_non_septic_max = len(non_septic_ids)
        
        n_septic = min(int(np.ceil(n / 2)), n_septic_max)
        n_non_septic = min(n - n_septic, n_non_septic_max)
        
        # Aggiusta se la quota iniziale ha lasciato un residuo (es. non ci sono abbastanza pazienti nell'altra classe)
        if n_septic + n_non_septic < n:
            # Riempi il restante dalla classe con più disponibilità, se possibile
            remaining = n - (n_septic + n_non_septic)
            if remaining > 0:
                 # Se c'è un residuo, cerca di prenderlo dalla classe non ancora piena
                if n_septic_max - n_septic > 0:
                    n_septic += min(remaining, n_septic_max - n_septic)
                elif n_non_septic_max - n_non_septic > 0:
                    n_non_septic += min(remaining, n_non_septic_max - n_non_septic)
                    
        if n_septic + n_non_septic != n:
            print(f"ATTENZIONE: Non è stato possibile estrarre esattamente n={n} pazienti bilanciati. Estratti {n_septic + n_non_septic}. Controllare la distribuzione dei label.")
        
        # 4. Selezione stratificata
        # Scelta casuale e senza sostituzione
        selected_septic_ids = rng.choice(septic_ids, size=n_septic, replace=False)
        selected_non_septic_ids = rng.choice(non_septic_ids, size=n_non_septic, replace=False)
        
        # Combina e ordina per stabilità cross-run del set finale (opzionale, ma utile)
        selected_ids = np.sort(np.concatenate([selected_septic_ids, selected_non_septic_ids]))
        
        # --- Estrazione e Salvataggio (il resto del codice rimane simile) ---
        
        # Estrazione dei pazienti selezionati
        subset_df = mimic_df[mimic_df['Index'].isin(selected_ids)].copy()
        remaining_df = mimic_df[~mimic_df['Index'].isin(selected_ids)].copy()

        # Cartella per i singoli CSV
        save_dir = os.path.join(mimic_path, "backend", "random_patients")
        os.makedirs(save_dir, exist_ok=True)

        # Salva ciascun paziente con indice fisso
        for i, pid in enumerate(selected_ids):
            patient_df = subset_df[subset_df["Index"] == pid]
            patient_df.to_csv(os.path.join(save_dir, f"patient_{i:02d}_id{pid}.csv"), index=False)

        mimic_df = remaining_df.copy()

        if verbose:
            print(f"Extracted {len(selected_ids)} patient(s) (Target: {n}) on a total of {len(unique_ids)}")
            print(f"Sepsis Label Distribution (Extracted): 1={n_septic}, 0={n_non_septic}")
            print(f"Selected IDs: {selected_ids.tolist()}")
            print(f"Saved individual CSVs in: {save_dir}")
            
    data_by_patient = {}
    # ... (il resto del codice della funzione prepare_mimic_data continua qui)
    for patient_id, group in mimic_df.groupby("Index"):
        x = group.drop(columns=["Index", "sepsis", "timestep"]).values
        y = group["sepsis"].iloc[0]

        # per MIMIC usiamo la lunghezza intera della sequenza
        sepsis_onset_index = group.shape[0]

        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)

        data_by_patient[patient_id] = {
            "x": pad_or_truncate_sequence(
                x_tensor,
                target_length=target_len,
                sepsis_onset_index=sepsis_onset_index,
                pred_length=pred_length,
                padding_value=-100,
                patient_id=patient_id
            ),
            "y": y_tensor
        }

    return data_by_patient, device, mimic_df, train_df, val_df, test_df

def sequence_smote(sequences, n_samples, random_state=42):
    """
    Semplice Sequence-SMOTE per LSTM.
    
    Args:
        sequences: lista di dict {"x": tensor, "y": tensor}, classe minoritaria
        n_samples: numero di sequenze sintetiche da generare
    """
    random.seed(random_state)
    synthetic_sequences = []
    seq_array = [s["x"].numpy() for s in sequences] 
    
    for _ in range(n_samples):
        s1, s2 = random.sample(seq_array, 2)
        alpha = random.random()
        new_seq = s1 + alpha * (s2 - s1)
        synthetic_sequences.append(torch.tensor(new_seq, dtype=torch.float32))
    
    return synthetic_sequences

def sequence_smote_flattened(sequences_minority, n_samples_to_generate, k_neighbors=5, random_state=42):
    """
    Implementa SMOTE su serie temporali appiattite.
    
    Args:
        sequences_minority: Lista di dict {"x": tensor, "y": tensor} della classe minoritaria.
        n_samples_to_generate: Numero di sequenze sintetiche da generare.
        k_neighbors: Numero di vicini K-NN da considerare.
    
    Returns:
        synthetic_sequences_list: Lista di torch.tensor per le sequenze sintetiche.
    """
    
    # --- 1. Conversione, Preparazione e Appiattimento (Flatten) ---
    X_minority_list = [s["x"].numpy() for s in sequences_minority]
    if not X_minority_list:
        return []
    
    X_minority = np.stack(X_minority_list)
    n_samples, n_timesteps, n_features = X_minority.shape
    
    # Appiattisci le sequenze da (N, T, F) a (N, T*F)
    X_flattened = X_minority.reshape(n_samples, n_timesteps * n_features)
    
    # Etichette fittizie
    y = np.ones(n_samples) 
    
    # --- 2. Preparazione del Resampler SMOTE ---
    # Creiamo un finto campione maggioritario per ingannare SMOTE e ottenere solo i sintetici.
    X_fake_majority = np.zeros((1, n_timesteps * n_features))
    y_fake_majority = np.array([0])
    
    X_full = np.concatenate([X_flattened, X_fake_majority])
    y_full = np.concatenate([y, y_fake_majority])

    smote = SMOTE(
        sampling_strategy={1: n_samples + n_samples_to_generate}, # Target per la classe 1
        k_neighbors=k_neighbors, 
        random_state=random_state
    )

    # --- 3. Oversampling ---
    X_resampled_flattened, y_resampled = smote.fit_resample(X_full, y_full)

    # --- 4. Rimodellamento ed Estrazione ---
    
    # Filtra solo i campioni con etichetta 1
    synthetic_X_flattened = X_resampled_flattened[y_resampled == 1]
    
    # Estrai i campioni sintetici (dopo gli originali)
    synthetic_sequences_np = synthetic_X_flattened[n_samples:]

    # Rimodella da (N_synth, T*F) a (N_synth, T, F)
    synthetic_sequences_3d = synthetic_sequences_np.reshape(-1, n_timesteps, n_features)

    # Converti in tensori di PyTorch
    synthetic_sequences_list = [torch.tensor(s, dtype=torch.float32) for s in synthetic_sequences_3d]
    
    return synthetic_sequences_list

def sequence_smoteenn(data_by_patient, random_state=42):
    """
    Sequence-SMOTEENN: oversample minoranza + pulizia maggioranza.
    
    Args:
        data_by_patient: dict {pid: {"x": tensor(24,43), "y": tensor}}
    Returns:
        balanced_data: dict con sequenze bilanciate
    """
    random.seed(random_state)
    
    # Separiamo le classi
    septic = [v for v in data_by_patient.values() if v["y"].item()==1]
    non_septic = [v for v in data_by_patient.values() if v["y"].item()==0]
    
    n_to_add = len(non_septic) - len(septic)
    
    # --- STEP 1: Sequence-SMOTE ---
    seq_array = [s["x"].numpy() for s in septic]
    synthetic_sequences = []
    for _ in range(n_to_add):
        s1, s2 = random.sample(seq_array, 2)
        alpha = random.random()
        new_seq = s1 + alpha * (s2 - s1)
        synthetic_sequences.append(torch.tensor(new_seq, dtype=torch.float32))
    
    septic_augmented = septic + [{"x": s, "y": torch.tensor(1)} for s in synthetic_sequences]
    
    # --- STEP 2: Pulizia della maggioranza (ENN) ---
    # Calcoliamo distanza tra ogni sequenza della maggioranza e tutte le minoranza sintetiche
    # Flatten timesteps per il calcolo della distanza
    septic_flat = np.array([s["x"].numpy().reshape(-1) for s in septic_augmented])
    non_septic_flat = np.array([s["x"].numpy().reshape(-1) for s in non_septic])
    
    # Distanza minima di ogni maggioranza dalla minoranza
    distances = cdist(non_septic_flat, septic_flat, metric="euclidean")
    min_dist = distances.min(axis=1)
    
    # Scegliamo una soglia per rimuovere esempi confondenti (ENN)
    threshold = np.percentile(min_dist, 25)  # rimuove la maggioranza troppo vicina alla minoranza
    cleaned_non_septic = [v for i, v in enumerate(non_septic) if min_dist[i] > threshold]
    
    # --- Ricostruisci dataset finale ---
    balanced_data = {}
    idx = 0
    for v in septic_augmented + cleaned_non_septic:
        balanced_data[idx] = {"x": v["x"], "y": v["y"]}
        idx += 1
    
    # Stampe
    n_septic_new = sum(v["y"].item()==1 for v in balanced_data.values())
    n_non_septic_new = sum(v["y"].item()==0 for v in balanced_data.values())
    print(f"(After sequence-smoteenn) Septic: {n_septic_new}, Non-septic: {n_non_septic_new}")
    
    return balanced_data

def sequence_smotetomek(data_by_patient, random_state=42, tomek_percentile=25):
    """
    Sequence-SMOTETomek: oversample minoranza + rimozione esempi confondenti della maggioranza.
    
    Args:
        data_by_patient: dict {pid: {"x": tensor(24,43), "y": tensor}}
        random_state: seed per riproducibilità
        tomek_percentile: percentile distanza minima per rimuovere maggioranza
    Returns:
        balanced_data: dict bilanciato LSTM-safe
    """
    random.seed(random_state)
    
    # separa minoranza e maggioranza
    septic = [v for v in data_by_patient.values() if v["y"].item() == 1]
    non_septic = [v for v in data_by_patient.values() if v["y"].item() == 0]
    
    n_to_add = len(non_septic) - len(septic)
    
    # --- STEP 1: Sequence-SMOTE ---
    seq_array = [s["x"].numpy() for s in septic]
    synthetic_sequences = []
    for _ in range(n_to_add):
        s1, s2 = random.sample(seq_array, 2)
        alpha = random.random()
        new_seq = s1 + alpha * (s2 - s1)
        synthetic_sequences.append(torch.tensor(new_seq, dtype=torch.float32))
    
    septic_augmented = septic + [{"x": s, "y": torch.tensor(1)} for s in synthetic_sequences]
    
    # --- STEP 2: Tomek link LSTM-safe ---
    # Flatten per calcolare distanza tra sequenze
    septic_flat = np.array([s["x"].numpy().reshape(-1) for s in septic_augmented])
    non_septic_flat = np.array([s["x"].numpy().reshape(-1) for s in non_septic])
    
    distances = cdist(non_septic_flat, septic_flat, metric="euclidean")
    min_dist = distances.min(axis=1)
    
    # Rimuovi maggioranza troppo vicina alla minoranza (simile a Tomek link)
    threshold = np.percentile(min_dist, tomek_percentile)
    cleaned_non_septic = [v for i, v in enumerate(non_septic) if min_dist[i] > threshold]
    
    # --- Ricostruzione dataset finale ---
    balanced_data = {}
    idx = 0
    for v in septic_augmented + cleaned_non_septic:
        balanced_data[idx] = {"x": v["x"], "y": v["y"]}
        idx += 1
    
    # Stampe
    n_septic_new = sum(v["y"].item() == 1 for v in balanced_data.values())
    n_non_septic_new = sum(v["y"].item() == 0 for v in balanced_data.values())
    print(f"(After sequence-smotetomek) Septic: {n_septic_new}, Non-septic: {n_non_septic_new}")
    
    return balanced_data

from tslearn.metrics import dtw_path 

def sequence_tsmote_dtw_manual(sequences_minority, n_samples_to_generate, k_neighbors=5, random_state=42):
    """
    Implementazione manuale di TSMOTE (DTW-based) per time series.
    
    Args:
        sequences_minority: Lista di dict {"x": tensor, "y": tensor} della classe minoritaria.
        n_samples_to_generate: Numero di sequenze sintetiche da generare.
        k_neighbors: Numero di vicini K-NN da considerare.
    
    Returns:
        synthetic_sequences_list: Lista di torch.tensor per le sequenze sintetiche.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    X_minority_list = [s["x"].numpy() for s in sequences_minority]
    if not X_minority_list:
        return []
    
    X_minority = np.stack(X_minority_list)
    n_samples, n_timesteps, n_features = X_minority.shape
    synthetic_sequences = []

    # --- 1. Trova K-NN usando DTW come metrica ---
    
    # Per semplicità (e per evitare la custom metric su KNN di sklearn che non supporta DTW),
    # replichiamo la logica: scegliamo un campione S1 e un vicino S2.
    
    # Nota: L'uso corretto di DTW K-NN richiede la pre-calcolazione della matrice di distanza DTW,
    # ma per semplicità, useremo la logica di SMOTE:
    
    # Creiamo un array 2D dove ogni riga è una sequenza appiattita, 
    # per usare KNN standard (solo per la selezione del vicino, non per la distanza finale)
    X_flat = X_minority.reshape(n_samples, -1)
    
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean').fit(X_flat)
    
    for _ in range(n_samples_to_generate):
        # 1. Scegli casualmente un campione S1
        idx_s1 = random.randint(0, n_samples - 1)
        S1 = X_minority[idx_s1]
        
        # 2. Trova i suoi vicini euclidei nello spazio appiattito
        # Usiamo KNN euclideo per la vicinanza, anche se non è l'ideale per DTW, è un compromesso
        # che funziona senza precalcolare l'intera matrice DTW.
        distances, indices = nn.kneighbors(X_flat[idx_s1].reshape(1, -1))
        
        # 3. Scegli casualmente un vicino S2 (escludendo S1 stesso, indice 0)
        idx_s2_in_neighbors = random.randint(1, k_neighbors)
        idx_s2 = indices[0][idx_s2_in_neighbors]
        S2 = X_minority[idx_s2]
        
        # --- 4. Genera la sequenza sintetica usando DTW ---
        alpha = random.random()
        
        # Calcola il percorso di allineamento DTW tra S1 e S2
        # `dtw_path` restituisce una lista di tuple (indice_S1, indice_S2) per i timesteps allineati.
        # Costrainiamo DTW a non fare warping estremo (window=None significa warping libero)
        path, _ = dtw_path(S1, S2, global_constraint="sakoe_chiba", sakoe_chiba_radius=1) 
        
        # Inizializza la nuova sequenza sintetica
        S_new = np.zeros_like(S1)
        
        # Per ogni feature, l'interpolazione viene applicata sui punti allineati
        for feature_idx in range(n_features):
            for t1, t2 in path:
                # Interpolazione lineare tra il punto S1[t1, feature] e S2[t2, feature]
                interpolated_value = S1[t1, feature_idx] + alpha * (S2[t2, feature_idx] - S1[t1, feature_idx])
                
                # Assegna il valore interpolato alla posizione del timestep S1 (o S2)
                # Qui c'è una semplificazione, in TSMOTE si usa una media dei punti interpolati
                # che mappano allo stesso timestep nella nuova sequenza.
                S_new[t1, feature_idx] = interpolated_value 

        synthetic_sequences.append(torch.tensor(S_new, dtype=torch.float32))
    
    return synthetic_sequences

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN


def imblearn_resampler(data_by_patient, resampler_class, random_state=42):
    """
    Funzione generica per applicare SMOTEENN/SMOTETomek di imblearn su dati appiattiti.
    
    CORREZIONE FINALE: Usa k_neighbors diretto, come richiesto dalla documentazione di imblearn.combine.
    """
    
    # --- 1. Preparazione Dati (Flatten) ---
    all_data = list(data_by_patient.values())
    X_list = [v["x"].numpy().reshape(-1) for v in all_data]
    Y_list = [v["y"].item() for v in all_data]

    X = np.stack(X_list)
    y = np.array(Y_list)
    
    first_x = data_by_patient[list(data_by_patient.keys())[0]]["x"]
    n_timesteps = first_x.shape[0]
    n_features = first_x.shape[1]

    # --- 2. Applicazione del Resampler (Correzione del TypeError) ---
    
    if resampler_class in [SMOTEENN, SMOTETomek]:
        resampler = resampler_class(
            random_state=random_state,
            sampling_strategy='all'
        )
    else:
        # Fallback per altre classi imblearn
        resampler = resampler_class(
            random_state=random_state,
            sampling_strategy='all'
        )
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    # --- 3. Rimodellamento e Ricostruzione ---
    
    X_resampled_3d = X_resampled.reshape(-1, n_timesteps, n_features)

    balanced_data = {}
    for i in range(len(X_resampled_3d)):
        balanced_data[i] = {
            "x": torch.tensor(X_resampled_3d[i], dtype=torch.float32), 
            "y": torch.tensor(int(y_resampled[i]))
        }
    
    # Stampa i risultati del bilanciamento combinato
    n_septic_new = sum(v["y"].item() == 1 for v in balanced_data.values())
    n_non_septic_new = sum(v["y"].item() == 0 for v in balanced_data.values())
    print(f"(After balancing) Septic patients: {n_septic_new}, Non septic patients: {n_non_septic_new}")
    
    return balanced_data

def balance_patients(data_by_patient, method="oversample", random_state=42):
    """
    Bilancia un dataset di pazienti settici/non-settici.
    Supporta: oversample, undersample, sequence_smote, sequence_smoteenn, sequence_smotetomek
    """
    # Conta prima del bilanciamento
    n_septic = sum(v["y"].item() == 1 for v in data_by_patient.values())
    n_non_septic = sum(v["y"].item() == 0 for v in data_by_patient.values())
    print(f"\n(Pre balancing) Septic patients: {n_septic}, Non septic patients: {n_non_septic}")
    
    # Separiamo le classi
    septic = [v for v in data_by_patient.values() if v["y"].item() == 1]
    non_septic = [v for v in data_by_patient.values() if v["y"].item() == 0]
    
    balanced_data = {}
    idx = 0

    if method is None:
        return data_by_patient
    
    elif method == "undersample":
        # campionamento casuale dalla maggioranza
        random.seed(random_state)
        non_septic_sampled = random.sample(non_septic, len(septic))
        combined = septic + non_septic_sampled

    elif method == "oversample":
        random.seed(random_state)
        n_to_add = len(non_septic) - len(septic)
        septic_augmented = septic + [random.choice(septic) for _ in range(n_to_add)]
        combined = non_septic + septic_augmented

    elif method == "sequence_dtw_smote": 
        n_to_add = len(non_septic) - len(septic)
        
        # CHIAMA LA NUOVA FUNZIONE DTW-BASED MANUALE
        synthetic = sequence_tsmote_dtw_manual(septic, n_to_add, random_state=random_state) 
        
        septic_augmented = septic + [{"x": s, "y": torch.tensor(1)} for s in synthetic]
        combined = non_septic + septic_augmented
    
    # --- Metodi che usano imblearn (Restituiscono già il dict finale) ---
    elif method == "sequence_smoteenn_new":
        # Usa la funzione generica imblearn_resampler con SMOTEENN
        return imblearn_resampler(data_by_patient, SMOTEENN, random_state=random_state)

    elif method == "sequence_smotetomek_new":
        # Usa la funzione generica imblearn_resampler con SMOTETomek
        return imblearn_resampler(data_by_patient, SMOTETomek, random_state=random_state)
    
    # elif method == "sequence_smote":
    #     n_to_add = len(non_septic) - len(septic)
    #     synthetic = sequence_smote(septic, n_to_add, random_state=random_state)
    #     septic_augmented = septic + [{"x": s, "y": torch.tensor(1)} for s in synthetic]
    #     combined = non_septic + septic_augmented

    # elif method == "sequence_tsmote":
    #     n_to_add = len(non_septic) - len(septic)
    #     synthetic = sequence_smote_flattened(septic, n_to_add, random_state=random_state) 
    #     septic_augmented = septic + [{"x": s, "y": torch.tensor(1)} for s in synthetic]
    #     combined = non_septic + septic_augmented

    # elif method == "sequence_smoteenn":
    #     balanced_data = sequence_smoteenn(data_by_patient, random_state=random_state)
    #     return balanced_data

    # elif method == "sequence_smotetomek":
    #     balanced_data = sequence_smotetomek(data_by_patient, random_state=random_state)
    #     return balanced_data

    else:
        raise ValueError(
            "method deve essere 'oversample', 'undersample', 'sequence_smote', "
            "'sequence_smoteenn' o 'sequence_smotetomek'"
        )
    
    # Ricostruisci dict (solo per undersample, oversample e sequence_smote)
    for v in combined:
        balanced_data[idx] = {"x": v["x"], "y": v["y"]}
        idx += 1
    
    # Conta dopo bilanciamento
    n_septic_new = sum(v["y"].item() == 1 for v in balanced_data.values())
    n_non_septic_new = sum(v["y"].item() == 0 for v in balanced_data.values())
    print(f"(After balancing) Septic patients: {n_septic_new}, Non septic patients: {n_non_septic_new}")
    
    return balanced_data

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def train_final_model(
    data_by_patient,
    model_class,
    device,
    hidden_dim=4,
    num_layers=4,
    dropout=0.3,
    pooling="mean",
    lr=5e-3,
    max_epochs=100,
    val_ratio=0.2,
    patience=10,
    loss_fn=None,
    model_name="final_model",
    seed=42, 
    method='undersample',
    error_analysis=False,
    feature_names = None, 
    batch_size=64,
    weight_decay_value=1e-4):
    """
    Allena il modello finale in modo deterministico su GPU/CPU.
    Il bilanciamento avviene SOLO sul training set. """
    # --- fissare seed e comportamento deterministico ---
    set_deterministic(seed)

    print("\nFinal training:")

    # --- preparazione dati ---
    patient_ids = list(data_by_patient.keys())
    X_full = torch.stack([data_by_patient[pid]["x"] for pid in patient_ids])
    y_full = torch.stack([data_by_patient[pid]["y"] for pid in patient_ids])
    full_dataset = TensorDataset(X_full, y_full)
    
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_gen)

    # --- ricostruisci dizionari train/val per bilanciamento e DataLoader ---
    train_data_by_patient = {
        i: {"x": x, "y": y} for i, (x, y) in enumerate(train_dataset)
    }

    # --- bilanciamento SOLO sul training set ---
    print("     Applying class balancing on training data only...")
    train_data_by_patient = balance_patients(train_data_by_patient, method=method, random_state=seed)

    # --- DataLoader ---
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.stack([v["x"] for v in train_data_by_patient.values()]),
            torch.stack([v["y"] for v in train_data_by_patient.values()])
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_gen
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- stampa per controllo ---
    print(f"\nTrain size (after balancing): {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")

    """# Il Class Weighting viene ricalcolato sullo specifico training set
    y_train_full_indices = torch.stack([y for _, y in train_dataset])
    y_train_counts = y_train_full_indices.float().sum(dim=0)
    n_samples = len(train_dataset)
    
    # 2. Calcola i pesi inversi della frequenza (per bilanciare l'imbalance)
    # Peso per la classe 0 (Non-sepsi)
    weight_0 = n_samples / (2 * (n_samples - y_train_counts))
    # Peso per la classe 1 (Sepsi)
    weight_1 = n_samples / (2 * y_train_counts) 
    
    # Il BCEWithLogitsLoss in PyTorch richiede solo il peso per la classe positiva (pos_weight)
    # Calcoliamo pos_weight come rapporto tra il peso della classe 0 e il peso della classe 1.
    pos_weight = weight_1 / weight_0"""

    pos_weight=torch.Tensor([1.0])
    
    print(f"Classe Sepsi (1) Weight (pos_weight): {pos_weight.item():.2f}")
    # --- modello ---
    model = model_class(
        input_dim=X_full.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=True,
        dropout=dropout,
        pooling=pooling
    ).to(device)

    criterion = loss_fn or nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)

    # --- early stopping ---
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch.float())
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{max_epochs} | TrainLoss={total_loss/len(train_loader):.4f} | ValLoss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

    # --- salva best model ---
    os.makedirs("models", exist_ok=True)
    torch.save(best_state, os.path.join("models", model_name + ".pth"))
    print(f"✅ Final model saved in: models/{model_name}.pth")

    model.load_state_dict(best_state)

    # --- valutazione finale ---
    model.eval()
    y_true, y_pred, probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            prob = torch.sigmoid(logits).cpu().numpy()
            pred = (prob > 0.5).astype(int)

            y_true.extend(y_batch.numpy())
            y_pred.extend(pred)
            probs.extend(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.array(probs)

    # --- metriche ---
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, probs)
    cm = confusion_matrix(y_true, y_pred)

    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    class0_metrics = class_report["0"]
    class1_metrics = class_report["1"]

    print("\n📊 Final performance:")
    print(f"ACC={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | Brier={brier:.3f}")
    print(f"Non-sepsis (0): P={class0_metrics['precision']:.3f}, R={class0_metrics['recall']:.3f}, F1={class0_metrics['f1-score']:.3f}")
    print(f"Sepsis (1):     P={class1_metrics['precision']:.3f}, R={class1_metrics['recall']:.3f}, F1={class1_metrics['f1-score']:.3f}")
    print(f"Confusion matrix:\n{cm}")

    error_group_stats = None # Nuovo
    
    if error_analysis:
        print("\n" + "="*50)
        print("🔍 Esecuzione dell'Analisi di Errore")
        print("="*50)
        
        # 1. Analisi per Paziente (necessaria per i gruppi di errore)
        error_analysis_df = analyze_patient_errors(
            model,
            val_dataset, # val_dataset è l'oggetto Subset/TensorDataset creato dallo split
            device
        )


        # print("\nRisultati dell'Analisi per Paziente (Primi 5 FN):")
        # print(error_analysis_df[error_analysis_df["Error_Category"].str.contains("FN")].head())

        # 2. Statistiche di Gruppo e Confronto Feature
        error_group_stats = get_error_group_statistics(
            error_analysis_df,
            val_dataset
        )

        local_importance_results = None
    
  
    # ... (codice per la stampa delle statistiche di gruppo) ...
        
        if feature_names is None:
            print("\n⚠️ ATTENZIONE: 'feature_names' non fornito. Impossibile eseguire l'analisi di importanza locale.")
        else:
            # 3. Analisi di Importanza Locale per i Pazienti Critici
            # Si assume che X_full e y_full siano già tensori

            # 1. Estrai il tensore X_train dal train_dataset
            # Il train_dataset è un Subset. Possiamo iterare sugli elementi o estrarre i dati.
            # Assumiamo che tu voglia estrarre X e Y per comodità:
            X_train = torch.stack([x for x, y in train_dataset]) 

            # 2. Seleziona un campione casuale (p.es., 100 sequenze)
            sample_size = 100 
            if X_train.shape[0] < sample_size:
                sample_size = X_train.shape[0]

            # Genera indici casuali
            sample_indices = torch.randperm(X_train.shape[0])[:sample_size]

            # 3. Definisci X_background
            # Lo sposti sul dispositivo (CPU/GPU) dove verrà usato dal DeepExplainer (che poi lo sposterà su CPU)
            X_background = X_train[sample_indices].to('cpu')
            X_val = torch.stack([x for x, _ in val_dataset]) # Estrai il tensore X_val
            
            local_importance_results = analyze_local_feature_importance(
                model=model,
                X_background=X_background,
                X_val=X_val,
                error_analysis_df=error_analysis_df,
                device=device,
                feature_names=feature_names,
                num_patients_to_analyze=10
            )
            
            print("\nAnalisi di Importanza Locale per i Pazienti Misclassificati Critici:")
            for pid, res in local_importance_results.items():
                print(f"\n--- Paziente {pid} ({res['error_type']}) ---")
                print("Top 5 Feature che hanno guidato la predizione (media assoluta SHAP):")
                # print(res["summary"].to_markdown(index=False, floatfmt=".3f"))
                print(res["summary"].to_dict())

    # --- RITORNO DEI RISULTATI ---
    return_dict = {
        "model": model,
        "metrics": {
            "acc": acc, "auc": auc, "prec": prec, "rec": rec, "f1": f1,
            "mcc": mcc, "brier": brier,
            "class0": class0_metrics, "class1": class1_metrics,
            "cm": cm.tolist()
        },
        "val_tensors":{
            "X_val": torch.stack([x for x, _ in val_dataset]),
            "y_val": torch.stack([y for _, y in val_dataset])
        }
    }

    
    # Aggiungi l'analisi degli errori se eseguita
    if error_analysis:
        return_dict["error_analysis_df"] = error_analysis_df
        return_dict["error_group_stats"] = error_group_stats
        return_dict['importance_results'] = local_importance_results
        
    return return_dict


"""ERROR ANALYSIS"""
# Calcolo della categoria di errore
def get_error_category(row):
    true = row["True_Label"]
    pred = row["Prediction"]
    if true == 1 and pred == 1:
        return "True Positive (TP)"
    elif true == 0 and pred == 0:
        return "True Negative (TN)"
    elif true == 1 and pred == 0:
        return "False Negative (FN)" # Errore grave: Sepsis non rilevata
    elif true == 0 and pred == 1:
        return "False Positive (FP)" # Rilevato allarme non necessario
    return "Unknown" # Non dovrebbe succedere

def analyze_patient_errors(
    model,
    val_dataset,
    device):
    """
    Analizza gli errori di predizione del modello sul set di validazione,
    associandoli ai rispettivi ID paziente.

    Args:
        model (nn.Module): Il modello addestrato.
        val_dataset (Dataset): Il TensorDataset del set di validazione.
        device (torch.device): Dispositivo su cui eseguire l'inferenza.

    Returns:
        pd.DataFrame: Un DataFrame con ID paziente, Etichetta Vera, Predizione,
                      Probabilità, e Categoria di Errore.
    """
    model.eval()
    
    # Assumiamo che val_dataset sia un TensorDataset che contiene (X, y)
    X_val = torch.stack([x for x, _ in val_dataset])
    y_val = torch.stack([y for _, y in val_dataset])
    
    # Supponiamo che gli ID paziente nel set di validazione siano gli indici originali
    # prima dello split, ma in questo specifico codice non abbiamo gli ID originali
    # associati. Li creeremo come 'Paziente_0', 'Paziente_1', ecc.
    # Se avessi gli ID originali, useresti quelli!
    patient_ids = [f"Val_Paziente_{i}" for i in range(len(y_val))]

    y_true_list, y_pred_list, probs_list = [], [], []

    # DataLoader con batch_size=1 per analizzare un paziente alla volta (più semplice)
    # o potresti usare il val_loader esistente, ma perdere l'associazione paziente.
    # Qui usiamo un DataLoader dedicato per mantenere l'ordine paziente
    patient_loader = torch.utils.data.DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=1, # BATCH SIZE DI 1 PER MANTENERE L'ASSOCIAZIONE PAZIENTE-PREDIZIONE
        shuffle=False
    )
    
    print("\n🔍 Analisi degli errori per paziente sul set di validazione...")
    
    with torch.no_grad():
        for X_batch, y_batch in patient_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            prob = torch.sigmoid(logits).cpu().item() # item() perché batch_size=1
            pred = int(prob > 0.5)

            y_true_list.append(y_batch.item())
            y_pred_list.append(pred)
            probs_list.append(prob)

    # Creazione del DataFrame
    results_df = pd.DataFrame({
        "Patient_ID": patient_ids,
        "True_Label": y_true_list,
        "Prediction": y_pred_list,
        "Probability": probs_list
    })


    results_df["Error_Category"] = results_df.apply(get_error_category, axis=1)

    print(f"Risultati salvati per {len(results_df)} pazienti di validazione.")
    
    # Ritorna il DataFrame ordinato per facilità d'uso, p.es. per i FN
    return results_df.sort_values(by=["Error_Category", "Probability"], ascending=[False, True])

def get_error_group_statistics(
    error_analysis_df: pd.DataFrame,
    val_dataset: torch.utils.data.Dataset) -> dict:
    """
    Calcola le statistiche sulla distribuzione delle probabilità per i gruppi di errore
    (FN e FP) e genera un confronto statistico delle feature tra i gruppi.

    Args:
        error_analysis_df (pd.DataFrame): Risultato di analyze_patient_errors.
        val_dataset (Dataset): Il TensorDataset (o Subset) originale del set di validazione.

    Returns:
        dict: Un dizionario contenente 'probability_stats' e 'feature_comparison'.
    """

    # --- 1. Statistiche sulle Probabilità ---
    prob_stats = {}
    for group in ["False Negative (FN)", "False Positive (FP)", "True Positive (TP)", "True Negative (TN)"]:
        subset = error_analysis_df[error_analysis_df["Error_Category"] == group]["Probability"]

        if not subset.empty:
            prob_stats[group] = {
                "Count": len(subset),
                "Mean_Prob": subset.mean(),
                "Median_Prob": subset.median(),
                "Q1_Prob": subset.quantile(0.25),
                "Q3_Prob": subset.quantile(0.75)
            }

    # Trasforma in DataFrame per una migliore visualizzazione
    prob_stats_df = pd.DataFrame(prob_stats).T.rename_axis('Group').reset_index()

    # --- 2. Analisi Comparativa delle Caratteristiche (Feature) ---

    # Mappa i gruppi alle loro liste di indici nel val_dataset
    group_indices = {}
    for group in prob_stats.keys():
        # L'indice del DataFrame di analisi è l'indice nel val_dataset (per come è stato costruito)
        group_indices[group] = error_analysis_df[error_analysis_df["Error_Category"] == group].index.tolist()

    # Estrai i tensori dei dati X e y dal val_dataset
    # val_dataset è un Subset/TensorDataset; estraiamo X e lo stackiamo
    X_val = torch.stack([x for x, _ in val_dataset])
    
    num_features = X_val.shape[2]
    
    # DataFrame per i risultati della feature comparison
    feature_comparison = []

    # Confronto principale: FN vs TP e FP vs TN
    comparisons = [
        ("TN", "True Negative (TN)", "False Positive (FP)"), # Base: TN vs Comp: FP (Falso Allarme)
        ("TP", "True Positive (TP)", "False Negative (FN)")  # Base: TP vs Comp: FN (Sepsis Mancata)
    ]

    for base_label, base_group, comp_group in comparisons:
        # Estrai i tensori per i gruppi
        X_base = X_val[group_indices[base_group]]
        X_comp = X_val[group_indices[comp_group]]

        # Calcola la media aggregata (media su tempo e pazienti) per ogni feature
        # (N_patients, T_steps, N_features) -> (N_features)
        mean_base = X_base.mean(dim=[0, 1]).cpu().numpy()
        std_base = X_base.std(dim=[0, 1]).cpu().numpy()
        mean_comp = X_comp.mean(dim=[0, 1]).cpu().numpy()
        std_comp = X_comp.std(dim=[0, 1]).cpu().numpy()

        # Estrai i nomi brevi dei gruppi
        base_short = base_group.split(' ')[0][0:2].replace('Tr','T').replace('Fa','F') # -> TN, TP
        comp_short = comp_group.split(' ')[0][0:2].replace('Tr','T').replace('Fa','F') # -> FP, FN
        
        # Aggiustamenti manuali per chiarezza (es. True Negative diventa TN)
        if base_group.startswith("True Negative"): base_short = "TN"
        elif base_group.startswith("True Positive"): base_short = "TP"
        if comp_group.startswith("False Negative"): comp_short = "FN"
        elif comp_group.startswith("False Positive"): comp_short = "FP"

        current_comparison_label = f"{comp_short} vs {base_short}" # Esempio: "FN vs TP"

        for i in range(num_features):
            row = {
                "Feature_Index": f"F{i}",
                "Comparison": current_comparison_label, 
                f"Mean_{base_short}": mean_base[i],
                f"Mean_{comp_short}": mean_comp[i],
                "Mean_Difference": mean_comp[i] - mean_base[i],
                f"STD_{base_short}": std_base[i],
                f"STD_{comp_short}": std_comp[i],
                "STD_Ratio": std_comp[i] / std_base[i] if std_base[i] != 0 else np.inf
            }
            feature_comparison.append(row)

    feature_comparison_df = pd.DataFrame(feature_comparison)
    
    return {
        "probability_stats": prob_stats_df,
        "feature_comparison": feature_comparison_df
    }

def analyze_local_feature_importance(
    model,
    X_background: torch.Tensor, 
    X_val: torch.Tensor,
    error_analysis_df: pd.DataFrame,
    device,
    feature_names: list,
    num_patients_to_analyze: int = 3):
    
    model.eval()
    
    # Sposta il modello su CPU
    model_cpu = model.to('cpu') 
    
    # Estrai le dimensioni originali
    T_steps = X_background.shape[1]
    N_features = X_background.shape[2]
    
    # -----------------------------------------------------------
    # FLATTENING DEI DATI
    # -----------------------------------------------------------
    
    # 1. Background: [N_samples, T_steps, N_features] -> [N_samples, T_steps * N_features]
    X_background_flat = X_background.to('cpu').numpy().reshape(X_background.shape[0], -1)
    
    # 2. Creazione della lista estesa dei Nomi delle Feature
    # F0_t0, F0_t1, ..., F1_t0, F1_t1, ...
    extended_feature_names = []
    for f_name in feature_names:
        for t in range(T_steps):
            extended_feature_names.append(f"{f_name}_t{t}")
            
    # -----------------------------------------------------------
    # FUNZIONE WRAPPER (TARGET: KernelExplainer su Dati Piatti)
    # -----------------------------------------------------------
    def model_logit_wrapper_flat(X_flat):
        # X_flat ha forma [B, T*F]
        
        # 1. Ricostruisci la forma sequenziale per il modello LSTM: [B, T, F]
        X_tensor = torch.from_numpy(X_flat).float().to('cpu')
        X_tensor_reshaped = X_tensor.reshape(-1, T_steps, N_features)
        
        with torch.no_grad():
            # 2. Chiama il modello (output ha forma [B])
            output = model_cpu(X_tensor_reshaped) 
            
            # 3. Reshape in [B, 1] e ritorna NumPy
            output_reshaped = output.unsqueeze(-1) 
            
            return output_reshaped.cpu().numpy()

    try:
        # Crea l'Explainer usando la funzione wrapper e il background appiattito
        print("🟡 ATTENZIONE: Il KernelExplainer su dati appiattiti è molto lento.")
        explainer = shap.KernelExplainer(model_logit_wrapper_flat, X_background_flat)
    except Exception as e:
        print(f"ERRORE SHAP (Creazione): Impossibile creare KernelExplainer.")
        print(f"Dettagli: {e}")
        return {}


    # 1. Selezione dei Pazienti Critici (Logica invariata)
    # ... (omissis, selezione di fn_indices e fp_indices)
    # 1. Filtra solo i Falsi Negativi (FN)
    fn_critical = error_analysis_df[error_analysis_df["Error_Category"].str.contains("FN")]
    fn_critical = fn_critical.sort_values(by="Probability", ascending=True).head(num_patients_to_analyze)
    fn_indices = fn_critical.index.tolist()

    # 3. Filtra solo i Falsi Positivi (FP)
    fp_critical = error_analysis_df[error_analysis_df["Error_Category"].str.contains("FP")]
    fp_critical = fp_critical.sort_values(by="Probability", ascending=False).head(num_patients_to_analyze)
    fp_indices = fp_critical.index.tolist()
    
    # 5. Unisci i 20 indici (solo FN e FP)
    analysis_indices = fn_indices + fp_indices
        
    # Prepara i dati di analisi: CONVERSIONE E FLATTENING
    X_analysis_np_flat = X_val[analysis_indices].to('cpu').numpy().reshape(len(analysis_indices), -1)
    
    print(f"\n🧠 Esecuzione dell'analisi SHAP su {len(analysis_indices)} pazienti critici ({T_steps*N_features} features per paziente)...")
    
    # 2. Esecuzione dell'Explainer
    
    try:
        # Passa i dati appiattiti in NumPy
        # shap_values_raw ora ha la forma: [N_samples, T_steps * N_features]
        shap_values_raw = explainer.shap_values(X_analysis_np_flat) 
    except Exception as e:
        print(f"ERRORE SHAP (Esecuzione): Errore durante il calcolo dei valori SHAP.")
        print(f"Dettagli: {e}")
        return {}

    # ... (Logica per elaborare i risultati) ...
    shap_values_class1 = shap_values_raw 
    
    results = {}
    for i, global_idx in enumerate(analysis_indices):
        patient_id = error_analysis_df.loc[global_idx]["Patient_ID"]
        error_type = error_analysis_df.loc[global_idx]["Error_Category"]
        
        # Estrai i valori SHAP appiattiti per il paziente: [T*F]
        patient_shap_flat = shap_values_class1[i]
        
        # Riporta i valori SHAP alla forma [T_steps, N_features] per l'analisi clinica
        patient_shap_reshaped = patient_shap_flat.reshape(T_steps, N_features)
        
        # Calcola l'importanza media assoluta e media (con segno) nel tempo (media su T_steps)
        feature_importance_abs_mean = np.mean(np.abs(patient_shap_reshaped), axis=0)
        feature_importance_mean = np.mean(patient_shap_reshaped, axis=0)
        
        # Crea un DataFrame riassuntivo con i NOMI DELLE FEATURE ORIGINALI
        local_summary_df = pd.DataFrame({
            "Feature_Name": feature_names, # Usa la lista corta dei nomi!
            "SHAP_Local_Mean_Abs": feature_importance_abs_mean,
            "SHAP_Local_Mean": feature_importance_mean 
        }).sort_values(by="SHAP_Local_Mean_Abs", ascending=False).head(5)
        
        results[patient_id] = {
            "error_type": error_type,
            "summary": local_summary_df,
            "shap_values_raw": patient_shap_reshaped.tolist() 
        }

    return results

def automatic_pfi_feature_selection(
    importance_df: pd.DataFrame, 
    positive_threshold: float = 0.005
    ) -> List[str]:
    """
    Esegue la Feature Selection automatica basata sui risultati PFI.

    Args:
        importance_df: DataFrame con le colonne 'Feature' e 'Importance'.
        positive_threshold: La soglia minima di Importance (F1-Score drop) 
                            che un feature deve superare per essere mantenuto.

    Returns:
        Una lista contenente i nomi delle feature selezionate.
    """
    
    initial_count = len(importance_df)
    
    # 1. Rimuovi i feature con Importanza Negativa (Rumore dannoso)
    # Questi feature, quando mescolati, non peggiorano (o addirittura migliorano) la performance.
    # Sono quasi sempre da eliminare.
    negative_importance_df = importance_df[importance_df['Importance'] <= 0]
    
    # 2. Applica la soglia per l'Importanza Trascurabile
    # Manteniamo solo i feature con Importanza > 0 E superiore alla soglia minima.
    selected_df = importance_df[importance_df['Importance'] > positive_threshold]
    
    final_count = len(selected_df)
    
    print("\n--- Risultati Selezione Automatica PFI ---")
    print(f"Feature iniziali: {initial_count}")
    print(f"Feature rimosse (Negativa/Zero): {len(negative_importance_df)}")
    print(f"Soglia minima di Importanza (Positiva): > {positive_threshold}")
    print(f"Feature selezionate finali: {final_count}")
    print(f"Tasso di riduzione: {1 - final_count/initial_count:.2%} \n")
    
    return selected_df['Feature'].to_list()

from itertools import product
from typing import Dict, Any, List, Callable, Tuple
import numpy as np

def grid_search_lstm(
    train_fn: Callable[..., Dict[str, Any]],
    param_grid: Dict[str, List[Any]],
    data_by_patient: Dict[int, Dict[str, torch.Tensor]],
    model_class: nn.Module,
    device: torch.device,
    fixed_params: Dict[str, Any] = None,
    scoring_metric: str = 'mcc') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Esegue la Grid Search su una griglia di iperparametri utilizzando la funzione
    di training/valutazione K-Fold fornita.

    Args:
        train_fn: La funzione di cross-validation (e.g., train_and_evaluate_kfold_val_earlystop).
        param_grid: Dizionario con i nomi dei parametri e le liste dei valori da testare.
        data_by_patient: I dati di input sequenziali (già ridotti dalle PFI).
        model_class: La classe del modello da istanziare (e.g., LSTMClassifier).
        device: Dispositivo per l'addestramento (e.g., torch.device('cuda')).
        fixed_params: Parametri da mantenere fissi (e.g., n_splits, weight_decay=1e-4).
        scoring_metric: Metrica da massimizzare ('mcc', 'f1', 'auc', ecc.).

    Returns:
        Una tupla contenente: (i migliori parametri trovati, i risultati medi migliori).
    """
    
    # Unione dei parametri fissi per la chiamata finale
    if fixed_params is None:
        fixed_params = {}
    
    # 1. Costruzione della Griglia di Combinazioni
    keys = param_grid.keys()
    # Genera tutte le combinazioni possibili
    combinations = [dict(zip(keys, v)) for v in product(*param_grid.values())]

    best_score = -float('inf')
    best_params = None
    best_results = None

    print(f"--- Inizio Grid Search con {len(combinations)} combinazioni. Scoring: {scoring_metric.upper()} ---")

    for i, params in enumerate(combinations):
        
        # 2. Setup dei Parametri per la Chiamata
        full_params = {**fixed_params, **params}
        print(f"\n==================== Combinazione {i+1}/{len(combinations)} ====================")
        print(f"Test Parametri: {params}")

        try:
            # 3. Esecuzione della Cross-Validation
            results = train_fn(
                data_by_patient=data_by_patient,
                model_class=model_class,
                device=device,
                **full_params
            )
            
            # 4. Estrazione dello Score
            # Lo score viene preso dai risultati medi della classe 1 (Sepsi) o globali
            if scoring_metric in ['f1', 'precision', 'recall']:
                 # F1, precisione e recall della classe Sepsi (1)
                 current_score = results['mean_class1'][f'{scoring_metric}-score' if scoring_metric=='f1' else scoring_metric]
                 print(f"Risultato {scoring_metric.upper()} (Sepsi): {current_score:.4f}")
            else:
                 # MCC, AUC, ACC globali
                 current_score = results['mean_global'][scoring_metric]
                 print(f"Risultato {scoring_metric.upper()} (Global): {current_score:.4f}")


            # 5. Aggiornamento del Miglior Risultato
            if current_score > best_score:
                best_score = current_score
                best_params = params
                best_results = results
                print(f"🏆 Nuovo Best Score: {best_score:.4f} con parametri: {best_params}")

        except Exception as e:
            print(f"⚠️ Errore durante l'addestramento con parametri {params}: {e}")
            continue

    print("\n\n==================== GRID SEARCH COMPLETATA ====================")
    print(f"Migliori Parametri Trovati: {best_params}")
    print(f"Miglior {scoring_metric.upper()} Score: {best_score:.4f}")
    
    return best_params, best_results


if __name__=="__main__":

    print()

    # pop_patients()

    dataset = "mimic"
    target_len = 24
    pred_length = 6
    method="oversample"

    # features_selected = [
    #     "meanbp", "resprate", "heartrate", "spo2_pulsoxy", "tempc",
    #     "cardiacoutput", "o2flow", "fio2", "albumin", "bands",
    #     "bicarbonate", "bilirubin", "creatinine", "chloride", "glucose",
    #     "hemoglobin", "lactate", "platelet", "potassium", "ptt",
    #     "inr", "sodium", "wbc", "creatinekinase", "ck_mb",
    #     "fibrinogen", "ldh", "magnesium", "calcium_free", "po2_bloodgas",
    #     "ph_bloodgas", "pco2_bloodgas", "so2_bloodgas", "troponin_t"
    # ]

#     features_selected = [
#     "sysbp",
#     "diabp",
#     "meanbp",
#     "resprate",
#     "heartrate",
#     "spo2_pulsoxy",
#     "tempc",
#     "cardiacoutput",
#     "tvset",
#     "tvobserved",
#     "tvspontaneous",
#     "peakinsppressure",
#     "totalpeeplevel",
#     "o2flow",
#     "fio2",
#     "albumin",
#     "bands",
#     "bicarbonate",
#     "bilirubin",
#     "creatinine",
#     "chloride",
#     "glucose",
#     "hematocrit",
#     "hemoglobin",
#     "lactate",
#     "platelet",
#     "potassium",
#     "ptt",
#     "inr",
#     "pt",
#     "sodium",
#     "bun",
#     "wbc",
#     "creatinekinase",
#     "ck_mb",
#     "fibrinogen",
#     "ldh",
#     "magnesium",
#     "calcium_free",
#     "po2_bloodgas",
#     "ph_bloodgas",
#     "pco2_bloodgas",
#     "so2_bloodgas",
#     "troponin_t"
# ] 

    # features_selected = ['sysbp', 'diabp', 'hematocrit', 'meanbp', 'bun', 'ph_bloodgas', 'hemoglobin', 'fio2', 'so2_bloodgas', 'wbc', 'resprate', 
    # 'o2flow', 'ptt', 'totalpeeplevel', 'ck_mb', 'bicarbonate', 'magnesium', 'albumin', 'sodium', 'creatinekinase', 'pt', 'tvspontaneous', 
    # 'cardiacoutput', 'calcium_free', 'tvobserved']

    features_selected=['sysbp', 'ph_bloodgas', 'diabp', 'meanbp', 'bun', 'fio2', 'lactate', 'bicarbonate', 'resprate', 'wbc', 'spo2_pulsoxy', 
    'chloride', 'sodium', 'hematocrit', 'so2_bloodgas', 'po2_bloodgas', 'totalpeeplevel', 'troponin_t', 'potassium', 'pt', 'ldh', 'magnesium', 
    'peakinsppressure', 'cardiacoutput']

    features_selected = ["Index", "timestep", "sepsis"] + features_selected
    
    if dataset=="mimic":
        data_by_patient, device, mimic_df, _, _, _= prepare_mimic_data(target_len=target_len, pred_length=pred_length, 
        features_selected=features_selected, pop_patient=False, n=1, random_state=0)

        feature_names= mimic_df.drop(columns=["Index", "sepsis", "timestep"]).columns.to_list()

    else:
        raise ValueError(f"Dataset not recognized: '{dataset}'. Insert a valid name ('sepsisexp' or 'mimic').")
    

    train_and_evaluate_kfold_val_earlystop(data_by_patient=data_by_patient, model_class=LSTMClassifier, hidden_dim=8, num_layers=4, max_epochs=100, 
                                            patience=10, lr=5e-3, dropout=0.3, batch_size=64, weight_decay_value=0, device=device, method=method)
    
    # model_dict = train_final_model(data_by_patient=data_by_patient, model_class=LSTMClassifier, method=method, 
    # hidden_dim=8, num_layers=4, pooling='mean', dropout=0.3, lr=5e-3, weight_decay_value=0, batch_size=64, device=device, 
    # max_epochs=100, patience=10, error_analysis=False, feature_names=feature_names, model_name=f"{dataset}_{target_len}-{pred_length}_{method}")

    # 1. Definisci la Griglia di Parametri da Esplorare
    # param_grid_full = {
    # # 1. PARAMETRI DI ARCHITETTURA (Architettura della LSTM)
    # 'hidden_dim': [8, 32],  # Dimensione del vettore nascosto LSTM
    # 'num_layers': [4, 16],      # Numero di strati LSTM (profondità)
    # 'pooling': ['mean'],  # Metodo di pooling (l'abbiamo fissato a 'mean' ma si può testare)
    
    # # 2. PARAMETRI DI REGOLARIZZAZIONE (Regularization)
    # 'dropout': [0.3, 0.5], # Il tuo 0.3 ottimale e alternative più aggressive
    
    # # 3. PARAMETRI DI ADDESTRAMENTO (Optimization)
    # 'lr': [1e-3, 5e-3],          # Learning Rate (i due valori che stavi testando)
    # 'weight_decay_value': [1e-3, 1e-4], # Valore L2 (il tuo 1e-4 ottimale + alternative)
    # 'batch_size': [32, 64]
    # }

    # # Esegui la Grid Search con la griglia completa
    # # Scegli la metrica di scoring più robusta (MCC)
    # best_params, best_results = grid_search_lstm(
    #     train_fn=train_and_evaluate_kfold_val_earlystop,
    #     param_grid=param_grid_full,
    #     data_by_patient=data_by_patient, # Dati con feature ridotte!
    #     model_class=LSTMClassifier,
    #     device=device,
    #     scoring_metric='mcc' 
    # )

    # print(f"I migliori parametri trovati per MCC: {best_params}")

    #FEATURE SELECTION WITH PFI
    # importance_df=permutation_feature_importance(model=model_dict['model'], X=model_dict["val_tensors"]['X_val'], y=model_dict["val_tensors"]['y_val'], 
    #                                n_repeats=1000, metric_fn = matthews_corrcoef, feature_names=feature_names)
    
    # print(importance_df.to_dict())
    # importance_df.to_csv(f"{dataset}_{target_len}-{pred_length}_{method}_importance_df.csv")

    # # Applica la selezione automatica con la soglia consigliata di 0.005
    # selected_feature_names = automatic_pfi_feature_selection(importance_df, positive_threshold=0.005)

    # print("Lista dei Feature da Mantenere:")
    # print(selected_feature_names)
    
    """error_analysis_df = model_dict['error_analysis_df']
    error_analysis_df.to_csv("error_analysis.csv")

    prob_stats = model_dict['error_group_stats']['probability_stats']
    prob_stats.to_csv("prob_stats.csv")
    feature_df = model_dict['error_group_stats']['feature_comparison']

    # Estrai l'indice numerico da F0, F1, F2...
    feature_df['Feature_Number'] = feature_df['Feature_Index'].str.replace('F', '').astype(int)

    # Verifica se il numero di feature corrisponde
    num_features_in_df = feature_df['Feature_Number'].max() + 1
    if num_features_in_df != len(feature_names):
        print(f"⚠️ ATTENZIONE: Il numero di feature nel DF ({num_features_in_df}) non corrisponde alla lista dei nomi ({len(feature_names)}). La mappatura potrebbe essere errata.")
        # Continua comunque, ma con cautela

    # Mappa l'indice numerico al nome della feature
    feature_df['Feature_Name'] = feature_df['Feature_Number'].apply(lambda x: feature_names[x])

    # Riorganizza e pulisci le colonne
    feature_df = feature_df.drop(columns=['Feature_Index', 'Feature_Number'])
    cols = ['Feature_Name', 'Comparison', 'Mean_Difference'] + [col for col in feature_df.columns if col not in ['Feature_Name', 'Comparison', 'Mean_Difference']]
    feature_df = feature_df[cols]

    fp_tn_comp = feature_df[feature_df['Comparison'] == 'FP vs TN'].copy()
    top_fp_triggers = fp_tn_comp.sort_values(by='Mean_Difference', ascending=False).head(5)

    print("\n🚨 Top 5 Triggers del Falso Allarme (FP vs TN): Segnali Anomali in Pazienti Sani")
    print(top_fp_triggers[['Feature_Name', 'Mean_FP', 'Mean_TN', 'Mean_Difference', 'STD_Ratio']].to_markdown(index=False, floatfmt=".3f"))

    fn_tp_comp = feature_df[feature_df['Comparison'] == 'FN vs TP'].copy()
    top_fn_misses = fn_tp_comp.sort_values(by='Mean_Difference', ascending=True).head(5)

    print("\n📉 Top 5 Segnali Smorzati/Mancati (FN vs TP): Feature che non hanno Innescato l'Allarme")
    print(top_fn_misses[['Feature_Name', 'Mean_FN', 'Mean_TP', 'Mean_Difference', 'STD_Ratio']].to_markdown(index=False, floatfmt=".3f"))

    feature_df.to_csv("feature_comp_per_group.csv")
    """

