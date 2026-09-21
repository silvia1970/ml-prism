import os
# DEVE essere fatto prima di importare torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, roc_curve, matthews_corrcoef, brier_score_loss

import shap
import lime
import lime.lime_tabular
import glob

import re
from scipy import stats
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import PartialDependenceDisplay
from sklearn import tree
from sklearn.tree import plot_tree
from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator 
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

import math

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from collections import Counter

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.spatial.distance import cdist

import json

from sklearn.mixture import GaussianMixture

from mpl_toolkits.mplot3d import Axes3D  # necessario per 3D plots
import plotly.graph_objects as go
import plotly.io as pio

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error

import random
import torch.backends.cudnn as cudnn

from captum.attr import IntegratedGradients

from torchinfo import summary
from torchviz import make_dot

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix
)
from torch.utils.data import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F

from imblearn.over_sampling import SMOTE # Il classico SMOTE

from itertools import product
from typing import Dict, Any, List, Callable, Tuple

from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""Data Analysis"""

def from_tsv_to_csv(path_to_dataset='datasets/SepsisExp'):
    """
    Converte tutti i file .tsv presenti nella cartella specificata in file .csv,
    sostituendo i tabulatori con virgole e salvando i nuovi file nella stessa cartella.
    
    Args:
        path_to_dataset (str): Percorso alla cartella contenente i file .tsv. Default: 'datasets/SepsisExp'.
    """

    # Scorre tutti i file nella directory specificata
    for partition in os.listdir(path_to_dataset):
        # Considera solo i file con estensione .tsv
        if partition.endswith(".tsv"):
            print(f"Partition name: {partition}")

            # Apre il file .tsv in modalità lettura
            with open(os.path.join(path_to_dataset, partition), 'r') as tsv_file:
                # Rimuove l'estensione .tsv per creare il nome del file .csv
                partition_root, _ = os.path.splitext(partition)
                partition_csv = partition_root + ".csv"

                # Apre (o crea) il file .csv in modalità scrittura
                with open(os.path.join(path_to_dataset, partition_csv), 'w') as csv_file:
                    # Legge ogni riga del file .tsv
                    for line in tsv_file:
                        # Sostituisce i tabulatori (\t) con virgole (,) per formattazione CSV
                        fileContent = re.sub("\t", ",", line)
                        # Scrive la riga convertita nel nuovo file .csv
                        csv_file.write(fileContent)

    print("Successfully made csv files \n")



    # Set plot style for better visualization
    sns.set_style(style="whitegrid")
    
    # Step 2: Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Step 3: Initialize an empty list to store DataFrames
    dfs = []

    # Step 4: Loop through all CSV files and read them into DataFrames
    for file in csv_files:
        df = pd.read_csv(file)  # Read each CSV file
        dfs.append(df)          # Append the DataFrame to the list

    # Step 5: Concatenate all DataFrames into one DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)

    # Step 6: Display the merged DataFrame
    # print(merged_df)
   
    data = merged_df.drop('id',axis=1)
    data = data.drop('sepsis',axis=1)
    data = data.drop('severity',axis=1)
    data = data.drop('timestep',axis=1)
    data = data.drop('age',axis=1)

    #data = merged_df

    # 2. Basic Information
    print("### Basic Information ###")
    print(f"Shape of the dataset: {data.shape}")
    print("\nColumn names and data types:")
    print(data.dtypes)
    
    # 3. Identify missing values
    print("\n### Missing Values ###")
    missing_values = data.isna().sum()
    print(missing_values[missing_values > 0])
    
    # 4. Unique values in each column
    print("\n### Unique Values per Column ###")
    print(data.nunique())

    # 5. Checking for duplicated rows
    print("\n### Duplicated Rows ###")
    duplicate_rows = data.duplicated().sum()
    print(f"Number of duplicated rows: {duplicate_rows}")

    duplicates_all = data[data.duplicated(keep=False)]

    print(f"\nTutte le righe duplicate (inclusa la prima occorrenza): {len(duplicates_all)}")
    print(duplicates_all)
    duplicates_all.to_csv("tutte_le_righe_duplicate.csv", index=False)

    # 6. Descriptive statistics
    print("\n### Descriptive Statistics ###")
    print(data.describe(include='all'))  # Include categorical and numerical data

    # 7. Check for outliers (Optional but useful for numerical data)
    print(f"\n### Outliers Detection con Scarto Interquartile ###")
    outliers_list = []
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
        print(f"{(Q1 - 1.5 * IQR)} , {(Q3 + 1.5 * IQR)}")
        outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
        print(f"Outliers in '{col}': {outliers}")
        outliers_list.append(outliers)
    
    
    numeric_data = data.select_dtypes(include=[np.number])

    # Boxplot multiplo
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_data, orient='v', palette='Set2', fliersize=4, flierprops={'markerfacecolor': 'red'})
    plt.title('Boxplot di tutte le variabili numeriche')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 8. Correlation Matrix (for numerical columns)
    print("\n### Correlation Matrix ###")
    corr_matrix = data.corr(numeric_only=True)
    print(corr_matrix)

    # 9. Data Visualization (Optional, but useful for deeper insights)
    
    # 9.1 Distribution of numerical features
    print("\n### Numerical Features Distribution ###")
    data.select_dtypes(include=[np.number]).hist(figsize=(10, 10), bins=20, edgecolor='black')
    plt.tight_layout()
    plt.show()

    # 9.2 Correlation heatmap (for numerical features)    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(18, 14))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        annot_kws={"size": 6},
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.3,
        linecolor='gray'
    )
    plt.title("Correlation Heatmap (Upper Triangle)", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

    return data, merged_df, outliers_list

def analyze_csvs(folder_path, verbose=True):
    """
    Analizza i CSV in una cartella: unisce i file, rimuove colonne non rilevanti, 
    esegue analisi descrittive, identifica outlier e visualizza statistiche.

    Args:
        folder_path (str): Percorso alla cartella contenente i CSV.
        verbose (bool): Se True, stampa dettagli e mostra grafici. Altrimenti usa solo una progress bar.

    Returns:
        data (pd.DataFrame): Dati pre-elaborati (senza colonne rimosse).
        merged_df (pd.DataFrame): DataFrame unito da tutti i CSV.
        outliers_list (list): Numero di outlier per ogni colonna numerica.
    """
    if verbose:
        sns.set_style(style="whitegrid")

    # Step 1: Caricamento CSV
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if verbose:
        print(f"📄 Trovati {len(csv_files)} file CSV")
        iterator = csv_files
    else:
        iterator = tqdm(csv_files, desc="📊 Analisi CSV", unit="file")

    dfs = [pd.read_csv(file) for file in iterator]
    merged_df = pd.concat(dfs, ignore_index=True)

    # Step 2: Preprocessing colonne
    data = merged_df.drop(columns=['id', 'sepsis', 'severity', 'timestep', 'age'])

    if verbose:
        print("### Basic Information ###")
        print(f"Shape of the dataset: {data.shape}")
        print("\nColumn names and data types:")
        print(data.dtypes)

        print("\n### Missing Values ###")
        missing_values = data.isna().sum()
        print(missing_values[missing_values > 0])

        print("\n### Unique Values per Column ###")
        print(data.nunique())

        print("\n### Duplicated Rows ###")
    duplicate_rows = data.duplicated().sum()
    if verbose:
        print(f"Number of duplicated rows: {duplicate_rows}")
        duplicates_all = data[data.duplicated(keep=False)]
        print(f"Tutte le righe duplicate (inclusa la prima occorrenza): {len(duplicates_all)}")
        print(duplicates_all)
        duplicates_all.to_csv("tutte_le_righe_duplicate.csv", index=False)

        print("\n### Descriptive Statistics ###")
        print(data.describe(include='all'))

    # Step 3: Outlier detection
    if verbose:
        print("\n### Outliers Detection con Scarto Interquartile ###")

    outliers_list = []
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outliers_list.append(outliers)
        if verbose:
            print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
            print(f"{lower_bound} , {upper_bound}")
            print(f"Outliers in '{col}': {outliers}")

    # Step 4: Boxplot numerici
    if verbose:
        numeric_data = data.select_dtypes(include=[np.number])
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=numeric_data, orient='v', palette='Set2', fliersize=4, flierprops={'markerfacecolor': 'red'})
        plt.title('Boxplot di tutte le variabili numeriche')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Step 5: Correlazione
    corr_matrix = data.corr(numeric_only=True)
    if verbose:
        print("\n### Correlation Matrix ###")
        print(corr_matrix)

        print("\n### Numerical Features Distribution ###")
        data.select_dtypes(include=[np.number]).hist(figsize=(10, 10), bins=20, edgecolor='black')
        plt.tight_layout()
        plt.show()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(18, 14))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            annot_kws={"size": 6},
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.3,
            linecolor='gray'
        )
        plt.title("Correlation Heatmap (Upper Triangle)", fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()

    return data, merged_df, outliers_list

def analyze_df(data, verbose=True):

    if verbose:
        print("### Basic Information ###")
        print(f"Shape of the dataset: {data.shape}")
        print("\nColumn names and data types:")
        print(data.dtypes)

        print("\n### Missing Values ###")
        missing_values = data.isna().sum()
        print(missing_values[missing_values > 0])

        print("\n### Unique Values per Column ###")
        print(data.nunique())

        print("\n### Duplicated Rows ###")
    duplicate_rows = data.duplicated().sum()
    if verbose:
        print(f"Number of duplicated rows: {duplicate_rows}")
        duplicates_all = data[data.duplicated(keep=False)]
        print(f"Tutte le righe duplicate (inclusa la prima occorrenza): {len(duplicates_all)}")
        print(duplicates_all)
        duplicates_all.to_csv("tutte_le_righe_duplicate.csv", index=False)

        print("\n### Descriptive Statistics ###")
        print(data.describe(include='all'))

    # Step 3: Outlier detection
    if verbose:
        print("\n### Outliers Detection con Scarto Interquartile ###")

    outliers_list = []
    for col in data.select_dtypes(include=[np.number]).dropna(axis=1, how='all'):
        if data[col].dropna().empty:
            if verbose:
                print(f"Colonna '{col}' contiene solo NaN. Skippata.")
            continue
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outliers_list.append(outliers)
        if verbose:
            print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
            print(f"{lower_bound} , {upper_bound}")
            print(f"Outliers in '{col}': {outliers}")

    # Step 4: Boxplot numerici
    if verbose:
        numeric_data = data.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=numeric_data, orient='v', palette='Set2', fliersize=4, flierprops={'markerfacecolor': 'red'})
        plt.title('Boxplot di tutte le variabili numeriche')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Step 5: Correlazione
    corr_matrix = data.corr(numeric_only=True)
    if verbose:
        print("\n### Correlation Matrix ###")
        print(corr_matrix)

        print("\n### Numerical Features Distribution ###")
        data.select_dtypes(include=[np.number]).hist(figsize=(10, 10), bins=20, edgecolor='black')
        plt.tight_layout()
        plt.show()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(18, 14))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            annot_kws={"size": 6},
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.3,
            linecolor='gray'
        )
        plt.title("Correlation Heatmap (Lower Triangle)", fontsize=14)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig("./corr_matrix.png")
        plt.show()

    return outliers_list

def analyze_outliers_with_confidence(data: pd.DataFrame, outliers_list: list, confidence: float = 0.95):
    """
    Calcola intervalli di confidenza per ciascuna colonna di un DataFrame e crea un DataFrame riepilogativo
    con numero di outlier e relativi intervalli.

    Args:
        data (pd.DataFrame): Dataset con dati numerici.
        outliers_list (list): Lista con il numero di outlier per ciascuna colonna, nello stesso ordine di `data.columns`.
        confidence (float): Livello di confidenza per l'intervallo (default 0.95).
    """
    print("\n### Intervalli di confidenza e .csv riepilogativo con Outliers ###\n")
    conf_int_list = []
    dfs_list = []

    for col in data.columns:
        mean = data[col].mean()
        s = data[col].std()
        n = len(data[col])

        conf_interval = stats.t.interval(
            confidence=confidence,
            df=n-1,
            loc=mean,
            scale=s / np.sqrt(n)
        )

        print(f"Intervallo di confidenza per {col}: ({conf_interval[0]}, {conf_interval[1]})")
        conf_int_list.append((float(conf_interval[0]), float(conf_interval[1])))

    outliers_info = [
        (
            f"{outliers_list[i]} ({(outliers_list[i] / data.shape[0]) * 100:.2f}%)",
            conf_int_list[i]
        )
        for i in range(len(outliers_list))
    ]

    outliers_df = pd.DataFrame(
        outliers_info,
        index=data.columns.to_list(),
        columns=["N. of outliers", "Confidence intervals"]
    )

    dfs_list.append(outliers_df)
    pd.concat(dfs_list, axis=1).to_excel("./data_analysis/outliers_test.xlsx")

def prepare_hourly_data(merged_df):
    
    # 1. Rimozione dei timestep con .5
    hour_df = merged_df[merged_df["timestep"] % 1 != 0.5].reset_index(drop=True)

    # 2. Preprocessing
    hour_df = hour_df.drop(['sepsis', 'timestep'], axis=1).drop_duplicates()
    X = hour_df.drop(['id', 'severity'], axis=1)
    y = hour_df['severity']

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_random_forest_from_hourly_data(X_train, X_test, y_train, y_test, output_tree_dir='trees', verbose=True):
    """
    Addestra un modello Random Forest e visualizza risultati e importanza delle feature.

    Args:
        merged_df (pd.DataFrame): Il dataframe unificato contenente i dati orari dei pazienti.
        output_tree_dir (str): Cartella in cui salvare i plot degli alberi.
        verbose (bool): Se True, stampa info dettagliate e mostra i grafici. Altrimenti solo progress bar sintetica.

    Returns:
        RandomForestClassifier: Il modello Random Forest addestrato.
    """

    # 4. Modellazione Random Forest
    if verbose:
        print("\n🔁 Training Random Forest...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        criterion='entropy',
        random_state=42,
        verbose=verbose
    )

    if verbose:
        rf_model.fit(X_train, y_train)
    else:
        with tqdm(total=1, desc="🌲 Addestramento RF", unit="step") as pbar:
            rf_model.fit(X_train, y_train)
            pbar.update(1)

    # 5. Valutazione
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    if verbose:
        print(f"\n✅ Accuracy: {accuracy:.4f}")
        print(f"✅ ROC AUC (ovr): {roc_auc:.4f}\n")

    # 6. Importanza delle feature
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X_train.columns

    if verbose:
        plt.figure(figsize=(12, 6))
        plt.title("Importanza delle feature nella Random Forest")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # 7. Plot alberi (3)
    os.makedirs(output_tree_dir, exist_ok=True)
    if verbose:
        plt.rcParams.update({'font.size': 12})
        fig, axes = plt.subplots(1, 3, figsize=(60, 20))

        for i in range(3):
            estimator = rf_model.estimators_[i]
            ax = axes[i]
            plot_tree(estimator, feature_names=feature_names, max_depth=1, ax=ax)
            ax.set_title(f"Albero {i}", pad=30)

        plt.tight_layout()
        tree_plot_path = os.path.join(output_tree_dir, "forest_with_lower_titles.png")
        plt.savefig(tree_plot_path, dpi=150)
        plt.show()
        print(f"🌳 Salvato il plot degli alberi in: {tree_plot_path}")

    return rf_model

def explain_model_with_shap(rf_model, X_test, verbose=True):
    """
    Esegue l'analisi di explainability del modello Random Forest usando SHAP.

    Args:
        rf_model (RandomForestClassifier): Il modello addestrato.
        X_test (pd.DataFrame): Dati di test usati per l'analisi SHAP.
        verbose (bool): Se True stampa dettagli e genera i plot. Altrimenti mostra solo una barra di avanzamento.
    """
    if verbose:
        print("🧠 Calcolo SHAP values...\n")

    # Campionamento per performance
    X_sample = X_test.sample(n=100 if verbose else 500, random_state=42)

    # Costruzione dell'explainer
    explainer = shap.TreeExplainer(rf_model, feature_perturbation="auto")
    shap_values = explainer(X_sample)

    # Numero di classi
    num_classes = shap_values.values.shape[-1]

    if verbose:
        for i in range(num_classes):
            print(f"📊 SHAP - Classe {i}")
            shap.plots.bar(shap_values[..., i], max_display=len(shap_values.feature_names))

        print()

        for i in range(num_classes):
            print(f"🐝 Beeswarm - Classe {i}")
            shap.summary_plot(shap_values[..., i], X_sample)

    else:
        for _ in tqdm(range(num_classes), desc="SHAP plotting", ncols=80):
            # Generazione invisibile dei plot
            plt.ioff()
            shap.plots.bar(shap_values[..., _], show=False)
            plt.close()

            shap.summary_plot(shap_values[..., _], X_sample, show=False)
            plt.close()

def run_eli5_analysis(rf_model, X_test, y_test, output_path="eli5_feature_importance.html", verbose=True):
    """
    Esegue l'analisi di importanza delle feature con ELI5 (Permutation Importance).

    Args:
        rf_model (RandomForestClassifier): Modello Random Forest addestrato.
        X_test (pd.DataFrame): Feature di test.
        y_test (pd.Series): Target di test.
        output_path (str): Percorso del file HTML da salvare.
        verbose (bool): Se True, stampa output testuale e messaggi. Se False, esegue in silenzio.
    """
    if verbose:
        print("⚙️ Calcolo Permutation Importance con ELI5...")

    # Calcolo importanza con permutazione
    perm = PermutationImportance(rf_model, random_state=42).fit(X_test, y_test)

    # Genera HTML e testo
    html = eli5.format_as_html(eli5.explain_weights(perm, feature_names=X_test.columns.tolist()))
    text = eli5.format_as_text(eli5.explain_weights(perm, feature_names=X_test.columns.tolist()))

    # Stampa (opzionale)
    if verbose:
        print("\n📋 Importanza delle feature (testo):")
        print(text)

    # Salva in file HTML
    with open(output_path, "w") as f:
        f.write(html)

    if verbose:
        print(f"💾 Salvato in: {output_path}")

def run_pdp_analysis(rf_model, X_test, features_to_plot, output_dir="pdp_plots", verbose=True):
    """
    Genera e salva i Partial Dependence Plots (PDP) per ciascuna classe.

    Args:
        rf_model (RandomForestClassifier): Il modello addestrato.
        X_test (pd.DataFrame): Il dataset di test.
        features_to_plot (list): Lista delle feature da plottare.
        output_dir (str): Directory dove salvare i plot.
        verbose (bool): Se True, stampa messaggi di stato.
    """
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n📊 Generazione PDP per le feature: {features_to_plot}")

    for class_id in rf_model.classes_:
        if verbose:
            print(f"  ➤ Classe {class_id}...")

        disp = PartialDependenceDisplay.from_estimator(
            rf_model,
            X_test,
            features=features_to_plot,
            kind="average",
            target=class_id
        )

        plt.title(f"PDP for class {class_id}")
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"pdp_class{class_id}.png")
        plt.savefig(output_path)
        plt.clf()

        if verbose:
            print(f"    ✔️ Salvato: {output_path}")

def safe_predict_proba(x):
    x_df = pd.DataFrame(x, columns=X_train.columns)
    return rf_model.predict_proba(x_df)

def run_lime_global_analysis(rf_model, X_train, y_train, X_test, n_samples=100, output_path="lime_global_feature_importance.png", verbose=True):
    """
    Calcola l'importanza globale delle feature usando LIME (aggregando le spiegazioni locali).

    Args:
        rf_model: Modello Random Forest addestrato.
        X_train (pd.DataFrame): Dati di training.
        y_train (pd.Series): Target di training.
        X_test (pd.DataFrame): Dati di test.
        n_samples (int): Numero di istanze di test da usare per l'analisi LIME.
        output_path (str): Percorso per salvare il grafico.
        verbose (bool): Se True, stampa messaggi intermedi.
    """

    if verbose:
        print("🧠 Inizializzazione LIME explainer...")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=[str(c) for c in np.unique(y_train)],
        mode='classification'
    )

    sample = X_test.sample(n=n_samples, random_state=42)
    feature_scores = {}

    if verbose:
        print(f"    🔍 Analisi di {n_samples} istanze di test...")

    for i in range(len(sample)):
        exp = explainer.explain_instance(
            data_row=sample.iloc[i].values,
            predict_fn=safe_predict_proba,
            num_features=len(X_train.columns)
        )
        for feature, weight in exp.as_list():
            feature_scores[feature] = feature_scores.get(feature, []) + [abs(weight)]

    global_scores = {feat: np.mean(weights) for feat, weights in feature_scores.items()}
    global_importance = pd.DataFrame.from_dict(global_scores, orient='index', columns=['mean_abs_weight'])

    # Estrai nome della feature da condizioni tipo "feature > x"
    global_importance['feature'] = global_importance.index.map(extract_feature_name)

    # Aggregazione per nome della feature
    feature_level_importance = global_importance.groupby('feature')['mean_abs_weight'].mean()
    feature_level_importance = feature_level_importance.sort_values(ascending=False)

    if verbose:
        print("     📊 Numero di feature trovate:", feature_level_importance.shape[0])
        print(      feature_level_importance.head())

    # Plot
    plt.figure(figsize=(10, 6))
    feature_level_importance.head(20).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |Weight| (LIME)")
    plt.title("Global Feature Importance via LIME (aggregated)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    if verbose:
        print(f"✅ Salvato in {output_path}")

def extract_feature_name(condition_str):
    match = re.match(r"[<>=\s\-\.\d]*(.*?)(?:\s[<>=])", condition_str)
    if match:
        return match.group(1).strip()
    else:
        return condition_str.strip()  # fallback: restituisce tutto

"""Clustering"""

def get_patients_with_max_severity(merged_df):
    """
    Estrae una singola osservazione per paziente corrispondente alla massima severity.
    
    Args:
        merged_df (pd.DataFrame): Dataset con colonne 'id', 'severity', 'timestep', ecc.

    Returns:
        X (pd.DataFrame): Feature set senza 'id' e 'severity'
        y (pd.Series): Target con la massima severity per paziente
    """
    # 1. Filtro: solo i timestep interi
    hour_df = merged_df[merged_df["timestep"] % 1 != 0.5].reset_index(drop=True)
    hour_df = hour_df.drop(['sepsis', 'timestep'], axis=1).drop_duplicates()

    # 2. Estrai per ogni paziente la riga con severity massima
    single_patients_list = [
        hour_df.loc[hour_df[hour_df["id"] == pid]["severity"].idxmax()]
        for pid in hour_df['id'].unique()
    ]

    single_patients_df = pd.DataFrame(single_patients_list)

    # 3. Separazione feature e target
    X = single_patients_df.drop(['id', 'severity'], axis=1)
    y = single_patients_df['severity']

    return X, y

def elbow_method_analysis(X, k_range=range(2, 10), save_dir="./clustering", prefix="elbow_method", verbose=True):
    """
    Esegue il metodo del gomito su un dataset X e restituisce il miglior numero
    di cluster secondo inertia, distortion e silhouette.

    Args:
        X (pd.DataFrame or np.ndarray): Dataset su cui eseguire il clustering.
        k_range (range): Range di valori di K da testare (default=range(2, 10)).
        save_dir (str): Cartella dove salvare i grafici.
        prefix (str): Prefisso per i nomi dei file grafici.
        verbose (bool): Se True, mostra i grafici e stampa lo stato.

    Returns:
        dict: Dizionario con il miglior k secondo ciascuna metrica.
    """
    os.makedirs(save_dir, exist_ok=True)

    inertias = []
    distortions = []
    silhouettes = []

    if verbose:
        print(f"📊 Avvio del metodo del gomito per k in {list(k_range)}...")
        iterator = k_range
    else:
        iterator = tqdm(k_range, desc="🔍 Analisi cluster", unit="k")

    for k in iterator:
        if verbose:
            print(f"🔹 Calcolo per k = {k}...")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)

        sil_score = silhouette_score(X, kmeans.labels_, metric='euclidean')
        silhouettes.append(sil_score)

        dist = np.min(cdist(X, kmeans.cluster_centers_, metric='cosine'), axis=1)
        distortions.append(np.sum(dist ** 2) / X.shape[0])

        inertias.append(kmeans.inertia_)

    # === Plot: Inertia ===
    plt.figure()
    plt.plot(k_range, inertias, marker='o')
    plt.title('Elbow Method: Inertia')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_inertia.png")
    if verbose:
        plt.show()
    plt.close()

    # === Plot: Distortion ===
    plt.figure()
    plt.plot(k_range, distortions, marker='o')
    plt.title('Elbow Method: Distortion (cosine)')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_distortion.png")
    if verbose:
        plt.show()
    plt.close()

    # === Plot: Silhouette ===
    plt.figure()
    sns.lineplot(x=list(k_range), y=silhouettes, marker='o')
    plt.title('Elbow Method: Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{prefix}_silhouette.png")
    if verbose:
        plt.show()
    plt.close()

    # === Migliori k ===
    best_k_inertia = k_range[np.argmin(inertias)]
    best_k_distortion = k_range[np.argmin(distortions)]
    best_k_silhouette = k_range[np.argmax(silhouettes)]

    if verbose:
        print("\n📌 Miglior numero di cluster:")
        print(f"  ▪️ Inertia     → k = {best_k_inertia}")
        print(f"  ▪️ Distortion  → k = {best_k_distortion}")
        print(f"  ▪️ Silhouette  → k = {best_k_silhouette}")
        print("✅ Grafici salvati in:", save_dir)
        print()

    return {
        "best_k_inertia": best_k_inertia,
        "best_k_distortion": best_k_distortion,
        "best_k_silhouette": best_k_silhouette
    }

def cluster_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    # Hungarian algorithm (optimal matching)
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched_cm = cm[:, col_ind]
    acc = matched_cm.diagonal().sum() / cm.sum()
    return acc, cm, matched_cm

def evaluate_kmeans_clustering(X, y, n_clusters=2, save_dir="./clustering", verbose=True):
    """
    Esegue il clustering con KMeans, calcola metriche, salva plot e risultati.

    Args:
        X (pd.DataFrame or np.ndarray): Dati di input.
        y (pd.Series or np.ndarray): Etichette reali.
        n_clusters (int): Numero di cluster da usare.
        save_dir (str): Cartella dove salvare i risultati.
        verbose (bool): Se True, mostra i plot.

    Returns:
        dict: Dizionario con n_clusters, accuracy, ARI, AMI.
    """
    os.makedirs(save_dir, exist_ok=True)

    kmeans_dict = {
        "n_clusters": n_clusters,
        "accuracy": None,
        "ari": None,
        "ami": None
    }

    # PCA per visualizzazione
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)

    # Accuracy (con matching)
    acc, cm, matched_cm = cluster_accuracy(y, labels)
    kmeans_dict["accuracy"] = acc
    kmeans_dict["ari"] = adjusted_rand_score(y, labels)
    kmeans_dict["ami"] = adjusted_mutual_info_score(y, labels)

    print(f"🧩 KMeans Clustering (n_clusters={n_clusters})")
    print(f"   - Accuracy (matched): {acc:.4f}")
    print(f"   - ARI: {kmeans_dict['ari']:.4f}")
    print(f"   - AMI: {kmeans_dict['ami']:.4f}")

    # Heatmap Confusion Matrix
    plt.figure()
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title("Confusion Matrix (matching)")
    cm_path = os.path.join(save_dir, f"kmeans_c{n_clusters}_confusion_matrix.png")
    plt.savefig(cm_path)
    if verbose:
        plt.show()
    else:
        plt.close()

    # Scatter PCA - Cluster
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA + KMeans Clustering')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)

    # Scatter PCA - Etichette reali
    plt.subplot(1, 2, 2)
    for label in np.unique(y):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Classe {label}', s=50)
    plt.title('Etichette Reali')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)

    cluster_path = os.path.join(save_dir, f"kmeans_c{n_clusters}.png")
    plt.savefig(cluster_path)
    if verbose:
        plt.show()
    else:
        plt.close()

    # Salvataggio JSON
    json_path = os.path.join(save_dir, f"kmeans_c{n_clusters}.json")
    with open(json_path, "w") as f:
        json.dump(kmeans_dict, f, indent=4)

    return kmeans_dict

def evaluate_kmeans_3D_clustering(X, y, n_clusters=2, save_dir="./clustering", verbose=True):
    os.makedirs(save_dir, exist_ok=True)

    # Result dictionary
    kmeans_dict = {
        "n_clusters": n_clusters,
        "accuracy": None,
        "ari": None,
        "ami": None
    }

    # PCA 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X)

    # Accuracy ottimizzata
    acc, cm, matched_cm = cluster_accuracy(y, labels)
    kmeans_dict["accuracy"] = acc
    kmeans_dict["ari"] = adjusted_rand_score(y, labels)
    kmeans_dict["ami"] = adjusted_mutual_info_score(y, labels)

    print(f"🧩 KMeans Clustering (n_clusters={n_clusters})")
    print(f"   - Accuracy (matched): {acc:.4f}")
    print(f"   - ARI: {kmeans_dict['ari']:.4f}")
    print(f"   - AMI: {kmeans_dict['ami']:.4f}")

    # Heatmap confusion matrix
    plt.figure()
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title("Confusion Matrix (matching)")
    cm_path = os.path.join(save_dir, f"kmeans_c{n_clusters}_confusion_matrix.png")
    plt.savefig(cm_path)
    if verbose:
        plt.show()
    else:
        plt.close()

    # === Plot: Cluster assegnati ===
    fig_clusters = go.Figure()
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        fig_clusters.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode='markers',
            marker=dict(size=5),
            name=f"Cluster {cluster_id}",
            legendgroup=f"cluster_{cluster_id}",
            showlegend=True
        ))

    fig_clusters.update_layout(
        title=f"KMeans Clustering (n_clusters={n_clusters})",
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        legend=dict(itemsizing='constant', title='Cluster'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    clusters_path = os.path.join(save_dir, f"kmeans_c{n_clusters}_clusters.html")
    fig_clusters.write_html(clusters_path)
    if verbose:
        fig_clusters.show()

    # === Plot: Etichette reali ===
    fig_labels = go.Figure()
    for class_id in np.unique(y):
        mask = y == class_id
        fig_labels.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode='markers',
            marker=dict(size=5, symbol='diamond'),
            name=f"Classe {class_id}",
            legendgroup=f"label_{class_id}",
            showlegend=True
        ))

    fig_labels.update_layout(
        title="Etichette Reali (PCA 3D)",
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        legend=dict(itemsizing='constant', title='Classi'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    labels_path = os.path.join(save_dir, f"kmeans_c{n_clusters}_labels.html")
    fig_labels.write_html(labels_path)
    if verbose:
        fig_labels.show()

    # Save JSON
    json_path = os.path.join(save_dir, f"kmeans_c{n_clusters}.json")
    with open(json_path, "w") as f:
        json.dump(kmeans_dict, f, indent=4)

    return kmeans_dict

def find_elbow_point(x, y):
    """
    Trova il punto di massimo scostamento dalla linea che congiunge gli estremi di (x,y).
    Restituisce l'indice del punto "gomito".
    """
    # Punti estremi
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    # Distanze perpendicolari dei punti dalla linea p1-p2
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    vecs = np.vstack([x, y]).T - p1
    scalar_product = np.dot(vecs, line_vec_norm)
    proj = np.outer(scalar_product, line_vec_norm)
    dist_vec = vecs - proj
    distances = np.linalg.norm(dist_vec, axis=1)
    
    elbow_index = np.argmax(distances)
    return elbow_index

def plot_dbscan_kdistance_optimal_eps_manual(X, n_neighbors=20, verbose=True, save_path="./dbscan_kdistance_plot_manual.png"):
    if verbose:
        print("🧠 Calcolo dell'eps ottimale per DBSCAN..")
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    k_distances = np.sort(distances[:, n_neighbors - 1])
    x = np.arange(len(k_distances))
    
    elbow_idx = find_elbow_point(x, k_distances)
    eps_optimal = k_distances[elbow_idx]
    
    if verbose:
        plt.figure(figsize=(8,5))
        plt.plot(x, k_distances, label=f'K-distance (k={n_neighbors})')
        plt.axvline(elbow_idx, color='red', linestyle='--', label=f'Optimal eps ≈ {eps_optimal:.3f}')
        plt.title('K-distance Graph for DBSCAN (Manual Elbow Detection)')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{n_neighbors}th Nearest Neighbor Distance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    else:
        for _ in tqdm(range(1), desc="📈 Calcolando DBSCAN k-distance plot..."):
            plt.figure(figsize=(8,5))
            plt.plot(x, k_distances, label=f'K-distance (k={n_neighbors})')
            plt.axvline(elbow_idx, color='red', linestyle='--', label=f'Optimal eps ≈ {eps_optimal:.3f}')
            plt.title('K-distance Graph for DBSCAN (Manual Elbow Detection)')
            plt.xlabel('Points sorted by distance')
            plt.ylabel(f'{n_neighbors}th Nearest Neighbor Distance')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
    
    return eps_optimal

def dbscan_clustering(X, y, eps=1, min_samples=57, save_path="./clustering/", verbose=True):
    """
    🔍 Esegue DBSCAN clustering con visualizzazioni e metriche di valutazione.
    
    Args:
        X (pd.DataFrame or np.array): Feature set
        y (pd.Series or np.array): Etichette reali
        eps (float): parametro eps per DBSCAN
        min_samples (int): parametro min_samples per DBSCAN
        save_path (str): cartella dove salvare risultati e immagini
        verbose (bool): se True mostra i plot, altrimenti li salva senza mostrarli
    
    Returns:
        dict: dizionario con parametri e metriche: eps, min_samples, accuracy, ARI, AMI
    """
    print("🧩 Inizio clustering DBSCAN...")
    
    # PCA per visualizzazione
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    dbscan_dict = {
        "eps": eps,
        "min_samples": min_samples,
        "accuracy": -1,
        "ari": -1,
        "ami": -1
    }

    print(f"⏳ Eseguo DBSCAN con eps={eps} e min_samples={min_samples}...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Usa tqdm per eventuali loop: qui no loop esplicito, ma simuliamo per esempio progress bar dummy
    for _ in tqdm(range(1), desc="DBSCAN clustering"):
        labels = dbscan.fit_predict(X)
    
    # Calcolo accuracy con matching clusters-classi (funzione esterna)
    print("🔎 Calcolo accuratezza e confusion matrix...")
    acc, cm, matched_cm = cluster_accuracy(y, labels)
    print(f"✅ Accuracy ottimizzata (dopo matching cluster-classi): {acc:.4f}")
    dbscan_dict["accuracy"] = acc

    # Plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title(f"Confusion Matrix (DBSCAN eps={eps}, min_samples={min_samples})")
    cm_path = f"{save_path}dbscan_eps{eps}_minsamples{min_samples}_confusion_matrix.png"
    plt.savefig(cm_path)
    if verbose:
        plt.show()
    plt.close()

    # Metriche cluster
    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    dbscan_dict["ari"] = ari
    dbscan_dict["ami"] = ami
    print(f"📊 Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"📊 Adjusted Mutual Information (AMI): {ami:.4f}")

    # Plot cluster DBSCAN (a destra)
    plt.figure(figsize=(14,6))
    plt.subplot(1, 2, 2)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        label_name = f'Cluster {label}' if label != -1 else 'Outlier'
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label_name, s=50)
    plt.title('Cluster DBSCAN')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)

    # Plot etichette reali (a sinistra)
    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Classe {label}', s=50)
    plt.title('Etichette reali')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = f"{save_path}dbscan_eps{eps}_minsamples{min_samples}.png"
    plt.savefig(plot_path)
    if verbose:
        plt.show()
    plt.close()

    # Salvataggio dei risultati in JSON
    json_path = f"{save_path}dbscan_eps{eps}_minsamples{min_samples}.json"
    with open(json_path, "w") as f:
        json.dump(dbscan_dict, f)

    print(f"💾 Risultati salvati in:\n  - {cm_path}\n  - {plot_path}\n  - {json_path}")
    print("🎉 Clustering completato con successo!")
    
    return dbscan_dict

"""Parentesi Lasso"""
def lasso_regression_with_cv(X, y, verbose=True, test_size=0.2, random_state=42):
    """
    Esegue split, GridSearchCV per alpha Lasso, allena modello e valuta su test.

    Args:
        X (pd.DataFrame): feature
        y (pd.Series): target
        verbose (bool): se True mostra plot e dettagli, altrimenti progress bar
        test_size (float): percentuale test set
        random_state (int): seme per split riproducibile

    Returns:
        best_alpha (float), metrics (dict), lasso_model (Lasso)
    """
    print("🧩 Preparazione dati e split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    alphas = np.logspace(-5, 1, 50)
    params = {"alpha": alphas}
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    lasso = Lasso(max_iter=10000)

    if verbose:
        print("⏳ Eseguo GridSearchCV per trovare il miglior alpha di Lasso...")
        grid = GridSearchCV(lasso, param_grid=params, cv=kf, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
    else:
        # Con verbose=False uso tqdm per progress bar custom
        print("⏳ Eseguo GridSearchCV con progress bar... 🔄")
        grid = GridSearchCV(lasso, param_grid=params, cv=kf, n_jobs=-1, verbose=0)

        # wrapper per mostrare progress bar su n_iter:
        from sklearn.model_selection._validation import _fit_and_score

        # funzione custom per tqdm (non standard, ma workaround)
        class GridSearchProgressBar(GridSearchCV):
            def _run_search(self, evaluate_candidates):
                # tqdm wrapper per evaluate_candidates
                def tqdm_evaluate_candidates(candidate_params, cv, more_results=None):
                    pbar = tqdm(total=len(candidate_params), desc="🔍 GridSearchCV")
                    results = {}
                    for param in candidate_params:
                        res = evaluate_candidates([param], cv, more_results)
                        results.update(res)
                        pbar.update(1)
                    pbar.close()
                    return results
                evaluate_candidates = tqdm_evaluate_candidates
                super()._run_search(evaluate_candidates)

        grid = GridSearchProgressBar(lasso, param_grid=params, cv=kf, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

    best_alpha = grid.best_params_['alpha']

    print(f"🎯 Miglior alpha trovato: {best_alpha:.6f}")

    # Allena modello finale
    lasso_best = Lasso(alpha=best_alpha, max_iter=10000)
    lasso_best.fit(X_train, y_train)

    y_pred = lasso_best.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "mae": mae, "r2": r2}

    if verbose:
        print(f"📊 Mean Squared Error (MSE): {mse:.4f}")
        print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
        print(f"📊 R^2 Score: {r2:.4f}")

        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Valori Reali")
        plt.ylabel("Valori Predetti")
        plt.title("Confronto tra Valori Reali e Predetti")
        plt.grid(True)
        plt.show()

        lasso_coef = np.abs(lasso_best.coef_)
        feature_names = X.columns

        plt.figure(figsize=(10,5))
        plt.bar(feature_names, lasso_coef)
        plt.xticks(rotation=90)
        plt.xlabel("Feature")
        plt.ylabel("Importanza (valore assoluto coefficiente)")
        plt.title("Feature selection basata su Lasso")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"✅ Test set evaluation — MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return best_alpha, metrics, lasso_best

def pairplots_with_labels(X, y, group_size=6, verbose=True):
    """
    Crea pairplot a blocchi di feature colorati per label.
    
    Args:
        X (pd.DataFrame): dataframe con feature
        y (pd.Series): target / label
        group_size (int): numero di feature per blocco
        verbose (bool): se True mostra i plot, altrimenti salva solo

    Salva i plot in 'pairplot_i.png' e li mostra solo se verbose=True.
    """
    print("🔢 Pairplots tra le features del dataset..")
    features = X.columns.tolist()
    n_features = len(features)

    df_plot = X.copy()
    df_plot['label'] = y

    n_blocks = math.ceil(n_features / group_size)

    if verbose:
        iterator = range(n_blocks)
    else:
        iterator = tqdm(range(n_blocks), desc="🧩 Pairplot progress")

    for i in iterator:
        start = i * group_size
        end = min((i + 1) * group_size, n_features)
        subset_features = features[start:end]

        if verbose:
            print(f"    📈 Plotting features {start + 1} to {end} (block {i+1}/{n_blocks})...")

            sns.pairplot(df_plot[subset_features + ['label']], hue='label', corner=True, palette='Set2')
            plt.savefig(f"pairplot_{i}.png")
            plt.show()
        else:
            # Solo salva senza mostrare
            sns.pairplot(df_plot[subset_features + ['label']], hue='label', corner=True, palette='Set2')
            plt.savefig(f"pairplot_{i}.png")
            plt.close()

def evaluate_agglomerative_clustering(X, verbose=True, save_path="./clustering/agglomerative_silhouette.png"):
    """
    Valuta l'Agglomerative Clustering per diversi valori di k e restituisce il migliore secondo la silhouette.
    
    Args:
        X (pd.DataFrame or np.ndarray): Dataset di input (solo feature).
        verbose (bool): Se True, stampa e mostra i plot. Se False, mostra solo progress bar e salva il grafico.
        save_path (str): Path dove salvare il grafico della silhouette.

    Returns:
        best_k (int): Numero di cluster con silhouette score massimo.
    """
    silhouette_scores = []
    cluster_range = range(2, 11)

    if verbose:
        print("🧩 Inizio valutazione Agglomerative Clustering...")
        iterator = cluster_range
    else:
        iterator = tqdm(cluster_range, desc="🔗 Clustering...")

    for k in iterator:
        if verbose:
            print(f"    ⏳ Calcolo silhouette per k={k}...")

        clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    best_k = np.argmax(silhouette_scores) + 2  # +2 perché partiamo da k=2

    # Plot
    if verbose:
        print(f"✅ Miglior numero di cluster secondo silhouette: {best_k}")
    else:
        tqdm.write(f"   ✅ Miglior numero di cluster secondo silhouette: {best_k}")

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o', color='teal')
    plt.xlabel('Numero di cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette per Agglomerative Clustering')
    plt.grid(True)
    plt.savefig(save_path)

    if verbose:
        plt.show()
    else:
        plt.close()

    return best_k

def evaluate_davies_bouldin(X, verbose=True, save_path="./clustering/agglomerative_davies_bouldin.png"):
    """
    Valuta Agglomerative Clustering tramite il Davies-Bouldin Index per diversi k.

    Args:
        X (pd.DataFrame or np.ndarray): Dataset con sole feature.
        verbose (bool): Se True, stampa log e mostra il grafico; se False, solo progress bar.
        save_path (str): Path dove salvare il grafico.

    Returns:
        best_k (int): Valore di k con Davies-Bouldin index minimo.
    """
    db_scores = []
    cluster_range = range(2, 11)

    if verbose:
        print("📊 Inizio valutazione Agglomerative Clustering (Davies-Bouldin Index)...")
        iterator = cluster_range
    else:
        iterator = tqdm(cluster_range, desc="🔗 Calcolo DB Index")

    for k in iterator:
        if verbose:
            print(f"    ⏳ Calcolo DB Index per k={k}...")

        clustering = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
        labels = clustering.fit_predict(X)
        db_scores.append(davies_bouldin_score(X, labels))

    best_k = np.argmin(db_scores) + 2  # +2 perché parte da k=2

    if verbose:
        print(f"    ✅ Miglior numero di cluster secondo Davies-Bouldin: {best_k}")
    else:
        tqdm.write(f"   ✅ Miglior numero di cluster secondo Davies-Bouldin: {best_k}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, db_scores, marker='o', color='coral')
    plt.xlabel('Numero di cluster')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index per Agglomerative')
    plt.grid(True)
    plt.savefig(save_path)

    if verbose:
        plt.show()
    else:
        plt.close()

    return best_k

def agglomerative_clustering(X, y, n_clusters=4, verbose=True, save_dir="./clustering"):
    os.makedirs(save_dir, exist_ok=True)
    
    print("🧩 Inizio Agglomerative Clustering...")

    X_pca = PCA(n_components=2).fit_transform(X)

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    cluster_counts = Counter(labels)
    if verbose:
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"🔹 Cluster {cluster_id}: {count} punti")

    def cluster_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cost_matrix = -cm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_cm = cm[:, col_ind]
        acc = matched_cm.diagonal().sum() / matched_cm.sum()
        return acc, cm, matched_cm

    acc, cm, matched_cm = cluster_accuracy(y, labels)
    if verbose:
        print(f"    ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")
    else:
        tqdm.write(f"   ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title("Confusion Matrix (matching)")
    conf_matrix_path = os.path.join(save_dir, f"agglomerative_c:{n_clusters}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    if verbose:
        plt.show()
    plt.close()

    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    if verbose:
        print(f"    📊 Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"    📊 Adjusted Mutual Information (AMI): {ami:.4f}")
    else:
        tqdm.write(f"   📊 ARI: {ari:.4f} | AMI: {ami:.4f}")

    # Save results
    results = {
        "n_clusters": n_clusters,
        "accuracy": acc,
        "ari": ari,
        "ami": ami
    }
    with open(os.path.join(save_dir, f"agglomerative_c:{n_clusters}.json"), "w") as f:
        json.dump(results, f)

    # Plot PCA clusters & true labels
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Classe {label}", s=50)
    plt.title("Etichette reali")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {label}", s=50)
    plt.title("Cluster Agglomerative")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"agglomerative_c:{n_clusters}.png")
    plt.savefig(fig_path)
    if verbose:
        plt.show()
    plt.close()

def agglomerative_3D_clustering(X, y, n_clusters=4, verbose=True, save_dir="./clustering"):
    os.makedirs(save_dir, exist_ok=True)
    
    print("🧩 Inizio Agglomerative Clustering...")

    # PCA con 3 componenti per visualizzazione 3D
    X_pca = PCA(n_components=3).fit_transform(X)

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(X)

    cluster_counts = Counter(labels)
    if verbose:
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"🔹 Cluster {cluster_id}: {count} punti")

    def cluster_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cost_matrix = -cm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_cm = cm[:, col_ind]
        acc = matched_cm.diagonal().sum() / matched_cm.sum()
        return acc, cm, matched_cm

    acc, cm, matched_cm = cluster_accuracy(y, labels)
    if verbose:
        print(f"    ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")
    else:
        tqdm.write(f"   ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")

    # Heatmap Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title("Confusion Matrix (matching)")
    conf_matrix_path = os.path.join(save_dir, f"agglomerative_c:{n_clusters}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    if verbose:
        plt.show()
    plt.close()

    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    if verbose:
        print(f"    📊 Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"    📊 Adjusted Mutual Information (AMI): {ami:.4f}")
    else:
        tqdm.write(f"   📊 ARI: {ari:.4f} | AMI: {ami:.4f}")

    # Salvataggio risultati
    results = {
        "n_clusters": n_clusters,
        "accuracy": acc,
        "ari": ari,
        "ami": ami
    }
    with open(os.path.join(save_dir, f"agglomerative_c:{n_clusters}.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Visualizzazione 3D: Etichette reali e cluster
    fig = plt.figure(figsize=(14, 6))

    # Etichette reali
    ax1 = fig.add_subplot(121, projection='3d')
    for label in np.unique(y):
        mask = y == label
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=f"Classe {label}", s=50)
    ax1.set_title("Etichette reali (PCA 3D)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()

    # Cluster assegnati
    ax2 = fig.add_subplot(122, projection='3d')
    for label in np.unique(labels):
        mask = labels == label
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], label=f"Cluster {label}", s=50)
    ax2.set_title("Agglomerative Clustering (PCA 3D)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"agglomerative_c:{n_clusters}_3d.png")
    plt.savefig(fig_path)
    if verbose:
        plt.show()
    plt.close()

def optimal_gmm_k(X, max_k=10, verbose=True, save_dir="./gmm"):
    os.makedirs(save_dir, exist_ok=True)

    aics = []
    bics = []
    Ks = range(1, max_k + 1)

    iterator = Ks if verbose else tqdm(Ks, desc="🔍 Calcolo AIC/BIC")

    for k in iterator:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        aic = gmm.aic(X)
        bic = gmm.bic(X)
        aics.append(aic)
        bics.append(bic)

        if verbose:
            print(f"    📊 k={k} | AIC={aic:.2f}, BIC={bic:.2f}")

    # Plot AIC/BIC
    plt.figure(figsize=(8, 5))
    plt.plot(Ks, aics, label='AIC', marker='o')
    plt.plot(Ks, bics, label='BIC', marker='o')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Score")
    plt.title("AIC e BIC per GMM")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(save_dir, "gmm_aic_bic.png")
    plt.savefig(plot_path)
    if verbose:
        plt.show()
    plt.close()

    best_k_bic = Ks[np.argmin(bics)]
    best_k_aic = Ks[np.argmin(aics)]

    if verbose:
        print(f"    ✅ Numero ottimale di cluster secondo BIC: {best_k_bic}")
        print(f"    ✅ Numero ottimale di cluster secondo AIC: {best_k_aic}")
    else:
        tqdm.write(f"✅ Miglior k BIC: {best_k_bic} | Miglior k AIC: {best_k_aic}")

    return best_k_bic, best_k_aic

def gmm_clustering(X, y, n_components=2, verbose=True, save_dir="./clustering"):

    os.makedirs(save_dir, exist_ok=True)
    prefix = f"gmm_c:{n_components}"

    log = tqdm if not verbose else lambda x, **kwargs: x
    log_write = tqdm.write if not verbose else print

    log_write(f"🤖 Inizio clustering GMM con {n_components} componenti...")

    gmm = GaussianMixture(n_components=n_components, tol=1e-3, init_params='random', random_state=42)
    labels = gmm.fit_predict(X)

    cluster_counts = Counter(labels)
    if verbose:
        for cluster_id, count in sorted(cluster_counts.items()):
            print(f"🔹 Cluster {cluster_id}: {count} punti")

    acc, cm, matched_cm = cluster_accuracy(y, labels)
    if verbose:
        print(f"    ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")
    else:
        tqdm.write(f"   ✅ Accuracy ottimizzata (matching cluster-classi): {acc:.4f}")

    # Heatmap della confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(matched_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Cluster assegnato")
    plt.ylabel("Classe reale")
    plt.title("Confusion Matrix (matching)")
    conf_matrix_path = os.path.join(save_dir, f"{prefix}_confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    if verbose:
        plt.show()
    plt.close()

    # ARI & AMI
    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    if verbose:
        print(f"    📊 Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"    📊 Adjusted Mutual Information (AMI): {ami:.4f}")
    else:
        tqdm.write(f"   📊 ARI: {ari:.4f} | AMI: {ami:.4f}")

    # Salva i risultati
    results = {
        "n_clusters": n_components,
        "accuracy": acc,
        "ari": ari,
        "ami": ami
    }
    with open(os.path.join(save_dir, f"{prefix}.json"), "w") as f:
        json.dump(results, f)

    # PCA per il plot
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(12, 5))

    # Plot etichette reali
    plt.subplot(1, 2, 1)
    for label in np.unique(y):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Classe {label}", s=50)
    plt.title("Etichette reali")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    # Plot cluster GMM
    plt.subplot(1, 2, 2)
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"Cluster {label}", s=50)
    plt.title("Cluster GMM")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{prefix}.png")
    plt.savefig(fig_path)
    if verbose:
        plt.show()
    plt.close()

    return results

def binarize_series(series):
    """
    Binarizza una pandas Series:
    - 0 o 1 → 0
    - 2, 3, 4 → 1

    Args:
        serie (pd.Series): Serie da binarizzare.

    Returns:
        pd.Series: Serie binarizzata.
    """
    return series.apply(lambda x: 1 if x in [2, 3, 4] else 0)

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

def find_sepsis_onset(arr):
    arr = np.array(arr)
    
    # Trova gli indici dei valori >= 2
    idx = np.where(np.isin(arr, [2, 3, 4]))[0]
    if len(idx) > 0:
        return idx[0]  # il primo valore tra 2, 3, 4 in ordine

    # Se non ci sono 2, 3 o 4 → ritorna indice del valore massimo (tra 0 o 1)
    val_max = np.max(arr)
    return np.argmax(arr == val_max)

"""class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 1 solo logit per esempio

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)          # h_n: [1, B, H]
        out = self.fc(h_n.squeeze(0))       # out: [B, 1]
        return out.squeeze(-1)  """

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 bidirectional=True, dropout=0.3, pooling="mean"):
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

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 1)
        )

    def forward(self, x, lengths):
        # x: [B, T, F]
        # lengths: [B]
        if not torch.is_tensor(lengths):
            lengths = torch.tensor(lengths, device=x.device)

        # Ordina per lunghezza
        lengths_sorted, idx = lengths.sort(descending=True)
        x_sorted = x[idx]

        # Impacchetta la sequenza (ignora padding)
        packed = pack_padded_sequence(
            x_sorted,
            lengths_sorted.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        # LSTM su sequenze reali
        packed_out, (h_n, c_n) = self.lstm(packed)

        # Unpack se serve pooling
        output, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Ripristina l’ordine originale
        _, unidx = idx.sort()
        output = output[unidx]
        h_n = h_n[:, unidx, :]

        # Pooling corretto
        if self.pooling == "last":
            # Ultimo hidden state (corretto anche per bidirectional)
            if self.bidirectional:
                out = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                out = h_n[-1]

        elif self.pooling == "mean":
            # Media solo sui timesteps validi
            mask = torch.arange(output.size(1), device=lengths.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            out = (output * mask).sum(dim=1) / lengths.unsqueeze(1)

        elif self.pooling == "max":
            mask = torch.arange(output.size(1), device=lengths.device)[None, :] < lengths[:, None]
            masked_output = output.clone()
            masked_output[~mask] = -1e9
            out, _ = masked_output.max(dim=1)

        else:
            raise ValueError("Pooling non supportato: ['last', 'mean', 'max']")

        out = self.fc(out)
        return out.view(-1)
   
def plot_confusion(y_true, y_pred, labels):
    os.makedirs('./eval', exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("./eval/confusion_matrix.png")
    plt.show()

def evaluate_predictions(y_true, y_pred, labels, probs=None):
    # Flatten e filtra padding
    mask = y_true != -100
    y_true_flat = y_true[mask].cpu().numpy()
    y_pred_flat = y_pred[mask].cpu().numpy()
    
    if probs is not None:
        probs_flat = probs[mask].cpu().numpy()

    # Accuracy
    print(f"\nAccuracy: ")
    print(accuracy_score(y_true_flat, y_pred_flat))

    # MCC
    print(f"\nMatthews Correlation Coefficient (MCC): ")
    print(matthews_corrcoef(y_true_flat, y_pred_flat))

    # Brier Score (solo se passate le probabilità)
    if probs is not None:
        print(f"\nBrier Score: ")
        print(brier_score_loss(y_true_flat, probs_flat))

    # Classification Report
    print("\nClassification Report:")
    report_dict = classification_report(
        y_true_flat, y_pred_flat, labels=labels, zero_division=0, output_dict=True
    )
    print(classification_report(y_true_flat, y_pred_flat, labels=labels, zero_division=0))

    # Salva su file JSON
    with open("./eval/classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=4)

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_flat, y_pred_flat, labels=labels))
    plot_confusion(y_true_flat, y_pred_flat, labels=labels)

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
    metric_fn=accuracy_score,
    n_repeats=5,
    baseline_score=None,
    feature_names=None,
    verbose=True):

    model.eval()
    device = next(model.parameters()).device

    X = X.clone().to(device)
    y = y.clone().to(device)

    B, T, F = X.shape
    lengths = [T] * B  # tutte le sequenze complete

    # -----------------------------------------
    #           BASELINE
    # -----------------------------------------
    with torch.no_grad():
        logits = model(X, lengths)

        if logits.dim() == 2:
            logits = logits.squeeze(1)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()

    if baseline_score is None:
        baseline_score = metric_fn(
            y.cpu().numpy(),
            preds.cpu().numpy()
        )

    if verbose:
        print(f"\nBaseline {metric_fn.__name__}: {baseline_score:.4f}")

    # -----------------------------------------
    #       PERMUTATION IMPORTANCE
    # -----------------------------------------
    importances = []
    std_devs = []

    for feature_idx in range(F):
        scores = []

        for _ in range(n_repeats):
            X_perm = X.clone()

            # permutazione tra pazienti
            perm = torch.randperm(B)
            X_perm[:, :, feature_idx] = X[perm, :, feature_idx]

            with torch.no_grad():
                logits = model(X_perm, lengths)
                if logits.dim() == 2:
                    logits = logits.squeeze(1)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

            score = metric_fn(
                y.cpu().numpy(),
                preds.cpu().numpy()
            )
            scores.append(score)

        diffs = baseline_score - np.array(scores)
        importances.append(diffs.mean())
        std_devs.append(diffs.std())

        if verbose:
            fname = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
            print(f"{fname}: Δ = {diffs.mean():.4f} ± {diffs.std():.4f}")

    df_importance = pd.DataFrame({
        "Feature": feature_names if feature_names else [f"f{i}" for i in range(F)],
        "Importance": importances,
        "StdDev": std_devs
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return df_importance

def random_feature_importance(
    model, X, y, metric_fn, n_repeats=5, baseline_score=None, verbose=True):
    """
    Calcola la feature importance sostituendo ogni feature con valori random (rumore),
    piuttosto che permutarla tra pazienti.
    """
    model.eval()
    device = next(model.parameters()).device
    X = X.clone().to(device)
    y = y.clone().to(device)

    # Calcola baseline se non fornita
    with torch.no_grad():
        base_probs = torch.sigmoid(model(X))
        base_preds = (base_probs > 0.5).long()
    if baseline_score is None:
        baseline_score = metric_fn(y.cpu().numpy(), base_preds.cpu().numpy())
    if verbose:
        print(f"Baseline score: {baseline_score:.4f}")

    importances = []
    std_devs = []

    for feature_idx in range(X.shape[2]):
        scores = []

        for _ in range(n_repeats):
            X_perturbed = X.clone()

            # Sostituisci la feature con rumore casuale (distribuito come la feature originale)
            mean = X[:, :, feature_idx].mean()
            std = X[:, :, feature_idx].std()
            noise = torch.normal(mean=mean, std=std, size=X[:, :, feature_idx].shape).to(device)
            X_perturbed[:, :, feature_idx] = noise

            with torch.no_grad():
                probs = torch.sigmoid(model(X_perturbed))
                preds = (probs > 0.5).long()
                score = metric_fn(y.cpu().numpy(), preds.cpu().numpy())
                scores.append(score)

        diffs = baseline_score - np.array(scores)
        importances.append(np.mean(diffs))
        std_devs.append(np.std(diffs))

        if verbose:
            print(f"[Random] Feature {feature_idx}: Δ = {np.mean(diffs):.4f} ± {np.std(diffs):.4f}")

    return importances, std_devs

def forward_fn(x):
    model.train()
    logits = model(x)
    probs = torch.sigmoid(logits)
    return probs

def compute_avg_attributions(model, X, feature_names, baseline_value=0.0, num_samples=20):

    ig = IntegratedGradients(forward_fn)
    attr_total = torch.zeros(X.shape[2])  # F

    for i in range(num_samples):
        input_tensor = X[i].unsqueeze(0).to(device)
        baseline = torch.zeros_like(input_tensor).fill_(baseline_value).to(device)

        attr, _ = ig.attribute(inputs=input_tensor, baselines=baseline, return_convergence_delta=True)
        attr_total += attr.squeeze(0).abs().mean(dim=0).detach().cpu()

    attr_avg = attr_total / num_samples
    return dict(zip(feature_names, attr_avg.tolist()))

def plot_ig_heatmap(model, input_tensor, feature_names, baseline_value=0.0, title="Attribution Heatmap"):
    model.train()
    device = next(model.parameters()).device
    input_tensor = input_tensor.unsqueeze(0).to(device)  # [1, T, F]
    baseline = torch.zeros_like(input_tensor).fill_(baseline_value).to(device)

    # model.eval()
    ig = IntegratedGradients(lambda x: torch.sigmoid(model(x)))
    attributions, _ = ig.attribute(inputs=input_tensor, baselines=baseline, return_convergence_delta=True)

    # Convert to shape [T, F] and detach
    attr = attributions.squeeze(0).detach().cpu().numpy().T  # [F, T] for heatmap

    # Plot
    plt.figure(figsize=(12, max(6, len(feature_names) * 0.3)))
    sns.heatmap(attr, xticklabels=range(attr.shape[1]), yticklabels=feature_names, cmap="viridis")
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("captum_ig_heatmap.png")
    plt.show()

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
    data,
    model_class,
    device,
    min_length,
    prediction_horizon,
    target_len,
    gender='male',
    gender_col='gender',
    id_col='icustay_id',
    label_col='label',
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
    seed=42,
    weight_decay_value=0):
    """
    Cross-validation con validation set interno, early stopping e metriche complete.
    Il bilanciamento viene applicato solo sul sottoinsieme di training interno.
    """
    set_deterministic(seed)

    data=data[['icustay_id','chart_time','gender','ph_bloodgas', 'peakinsppressure', 'hematocrit', 'wbc', 'bun', 'po2_bloodgas', 'sysbp', 'so2_bloodgas', 'platelet', 
                            'fio2', 'bicarbonate', 'cardiacoutput', 'pco2_bloodgas', 'lactate', 'diabp', 'potassium', 'heartrate', 'tvobserved', 
                            'ck_mb', 'totalpeeplevel', 'troponin_t', 'o2flow', 'tvset', 'label']]

    static_cols = ['label']

    measurement_cols = data.columns.drop(static_cols+['icustay_id', 'chart_time', 'gender'])

    # if gender == 'male':
    #     data = data[data[gender_col] == 0]
    # else:
    #     data = data[data[gender_col] == 1]

    # Prepara liste una volta sola
    grouped = data.groupby(id_col)
    patient_ids = list(grouped.groups.keys())
    patient_labels = [grouped.get_group(pid)[label_col].iloc[0] for pid in patient_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results, all_y_true_folds, all_y_pred_folds = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")

        train_loss_trend, val_loss_trend = [], []

        train_ids = [patient_ids[i] for i in train_idx]
        test_ids  = [patient_ids[i] for i in test_idx]

        train_df = data[data[id_col].isin(train_ids)].drop(columns=['gender'])
        test_df  = data[data[id_col].isin(test_ids)].drop(columns=['gender'])

        print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
        print(f" N. of train pat. : {len(train_ids)}, N. of test pat.: {len(test_ids)}")
        print(f" N. of septic pat. (train): {train_df.drop_duplicates(id_col)[label_col].sum()}, N. of septic pat. (test): {test_df.drop_duplicates(id_col)[label_col].sum()}")
        print()
        train_df, test_df = normalize_df(train_df, test_df, measurement_cols, save_path='scaler_stats.json')

        train_df = resample_propagate_mask(train_df, measurement_cols, static_cols, mask_value=0, desc='Resampling and Carry forward for Training')
        test_df = resample_propagate_mask(test_df, measurement_cols, static_cols, mask_value=0, desc='Resampling and Carry forward for Test')

        # Window Size filtering
        patient_lengths = train_df.groupby('icustay_id').size()
        valid_train_ids = patient_lengths[patient_lengths >= prediction_horizon + min_length].index
        train_min_len_df = train_df[train_df['icustay_id'].isin(valid_train_ids)]

        patient_lengths = test_df.groupby('icustay_id').size()
        valid_test_ids = patient_lengths[patient_lengths >= prediction_horizon + min_length].index
        test_min_len_df = test_df[test_df['icustay_id'].isin(valid_test_ids)]

        print(f" N. of (remained) train pat. : {len(valid_train_ids)}, N. of test pat.: {len(valid_test_ids)}")
        print(f" N. of (remained) septic pat. (train): {train_min_len_df.drop_duplicates(id_col)[label_col].sum()}, N. of septic pat. (test): {test_min_len_df.drop_duplicates(id_col)[label_col].sum()}")

        print()
        train_data_by_patient = prepare_mimic_data(train_min_len_df, target_len=target_len, pred_length=prediction_horizon, desc="Prepare (padding, cut, truncate, ecc.) training data")
        test_data_by_patient = prepare_mimic_data(test_min_len_df, target_len=target_len, pred_length=prediction_horizon,  desc="Prepare (padding, cut, truncate, ecc.) test data")

        X_train_full = torch.stack([train_data_by_patient[pid]["x"] for pid in valid_train_ids])
        y_train_full = torch.stack([train_data_by_patient[pid]["y"] for pid in valid_train_ids])
        X_test = torch.stack([test_data_by_patient[pid]["x"] for pid in valid_test_ids])
        y_test = torch.stack([test_data_by_patient[pid]["y"] for pid in valid_test_ids])

        # --- Split interno train/val (per early stopping) ---
        full_dataset = TensorDataset(X_train_full, y_train_full)
        val_size = int(val_ratio * len(full_dataset))
        train_size = len(full_dataset) - val_size

        split_gen = torch.Generator().manual_seed(42 + fold)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_gen)

        # --- DataLoader ---
        loader_gen = torch.Generator().manual_seed(42 + fold)

        if method:
            print("\nApplying balancing to training subset only...")
            train_data_by_patient = {i: {"x": x, "y": y} for i, (x, y) in enumerate(train_dataset)}
            train_data_by_patient = balance_patients(train_data_by_patient, method=method, random_state=42 + fold)
            train_loader = DataLoader(
            TensorDataset(
                torch.stack([v["x"] for v in train_data_by_patient.values()]),
                torch.stack([v["y"] for v in train_data_by_patient.values()])
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=loader_gen)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_gen)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        print(f" Train size: {len(train_loader.dataset)}, Val size: {len(val_loader.dataset)}, Test size: {len(test_loader.dataset)}")

        pos_weight=torch.Tensor([1.0])

        # Modello
        model = model_class(
            input_dim=X_train_full.shape[2],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            pooling=pooling
        ).to(device)

        criterion = loss_fn or nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)

        # --- Training Loop ---
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_loss_trend, val_loss_trend = [], []

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # lunghezze reali
                lengths = (batch_x != -100.0).any(dim=2).sum(dim=1)

                optimizer.zero_grad()
                logits = model(batch_x, lengths)
                loss = criterion(logits, batch_y.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    lengths = (batch_x != -100.0).any(dim=2).sum(dim=1)
                    logits = model(batch_x, lengths)
                    loss = criterion(logits, batch_y.float())
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_loss_trend.append(train_loss)
            val_loss_trend.append(val_loss)

            print(f"Epoch {epoch+1}/{max_epochs} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch+1}")
                    break

        # Plot loss trend
        plt.plot(train_loss_trend, label="Train")
        plt.plot(val_loss_trend, label="Val")
        plt.title(f"Loss trend fold {fold+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./loss_trend_fold-{fold+1}.png", dpi=300)
        plt.clf()

        # Ripristina best model
        model.load_state_dict(best_state)
        model.eval()

        # ----------------------
        # Test finale per fold
        # ----------------------
        all_y_true, all_y_pred, all_probs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                lengths = (batch_x != -100.0).any(dim=2).sum(dim=1)
                logits = model(batch_x, lengths)
                probs = torch.sigmoid(logits)
                y_pred = (probs > 0.5).int()
                all_y_true.append(batch_y.cpu())
                all_y_pred.append(y_pred.cpu())
                all_probs.append(probs.cpu())

        # Concatenazione batch
        all_y_true = torch.cat(all_y_true).numpy()
        all_y_pred = torch.cat(all_y_pred).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # Metriche fold
        acc = accuracy_score(all_y_true, all_y_pred)
        auc = roc_auc_score(all_y_true, all_probs)
        prec = precision_score(all_y_true, all_y_pred, zero_division=0)
        rec = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        mcc = matthews_corrcoef(all_y_true, all_y_pred)
        brier = brier_score_loss(all_y_true, all_probs)
        cm = confusion_matrix(all_y_true, all_y_pred)

        # Metriche per classe
        class_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
        class0_metrics = class_report["0"]
        class1_metrics = class_report["1"]

        print(f"\nFold {fold+1} results:")
        print(f"ACC={acc:.3f} | AUC={auc:.3f} | F1={f1:.3f} | MCC={mcc:.3f} | Brier={brier:.3f}")
        print(f"Non-sepsis (0): P={class0_metrics['precision']:.3f}, R={class0_metrics['recall']:.3f}, F1={class0_metrics['f1-score']:.3f}")
        print(f"Sepsis (1):     P={class1_metrics['precision']:.3f}, R={class1_metrics['recall']:.3f}, F1={class1_metrics['f1-score']:.3f}")
        print(f"Confusion matrix:\n{cm}")

        fold_results.append({
            "acc": acc, "auc": auc, "f1": f1, "mcc": mcc, "brier": brier,
            "prec": prec, "rec": rec, "cm": cm,
            "class0": class0_metrics,
            "class1": class1_metrics
        })

        all_y_true_folds.append(all_y_true)
        all_y_pred_folds.append(all_y_pred)

    # ----------------------
    # Metriche medie tra fold
    # ----------------------
    mean_metrics = {k: np.mean([f[k] for f in fold_results]) for k in ["acc", "auc", "prec", "rec", "f1", "mcc", "brier"]}
    mean_class0 = {m: np.mean([f["class0"][m] for f in fold_results]) for m in ["precision", "recall", "f1-score"]}
    mean_class1 = {m: np.mean([f["class1"][m] for f in fold_results]) for m in ["precision", "recall", "f1-score"]}

    print("\n================= RISULTATI MEDI TRA I FOLD =================")
    print(f"ACC={mean_metrics['acc']:.3f} | AUC={mean_metrics['auc']:.3f} | MCC={mean_metrics['mcc']:.3f} | Brier={mean_metrics['brier']:.3f}")
    print(f"\nClasse 0 (non sepsis): P={mean_class0['precision']:.3f}, R={mean_class0['recall']:.3f}, F1={mean_class0['f1-score']:.3f}")
    print(f"Classe 1 (sepsis):     P={mean_class1['precision']:.3f}, R={mean_class1['recall']:.3f}, F1={mean_class1['f1-score']:.3f}")

    # Confusion matrix totale (tutti i fold)
    all_y_true_total = np.concatenate(all_y_true_folds)
    all_y_pred_total = np.concatenate(all_y_pred_folds)
    final_cm = confusion_matrix(all_y_true_total, all_y_pred_total)
    print(f"Confusion matrix finale:\n{final_cm}")

    return {
        "folds": fold_results,
        "mean_global": mean_metrics,
        "mean_class0": mean_class0,
        "mean_class1": mean_class1
    }

def compute_permutation_importance(
    model,
    X,
    y,
    feature_names,
    metric_fn,
    n_repeats=100,
    output_path="./permutation_importance.png",
    verbose=True,
    show_plot=True,
    permutation_feature_importance=None,):
    """
    Calcola e visualizza la Permutation Feature Importance con deviazioni standard.

    Args:
        model: modello già addestrato
        X (np.ndarray o torch.Tensor): feature di test
        y (np.ndarray o torch.Tensor): target di test
        feature_names (list): nomi delle feature
        metric_fn (callable): metrica da usare (es. accuracy_score, roc_auc_score)
        n_repeats (int): numero di permutazioni per ogni feature
        output_path (str): path per salvare il grafico
        verbose (bool): se True stampa risultati
        show_plot (bool): se True mostra il grafico a schermo
        permutation_feature_importance (callable): funzione che calcola le importanze

    Returns:
        dict con importances, std_devs e ordinamento delle feature
    """
    if permutation_feature_importance is None:
        raise ValueError("Devi passare la funzione `permutation_feature_importance` come parametro.")

    if verbose:
        print("\n### Feature Permutation Importance ###")

    importances, std_devs = permutation_feature_importance(
        model=model,
        X=X,
        y=y,
        metric_fn=metric_fn,
        n_repeats=n_repeats,
        verbose=verbose
    )

    # Ordina in base all’importanza
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_stddevs = std_devs[sorted_idx]
    sorted_feature_names = [feature_names[i] for i in sorted_idx]

    if verbose:
        print()
        for name, imp, std in zip(sorted_feature_names, sorted_importances, sorted_stddevs):
            print(f"{name} -> (Avg) Importance: {imp:.6f} - Std: {std:.6f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, sorted_importances, xerr=sorted_stddevs, color='skyblue')
    plt.xlabel(f"Permutation Importance (Δ {metric_fn.__name__})")
    plt.title("Feature Importance with confidence interval")
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "importances": sorted_importances,
        "std_devs": sorted_stddevs,
        "feature_names": sorted_feature_names
    }

"""BALANCING"""

def prepare_sepsisexp_data(
    sepsisexp_path="datasets/SepsisExp",
    target_len=24,
    pred_length=1,
    padding_value=-100,
    verbose=True,
    pop_patient=False,
    random_state=42,   # default seed fisso
    n=10):

    # --- analisi dataset ---
    # La funzione analyze_csvs è assunta essere disponibile
    data, merged_df, outliers_list = analyze_csvs(folder_path=sepsisexp_path, verbose=False)

    hour_df = merged_df[merged_df["timestep"] % 1 != 0.5].reset_index(drop=True)
    if verbose:
        print(f"\nDf shape: {hour_df.shape}")

    if pop_patient:
        # --- Modifica: Selezione Bilanciata per il Pop-out ---
        rng = np.random.default_rng(random_state)
        unique_ids = hour_df['id'].unique()
        
        # 1. Trova l'etichetta 'sepsis' (y) per ogni paziente
        patient_labels = hour_df.groupby('id')['sepsis'].first()
        
        # 2. Suddividi gli ID in base all'etichetta
        septic_ids = patient_labels[patient_labels == 1].index.tolist()
        non_septic_ids = patient_labels[patient_labels == 0].index.tolist()

        if n > len(unique_ids):
            raise ValueError(f"n={n} è maggiore del numero di pazienti disponibili ({len(unique_ids)}).")

        # 3. Calcola il numero di pazienti da estrarre per ogni classe
        # L'obiettivo è n/2 per classe, ma rispettando i limiti di ciascuna
        n_septic_max = len(septic_ids)
        n_non_septic_max = len(non_septic_ids)
        
        # Inizialmente, cerca di prendere la metà (arrotondando per eccesso per la classe 1)
        n_septic = min(int(np.ceil(n / 2)), n_septic_max)
        n_non_septic = min(n - n_septic, n_non_septic_max)
        
        # Aggiusta se non è stato raggiunto il totale 'n' a causa dei limiti di una classe
        if n_septic + n_non_septic < n:
            remaining = n - (n_septic + n_non_septic)
            if remaining > 0:
                 # Riempi il restante dalla classe non ancora piena
                if n_septic_max - n_septic > 0:
                    n_septic += min(remaining, n_septic_max - n_septic)
                elif n_non_septic_max - n_non_septic > 0:
                    n_non_septic += min(remaining, n_non_septic_max - n_non_septic)
                    
        # 4. Controllo finale
        if n_septic + n_non_septic != n:
            print(f"ATTENZIONE: Non è stato possibile estrarre esattamente n={n} pazienti bilanciati. Estratti {n_septic + n_non_septic}. Controllare la distribuzione dei label.")
        
        # 5. Selezione stratificata
        # Scelta casuale e senza sostituzione
        selected_septic_ids = rng.choice(septic_ids, size=n_septic, replace=False)
        selected_non_septic_ids = rng.choice(non_septic_ids, size=n_non_septic, replace=False)
        
        # Combina e ordina per stabilità cross-run
        selected_ids = np.sort(np.concatenate([selected_septic_ids, selected_non_septic_ids]))
        
        # --- Estrazione e Salvataggio (il resto del codice rimane simile) ---
        
        # estrazione dei pazienti selezionati
        subset_df = hour_df[hour_df['id'].isin(selected_ids)].copy()
        remaining_df = hour_df[~hour_df['id'].isin(selected_ids)].copy()

        # cartella per i singoli CSV
        save_dir = os.path.join(sepsisexp_path, "backend", "random_patients")
        os.makedirs(save_dir, exist_ok=True)

        # salva ciascun paziente con indice fisso
        for i, pid in enumerate(selected_ids):
            patient_df = subset_df[subset_df["id"] == pid]
            patient_df.to_csv(os.path.join(save_dir, f"patient_{i:02d}_id{pid}.csv"), index=False)

        hour_df = remaining_df.copy()

        if verbose:
            print(f"Extracted {len(selected_ids)} patient(s) (Target: {n}) on a total of {len(unique_ids)}")
            print(f"Sepsis Label Distribution (Extracted): 1={n_septic}, 0={n_non_septic}")
            print(f"Selected IDs: {selected_ids.tolist()}")
            print(f"Saved individual CSVs in: {save_dir}")

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("Using device:", device)

    # --- preparazione tensoriale ---
    data_by_patient = {}
    for patient_id, group in hour_df.groupby("id"):
        x = group.drop(columns=["id", "sepsis", "timestep", "severity"]).values
        y = group["sepsis"].iloc[0]
        sepsis_onset_index = np.argmax(group["severity"])

        x_tensor = torch.tensor(x, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)

        data_by_patient[patient_id] = {
            "x": pad_or_truncate_sequence(
                x_tensor,
                target_length=target_len,
                sepsis_onset_index=sepsis_onset_index,
                pred_length=pred_length,
                padding_value=padding_value
            ),
            "y": y_tensor
        }

    if verbose:
        print(f"Prepared data for {len(data_by_patient)} patients")

    return data_by_patient, device, hour_df, outliers_list

def prepare_mimic_data(mimic_df, target_len=24, pred_length=1, pop_patient=False, n=1, random_state=42, verbose=True, desc=None):

    set_seed(42)

    # original_features = ['icustay_id','chart_time','gender','ph_bloodgas', 'peakinsppressure', 'hematocrit', 'wbc', 'bun', 'po2_bloodgas', 'sysbp', 'so2_bloodgas', 'platelet', 
    #                         'fio2', 'bicarbonate', 'cardiacoutput', 'pco2_bloodgas', 'lactate', 'diabp', 'potassium', 'heartrate', 'tvobserved', 
    #                         'ck_mb', 'totalpeeplevel', 'troponin_t', 'o2flow', 'tvset', 'label']

    # mimic_df= mimic_df[original_features]

    if pop_patient:
        # --- Modifica: Selezione Bilanciata per il Pop-out ---
        rng = np.random.default_rng(random_state)
        unique_ids = mimic_df['icustay_id'].unique()
        
        # 1. Trova l'etichetta 'sepsis' (y) per ogni paziente (è la stessa per tutte le timestep)
        # Usiamo groupby e take(0) per ottenere la prima (e unica) etichetta per ogni ID
        patient_labels = mimic_df.groupby('icustay_id')['label'].first()
        
        # 2. Suddividi gli ID in base all'etichetta
        septic_ids = patient_labels[patient_labels == 1].index.tolist()
        non_septic_ids = patient_labels[patient_labels == 0].index.tolist()

        if n > len(unique_ids):
            raise ValueError(f"n={n} is greater than the number of available patients({len(unique_ids)}).")

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
            print(f"Warninf: Ita was not possible to perfectly extract n={n} patients. Patientes extracted: {n_septic + n_non_septic}. Check labels distribution.")
        
        # 4. Selezione stratificata
        # Scelta casuale e senza sostituzione
        selected_septic_ids = rng.choice(septic_ids, size=n_septic, replace=False)
        selected_non_septic_ids = rng.choice(non_septic_ids, size=n_non_septic, replace=False)
        
        # Combina e ordina per stabilità cross-run del set finale (opzionale, ma utile)
        selected_ids = np.sort(np.concatenate([selected_septic_ids, selected_non_septic_ids]))
        
        # --- Estrazione e Salvataggio (il resto del codice rimane simile) ---
        
        # Estrazione dei pazienti selezionati
        subset_df = mimic_df[mimic_df['icustay_id'].isin(selected_ids)].copy()
        remaining_df = mimic_df[~mimic_df['icustay_id'].isin(selected_ids)].copy()

        # Cartella per i singoli CSV
        save_dir = os.path.join(mimic_path, "backend", "random_patients")
        os.makedirs(save_dir, exist_ok=True)

        # Salva ciascun paziente con indice fisso
        for i, pid in enumerate(selected_ids):
            patient_df = subset_df[subset_df["icustay_id"] == pid]
            patient_df.to_csv(os.path.join(save_dir, f"patient_{i:02d}_id{pid}.csv"), index=False)

        mimic_df = remaining_df.copy()

        if verbose:
            print(f"Extracted {len(selected_ids)} patient(s) (Target: {n}) on a total of {len(unique_ids)}")
            print(f"Sepsis Label Distribution (Extracted): 1={n_septic}, 0={n_non_septic}")
            print(f"Selected IDs: {selected_ids.tolist()}")
            print(f"Saved individual CSVs in: {save_dir}")
            
    data_by_patient = {}

    for patient_id, group in tqdm(mimic_df.groupby("icustay_id"), desc=desc):
        x = group.drop(columns=["icustay_id", "label", "chart_time"]).values
        y = group["label"].iloc[0]

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

    return data_by_patient

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
    data,
    model_class,
    device,
    min_length,
    prediction_horizon,
    target_len,
    gender='male',
    gender_col='gender',
    id_col='icustay_id',
    label_col='label',
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
    seed=42,
    weight_decay_value=0):
    """
    Allena il modello finale usando tutto il dataset disponibile
    con uno split train/validation interno.
    Restituisce: modello finale, metriche complete, tensori di validazione.
    """

    set_deterministic(seed)

    static_cols = ['label']
    measurement_cols = data.columns.drop(static_cols + [id_col, 'chart_time', gender_col])

    # --------------------------
    # Filtra per genere
    # --------------------------
    """if gender == 'male':
        data = data[data[gender_col] == 0].drop(columns=[gender_col])
    else:
        data = data[data[gender_col] == 1].drop(columns=[gender_col])"""

    data_for_test = resample_propagate_mask(
    data, measurement_cols, static_cols,
    mask_value=0,
    desc="Just for test"
    )

    data_for_test.to_csv("datasets/MIMIC/my_mimic_resampled_propagate.csv", index=False)

    # --------------------------
    # Normalizzazione
    # --------------------------
    data_norm, _ = normalize_df(
        data, data, measurement_cols,
        save_path="scaler_stats_final.json"
    )

    # --------------------------
    # Resampling + carry forward
    # --------------------------
    data_proc = resample_propagate_mask(
        data_norm, measurement_cols, static_cols,
        mask_value=0,
        desc="Resampling and Carry forward (final)"
    )

    # --------------------------
    # Filtra pazienti troppo corti
    # --------------------------
    patient_lengths = data_proc.groupby(id_col).size()
    valid_ids = patient_lengths[patient_lengths >= prediction_horizon + min_length].index
    data_proc = data_proc[data_proc[id_col].isin(valid_ids)]

    # --------------------------
    # Prepara tensor per paziente
    # --------------------------
    data_by_patient = prepare_mimic_data(
        data_proc,
        target_len=target_len,
        pred_length=prediction_horizon,
        desc="Prepare (padding, cut, truncate, ecc.) final training data"
    )

    X_all = torch.stack([data_by_patient[pid]["x"] for pid in valid_ids])
    y_all = torch.stack([data_by_patient[pid]["y"] for pid in valid_ids])

    full_dataset = TensorDataset(X_all, y_all)

    # --------------------------
    # Train/val split interno
    # --------------------------
    val_size = int(val_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=split_gen
    )

    loader_gen = torch.Generator().manual_seed(seed)

    # --------------------------
    # Balancing solo sul train
    # --------------------------
    if method:
        print("Applying balancing to training subset only...")
        tmp = {i: {"x": x, "y": y} for i, (x, y) in enumerate(train_dataset)}
        tmp = balance_patients(tmp, method=method, random_state=seed)

        train_loader = DataLoader(
            TensorDataset(
                torch.stack([v["x"] for v in tmp.values()]),
                torch.stack([v["y"] for v in tmp.values()])
            ),
            batch_size=batch_size, shuffle=True, generator=loader_gen
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, generator=loader_gen
        )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --------------------------
    # Modello
    # --------------------------
    pos_weight = torch.Tensor([1.0]).to(device)

    model = model_class(
        input_dim=X_all.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        bidirectional=True,
        dropout=dropout,
        pooling=pooling
    ).to(device)

    criterion = loss_fn or nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_value)

    # --------------------------
    # Training con early stopping
    # --------------------------
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            lengths = (batch_x != -100.0).any(dim=2).sum(dim=1)

            optimizer.zero_grad()
            logits = model(batch_x, lengths)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        all_y_true, all_y_pred, all_probs = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                lengths = (batch_x != -100.0).any(dim=2).sum(dim=1)

                logits = model(batch_x, lengths)
                probs = torch.sigmoid(logits)

                val_loss += criterion(logits, batch_y.float()).item()

                all_y_true.append(batch_y.cpu())
                all_y_pred.append((probs > 0.5).int().cpu())
                all_probs.append(probs.cpu())

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{max_epochs} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

    # --------------------------
    # Ripristina il best model
    # --------------------------
    model.load_state_dict(best_state)
    model.eval()

    # --------------------------
    # METRICHE VALIDATION FINALI
    # --------------------------
    all_y_true = torch.cat(all_y_true).numpy()
    all_y_pred = torch.cat(all_y_pred).numpy()
    all_probs = torch.cat(all_probs).numpy()

    acc = accuracy_score(all_y_true, all_y_pred)
    auc = roc_auc_score(all_y_true, all_probs)
    prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    mcc = matthews_corrcoef(all_y_true, all_y_pred)
    brier = brier_score_loss(all_y_true, all_probs)
    cm = confusion_matrix(all_y_true, all_y_pred)

    class_report = classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)
    class0_metrics = class_report["0"]
    class1_metrics = class_report["1"]

    # --------------------------
    # RITORNO DEI RISULTATI
    # --------------------------
    return {
        "model": model,
        "metrics": {
            "acc": acc, "auc": auc, "prec": prec, "rec": rec, "f1": f1,
            "mcc": mcc, "brier": brier,
            "class0": class0_metrics,
            "class1": class1_metrics,
            "cm": cm.tolist()
        },
        "val_tensors": {
            "X_val": torch.stack([x for x, _ in val_dataset]),
            "y_val": torch.stack([y for _, y in val_dataset])
        }
    }


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

            if results is None:
                print("MCC non valida!")
                current_score=-float('inf')
            
            # 4. Estrazione dello Score
            # Lo score viene preso dai risultati medi della classe 1 (Sepsi) o globali
            elif scoring_metric in ['f1', 'precision', 'recall']:
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

                best_params['mcc']=best_score

                with open("./best_params.json", "w") as f:
                    json.dump(best_params, f, indent=4)

        except Exception as e:
            print(f"⚠️ Errore durante l'addestramento con parametri {params}: {e}")
            continue

    print("\n\n==================== GRID SEARCH COMPLETATA ====================")
    print(f"Migliori Parametri Trovati: {best_params}")
    print(f"Miglior {scoring_metric.upper()} Score: {best_score:.4f}")
    
    return best_params, best_results

"""GENDER"""
# --- Funzione per normalizzazione per fold ---
def normalize_df(train_df, test_df, feature_cols, save_path='scaler_stats.json'):
    scaler = StandardScaler() #z-score
    train_df_copy=train_df.copy()
    test_df_copy=test_df.copy()
    train_df_copy[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df_copy[feature_cols] = scaler.transform(test_df[feature_cols])

    # Salva mean e std in JSON
    stats = {col: {'mean': float(scaler.mean_[i]), 'std': float(np.sqrt(scaler.var_[i]))} 
             for i, col in enumerate(feature_cols)}
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    return train_df_copy, test_df_copy

# --- Funzione aggiornata di resampling per paziente ---
def resample_propagate_mask(df, measurement_cols, static_cols, freq='1h', mask_value=-100, desc='Resampling', fixed_length=50):
    """
    Resample, imputa valori mancanti e aggiunge feature statiche.
    Adatto per LSTM con masking.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe originale con colonne temporali e statiche.
    measurement_cols : list of str
        Colonne variabili (da resample/imputare).
    static_cols : list of str
        Colonne statiche (ad esempio età, sesso).
    freq : str
        Frequenza di resampling (default '1h').
    mask_value : float
        Valore da usare per indicare timestep da ignorare.
    
    Returns
    -------
    pd.DataFrame
        Dataframe resampled e pronto per LSTM.
    """
    
    all_patients = []
    
    for icustay_id, group in tqdm(df.groupby('icustay_id'), desc=desc):
        
        # Salva feature statiche
        static_dict = {col: group[col].iloc[0] for col in static_cols}

        group['chart_time'] = pd.to_datetime(group['chart_time'], format = "%Y-%m-%d %H:%M:%S")
        
        # Imposta indice temporale
        temp_series = group[measurement_cols].set_index(group['chart_time']).sort_index()
        
        # Se il primo timestamp > freq.min, aggiungi riga iniziale NaN
        start = temp_series.index.min()
        if start > temp_series.index.min():  # solitamente start = primo timestamp disponibile
            first_row = pd.DataFrame({col: [np.nan] for col in measurement_cols}, index=[start])
            temp_series = pd.concat([first_row, temp_series])
        
        # Resampling + Ffill + Bfill
        resampled_data = temp_series.resample(freq).first().ffill().bfill()
        
        # Imputazione dummy per eventuali NaN rimanenti
        for col in measurement_cols:
            if resampled_data[col].isnull().all():
                # tutta la colonna è NaN → metti valore di masking
                resampled_data[col] = mask_value
            else:
                # alcune righe NaN → metti valore di masking
                resampled_data[col] = resampled_data[col].fillna(mask_value)
        
        # Aggiungi feature statiche
        for col, val in static_dict.items():
            resampled_data[col] = val
        
        # Mantieni icustay_id
        resampled_data['icustay_id'] = icustay_id
        
        # Reset index
        resampled_data = resampled_data.reset_index().rename(columns={'index': 'chart_time'})

        diff_len = fixed_length-resampled_data.shape[0]

        if diff_len>0:
            dummy_index = [i for i in range(diff_len)]
            dummy_df = pd.DataFrame([resampled_data.drop(columns=['chart_time', 'icustay_id', 'label']).iloc[0].values]*diff_len, index=dummy_index, columns=measurement_cols)

            later = resampled_data['chart_time'].to_list()
            begin_day = later[0]
            former = pd.date_range(begin_day, periods=diff_len, freq="h"). shift(-diff_len).to_list()
            dummy_df['chart_time']=former
            dummy_df['label']=resampled_data['label'].iloc[0]
            dummy_df['icustay_id']=resampled_data['icustay_id'].iloc[0]
            dummy_df = dummy_df[['label', 'icustay_id', 'chart_time']+dummy_df.drop(columns=['icustay_id', 'label', 'chart_time']).columns.to_list()]

            resampled_data = pd.concat([dummy_df, resampled_data], ignore_index=True)
        
        else:
            resampled_data=resampled_data.iloc[-fixed_length:]
            print(resampled_data)
        
        all_patients.append(resampled_data)
    
    # Concatenazione finale
    return pd.concat(all_patients, ignore_index=True)


if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    target_len = 24
    pred_length = 6
    min_input_len = 2
    method=None

    my_mimic_df = pd.read_csv("datasets/MIMIC/my_mimic.csv")

    my_mimic_df = my_mimic_df.drop(columns=['adm_age_(0,50]', 'adm_age_(50,70]', 'adm_age_(70,inf)'])

    feature_names= [
                "sysbp",
                "diabp",
                "meanbp",
                "resprate",
                "heartrate",
                "spo2_pulsoxy",
                "tempc",
                "cardiacoutput",
                "tvset",
                "tvobserved",
                "tvspontaneous",
                "peakinsppressure",
                "totalpeeplevel",
                "o2flow",
                "fio2",
                "albumin",
                "bands",
                "bicarbonate",
                "bilirubin",
                "creatinine",
                "chloride",
                "glucose",
                "hematocrit",
                "hemoglobin",
                "lactate",
                "platelet",
                "potassium",
                "ptt",
                "inr",
                "pt",
                "sodium",
                "bun",
                "wbc",
                "creatinekinase",
                "ck_mb",
                "fibrinogen",
                "ldh",
                "magnesium",
                "calcium_free",
                "po2_bloodgas",
                "ph_bloodgas",
                "pco2_bloodgas",
                "so2_bloodgas",
                "troponin_t"
                ]

    print(f"Len. features: {len(feature_names)}")

    # train_and_evaluate_kfold_val_earlystop(data=my_mimic_df, min_length=min_input_len, prediction_horizon=pred_length, target_len = target_len, 
    #                                         model_class=LSTMClassifier, hidden_dim=8, num_layers=4, max_epochs=100, patience=10, lr=5e-3, dropout=0.3, 
    #                                         weight_decay_value=0, pooling='mean', batch_size=64, device=device, method=method)
    
    model_dict = train_final_model(data=my_mimic_df, min_length=min_input_len, prediction_horizon=pred_length, target_len = target_len, 
                                            model_class=LSTMClassifier, hidden_dim=8, num_layers=4, max_epochs=100, patience=10, lr=5e-3, dropout=0.3, 
                                            weight_decay_value=0, pooling='mean', batch_size=64, device=device, method=method)

    # #FEATURE SELECTION WITH PFI
    # importance_df=permutation_feature_importance(model=model_dict['model'], X=model_dict["val_tensors"]['X_val'], y=model_dict["val_tensors"]['y_val'], 
    #                                n_repeats=1000, metric_fn = f1_score, feature_names=feature_names)
    
    # print(importance_df)
    # importance_df.to_csv("female_male_importance_df.csv")

    # # Applica la selezione automatica con la soglia consigliata di 0.005
    # selected_feature_names = automatic_pfi_feature_selection(importance_df, positive_threshold=0.005)

    # print("Lista dei Feature da Mantenere:")
    # print(selected_feature_names)
    
    # error_analysis_df = model_dict['error_analysis_df']
    # error_analysis_df.to_csv("error_analysis.csv")

    # prob_stats = model_dict['error_group_stats']['probability_stats']
    # prob_stats.to_csv("prob_stats.csv")
    # feature_df = model_dict['error_group_stats']['feature_comparison']

    # # Estrai l'indice numerico da F0, F1, F2...
    # feature_df['Feature_Number'] = feature_df['Feature_Index'].str.replace('F', '').astype(int)

    # # Verifica se il numero di feature corrisponde
    # num_features_in_df = feature_df['Feature_Number'].max() + 1
    # if num_features_in_df != len(feature_names):
    #     print(f"⚠️ ATTENZIONE: Il numero di feature nel DF ({num_features_in_df}) non corrisponde alla lista dei nomi ({len(feature_names)}). La mappatura potrebbe essere errata.")
    #     # Continua comunque, ma con cautela

    # # Mappa l'indice numerico al nome della feature
    # feature_df['Feature_Name'] = feature_df['Feature_Number'].apply(lambda x: feature_names[x])

    # # Riorganizza e pulisci le colonne
    # feature_df = feature_df.drop(columns=['Feature_Index', 'Feature_Number'])
    # cols = ['Feature_Name', 'Comparison', 'Mean_Difference'] + [col for col in feature_df.columns if col not in ['Feature_Name', 'Comparison', 'Mean_Difference']]
    # feature_df = feature_df[cols]

    # fp_tn_comp = feature_df[feature_df['Comparison'] == 'FP vs TN'].copy()
    # top_fp_triggers = fp_tn_comp.sort_values(by='Mean_Difference', ascending=False).head(5)

    # print("\n🚨 Top 5 Triggers del Falso Allarme (FP vs TN): Segnali Anomali in Pazienti Sani")

    # fn_tp_comp = feature_df[feature_df['Comparison'] == 'FN vs TP'].copy()
    # top_fn_misses = fn_tp_comp.sort_values(by='Mean_Difference', ascending=True).head(5)

    # print("\n📉 Top 5 Segnali Smorzati/Mancati (FN vs TP): Feature che non hanno Innescato l'Allarme")
    # print(top_fn_misses[['Feature_Name', 'Mean_FN', 'Mean_TP', 'Mean_Difference', 'STD_Ratio']].to_markdown(index=False, floatfmt=".3f"))

    # feature_df.to_csv("feature_comp_per_group.csv")
    
    # print()

    # # recupera i tensori
    # patient_ids = list(balanced_data.keys())
    
    # model = model_dict["model"]
    # X_val = model_dict["val_tensors"]["X_val"]
    # y_val = model_dict["val_tensors"]["y_val"]

    # # Calcola la PFI
    # df_importance = permutation_feature_importance(
    #     model=model,
    #     X=X_val,
    #     y=y_val,
    #     metric_fn=accuracy_score,
    #     feature_names=feature_names,
    #     n_repeats=1000,
    #     verbose=False
    #     )

    # print(df_importance)

    # # 1. Definisci la Griglia di Parametri da Esplorare
    # param_grid_full = {
    # # 1. PARAMETRI DI ARCHITETTURA (Architettura della LSTM)
    # 'hidden_dim': [8, 16],  # Dimensione del vettore nascosto LSTM
    # 'num_layers': [4, 8, 16],      # Numero di strati LSTM (profondità)
    # 'pooling': ['mean', 'max', 'last'],  # Metodo di pooling (l'abbiamo fissato a 'mean' ma si può testare)
    
    # # 2. PARAMETRI DI REGOLARIZZAZIONE (Regularization)
    # 'dropout': [0.3, 0.4, 0.5], 

    # # 3. PARAMETRI DI ADDESTRAMENTO (Optimization)
    # 'lr': [1e-3, 5e-3, 1e-4],          # Learning Rate (i due valori che stavi testando)
    # 'weight_decay_value': [0, 1e-3, 1e-2], # Valore L2 (il tuo 1e-4 ottimale + alternative)
    # 'batch_size': [16, 32, 64],
    # 'patience' : [10, 15, 20],
    # 'max_epochs': [100],
    # 'method':['undersample','oversample']
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

