import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score

import shap
import lime
import lime.lime_tabular
import glob

# @author: Agostino
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

import pickle
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact
import statsmodels.api as sm

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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

    merged_df.to_csv("./SepsisExp_merged.tsv", sep="\t", index=False)

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
            print(f"Outliers in '{col}': {outliers}")
            print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
            print(f"{lower_bound} , {upper_bound}")
            

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
        print(corr_matrix.to_dict())
        corr_matrix.to_csv("correlation_matrix.tsv", sep="\t")

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

    Returns:
        pd.Series: Somma totale dei contributi SHAP (valori assoluti) per ciascuna feature su tutte le classi.
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
    feature_names = shap_values.feature_names

    if verbose:
        for i in range(num_classes):
            print(f"📊 SHAP - Classe {i}")
            shap.plots.bar(shap_values[..., i], max_display=len(feature_names))
            
        print()

        for i in range(num_classes):
            print(f"🐝 Beeswarm - Classe {i}")
            shap.summary_plot(shap_values[..., i], X_sample)

    else:
        for _ in tqdm(range(num_classes), desc="SHAP plotting", ncols=80):
            plt.ioff()
            shap.plots.bar(shap_values[..., _], show=False)
            plt.close()

            shap.summary_plot(shap_values[..., _], X_sample, show=False)
            plt.close()

    # Somma assoluta lungo i campioni e lungo le classi
    total_contributions = np.abs(shap_values.values).sum(axis=(0, 2))  # shape: (n_features,)

    # Creiamo una serie ordinata
    shap_summary_series = pd.Series(total_contributions, index=feature_names).sort_values(ascending=False)

    if verbose:
        print("\n📈 Somma totale dei contributi SHAP per ciascuna feature (tutte le classi sommate):")
        print(shap_summary_series)
    
    shap_summary_series.to_csv("./shapsum_df.tsv", sep="\t")

    return shap_summary_series

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

"""MIMIC"""

def load_and_reshape_data(
    folder_path,
    split='train',
    x_filename='x_dl.npy',
    y_filename='y.npy',
    col_names=None,
    transpose_shape=(0, 2, 1)
):
    """
    Carica i dati da file .npy, li trasforma in DataFrame e assegna i nomi delle colonne.

    Args:
        folder_path (str): Path principale della cartella.
        split (str): Sotto-cartella di split, es. 'train', 'test', 'val'.
        x_filename (str): Nome del file per i dati X.
        y_filename (str): Nome del file per le etichette Y.
        col_names (list, optional): Lista dei nomi delle colonne. Se None, usa un default.
        transpose_shape (tuple, optional): Shape per trasporre i dati X.

    Returns:
        tuple: (X_array, y_array, X_dataframe)
    """

    x_path = os.path.join(folder_path, split, x_filename)
    y_path = os.path.join(folder_path, split, y_filename)

    x_data = np.load(x_path)
    if transpose_shape:
        x_data = np.transpose(x_data, transpose_shape)

    y_data = np.load(y_path)

    print(f"{split.capitalize()} data: {x_data.shape}")
    print(f"{split.capitalize()} labels: {y_data.shape}")

    if col_names is None:
        col_names = ['sysbp', 'diabp', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                     'tempc', 'cardiacoutput', 'tvset', 'tvobserved', 'tvspontaneous',
                     'peakinsppressure', 'totalpeeplevel', 'o2flow', 'fio2', 'albumin',
                     'bands', 'bicarbonate', 'bilirubin', 'creatinine', 'chloride',
                     'glucose', 'hematocrit', 'hemoglobin', 'lactate', 'platelet',
                     'potassium', 'ptt', 'inr', 'pt', 'sodium', 'bun', 'wbc',
                     'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh', 'magnesium',
                     'calcium_free', 'po2_bloodgas', 'ph_bloodgas', 'pco2_bloodgas',
                     'so2_bloodgas', 'troponin_t']

    # Flatten temporally per sample
    df = pd.DataFrame([list(timestep) for timestep in x_data]).stack().apply(pd.Series).reset_index(1, drop=True)
    df.index.name = 'Index'
    df.columns = col_names

    df.to_csv(os.path.join(folder_path, split, 'X_'+split+'.tsv'), sep='\t')
    pd.DataFrame(y_data).to_csv(os.path.join(folder_path, split, 'y_'+split+'.tsv'), sep='\t')

    return df, y_data

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

def analyze_timeseries_df(df_data, df_labels=None, verbose=True):
    """
    Analizza un DataFrame time series con dati clinici per paziente (multivariate, many-to-one).
    
    Args:
        df_data (pd.DataFrame): Dati sequenziali con colonna 'Index' per i pazienti e 'timestep'.
        df_labels (pd.DataFrame or None): Etichette many-to-one (una riga per paziente).
        verbose (bool): Se True, stampa dettagli e mostra grafici.
        
    Returns:
        data (pd.DataFrame): Dati pre-elaborati.
        outliers_list (list): Numero di outlier per colonna numerica.
    """
    if verbose:
        sns.set_style(style="whitegrid")
    
    if verbose:
        print(f"👥 Numero di pazienti unici: {df_data['Index'].nunique()}")
        print(f"⏱ Numero medio di timestep per paziente: {df_data.groupby('Index').size().mean():.2f}")

    # Step 1: Preprocessing colonne
    data = df_data.copy()

    if verbose:
        print("\n### Informazioni di base ###")
        print(data.info())
        print("\n### Missing Values ###")
        print(data.isna().sum()[data.isna().sum() > 0])

    # Step 2: Statistiche descrittive globali
    if verbose:
        print("\n### Statistiche descrittive globali ###")
        print(data.describe())

    # Step 3: Outlier detection (su tutte le righe, ignorando la struttura temporale)
    if verbose:
        print("\n### Rilevamento outlier (IQR) ###")

    outliers_list = []
    for col in data.select_dtypes(include=[np.number]).columns:
        if col in ['Index', 'timestep']:
            continue
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outliers_list.append((col, outliers))
        if verbose:
            print(f"Outliers in '{col}': {outliers}")
            print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
            print(f"{lower_bound} , {upper_bound}")

    # Step 4.1: Istogrammi multipli in più figure
    if verbose:
        print("\n### Istogrammi delle variabili numeriche (divisi in più figure) ###")
        num_cols = data.select_dtypes(include=[np.number]).columns.drop(['Index', 'timestep'], errors='ignore')
        
        # Parametri di configurazione
        features_per_fig = 10  # Quante variabili per figura
        n_cols = 4  # Colonne per figura

        chunks = [num_cols[i:i + features_per_fig] for i in range(0, len(num_cols), features_per_fig)]

        for fig_num, chunk in enumerate(chunks):
            n_rows = int(np.ceil(len(chunk) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
            axes = axes.flatten()

            for i, col in enumerate(chunk):
                sns.histplot(data[col].dropna(), ax=axes[i], kde=True, bins=15, color='steelblue')
                axes[i].set_title(col)
                axes[i].set_ylabel("Frequenza")
                axes[i].grid(True)
                # mean = data[col].mean()
                # std = data[col].std()
                # axes[i].set_xlim(mean - 4*std, mean + 4*std)
                axes[i].set_xlim(-5, +5)
            # Rimuove i subplot vuoti
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Istogrammi delle variabili (figura {fig_num + 1})", fontsize=16, y=1.02)
            plt.show()

    # Step 5: Correlazioni globali
    if verbose:
        print("\n### Matrice di correlazione ###")
        corr_matrix = data.drop(columns=['Index', 'timestep'], errors='ignore').corr(method='spearman', numeric_only=True)
        print(corr_matrix.to_dict())
        corr_matrix.to_csv("correlation_matrix_mimic.tsv", sep="\t")
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.3)
        plt.title("Correlation Heatmap (Upper Triangle)")
        plt.tight_layout()
        plt.show()

    # Step 6: Distribuzione delle feature nel tempo (media su pazienti)
    if verbose:
        print("\n### Andamento medio delle variabili nel tempo ###")
        # septic_patients = df_labels[df_labels['0']==1].index
        # ts_features = data.loc[data.index[septic_patients]].drop(columns=['Index']).groupby('timestep').mean()
        # ts_features.plot(figsize=(15, 15), title="Andamento medio delle feature nel tempo")
        # plt.xlabel("Timestep")
        # plt.ylabel("Valore medio")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        plot_interactive_ts(data, df_labels, septic=True)   # Septic patients
        plot_interactive_ts(data, df_labels, septic=False)  # Non-septic patients

def plot_interactive_ts(data, df_labels, septic=True):

    label_val = 1 if septic else 0
    group_name = "Septic" if septic else "Non-Septic"

    selected_patients = df_labels[df_labels['sepsis'] == label_val].index
    df_filtered = data.loc[data.index.get_level_values(0).isin(selected_patients)]

    if 'Index' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['Index'])
    
    if 'sepsis' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['sepsis'])

    ts_features = df_filtered.groupby('timestep').mean()

    if ts_features.empty:
        print(f"[!] Nessun dato disponibile per il gruppo '{group_name}'")
        return

    fig = go.Figure()
    for col in ts_features.columns:
        fig.add_trace(go.Scatter(
            x=ts_features.index,
            y=ts_features[col],
            mode='lines',
            name=col,
            hovertemplate=f"{col}: %{{y:.2f}}<br>Timestep: %{{x}}"
        ))

    fig.update_layout(
        title=f"Andamento medio delle feature nel tempo - {group_name} patients",
        xaxis_title="Timestep",
        yaxis_title="Valore medio",
        height=700,
        width=1200,
        legend=dict(orientation="h", y=-0.3),
    )

    fig.show(renderer="browser")

def compute_corr_by_timestep(df_timestep_labels, plot=True):
    """
    Calcola la correlazione tra le feature e il label 'sepsis' per ogni timestep.

    Args:
        df_timestep_labels (pd.DataFrame): DataFrame con colonne 'Index', 'timestep', features e 'sepsis'.
        plot (bool): Se True, chiama plot_interactive_corr sul risultato.

    Returns:
        pd.DataFrame: correlazioni feature vs sepsis, con timestep come indice.
    """
    results = {}
    for t, sub in df_timestep_labels.groupby("timestep"):
        # calcola correlazione ignorando Index e timestep
        corr = sub.drop(columns=["Index", "timestep"]).corr()["sepsis"]
        results[t] = corr

    corr_by_time = pd.DataFrame(results).T.sort_index()

    if plot:
        plot_interactive_corr(corr_by_time)

    return corr_by_time

def plot_interactive_corr(corr_by_time, exclude_label=True):
    """
    Plotta interattivamente le correlazioni feature–label per timestep.
    
    Parameters
    ----------
    corr_by_time : pd.DataFrame
        DataFrame con index=timestep, colonne=feature + 'sepsis', valori=correlazione
    exclude_label : bool
        Se True, esclude la colonna 'sepsis' dal plot
    """
    df_plot = corr_by_time.copy()
    if exclude_label and 'sepsis' in df_plot.columns:
        df_plot = df_plot.drop(columns=['sepsis'])
    
    fig = go.Figure()
    for col in df_plot.columns:
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot[col],
            mode='lines',
            name=col,
            hovertemplate=f"{col}: %{{y:.2f}}<br>Timestep: %{{x}}"
        ))
    
    fig.update_layout(
        title="Correlazione feature–label per timestep",
        xaxis_title="Timestep (ore)",
        yaxis_title="Correlazione",
        height=700,
        width=1200,
        legend=dict(orientation="h", y=-0.3),
        yaxis=dict(range=[-1, 1])  # correlazioni vanno da -1 a 1
    )
    
    fig.show(renderer="browser")

def statistical_analysis(df, label_col="sepsis", logistic=True, verbose=True):

    results_ttest = []
    results_chi2 = []
    rr_results = []
    logistic_results = []

    features = [c for c in df.columns if c not in [label_col, 'Index', 'timestep']]

    # --- T-TEST PER VARIABILI CONTINUE ---
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            septic = df[df[label_col] == 1][col].dropna()
            non_septic = df[df[label_col] == 0][col].dropna()

            if len(septic) > 1 and len(non_septic) > 1:
                t_stat, p_val = ttest_ind(septic, non_septic, equal_var=False)
                mean_septic = septic.mean()
                mean_non_septic = non_septic.mean()

                direction = "↑ nei septic" if mean_septic > mean_non_septic else "↓ nei septic"

                results_ttest.append({
                    "Feature": col,
                    "Mean (Septic)": mean_septic,
                    "Mean (Non-Septic)": mean_non_septic,
                    "Direction": direction,
                    "p-value": p_val
                })

    # --- CHI-QUADRO + RELATIVE RISK PER VARIABILI CATEGORICHE ---
    for col in features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            contingency = pd.crosstab(df[col], df[label_col])
            if contingency.shape == (2, 2):  # solo binarie
                chi2, p_val, _, _ = chi2_contingency(contingency)

                # Relative Risk
                a = contingency.loc[1,1] if 1 in contingency.index and 1 in contingency.columns else 0
                b = contingency.loc[1,0] if 1 in contingency.index and 0 in contingency.columns else 0
                c = contingency.loc[0,1] if 0 in contingency.index and 1 in contingency.columns else 0
                d = contingency.loc[0,0] if 0 in contingency.index and 0 in contingency.columns else 0

                rr = (a / (a+b)) / (c / (c+d)) if (a+b) > 0 and (c+d) > 0 else np.nan

                results_chi2.append({"Feature": col, "Chi2 p-value": p_val})
                rr_results.append({"Feature": col, "Relative Risk": rr})

    # --- REGRESSIONE LOGISTICA MULTIVARIATA ---
    try:
        X = df[features].select_dtypes(include=[np.number]).dropna()
        y = df.loc[X.index, label_col]
        X = sm.add_constant(X)

        model = sm.Logit(y, X).fit(disp=False)
        summary = model.summary2().tables[1].reset_index()
        summary.rename(columns={"index": "Feature"}, inplace=True)
        logistic_results = summary
    except Exception as e:
        print("[!] Logistic regression failed:", e)

    # --- Ritorno come tabelle ---
    ttest_df = pd.DataFrame(results_ttest).sort_values("p-value")
    chi2_df = pd.DataFrame(results_chi2)
    rr_df = pd.DataFrame(rr_results)
    logistic_df = pd.DataFrame(logistic_results)

    print("\n=== RISULTATI NUMERICI ===")
    print(ttest_df)

    print("\n=== RISULTATI CATEGORICI ===")
    print(chi2_df, rr_df)

    print("\n=== RISULTATI LOGISTICA ===")
    print(logistic_df)

    return ttest_df, chi2_df, rr_df, logistic_df

"""Clustering"""

def make_3d_timeseries(df, id_col="Index", time_col="timestep", drop_cols=["sepsis"], resample_len=None):
    """
    Converte un DataFrame in array 3D per clustering di time series.
    
    Parametri:
        df : DataFrame
        id_col : nome colonna ID paziente
        time_col : nome colonna timestep
        drop_cols : colonne da rimuovere (es. label)
        resample_len : se specificato, tutte le serie vengono interpolate a questa lunghezza
    
    Ritorna:
        X : np.ndarray con shape (n_pazienti, timesteps, n_features)
        ids : lista con gli ID pazienti
    """
    # Seleziona solo le feature numeriche
    features = df.drop(columns=[id_col, time_col] + drop_cols, errors="ignore").columns
    
    series_list = []
    ids = []
    
    for pid, group in df.groupby(id_col):
        group = group.sort_values(time_col)
        s = group[features].values  # (timesteps, n_features)
        series_list.append(s)
        ids.append(pid)
    
    lengths = [len(s) for s in series_list]
    
    # Se tutte le serie hanno stessa lunghezza
    if len(set(lengths)) == 1 and resample_len is None:
        X = np.stack(series_list, axis=0)
    else:
        target_len = resample_len if resample_len is not None else max(lengths)
        X = TimeSeriesResampler(sz=target_len).fit_transform(series_list)
    
    return X, ids

def find_optimal_clusters(X, k_range=(2, 10), metric="dtw", max_samples=2000, sample_size=1000):
    """
    Trova numero ottimale di cluster usando silhouette score.
    """
    # Subsampling se troppi pazienti
    if X.shape[0] > max_samples:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    
    scores = []
    k_values = range(k_range[0], k_range[1])

    for k in k_values:
        model = TimeSeriesKMeans(n_clusters=k, metric=metric, random_state=0)
        labels = model.fit_predict(X_sub)
        
        score = silhouette_score(
            X_sub.reshape(X_sub.shape[0], -1),  # silhouette vuole 2D
            labels,
            metric="euclidean",
            sample_size=min(sample_size, X_sub.shape[0]),
            random_state=0
        )
        scores.append(score)
        print(f"k={k}, silhouette={score:.3f}")

    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette method")
    plt.savefig(f"./data-analysis/clustering/mimic_elbow_method_silhouette.png", dpi=300)
    plt.show()

    best_k = k_values[np.argmax(scores)]
    print(f"Numero ottimale di cluster: {best_k}")
    return best_k

def find_optimal_clusters_inertia(X, k_range=(2, 10), metric="dtw", random_state=42):
    """
    Usa l'elbow method (inertia) per stimare il numero ottimale di cluster.

    Parametri
    ---------
    X : array-like, shape (n_samples, n_timestamps, n_features)
        Dataset di serie temporali.
    k_range : tuple(int, int)
        Range dei cluster da provare (es. (2, 10)).
    metric : str
        Distanza da usare (es. "euclidean" o "dtw").
    random_state : int
        Per riproducibilità.

    Ritorna
    -------
    best_k : int
        Numero ottimale di cluster secondo il metodo del gomito.
    inertias : list
        Lista dei valori di inertia per ogni k.
    """
    inertias = []
    ks = range(k_range[0], k_range[1] + 1)

    for k in tqdm(ks, desc="Finding optimal n. of clusters (inertia)"):
        model = TimeSeriesKMeans(n_clusters=k, metric=metric, random_state=random_state)
        model.fit(X)
        inertias.append(model.inertia_)
        print(f"k={k}, inertia={model.inertia_:.2f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, "o-", color="blue")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method with Inertia")
    plt.grid(True)
    plt.savefig(f"./data-analysis/clustering/mimic_elbow_method_inertia.png", dpi=300)
    plt.show()

    # Calcolo gomito con differenze finite (semplice euristica)
    deltas = np.diff(inertias)
    second_deltas = np.diff(deltas)
    elbow_k = ks[np.argmin(second_deltas) + 1]

    print(f"Best k (elbow method) ≈ {elbow_k}")

    return elbow_k, inertias

def cluster_time_series(X, n_clusters, metric="dtw"):
    """
    Esegue clustering TS-KMeans.
    """
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric, random_state=0)
    labels = model.fit_predict(X)
    return model, labels

def plot_clusters_pca(X, labels, clustering_method, best_k, model=None):
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="tab10", alpha=0.6,
        label="Pazienti"
    )

    if model is not None:
        centroids = model.cluster_centers_.reshape(model.n_clusters, -1)
        centroids_pca = pca.transform(centroids)
        plt.scatter(
            centroids_pca[:, 0],
            centroids_pca[:, 1],
            c=range(model.n_clusters),
            cmap="tab10",
            marker="X",
            s=200,
            edgecolor="k",
            linewidth=1.5,
            label="Centroidi"
        )

    plt.title("Clustering time series (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(f"./data-analysis/clustering/{clustering_method}_k:{best_k}_sepsis_rate_per_cluster.png", dpi=300)
    plt.show()

def subsample_data(X, max_samples=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    if X.shape[0] > max_samples:
        if X.shape[0] > max_samples:
            idx = rng.choice(X.shape[0], max_samples, replace=False)
            return X.iloc[idx]
    return X

def compare_clusters(cluster_list, summary, std_summary, method_name, k):
    """
    Confronta i cluster selezionati e produce heatmap e barplot delle differenze.
    
    cluster_list : list
        Lista dei numeri dei cluster da confrontare
    summary : pd.DataFrame
        Valori medi delle features per cluster
    std_summary : pd.DataFrame
        Deviazioni standard delle features per cluster
    method_name : str
        Nome del metodo di clustering, per salvare i plot
    k : int
        Numero di cluster
    """
    
    # Filtra i cluster
    summary_sel = summary.loc[cluster_list]
    std_sel = std_summary.loc[cluster_list]
    
    print(f"\nValori medi delle features per i cluster {cluster_list}:")
    print(summary_sel)
    
    print(f"\nDeviazione standard delle features per i cluster {cluster_list}:")
    print(std_sel)
    
    # Differenze assolute tra tutti i cluster selezionati
    # Creiamo una matrice delle differenze a coppie
    n = len(cluster_list)
    diff_matrix = pd.DataFrame(index=summary_sel.columns, columns=[f"{i}-{j}" for i in range(n) for j in range(i+1, n)])
    
    for col_idx, (i, j) in enumerate([(i, j) for i in range(n) for j in range(i+1, n)]):
        diff_matrix.iloc[:, col_idx] = (summary_sel.iloc[i] - summary_sel.iloc[j]).abs()
    
    print("\nDifferenze assolute tra i cluster selezionati (colonne = coppie di cluster):")
    print(diff_matrix)
    
    # Heatmap comparativa
    plt.figure(figsize=(12, 6))
    sns.heatmap(summary_sel.T, cmap="coolwarm", annot=True)
    plt.title(f"Confronto medio delle features: cluster {cluster_list}")
    plt.ylabel("Feature")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/{method_name}_k:{k}_clusters_{'_'.join(map(str, cluster_list))}_heatmap.png", dpi=300)
    plt.show()
    
    # Barplot delle differenze più significative (media delle differenze a coppie)
    diff_mean = diff_matrix.mean(axis=1).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=diff_mean.index, y=diff_mean.values, palette="viridis", hue=diff_mean.index, legend=False)
    plt.xticks(rotation=90)
    plt.title(f"Differenze medie tra i cluster selezionati {cluster_list}")
    plt.ylabel("Differenza media")
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/{method_name}_k:{k}_clusters_{'_'.join(map(str, cluster_list))}_diff.png", dpi=300)
    plt.show()


if __name__=="__main__":

    # folder_path = 'datasets/SepsisExp'

    # data, merged_df, outliers_list = analyze_csvs(folder_path=folder_path, verbose=True)

    # sepsis_patient=0
    # non_sepsis_patient=0

    # for _, group in merged_df.groupby(by='id'):
    #     if group['sepsis'].iloc[0]==1:
    #         sepsis_patient+=1
    #     else:
    #         non_sepsis_patient+=1
    
    # print(f"N. of sepsis patients: {sepsis_patient}")
    # print(f"N. of non sepsis patients: {non_sepsis_patient}")

    """Data Analysis"""
    # X_train, X_test, y_train, y_test = prepare_hourly_data(merged_df)
    # rf_model = train_random_forest_from_hourly_data(X_train, X_test, y_train, y_test,verbose=False)
    # explain_model_with_shap(rf_model, X_test, verbose=True)
    # run_eli5_analysis(rf_model, X_test, y_test, verbose=False)
    # run_pdp_analysis(rf_model, X_test, features_to_plot=['procalcitonin', 'c-reactive_protein'], verbose=False)
    # run_lime_global_analysis(rf_model, X_train, y_train, X_test)

    """Clustering"""
    # X, y = get_patients_with_max_severity(merged_df)
    # y_binarized = binarize_series(y)

    # elbow_method_analysis(X, k_range=range(2,10), verbose=True)
    # evaluate_kmeans_clustering(X, y, n_clusters=5, verbose=True)
    # evaluate_kmeans_3D_clustering(X, y_binarized, n_clusters=2, verbose=True)
    # plot_dbscan_kdistance_optimal_eps_manual(X, verbose=True)
    # dbscan_clustering(X, y, eps=4, min_samples=5, verbose=True)

    """Parentesi Lasso"""
    # lasso_regression_with_cv(X, y)

    # pairplots_with_labels(X, y)
    # evaluate_agglomerative_clustering(X, verbose=True)
    # evaluate_davies_bouldin(X, verbose=True)
    # agglomerative_clustering(X, y, n_clusters=3, verbose=True)
    # agglomerative_3D_clustering(X, y, n_clusters=4, verbose=True)

    # optimal_gmm_k(X)
    # gmm_clustering(X, y, n_components=3)

    """MIMIC"""
    folder_path = './datasets/MIMIC/carry_forward/mean'
    mimic_df, train_df, val_df, test_df = load_and_prepare_data(folder_path)

    print(mimic_df.shape)

    sepsis_patient=0
    non_sepsis_patient=0

    for _, group in mimic_df.groupby(by='Index'):
        if group['sepsis'].iloc[0]==1:
            sepsis_patient+=1
        else:
            non_sepsis_patient+=1
    
    print(f"N. of sepsis patients: {sepsis_patient}")
    print(f"N. of non sepsis patients: {non_sepsis_patient}")

    """Analisi dataset"""
    analyze_timeseries_df(df_data=mimic_df, df_labels=pd.DataFrame(mimic_df['sepsis']))

    """Correlazione nel tempo"""
    # corr_by_time = compute_corr_by_timestep(mimic_df)
    # ttest_df, chi2_df, rr_df, logistic_df = statistical_analysis(mimic_df, label_col="sepsis", logistic=True, verbose=True)

    """Clustering MIMIC"""
    # X, patient_ids = make_3d_timeseries(mimic_df)
    # metric = "euclidean"
    # clustering_method = "ts-Kmeans"
    # best_k, _ = find_optimal_clusters_inertia(X,metric=metric)
    # best_k = 5

    # model, labels = cluster_time_series(X, best_k, metric=metric)
    # cluster_df = pd.DataFrame({"Index": patient_ids, "Cluster": labels})
    # merged = mimic_df.merge(cluster_df, on="Index")

    # # ============================
    # # Patient-level aggregation
    # # ============================
    # feature_cols = [c for c in mimic_df.columns if c not in ["Index", "timestep", "sepsis"]]

    # patient_features = merged.groupby("Index")[feature_cols].mean().reset_index()
    # patient_cluster_sepsis = merged.groupby("Index").agg(
    #     Cluster=("Cluster", "first"),
    #     sepsis=("sepsis", "max")
    # ).reset_index()

    # patient_level = patient_cluster_sepsis.merge(patient_features, on="Index")

    # # ============================
    # # Conteggio sepsi per cluster
    # # ============================
    # sepsis_counts = patient_level.groupby("Cluster")["sepsis"].sum()
    # cluster_order = sepsis_counts.sort_values(ascending=False).index.tolist()

    # # Mapping numerico -> nome leggibile
    # cluster_name_map = {old: f"Cluster_{i+1}" for i, old in enumerate(cluster_order)}
    # patient_level["Cluster_name"] = patient_level["Cluster"].map(cluster_name_map)
    # merged["Cluster_name"] = merged["Cluster"].map(cluster_name_map)

    # # ============================
    # # Plot PCA semplice
    # # ============================
    # X_flat = X.reshape(X.shape[0], -1)
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_flat)

    # plt.figure(figsize=(10, 7))
    # cluster_colors = sns.color_palette("tab10", n_colors=best_k)

    # for cluster_id in np.unique(labels):
    #     mask = labels == cluster_id
    #     plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
    #                 alpha=0.6,
    #                 color=cluster_colors[cluster_id],
    #                 label=f"Cluster {cluster_id}")

    # plt.title("Clustering time series (PCA 2D)")
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.legend(fontsize=10)
    # plt.tight_layout()
    # plt.savefig(f"./data-analysis/clustering/{clustering_method}_k:{best_k}_clustering_simple.png", dpi=300)
    # plt.show()

    # # ============================
    # # Statistiche cluster
    # # ============================
    # cluster_counts = patient_level.groupby("Cluster")["sepsis"].count()
    # sepsis_counts = patient_level.groupby("Cluster")["sepsis"].sum()
    # sepsis_rate = sepsis_counts / cluster_counts

    # print("\nNumero pazienti per cluster:\n", cluster_counts)
    # print("\nNumero pazienti con sepsi per cluster:\n", sepsis_counts)
    # print("\nProporzione sepsi per cluster:\n", sepsis_rate)

    # summary = patient_level.groupby("Cluster")[feature_cols].mean()
    # std_summary = patient_level.groupby("Cluster")[feature_cols].std()

    # print("\nValori medi per cluster (livello paziente):")
    # print(summary.to_dict())
    # print("\nDeviazione standard per cluster (livello paziente):")
    # print(std_summary.to_dict())

    # plt.figure(figsize=(12, 6))
    # sns.heatmap(summary.T, cmap="coolwarm", annot=False)
    # plt.title("Mean values per cluster (livello paziente)")
    # plt.ylabel("Feature")
    # plt.xlabel("Cluster")
    # plt.tight_layout()
    # plt.savefig(f"./data-analysis/clustering/{clustering_method}_k:{best_k}_mean_values_per_cluster_patient.png", dpi=300)
    # plt.show()

    # # ============================
    # # Cluster con più sepsi (assoluto)
    # # ============================
    # max_val = sepsis_counts.max()
    # max_sepsis_clusters = sepsis_counts[sepsis_counts == max_val].index.tolist()

    # if len(max_sepsis_clusters) == 1:
    #     max_sepsis_cluster = max_sepsis_clusters[0]
    # else:
    #     print(f"\nPareggio tra i cluster con {max_val} pazienti settici: {max_sepsis_clusters}")
    #     max_sepsis_cluster = max_sepsis_clusters[0]
    #     print(f"Proseguirò usando: {max_sepsis_cluster}")

    # rest_mean = patient_level[patient_level["Cluster"] != max_sepsis_cluster][feature_cols].mean()
    # diff = summary.loc[max_sepsis_cluster] - rest_mean
    # diff_sorted = diff.abs().sort_values(ascending=False)

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x=diff_sorted.index, y=diff_sorted.values)
    # plt.xticks(rotation=90)
    # plt.title(f"Differenze medie tra Cluster {max_sepsis_cluster} e gli altri (livello paziente)")
    # plt.ylabel("Differenza media")
    # plt.tight_layout()
    # plt.savefig(f"./data-analysis/clustering/{clustering_method}_k:{best_k}_Cluster{max_sepsis_cluster}_diff_patient.png", dpi=300)
    # plt.show()

    # # ============================
    # # Riepilogo cluster
    # # ============================
    # cluster_summary = pd.DataFrame({
    #     "Totale pazienti": cluster_counts,
    #     "Pazienti con sepsi": sepsis_counts,
    #     "Rate sepsi": sepsis_rate
    # }).sort_index()

    # # Plot riepilogo cluster
    # plt.figure(figsize=(10,6))
    # sns.barplot(x=cluster_summary.index, y=cluster_summary["Totale pazienti"], color="skyblue", label="Totale pazienti")
    # sns.barplot(x=cluster_summary.index, y=cluster_summary["Pazienti con sepsi"], color="red", label="Pazienti con sepsi")
    # plt.title("Numero pazienti totali e settici per cluster")
    # plt.xlabel("Cluster")
    # plt.ylabel("Numero pazienti")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"./data-analysis/clustering/{clustering_method}_k:{best_k}_septic_patients_wrt_total.png", dpi=300)
    # plt.show()

    # # Barplot rate sepsi
    # plt.figure(figsize=(8,5))
    # colors = sns.color_palette("Reds", n_colors=len(cluster_summary))
    # sns.barplot(x=cluster_summary.index, y=cluster_summary["Rate sepsi"], palette=colors, hue=cluster_summary.index, legend=False)
    # plt.title("Rate di sepsi per cluster")
    # plt.xlabel("Cluster")
    # plt.ylabel("Proporzione pazienti settici")
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.show()

    """
    clusters_to_compare = [3, 4]  # puoi cambiare i cluster che vuoi confrontare
    compare_clusters(clusters_to_compare, summary, std_summary, clustering_method, best_k)"""

    """Clustering SepsisExp"""
    # print(merged_df)
    """X, patient_ids = make_3d_timeseries(merged_df, id_col='id', drop_cols=["sepsis", "severity"])
    metric = "euclidean"
    clustering_method = "ts-Kmeans"
    # best_k, _ = find_optimal_clusters_inertia(X,metric=metric)
    best_k = 5

    model, labels = cluster_time_series(X, best_k, metric=metric)
    cluster_df = pd.DataFrame({"id": patient_ids, "Cluster": labels})
    merged = merged_df.merge(cluster_df, on="id")

    # ============================
    # Patient-level aggregation
    # ============================
    feature_cols = [c for c in merged_df.columns if c not in ["id", "timestep", "sepsis", "severity"]]

    patient_features = merged.groupby("id")[feature_cols].mean().reset_index()
    patient_cluster_sepsis = merged.groupby("id").agg(
        Cluster=("Cluster", "first"),
        sepsis=("sepsis", "max")
    ).reset_index()

    patient_level = patient_cluster_sepsis.merge(patient_features, on="id")

    # ============================
    # Conteggio sepsi per cluster
    # ============================
    sepsis_counts = patient_level.groupby("Cluster")["sepsis"].sum()
    cluster_order = sepsis_counts.sort_values(ascending=False).index.tolist()

    # Mapping numerico -> nome leggibile
    cluster_name_map = {old: f"Cluster_{i+1}" for i, old in enumerate(cluster_order)}
    patient_level["Cluster_name"] = patient_level["Cluster"].map(cluster_name_map)
    merged["Cluster_name"] = merged["Cluster"].map(cluster_name_map)

    # ============================
    # Plot PCA semplice
    # ============================
    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.figure(figsize=(10, 7))
    cluster_colors = sns.color_palette("tab10", n_colors=best_k)

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    alpha=0.6,
                    color=cluster_colors[cluster_id],
                    label=f"Cluster {cluster_id}")

    plt.title("Clustering time series (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/sepsisexp_{clustering_method}_k:{best_k}_clustering_simple.png", dpi=300)
    plt.show()

    # ============================
    # Statistiche cluster
    # ============================
    cluster_counts = patient_level.groupby("Cluster")["sepsis"].count()
    sepsis_counts = patient_level.groupby("Cluster")["sepsis"].sum()
    sepsis_rate = sepsis_counts / cluster_counts

    print("\nNumero pazienti per cluster:\n", cluster_counts)
    print("\nNumero pazienti con sepsi per cluster:\n", sepsis_counts)
    print("\nProporzione sepsi per cluster:\n", sepsis_rate)

    summary = patient_level.groupby("Cluster")[feature_cols].mean()
    std_summary = patient_level.groupby("Cluster")[feature_cols].std()

    print("\nValori medi per cluster (livello paziente):")
    print(summary)
    print("\nDeviazione standard per cluster (livello paziente):")
    print(std_summary)

    plt.figure(figsize=(12, 6))
    sns.heatmap(summary.T, cmap="coolwarm", annot=False)
    plt.title("Mean values per cluster (livello paziente)")
    plt.ylabel("Feature")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/sepsisexp_{clustering_method}_k:{best_k}_mean_values_per_cluster_patient.png", dpi=300)
    plt.show()

    # ============================
    # Cluster con più sepsi (assoluto)
    # ============================
    max_val = sepsis_counts.max()
    max_sepsis_clusters = sepsis_counts[sepsis_counts == max_val].index.tolist()

    if len(max_sepsis_clusters) == 1:
        max_sepsis_cluster = max_sepsis_clusters[0]
    else:
        print(f"\nPareggio tra i cluster con {max_val} pazienti settici: {max_sepsis_clusters}")
        max_sepsis_cluster = max_sepsis_clusters[0]
        print(f"Proseguirò usando: {max_sepsis_cluster}")

    rest_mean = patient_level[patient_level["Cluster"] != max_sepsis_cluster][feature_cols].mean()
    diff = summary.loc[max_sepsis_cluster] - rest_mean
    diff_sorted = diff.abs().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=diff_sorted.index, y=diff_sorted.values)
    plt.xticks(rotation=90)
    plt.title(f"Differenze medie tra Cluster {max_sepsis_cluster} e gli altri (livello paziente)")
    plt.ylabel("Differenza media")
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/sepsisexp_{clustering_method}_k:{best_k}_Cluster{max_sepsis_cluster}_diff_patient.png", dpi=300)
    plt.show()

    # ============================
    # Riepilogo cluster
    # ============================
    cluster_summary = pd.DataFrame({
        "Totale pazienti": cluster_counts,
        "Pazienti con sepsi": sepsis_counts,
        "Rate sepsi": sepsis_rate
    }).sort_index()

    # Plot riepilogo cluster
    plt.figure(figsize=(10,6))
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Totale pazienti"], color="skyblue", label="Totale pazienti")
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Pazienti con sepsi"], color="red", label="Pazienti con sepsi")
    plt.title("Numero pazienti totali e settici per cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Numero pazienti")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/sepsisexp_{clustering_method}_k:{best_k}_septic_patients_wrt_total.png", dpi=300)
    plt.show()

    # Barplot rate sepsi
    plt.figure(figsize=(8,5))
    colors = sns.color_palette("Reds", n_colors=len(cluster_summary))
    sns.barplot(x=cluster_summary.index, y=cluster_summary["Rate sepsi"], palette=colors, hue=cluster_summary.index, legend=False)
    plt.title("Rate di sepsi per cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Proporzione pazienti settici")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"./data-analysis/clustering/sepsisexp_{clustering_method}_k:{best_k}_sepsis_rate_per_cluster.png", dpi=300)
    plt.show()"""

