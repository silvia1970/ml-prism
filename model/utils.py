import glob
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tqdm import tqdm 

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