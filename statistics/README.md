# statistics — data study & analysis (pre-model)

Modulo di **analisi esplorativa** dei dati, da eseguire **prima** del training.
Produce i dataset puliti e le analisi (clustering, feature importance, time-series)
su cui si basa la scelta delle feature per il modello.

> EN: Exploratory data analysis module, run **before** training. Produces the cleaned
> datasets and the analyses (clustering, feature importance, time-series) that drive
> feature selection for the model.

## Contents / Contenuto

| File | Purpose / Scopo |
|------|-----------------|
| `load_mimic.py` | Collects and preprocesses raw MIMIC-III records into the `datasets/MIMIC/...` arrays. / Raccoglie e preelabora i record MIMIC-III grezzi. |
| `data_analysis.py` | Library of analyses: outlier detection, RF + SHAP/eli5/PDP/LIME, clustering (KMeans, GMM, DBSCAN, agglomerative), time-series & correlation analysis. / Libreria di analisi: outlier, SHAP/eli5/PDP/LIME, clustering, time-series. |

## How to run / Come eseguire

Both scripts are **monolithic**: you edit the `__main__` block to choose what to run
(like `model/ML_Prism.py`). / Entrambi sono **monolitici**: si modifica il blocco `__main__`
per scegliere cosa eseguire.

```bash
# 1) (MIMIC only) build the processed arrays from raw MIMIC files
python statistics/load_mimic.py

# 2) run the exploratory analyses (uncomment the blocks you need in data_analysis.py)
python statistics/data_analysis.py
```

## Inputs / Inputs
- `datasets/SepsisExp/` (public) or `datasets/MIMIC/` (credentialed) — see `datasets/README.md`.

## Outputs / Output
- Plots and reports in `clustering/`, `trees/`, `gmm/` and top-level `*.png` / `*.html`
  (all gitignored — regenerable).
