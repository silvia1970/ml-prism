# Datasets

This folder holds the **training data** used by the `model/` and `statistics/` modules.
The raw datasets are **large (~1.6 GB)** and are therefore **not committed** to git
(see `.gitignore`). Obtain them with:

```bash
bash scripts/prepare_data.sh
```

## Expected layout

```
datasets/
├── MIMIC/
│   ├── carry_forward/mean/
│   │   ├── train/   (x_ml.npy, x_dl.npy, X_train.tsv, y.npy, y_train.tsv)
│   │   ├── val/
│   │   └── test/
│   └── original/    (train_data.pkl, val_data.pkl, test_data.pkl)
└── SepsisExp/
    ├── sepsisexp_timeseries_partition-A.{csv,tsv}
    ├── sepsisexp_timeseries_partition-B.{csv,tsv}
    ├── sepsisexp_timeseries_partition-C.{csv,tsv}
    └── sepsisexp_timeseries_partition-D.{csv,tsv}
```

## Sources

### SepsisExp (public, CC-BY 4.0)
- Download: <https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/>
- Keep the four time-series partitions (A/B/C/D) as `.csv`/`.tsv` under `datasets/SepsisExp/`.
- Reference: Lindner et al., *J. Transl. Med.* (2022); Schamoni et al., *PMLR* (2022).

### MIMIC-III (credentialed access required)
- Source: PhysioNet — <https://physionet.org/content/mimiciii/1.4/>
- You must complete the PhysioNet credentialed-data course and accept the license
  before downloading.
- The preprocessed `carry_forward/mean/{train,val,test}` arrays are produced by the
  data-loading/preprocessing step in `statistics/load_mimic.py` (see `statistics/README.md`).

> **Note:** the small JSON files under `backend/datasets/` (normalization / scaler stats)
> **are** committed to git — they are required by the inference API and are tiny.
