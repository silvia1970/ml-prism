# model — LSTM training & testing

Modulo di **training e validazione** del modello LSTM (PyTorch) per la predizione
della sepsi, su SepsisExp e MIMIC-III.

> EN: LSTM (PyTorch) training & validation module for sepsis prediction on SepsisExp
> and MIMIC-III.

## Contents / Contenuto

| File | Purpose / Scopo |
|------|-----------------|
| `ML_Prism.py` | Main entry point: data prep, k-fold CV, early-stopping, final model, PFI. / Punto d'ingresso principale. |
| `ML_Prism_sepsisexp.py` / `ML_Prism_sepsisexp_norm.py` | SepsisExp-specific runs (raw / normalized). |
| `ML_Prism_mimic.py` / `ML_Prism_mimic_multi_window.py` | MIMIC-III runs (single / multi-window). |
| `ML_Prism_gender.py` | Gender-stratified run. |
| `utils.py` | Shared helpers (LSTM classifier, metrics, plotting). / Helper condivisi. |
| `renormalization.py` | Renormalization utilities. / Utility di rinormalizzazione. |

## How to run / Come eseguire

`ML_Prism.py` is **monolithic**: set the configuration at the top of the `__main__`
block, then run it. / È **monolitico**: imposta la configurazione nel blocco `__main__`,
poi eseguilo.

```python
# inside ML_Prism.py  __main__
dataset     = "sepsisexp"   # or "mimic"
target_len  = 24            # input window (hours)
pred_length = 6             # prediction horizon (hours)
method      = None          # or "sequence_dtw_smote", "undersample", "oversample"
```

```bash
python model/ML_Prism.py
```

Outputs: trained checkpoints in `models/`, loss curves and PFI reports (gitignored).

## Inputs / Inputs
- `datasets/` (SepsisExp public / MIMIC-III credentialed) — see `datasets/README.md`.

## Dependencies / Dipendenze
- `requirements.txt` (common) + `requirements-torch.txt` (PyTorch + CUDA), see root `README.md`.
