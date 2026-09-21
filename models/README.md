# Trained models (`.pth`)

This folder holds the **trained LSTM checkpoints** produced by the `model/` module.
They are small and are committed to git.

| File | Dataset | Window (target-pred) | Notes |
|------|---------|----------------------|-------|
| `sepsisexp_24-1_None.pth` | SepsisExp | 24h → 1h | |
| `sepsisexp_24-3_None_paper.pth` | SepsisExp | 24h → 3h | paper configuration |
| `sepsisexp_24-6_None_paper.pth` | SepsisExp | 24h → 6h | paper configuration |

## Inference API (`backend/`)

The inference API loads its models from `backend/models/` (a separate folder, see
`backend/README.md`). It expects:

- `backend/models/mimic_24-6_None.pth`
- `backend/models/sepsisexp_24-6_None_29features_norm.pth`

If a file is missing, the API falls back to a **mock predictor** (it still starts and
serves requests, but predictions are heuristic). To get real predictions, train the
model with `model/` and copy the resulting checkpoint to the expected path above.
