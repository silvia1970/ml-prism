# PRISM API — Backend

REST API per predizioni PRISM ML con **LSTM PyTorch**, **CSV upload**, **Auth0 JWT** e **validazione completa**.

## Quick Start

### Docker

```bash
docker compose up -d
# Backend:  http://localhost:5000
# Swagger:  http://localhost:5000/apidocs
```

### Locale

```bash
pip install -r requirements/requirements-api.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python start_api.py
```

**Verifica**: `http://localhost:5000/health`

## Modelli ML

| Dataset | Features | Architettura | File |
|---------|----------|-------------|------|
| MIMIC | 34 | LSTM (34→4, 2 layers, mean) | `mimic_24-6_None.pth` |
| SepsisExp | 27 + age | LSTM (27→8, 4 layers, max) | `sepsisexp_24-6_None_29features_norm.pth` |

**Output**: Risk score 0.0–1.0 (LOW `<0.50`, MODERATE `0.50–0.75`, HIGH `≥0.75`)

## Endpoint Principali

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/health` | GET | Health check (pubblico) |
| `/api/v1/data` | POST | Submit patient data |
| `/api/v1/data/<sample_id>` | GET | Retrieve record |
| `/api/v1/data/<sample_id>` | PATCH | Update record |
| `/api/v1/upload-csv` | POST | CSV batch upload |
| `/api/v1/csv-template/<db>` | GET | Download template |
| `/api/v1/predict-csv` | POST | Predict from CSV |
| `/api/v1/submissions` | GET | List submissions |
| `/api/v1/submissions/<id>` | GET/DELETE | Submission details/delete |
| `/api/v1/submissions/<id>/recalculate` | POST | Recalculate |
| `/api/v1/patients` | GET | List patients |
| `/api/v1/patients/<id>/history` | GET | Patient history |
| `/api/v1/patients/<id>/statistics` | GET | Patient statistics |
| `/api/v1/patients/<id>/predict-sequence` | GET | Sliding window |
| `/api/v1/stats` | GET | Global statistics |
| `/api/v1/scores` | GET | Recent scores |
| `/api/v1/records` | GET | All records |
| `/api/v1/fields/ranges` | GET | Field validation ranges |
| `/api/v1/models` | GET | Loaded models info |
| `/api/v1/charts/*` | GET/POST | Charts |

## Architettura

```
api/
├── app.py              # Flask app, Swagger, error handlers
├── auth.py             # Auth0 JWT verification
├── database.py         # SQLite + JSON + MinIO persistence
├── csv_handler.py      # CSV parsing e validazione
├── ml_scorer.py        # ML scoring (PyTorch + mock fallback)
├── torch_models.py     # PyTorch model loader e inference
├── inference_engine.py # Sliding window predictions
├── field_mappings.py   # JSON schema ↔ PyTorch field mapping
├── field_ranges.py     # Clinical field validation ranges
├── chart_generator.py  # Matplotlib chart generation
├── minio_storage.py    # MinIO S3 storage client
├── utils.py            # LSTMClassifier, _classify_risk()
├── validators.py       # Schema validation
└── blueprints/         # Modular Flask blueprints
    ├── data.py         #   /api/v1/data/*
    ├── csv.py          #   /api/v1/upload-csv, /predict-csv
    ├── patients.py     #   /api/v1/patients/*
    ├── submissions.py  #   /api/v1/submissions/*
    ├── charts.py       #   /api/v1/charts/*
    └── stats.py        #   /api/v1/stats, /scores, /records, ...
```

### Componenti Chiave

- **ModelLoader** (`torch_models.py`) — Carica e gestisce modelli PyTorch LSTM
- **MLScorer** (`ml_scorer.py`) — Inference con fallback mock
- **InferenceEngine** (`inference_engine.py`) — Sliding window predictions da CSV
- **Database** (`database.py`) — Persistenza multi-backend (SQLite + JSON + MinIO)
- **Auth** (`auth.py`) — Auth0 JWT con cache JWKS

## Configurazione

| Variable | Description | Default |
|----------|-------------|---------|
| `AUTH0_ENABLED` | Enable Auth0 JWT | `false` |
| `AUTH0_DOMAIN` | Auth0 tenant domain | — |
| `AUTH0_AUDIENCE` | API identifier | `https://api.prism.local` |
| `CORS_ORIGINS` | Allowed origins | `http://localhost:3000,...` |
| `MAX_FILE_SIZE_MB` | Max CSV upload size | `100` |
| `MINIO_ENABLED` | Enable MinIO S3 | `true` (Docker) |
| `MINIO_ENDPOINT` | MinIO host:port | `minio:9000` |

## Autenticazione

**Auth0 JWT** — endpoint protetti (escluso `/health`):

```http
Authorization: Bearer <auth0_jwt_token>
```

- **Domain**: `dev-7w753vlas2njxci6.eu.auth0.com`
- **Audience**: `https://api.prism.local`
- **Algorithm**: RS256
- **Disable**: `AUTH0_ENABLED=false`