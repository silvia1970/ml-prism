# PRISM API — Endpoint Reference

> **Version**: 3.4.1 | **Base URL**: `http://localhost:5000` | **Swagger UI**: `/apidocs`

---

## Authentication

Tutti gli endpoint (escluso `/health`) richiedono **Auth0 JWT Bearer token**:

```http
Authorization: Bearer <auth0_jwt_token>
```

| Environment | Status |
|-------------|--------|
| Development | 🔐 Auth0 JWT (disabilita con `AUTH0_ENABLED=false`) |
| Production | 🔐 Auth0 JWT obbligatorio |

---

## Query Parameters

Gli endpoint che accettano un tipo di dataset usano il parametro **`db`** (valori: `mimic`, `sepsiexp`).

---

## Endpoints

### Health

| Method | Endpoint | Description | Auth |
|:------:|----------|-------------|:----:|
| GET | `/health` | Server health check | 🟢 Public |

### Data Submission

| Method | Endpoint | Description |
|:------:|----------|-------------|
| POST | `/api/v1/data` | Submit patient data per ML scoring |
| POST | `/api/v1/upload-csv` | Batch upload via CSV file |
| POST | `/api/v1/predict-csv` | CSV prediction con sliding window |

### Data Operations

| Method | Endpoint | Description |
|:------:|----------|-------------|
| GET | `/api/v1/data/<sample_id>` | Retrieve patient record |
| PATCH | `/api/v1/data/<sample_id>` | Update patient record |
| GET | `/api/v1/data/patient/<patient_id>` | Get record by patient ID |

### Submissions

| Method | Endpoint | Description |
|:------:|----------|-------------|
| GET | `/api/v1/submissions` | List submissions (paginazione + filtri) |
| GET | `/api/v1/submissions/<id>` | Get submission details |
| DELETE | `/api/v1/submissions/<id>` | Delete submission |
| POST | `/api/v1/submissions/<id>/recalculate` | Recalculate predictions |

**Filtri per GET /submissions**: `db`, `submission_id`, `sample_id`, `date_from`, `date_to`, `page`, `page_size`

### Patients

| Method | Endpoint | Description |
|:------:|----------|-------------|
| GET | `/api/v1/patients` | List patients (paginazione + search) |
| GET | `/api/v1/patients/<id>/history` | Patient prediction history |
| GET | `/api/v1/patients/<id>/statistics` | Patient statistics |
| GET | `/api/v1/patients/<id>/submissions` | Patient submissions |
| GET | `/api/v1/patients/<id>/predict-sequence` | Sliding window predictions |

**Sliding window params**: `target_len` (default 24), `stride` (default 6)

### Records & Statistics

| Method | Endpoint | Description |
|:------:|----------|-------------|
| GET | `/api/v1/stats` | Global database statistics |
| GET | `/api/v1/scores` | Recent scores |
| GET | `/api/v1/records` | All records (paginazione) |
| GET | `/api/v1/critical-statistics` | Critical statistics summary |

### Metadata & Utilities

| Method | Endpoint | Description |
|:------:|----------|-------------|
| GET | `/api/v1/fields/ranges` | Clinical field validation ranges |
| GET | `/api/v1/models` | Loaded models info |
| GET | `/api/v1/csv-template/<db_name>` | Download CSV template |
| GET | `/api/v1/charts/<filename>` | Chart PNG image |
| POST | `/api/v1/charts/generate/risk-distribution` | Generate risk pie chart |
| POST | `/api/v1/charts/generate/score-distribution` | Generate score histogram |
| POST | `/api/v1/charts/generate/combined` | Generate combined chart |

---

## Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized (missing/invalid JWT) |
| 404 | Not Found |
| 405 | Method Not Allowed |
| 413 | Payload Too Large |
| 500 | Internal Server Error |

---

## Risk Classification

| Score Range | Class | Description |
|-------------|-------|-------------|
| `< 0.50` | `low_risk` | Nessun rischio sepsi previsto |
| `0.50 – 0.74` | `moderate_risk` | Possibile sepsi |
| `≥ 0.75` | `high_risk` | Sepsi prevista |

---

## Supported Databases

| Database | Features | Model | File |
|----------|----------|-------|------|
| `mimic` | 34 clinical | LSTM (34→4, 2 layers, mean) | `mimic_24-6_None.pth` |
| `sepsiexp` | 27 clinical + age | LSTM (27→8, 4 layers, max) | `sepsisexp_24-6_None_29features_norm.pth` |

---

## Architecture

```
api/
├── app.py              # Flask app, Swagger, error handlers
├── auth.py             # Auth0 JWT verification
├── database.py         # SQLite + JSON + MinIO persistence
├── csv_handler.py      # CSV parsing e validazione
├── ml_scorer.py        # ML scoring (PyTorch + mock fallback)
├── torch_models.py     # PyTorch model loader e inference
├── inference_engine.py # CSV → sliding window predictions
├── field_mappings.py   # JSON schema ↔ PyTorch field mapping
├── field_ranges.py     # Clinical field validation ranges
├── chart_generator.py  # Matplotlib chart generation
├── minio_storage.py    # MinIO S3 storage client
├── utils.py            # LSTMClassifier, _classify_risk()
├── validators.py       # Schema validation (MIMIC/SepsisExp)
└── blueprints/         # Flask blueprints modulari
    ├── data.py         #   /api/v1/data/*
    ├── csv.py          #   /api/v1/upload-csv, /predict-csv
    ├── patients.py     #   /api/v1/patients/*
    ├── submissions.py  #   /api/v1/submissions/*
    ├── charts.py       #   /api/v1/charts/*
    └── stats.py        #   /api/v1/stats, /scores, /records, ...
```