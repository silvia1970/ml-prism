# PRISM API Documentation

## Overview

| Property | Value |
|----------|-------|
| **Version** | 3.0.0 |
| **Base URL** | `http://localhost:5000` |
| **Swagger UI** | `http://localhost:5000/apidocs` |
| **Format** | JSON |
| **Encoding** | UTF-8 |

PRISM (Predictive Risk Intelligence System for Medicine) provides ML-based clinical risk predictions for ICU patients.

---

## Table of Contents

1. [Authentication](#1-authentication)
2. [Databases](#2-databases)
3. [Endpoints](#3-endpoints)
4. [Data Schemas](#4-data-schemas)
5. [Error Handling](#5-error-handling)
6. [Examples](#6-examples)

---

## 1. Authentication

### Current Status

| Environment | Status |
|-------------|--------|
| Development | Disabled |
| Production | Auth0 JWT Required |

### Production Configuration

```http
Authorization: Bearer <auth0_jwt_token>
```

| Parameter | Value |
|-----------|-------|
| Provider | Auth0 |
| Domain | `dev-7w753vlas2njxci6.eu.auth0.com` |
| Audience | `https://api.prism.local` |
| Algorithm | RS256 |

### OAuth2 Scopes

| Scope | Access |
|-------|--------|
| `read:data` | GET endpoints |
| `write:data` | POST, PATCH, DELETE |
| `admin` | Full access |

---

## 2. Databases

| Database | Description | ML Features | Model File | SQLite File |
|----------|-------------|:-----------:|------------|-------------|
| **MIMIC** | ICU data | 34 | `mimic_24-6_None.pth` | `api_data/mimic.db` |
| **SepsisExp** | Sepsis-specialized | 27 | `sepsisexp_24-6_None_29features_norm.pth` | `api_data/sepsiexp.db` |

> **v3.0.0**: Ogni database usa un file SQLite separato. Il vecchio `prism.db` è mantenuto come fallback per dati legacy.

### Score & Risk Classes

| Score Range | Class | Description |
|:-----------:|-------|-------------|
| `0.00 – 0.49` | `low_risk` | Basso rischio |
| `0.50 – 0.74` | `moderate_risk` | Rischio moderato |
| `0.75 – 1.00` | `high_risk` | Alto rischio |

> Lo score è la probabilità grezza sigmoid (0.0–1.0), **non** scalata a 0–100.

---


---

## 3 Endpoints

### 3.1 Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-14T10:00:00.000000Z"
}
```

---

### 3.2 Submit Data

Submit patient data for ML risk prediction.

```http
POST /api/v1/data
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `db` | string | ✓ | `mimic` or `sepsiexp` |
| `submission_id` | string | ✓ | Unique submission ID |
| `submitted_at` | string | ✓ | ISO 8601 timestamp |
| `data` | array | ✓ | Patient records array |
| `meta` | object | | Optional metadata |

**Example Request:**
```json
{
  "db": "mimic",
  "submission_id": "SUB-2026-001",
  "submitted_at": "2026-01-14T10:00:00Z",
  "data": [
    {
      "sample_id": "P001",
      "meanbp": 92.5,
      "heartrate": 75,
      "lactate": 1.5
    }
  ]
}
```

**Response 200:**
```json
{
  "db": "mimic",
  "submission_id": "SUB-2026-001",
  "status": "success",
  "records_processed": 1,
  "chart_path": "mimic_combined_analysis_SUB-2026-001.png",
  "results": [
    {
      "sample_id": "P001",
      "score": 0.35,
      "class": "low_risk",
      "explanations": {}
    }
  ]
}
```

**Response 400:**
```json
{
  "status": "failure",
  "error_type": "ValidationError",
  "message": "Validation failed",
  "data_errors": [
    {
      "record_index": 0,
      "field": "lactate",
      "code": "out_of_range",
      "description": "Value out of allowed range"
    }
  ]
}
```

---

### 3.3 Upload CSV

Batch upload patient data via CSV file.

```http
POST /api/v1/upload-csv
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `file` | file | ✓ | CSV file (max 100MB) |
| `db` | string | ✓ | `mimic` or `sepsiexp` |

---

### 3.4 Predict from CSV

Upload CSV with time series data for sliding window predictions.

```http
POST /api/v1/predict-csv
Content-Type: multipart/form-data
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | - | CSV with time series |
| `model` | string | - | `mimic` or `sepsiexp` |
| `target_len` | int | 24 | Window size |
| `stride` | int | 6 | Sliding stride |

**Response:**
```json
{
  "status": "success",
  "filename": "patient_data.csv",
  "windows": [
    {"window_index": 0, "score": 0.28, "class": "low_risk"},
    {"window_index": 1, "score": 0.72, "class": "high_risk"}
  ],
  "summary": {
    "n_windows": 5,
    "avg_score": 0.45,
    "at_risk_windows": 2,
    "low_risk_windows": 3,
    "at_risk_ratio": 0.4,
    "final_class": "moderate_risk"
  }
}
```

---

### 3.5 Get Sample

```http
GET /api/v1/data/{sample_id}?db=mimic&timestamp=2026-01-14T10:00:00Z
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db` | string | mimic | Database |
| `timestamp` | string | - | Filter by timestamp |

---

### 3.6 Update Sample

Update fields and automatically recalculate prediction.

```http
PATCH /api/v1/data/{sample_id}?db=mimic
```

**Request:**
```json
{
  "heartrate": 85,
  "lactate": 2.0
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Sample updated and prediction recalculated",
  "updated_fields": ["heartrate", "lactate"],
  "original_values": {"heartrate": 75, "lactate": 1.5},
  "new_values": {"heartrate": 85, "lactate": 2.0},
  "prediction": {"score": 0.48, "class": "moderate_risk"}
}
```

---

### 3.7 Get Sample by Patient ID

```http
GET /api/v1/data/patient/{patient_id}?db=sepsiexp&timestep=24
```

---

### 3.8 Statistics

```http
GET /api/v1/stats
```

**Response:**
```json
[
  {
    "status": "success",
    "current_timestamp_utc": "2026-03-03T10:00:00Z",
    "db": "mimic",
    "counts": {
      "total_entries": 1250,
      "submitted_today": 85,
      "submitted_this_week": 320
    }
  },
  {
    "status": "success",
    "current_timestamp_utc": "2026-03-03T10:00:00Z",
    "db": "sepsiexp",
    "counts": {
      "total_entries": 340,
      "submitted_today": 12,
      "submitted_this_week": 75
    }
  }
]
```

---

### 3.9 Submissions

**List:**
```http
GET /api/v1/submissions?page=1&page_size=50&db=all
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 50 | Results per page |
| `db` | string | all | `mimic`, `sepsiexp`, or `all` |
| `submission_id` | string | - | Filter by submission ID (partial match) |
| `sample_id` | string | - | Filter by sample ID (partial match) |
| `date_from` | string | - | Filter from date (YYYY-MM-DD) |
| `date_to` | string | - | Filter to date (YYYY-MM-DD) |

**Examples:**
```http
# Filter by submission ID
GET /api/v1/submissions?submission_id=SUB-2026

# Filter by sample ID
GET /api/v1/submissions?sample_id=P001

# Filter by date range
GET /api/v1/submissions?date_from=2026-01-01&date_to=2026-03-31

# Combined filters
GET /api/v1/submissions?db=mimic&sample_id=P001&date_from=2026-01-01
```

**Details:**
```http
GET /api/v1/submissions/{submission_id}
```

**Response:**
```json
{
  "submission_id": "SUB-2026-001",
  "db": "mimic",
  "status": "completed",
  "submitted_at": "2026-03-03T10:00:00Z",
  "records_count": 3,
  "source": "json",
  "original_filename": null,
  "client_version": "web-3.0.0",
  "completion_time_ms": 142,
  "failure_reason": null,
  "records": [
    {"sample_id": "P001", "score": 0.28, "class": "low_risk", "timestep": 0},
    {"sample_id": "P001", "score": 0.54, "class": "moderate_risk", "timestep": 1},
    {"sample_id": "P001", "score": 0.81, "class": "high_risk", "timestep": 2}
  ]
}
```

**Delete:**
```http
DELETE /api/v1/submissions/{submission_id}
```

**Recalculate:**
```http
POST /api/v1/submissions/{submission_id}/recalculate
```
```json
{"updates": {"heartrate": 90}}
```

---

### 3.10 Patients

**List:**
```http
GET /api/v1/patients?db=mimic&page=1&page_size=50&search=MIM
```

**History:**
```http
GET /api/v1/patients/{sample_id}/history?db=mimic&limit=100
```

**Statistics:**
```http
GET /api/v1/patients/{sample_id}/statistics?db=mimic
```

**Response:**
```json
{
  "sample_id": "P001",
  "total_records": 12,
  "score_statistics": {"min": 0.28, "max": 0.72, "mean": 0.45},
  "class_distribution": {"low_risk": 5, "moderate_risk": 6, "high_risk": 1},
  "trend": "stable"
}
```

**Submissions:**
```http
GET /api/v1/patients/{sample_id}/submissions?db=mimic
```

**Predict Sequence:**
```http
GET /api/v1/patients/{sample_id}/predict-sequence?db=mimic&target_len=24&stride=6
```

---

### 3.11 Records

```http
GET /api/v1/records?db=mimic&page=1&page_size=50&sample_id=P001
```

---

### 3.12 Scores

```http
GET /api/v1/scores?db=mimic&limit=100
```

---

### 3.13 Critical Statistics

```http
GET /api/v1/critical-statistics?db=mimic
```

**Response:**
```json
{
  "total_records_analyzed": 1500,
  "critical_statistics": {
    "meanbp": {"min": 45.2, "max": 165.8, "avg": 82.3, "unit": "mmHg"},
    "lactate": {"min": 0.5, "max": 18.2, "avg": 2.1, "unit": "mmol/L"}
  }
}
```

---

### 3.14 Field Ranges

```http
GET /api/v1/fields/ranges?db=mimic
```

**Response:**
```json
{
  "fields": {
    "age": {"min": 0, "max": 120, "unit": "anni", "label": "Età"},
    "meanbp": {"min": 30, "max": 180, "unit": "mmHg", "label": "Pressione Arteriosa Media"},
    "heartrate": {"min": 20, "max": 250, "unit": "bpm", "label": "Frequenza Cardiaca"}
  }
}
```

> **Nota**: Il campo `age` è presente sia nei ranges MIMIC che SepsisExp (min: 0, max: 120).
```

---

### 3.15 Models

```http
GET /api/v1/models
```

**Response:**
```json
{
  "models": [
    {"model_name": "mimic", "loaded": true, "input_dim": 34},
    {"model_name": "sepsiexp", "loaded": true, "input_dim": 27}
  ]
}
```

---

### 3.16 CSV Template

```http
GET /api/v1/csv-template/{db_name}
```

Returns CSV file with required headers and example row.

---

### 3.17 Charts

```http
GET /api/v1/charts/{filename}
```

Returns PNG image of generated prediction chart.

---

## 4. Data Schemas

### 4.1 MIMIC (34 features)

| Category | Fields |
|----------|--------|
| **Demographics** | `age` |
| **Vitals** | `meanbp`, `resprate`, `heartrate`, `spo2_pulsoxy`, `tempc`, `cardiacoutput` |
| **Respiratory** | `o2flow`, `fio2` |
| **Lab** | `albumin`, `bands`, `bicarbonate`, `bilirubin`, `creatinine`, `chloride`, `glucose`, `hemoglobin`, `lactate`, `platelet`, `potassium`, `ptt`, `inr`, `sodium`, `wbc`, `creatinekinase`, `ck_mb`, `fibrinogen`, `ldh`, `magnesium`, `calcium_free`, `troponin_t` |
| **Blood Gas** | `po2_bloodgas`, `ph_bloodgas`, `pco2_bloodgas`, `so2_bloodgas` |

### 4.2 SepsisExp (27 features)

| Category | Fields (DataFlow names) |
|----------|------------------------|
| **Demographics** | `age` |
| **Vitals** | `heartrate`, `svri`, `meanbp`, `hearttimevolume`, `oxygensaturation`, `deltatemp`, `oxygenationsaturation` |
| **Lab** | `lactate`, `creatinine`, `bilirubin`, `sodium`, `potassium`, `hemoglobin`, `chloride`, `leukocytes`, `bicarbonate`, `pancreaticlipase`, `bun`, `pct`, `buncreatinineratio`, `ast`, `crp` |
| **Respiratory** | `respiratoryminutevolume`, `fio2` |
| **Blood Gas** | `arterialph`, `pa_o2` |
| **Optional** | `alt` (stored, not used by model) |

> **v3.0.0**: I nomi dei campi seguono il DataFlow Schema. I vecchi nomi snake_case (`heart_rate`, `mean_bp`, ecc.) sono ancora accettati come alias.

---

## 5. Error Handling

### HTTP Status Codes

| Code | Description |
|:----:|-------------|
| 200 | Success |
| 400 | Validation error |
| 404 | Resource not found |
| 413 | Payload too large |
| 500 | Internal server error |

### Error Response Format

```json
{
  "status": "failure",
  "error_type": "ValidationError",
  "message": "Description of the error",
  "data_errors": [
    {
      "record_index": 0,
      "sample_id": "P001",
      "field": "lactate",
      "code": "out_of_range",
      "description": "Value 50 is outside range (0.1-30)"
    }
  ]
}
```

### 404 Not Found

```json
{
  "status": "error",
  "error_type": "ResourceNotFound",
  "message": "The sample record for ID 'MIM-0001' at time '2025-11-19T10:00:00Z' was not found."
}
```

### 500 Internal Server Error

```json
{
  "status": "error",
  "error_type": "InternalServerError",
  "message": "An unexpected error occurred. Please try again later.",
  "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```
```

### Error Codes

| Code | Description |
|------|-------------|
| `missing_required_field` | Required field not provided |
| `invalid_type` | Wrong data type |
| `out_of_range` | Value outside allowed range |
| `invalid_value` | Invalid categorical value |
| `invalid_format` | Invalid format (e.g., timestamp) |

---

## 6. Examples

### Python

```python
import requests

BASE_URL = "http://localhost:5000"

# Submit data
response = requests.post(f"{BASE_URL}/api/v1/data", json={
    "db": "mimic",
    "submission_id": "PY-001",
    "submitted_at": "2026-01-14T10:00:00Z",
    "data": [{"sample_id": "P001", "meanbp": 92.5, "heartrate": 75}]
})
print(response.json()["results"][0]["score"])

# Upload CSV
with open("patients.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/upload-csv",
        files={"file": f},
        data={"db": "mimic"}
    )

# Get patient history
response = requests.get(
    f"{BASE_URL}/api/v1/patients/P001/history",
    params={"db": "mimic", "limit": 100}
)
```

### JavaScript

```javascript
const response = await fetch("http://localhost:5000/api/v1/data", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    db: "mimic",
    submission_id: "JS-001",
    submitted_at: new Date().toISOString(),
    data: [{sample_id: "P001", meanbp: 92.5, heartrate: 75}]
  })
});
const result = await response.json();
console.log(result.results[0].score);
```

### cURL

```bash
# Health check
curl http://localhost:5000/health

# Submit data
curl -X POST http://localhost:5000/api/v1/data \
  -H "Content-Type: application/json" \
  -d '{"db":"mimic","submission_id":"CURL-001","submitted_at":"2026-01-14T10:00:00Z","data":[...]}'

# Upload CSV
curl -X POST http://localhost:5000/api/v1/upload-csv \
  -F "file=@patients.csv" -F "db=mimic"

# Get statistics
curl http://localhost:5000/api/v1/stats
```

---

## Changelog

### v3.1.0 (2026-03-27)

**New Features:**
- `GET /api/v1/submissions`: nuovi parametri di filtro `submission_id`, `sample_id`, `date_from`, `date_to`
- Campo `age` aggiunto ai ranges MIMIC in `/api/v1/fields/ranges` (min: 0, max: 120)

**Configuration:**
- CORS: aggiunti domini produzione Firebase (`prism-data-ingestion-v1.web.app`, `prism-data-ingestion-v1.firebaseapp.com`)
- CORS: aggiunto `http://localhost:3003` ai default di sviluppo
- Auth0 domain aggiornato a `dev-7w753vlas2njxci6.eu.auth0.com`

### v3.0.1 (2026-03-03)

**Bug Fix:**
- Fix 404 su `GET /api/v1/submissions/{id}`: aggiunto `find_submission()` che ricerca su tutti i DB (`mimic.db`, `sepsiexp.db`, `prism.db`) con fallback case-insensitive
- Fix `DELETE /api/v1/submissions/{id}`: ora usa il DB corretto al posto del legacy `prism.db`
- `inference_engine.py`: score range `0.0–1.0`, classi v3 (`low_risk`/`moderate_risk`/`high_risk`), summary fields rinominati (`at_risk_windows`, `low_risk_windows`, `at_risk_ratio`)
- `torch_models.py`: summary fields allineati a `at_risk_windows`, `low_risk_windows`, `at_risk_ratio`
- `GET /api/v1/submissions/{id}`: risposta ora include array `records` con i campioni della submission
- `GET /api/v1/stats`: risposta include `status` e `current_timestamp_utc` per ogni DB

### v3.0.0 (2026-02-25)

**Breaking Changes:**
- Score format: `0.0–1.0` (era `0–100`)
- Risk classes: `low_risk` / `moderate_risk` / `high_risk` (era `positive` / `negative`)
- SepsisExp field names allineati al DataFlow Schema
- Database separati: `mimic.db` e `sepsiexp.db`
- 500 errors includono `trace_id` (UUID v4)
- 404 messages includono timestamp quando fornito

### v2.1.0 (2026-01-14)

**New Endpoints:**
- `GET /api/v1/patients/{id}/submissions`
- `GET /api/v1/patients/{id}/predict-sequence`
- `GET /api/v1/records`
- `GET /api/v1/critical-statistics`
- `POST /api/v1/predict-csv`
- `GET /api/v1/models`

### v2.0.0 (2025-01-09)

**New Endpoints:**
- CSV upload and templates
- Submission management (CRUD)
- Patient history and statistics
- Field validation ranges

---

## Links

| Resource | URL |
|----------|-----|
| Swagger UI | http://localhost:5000/apidocs |
| Health Check | http://localhost:5000/health |
| MIMIC Template | http://localhost:5000/api/v1/csv-template/mimic |
| SepsisExp Template | http://localhost:5000/api/v1/csv-template/sepsiexp |

---

*Last updated: 2026-03-03*
