# PRISM — Test Suite

## Esecuzione

```bash
# Dalla root del progetto
AUTH0_ENABLED=false python -m pytest tests/backend/ -v

# Con coverage
AUTH0_ENABLED=false python -m pytest tests/backend/ -v --cov=api --cov-report=html
```

## Struttura

```
tests/
├── backend/
│   ├── __init__.py
│   └── test_all_apis.py     # Suite completa (53 test)
├── cleanup_database.py       # Utility reset database
└── __init__.py
```

## Test Disponibili

**test_all_apis.py** (53 test, Flask test client):

| Classe | Endpoint | Test |
|--------|----------|------|
| TestHealthCheck | `/health` | 2 |
| TestDataSubmission | `POST /api/v1/data` | 8 |
| TestCSVUpload | `POST /api/v1/upload-csv` | 3 |
| TestCSVTemplate | `GET /api/v1/csv-template/<db>` | 3 |
| TestDataRetrieval | `GET /api/v1/data/<id>` | 2 |
| TestDataUpdate | `PATCH /api/v1/data/<id>` | 3 |
| TestStatistics | `GET /api/v1/stats` | 1 |
| TestSubmissions | `/api/v1/submissions*` | 7 |
| TestScores | `GET /api/v1/scores` | 2 |
| TestPatients | `/api/v1/patients*` | 4 |
| TestRecords | `GET /api/v1/records` | 1 |
| TestCriticalStatistics | `GET /api/v1/critical-statistics` | 1 |
| TestFieldRanges | `GET /api/v1/fields/ranges` | 2 |
| TestModels | `GET /api/v1/models` | 1 (skipped) |
| TestErrorHandlers | Error handlers | 2 |
| TestValidators | Unit validation | 9 |
| TestIntegration | Full workflow | 3 |

## Auth0 nei Test

L'autenticazione Auth0 è disabilitata automaticamente nei test tramite `os.environ['AUTH0_ENABLED'] = 'false'` all'inizio del file.