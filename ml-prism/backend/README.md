# backend — inference API

REST API (Flask) per la **predizione del rischio clinico** basata sui modelli LSTM
PyTorch addestrati in `model/`. Supporta upload CSV, persistenza SQLite + MinIO (S3)
e autenticazione Auth0 JWT (opzionale).

> EN: Flask REST API for clinical risk prediction based on the PyTorch LSTM models
> trained in `model/`. Supports CSV upload, SQLite + MinIO (S3) persistence, and
> optional Auth0 JWT authentication.

## Models / Modelli

| Dataset | Features | Architecture | Expected file |
|---------|----------|--------------|---------------|
| MIMIC | 34 ICU variables | LSTM (in=34, hidden=4, layers=2, mean pooling) | `backend/models/mimic_24-6_None.pth` |
| SepsisExp | 27 variables + age | LSTM (in=27, hidden=8, layers=4, max pooling) | `backend/models/sepsisexp_24-6_None_29features_norm.pth` |

Risk classes: `< 0.50` `low_risk` · `0.50–0.74` `moderate_risk` · `≥ 0.75` `high_risk`.

> If a `.pth` file is missing the API still **starts** and serves requests, using a
> heuristic **mock predictor** (see `models/README.md`). Place your trained
> checkpoints at the paths above for real predictions.

## Quick start / Avvio rapido

### Docker Compose (recommended / consigliato)
```bash
cd backend
docker compose up -d
```
| Service / Servizio | URL |
|--------------------|-----|
| Backend API | `http://localhost:5000` |
| Swagger UI | `http://localhost:5000/apidocs` |
| MinIO Console | `http://localhost:9001` |

### Local / Locale
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements-api.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env        # then edit values (SECRET_KEY, etc.)
python start_api.py
```

## Configuration / Configurazione
- Copy `.env.example` to `.env` and edit. Auth0 is **disabled by default**
  (`AUTH0_ENABLED=false`); MinIO is optional for basic inference.
- Full variable reference: `.env.example`. API reference: `docs/API_DOCUMENTATION.md`
  and `api/ENDPOINTS.md`.

## Structure / Struttura
```
backend/
├── api/                  # Flask app, blueprints, inference engine, torch loader
├── start_api.py          # local entry point
├── docker/  docker-compose.yml
├── datasets/             # normalization / scaler JSONs (committed)
├── models/               # place trained .pth here (see table above)
├── requirements/         # requirements-api.txt, requirements.txt, ...
├── docs/                 # API docs, Auth0 setup, changelog
└── tests/                # API tests
```
