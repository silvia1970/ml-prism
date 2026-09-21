"""
PRISM API - Main Application

Flask-based REST API for PRISM machine learning predictions.
Uses modular blueprints for clean separation of concerns.

Architecture:
  - api/blueprints/data.py       -> /api/v1/data/*
  - api/blueprints/csv.py        -> /api/v1/upload-csv, /api/v1/csv-template/*, /api/v1/predict-csv
  - api/blueprints/submissions.py -> /api/v1/submissions/*
  - api/blueprints/patients.py   -> /api/v1/patients/*
  - api/blueprints/charts.py     -> /api/v1/charts/*
  - api/blueprints/stats.py      -> /api/v1/stats, /api/v1/scores, /api/v1/records, /api/v1/critical-statistics, /api/v1/fields/ranges, /api/v1/models
"""

from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS
from datetime import datetime, timezone
from flasgger import Swagger
from pathlib import Path
import logging
import os
import io
import json
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress health-check noise from Docker (every 30s)
class _HealthCheckFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return '/health' not in msg

logging.getLogger('werkzeug').addFilter(_HealthCheckFilter())

# Import core components
from api.database import Database
from api.ml_scorer import MLScorer
from api.csv_handler import CSVHandler, process_csv_upload
from api.chart_generator import ChartGenerator
from api.auth import requires_auth
from api import minio_storage

# Import ML components
from api.torch_models import ModelLoader
from api.inference_engine import get_inference_engine

# Import blueprints
from api.blueprints import register_all_blueprints

# Initialize Flask app
app = Flask(__name__)
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key:
    import warnings
    _secret_key = 'dev-secret-key-change-in-production'
    warnings.warn(
        "SECRET_KEY env var is not set. Using insecure default — DO NOT use in production!",
        stacklevel=1,
    )
app.config['SECRET_KEY'] = _secret_key
# Max upload size: 100 MB default (for large clinical datasets up to 100k records)
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE_MB', 100)) * 1024 * 1024

# CORS configuration
cors_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3003,https://prism-data-ingestion-v1.web.app,https://prism-data-ingestion-v1.firebaseapp.com').split(',')]
CORS(app, origins=cors_origins, supports_credentials=True)


def _is_production_mode() -> bool:
    return os.getenv('FLASK_ENV', 'production').lower() != 'development'


def _validate_runtime_security_config() -> None:
    """Emit explicit warnings for insecure or incomplete runtime configuration."""
    import warnings

    if not os.getenv('SECRET_KEY'):
        warnings.warn(
            'SECRET_KEY is not set. The app is using an insecure fallback key.',
            stacklevel=1,
        )
    elif os.getenv('SECRET_KEY') == 'dev-secret-key-change-in-production':
        warnings.warn(
            'SECRET_KEY is using the default placeholder value. Replace it before production use.',
            stacklevel=1,
        )

    if os.getenv('AUTH0_ENABLED', 'false').lower() == 'true':
        if not os.getenv('AUTH0_DOMAIN') or not os.getenv('AUTH0_AUDIENCE'):
            raise RuntimeError('AUTH0_ENABLED=true requires AUTH0_DOMAIN and AUTH0_AUDIENCE to be set')

    if _is_production_mode() and any(origin == '*' for origin in cors_origins):
        warnings.warn(
            'CORS_ORIGINS contains a wildcard origin in production mode. Restrict allowed origins.',
            stacklevel=1,
        )


_validate_runtime_security_config()

# Swagger configuration
app.config['SWAGGER'] = {
    'title': 'PRISM API',
    'uiversion': 3,
    'version': '3.4.0',
    'description': '''REST API for PRISM (Predictive Risk Intelligence System for Medicine).

## Overview
PRISM provides machine learning-based clinical risk predictions for ICU patients.

## Databases Supported
- **MIMIC**: Intensive Care Unit data (34 ML features + age, icustay_id as identifier) — stored in `mimic.db`
- **SepsisExp**: Sepsis-specialized data (27 ML features + age, id as identifier) — stored in `sepsiexp.db`

Each dataset uses a separate SQLite database file for data isolation.

## Score & Classification
The ML model returns a **score** in the range `0.0 - 1.0` (raw sigmoid probability).

| Score Range | Class | Description |
|-------------|-------|-------------|
| < 0.50      | low_risk | Low risk — No sepsis/adverse outcome predicted |
| 0.50 - 0.75 | moderate_risk | Moderate risk — Possible sepsis/adverse outcome |
| ≥ 0.75      | high_risk | High risk — Sepsis/adverse outcome predicted |

## Authentication
**✅ Authentication is active via Auth0 JWT.**

All endpoints (except `/health`) require an Auth0 JWT Bearer token:

```
Authorization: Bearer <auth0_jwt_token>
```

**Auth0 Configuration:**
- Domain: `dev-7w753vlas2njxci6.eu.auth0.com`
- Audience: `https://api.prism.local`
- Algorithm: RS256

The `/health` endpoint remains public for monitoring purposes.

## CORS
Allowed origins:
- `http://localhost:3000` (development)
- `http://localhost:3003` (development / Docker)
- `https://prism-data-ingestion-v1.web.app` (production)
- `https://prism-data-ingestion-v1.firebaseapp.com` (production)

## Changelog v3.2.0
- **Blueprints**: Refactored into modular blueprints for better maintainability
- **Shared Utils**: `utils.py` with LSTMClassifier and `_classify_risk()` shared across modules
- **DateTime Fix**: Replaced deprecated `datetime.utcnow()` with `datetime.now(timezone.utc)`
- **Frontend Alignment**: All frontend components now use `db_name` parameter (was `db`)
- **Cleanup**: Removed unused files (`config.py`, `test_examples.py`, `export_models.py`, `CONFIG.md`, `normalization.py`, `sanitization.py`, `run_Prism.py`, `ML_Prism*.py`, `tkinter_prism.py`, migration scripts, local startup scripts)
- **Submissions filtering**: `GET /api/v1/submissions` supports `submission_id`, `sample_id`, `date_from`, `date_to`
- **Field ranges**: `age` field added to MIMIC ranges in `/api/v1/fields/ranges`
- **CORS**: Firebase production domains added
''',
    'termsOfService': '',
    'contact': {
        'name': 'PRISM API Support',
        'email': 'support@prism-api.local'
    },
    'license': {
        'name': 'MIT',
        'url': 'https://opensource.org/licenses/MIT'
    },
    'specs_route': '/apidocs/',
    'securityDefinitions': {
        'Bearer': {
            'type': 'apiKey',
            'name': 'Authorization',
            'in': 'header',
            'description': 'Auth0 JWT token. Format: Bearer <token>'
        },
        'OAuth2': {
            'type': 'oauth2',
            'flow': 'implicit',
            'authorizationUrl': f'https://{os.getenv("AUTH0_DOMAIN", "your-tenant.eu.auth0.com")}/authorize',
            'scopes': {
                'read:data': 'Read patient data',
                'write:data': 'Submit and update patient data',
                'admin': 'Administrative access'
            }
        }
    },
    'tags': [
        {'name': 'Health', 'description': 'API health and status checks (public)'},
        {'name': 'Data Submission', 'description': 'Submit patient data for ML scoring (requires auth)'},
        {'name': 'Data Retrieval', 'description': 'Retrieve patient records (requires auth)'},
        {'name': 'Data Update', 'description': 'Update patient records (requires auth)'},
        {'name': 'Submissions', 'description': 'Manage submissions (requires auth)'},
        {'name': 'Statistics', 'description': 'Aggregate statistics (requires auth)'},
        {'name': 'Scores', 'description': 'Score data for visualization (requires auth)'},
        {'name': 'Patients', 'description': 'Patient management (requires auth)'},
        {'name': 'Metadata', 'description': 'Field ranges and validation info (requires auth)'},
        {'name': 'Utilities', 'description': 'Templates and charts (requires auth)'}
    ]
}
swagger = Swagger(app)

# Initialize components
db = Database()
ml_scorer = MLScorer()
chart_generator = ChartGenerator()
# ModelLoader is already instantiated inside MLScorer; re-use it to avoid
# loading the same .pth files multiple times into RAM.
model_loader = ml_scorer.torch_loader
if model_loader is None:
    # Fallback: MLScorer failed to load PyTorch — create a standalone loader
    model_loader = ModelLoader()
    model_loader.load_all_models()
inference_engine = get_inference_engine()


@app.after_request
def add_security_headers(response):
    """Add baseline security and cache-control headers to API responses."""
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'DENY')
    response.headers.setdefault('Referrer-Policy', 'no-referrer')
    response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    response.headers.setdefault('Cache-Control', 'no-store, max-age=0')
    response.headers.setdefault('Pragma', 'no-cache')
    response.headers.setdefault('Expires', '0')

    if _is_production_mode() and request.is_secure:
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')

    if response.mimetype == 'application/json':
        response.headers.setdefault('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

    return response


# =============================================================================
# Health Check (Public - No Authentication Required)
# =============================================================================

@app.route('/health', methods=['GET'])
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """
    API Health Check
    ---
    tags:
      - Health
    summary: Verify API server status
    description: Returns the current health status and timestamp of the API server.
    operationId: healthCheck
    responses:
      200:
        description: API is healthy and operational
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
              description: Server status
            timestamp:
              type: string
              format: date-time
              example: '2026-01-14T10:00:00.000000Z'
              description: Current UTC timestamp
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }), 200


# =============================================================================
# Register Blueprints
# =============================================================================

register_all_blueprints(app, db, ml_scorer, model_loader, inference_engine, chart_generator)


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'status': 'error',
        'error_type': 'NotFound',
        'message': 'The requested resource was not found.'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}", exc_info=True)
    return jsonify({
        'status': 'error',
        'error_type': 'InternalServerError',
        'message': 'An unexpected error occurred. Please try again later.',
        'trace_id': str(uuid.uuid4())
    }), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    """Handle file too large errors"""
    return jsonify({
        'status': 'error',
        'error_type': 'RequestEntityTooLarge',
        'message': 'The uploaded file exceeds the maximum allowed size (100 MB).'
    }), 413


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors"""
    return jsonify({
        'status': 'error',
        'error_type': 'MethodNotAllowed',
        'message': 'The requested method is not allowed for this endpoint.'
    }), 405


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    _debug = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    app.run(host='0.0.0.0', port=5000, debug=_debug)
