"""
PRISM API - Main Application Factory.
"""
from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS
from datetime import datetime, timezone
from flasgger import Swagger
from pathlib import Path
import logging
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.api.auth import requires_auth
from src.api.model_loader import ModelLoader
from src.api.inference import get_inference_engine
from src.api.database import Database
from src.api.ml_scorer import MLScorer
from src.api.chart_generator import ChartGenerator


def _is_production_mode() -> bool:
    return os.getenv('FLASK_ENV', 'production').lower() != 'development'


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    _secret_key = os.getenv('SECRET_KEY')
    if not _secret_key:
        _secret_key = 'dev-secret-key-change-in-production'
        import warnings
        warnings.warn("SECRET_KEY not set. Using insecure default.")
    app.config['SECRET_KEY'] = _secret_key
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE_MB', 100)) * 1024 * 1024

    cors_origins = [o.strip() for o in os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3003').split(',')]
    CORS(app, origins=cors_origins, supports_credentials=True)

    app.config['SWAGGER'] = {
        'title': 'PRISM API',
        'uiversion': 3,
        'version': '3.4.0',
        'description': 'REST API for PRISM - Predictive Risk Intelligence System for Medicine.',
        'specs_route': '/apidocs/',
    }
    Swagger(app)

    # Initialize components
    db = Database()
    ml_scorer = MLScorer()
    model_loader = ml_scorer.torch_loader or ModelLoader()
    model_loader.load_all_models()
    inference_engine = get_inference_engine()
    chart_generator = ChartGenerator()

    # Register blueprints
    from src.api.blueprints.routes import register_all_routes
    register_all_routes(app, db, ml_scorer, model_loader, inference_engine, chart_generator)

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('X-Frame-Options', 'DENY')
        response.headers.setdefault('Cache-Control', 'no-store, max-age=0')
        return response

    @app.route('/health', methods=['GET'])
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat()}), 200

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'status': 'error', 'error_type': 'NotFound', 'message': 'Resource not found.'}), 404

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'error_type': 'InternalServerError',
                        'message': 'An unexpected error occurred.', 'trace_id': str(uuid.uuid4())}), 500

    return app


if __name__ == '__main__':
    _debug = os.getenv('FLASK_ENV', 'production').lower() == 'development'
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=_debug)