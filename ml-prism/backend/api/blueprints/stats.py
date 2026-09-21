"""
Stats Blueprint for PRISM API

Handles statistics, scores, records, and model information endpoints.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import logging

from api.auth import requires_auth

logger = logging.getLogger(__name__)

stats_bp = Blueprint('stats', __name__, url_prefix='/api/v1')


def register_stats_routes(app, db, ml_scorer, model_loader):
    """
    Register statistics-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
        model_loader: ModelLoader instance
    """

    @stats_bp.route('/stats', methods=['GET'])
    @requires_auth
    def get_stats():
        """
        Get overall statistics for a dataset.
        
        ---
        parameters:
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            default: mimic
            description: Dataset type
        responses:
          200:
            description: Statistics including record counts, risk distribution, etc.
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            
            stats = db.get_stats(db_name=db_name)
            
            return jsonify({
                'status': 'success',
                'current_timestamp_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'counts': stats
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_stats: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @stats_bp.route('/scores', methods=['GET'])
    @requires_auth
    def get_scores():
        """
        Get recent prediction scores.
        
        ---
        parameters:
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            default: mimic
            description: Dataset type
          - name: limit
            in: query
            type: integer
            default: 100
            description: Maximum number of scores to return
        responses:
          200:
            description: List of recent prediction scores
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            limit = request.args.get('limit', 100, type=int)
            
            scores = db.get_recent_scores(db_name=db_name, limit=limit)
            
            return jsonify({
                'status': 'success',
                'scores': scores,
                'count': len(scores)
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_scores: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @stats_bp.route('/records', methods=['GET'])
    @requires_auth
    def get_all_records():
        """
        List all records with pagination and filtering.
        
        ---
        parameters:
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            default: mimic
            description: Dataset type
          - name: page
            in: query
            type: integer
            default: 1
            description: Page number
          - name: page_size
            in: query
            type: integer
            default: 50
            description: Items per page
          - name: search
            in: query
            type: string
            description: Search in sample IDs
          - name: risk_class
            in: query
            type: string
            enum: [low_risk, moderate_risk, high_risk]
            description: Filter by risk class
        responses:
          200:
            description: List of records with pagination info
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            page = request.args.get('page', 1, type=int)
            page_size = request.args.get('page_size', 50, type=int)
            search = request.args.get('search')
            risk_class = request.args.get('risk_class')
            
            result = db.list_all_records(
                db_name=db_name,
                page=page,
                page_size=page_size,
                sample_id_filter=search
            )
            
            return jsonify({
                'status': 'success',
                'records': result.get('records', []),
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': result.get('total_records', 0),
                    'pages': result.get('total_pages', 0)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_all_records: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @stats_bp.route('/critical-statistics', methods=['GET'])
    @requires_auth
    def get_critical_statistics():
        """
        Get critical patient statistics based on high-risk predictions.
        
        ---
        parameters:
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            default: mimic
            description: Dataset type
          - name: sample_id
            in: query
            type: string
            description: Optional filter by specific patient
        responses:
          200:
            description: Critical statistics including high-risk patient counts
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            sample_id_filter = request.args.get('sample_id')
            
            stats = db.get_critical_statistics(
                db_name=db_name,
                sample_id_filter=sample_id_filter
            )
            
            return jsonify({
                'status': 'success',
                'statistics': stats
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_critical_statistics: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @stats_bp.route('/fields/ranges', methods=['GET'])
    @requires_auth
    def get_field_ranges():
        """
        Get clinical field ranges and validation information.
        
        ---
        parameters:
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            default: mimic
            description: Dataset type
        responses:
          200:
            description: Field ranges with min/max values and units
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            
            from api.field_ranges import get_all_fields_with_ranges
            ranges = get_all_fields_with_ranges(db_name)
            
            return jsonify({
                'status': 'success',
                'db_name': db_name,
                'fields': ranges
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_field_ranges: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @stats_bp.route('/models', methods=['GET'])
    @requires_auth
    def list_available_models():
        """
        List available ML models and their status.
        
        ---
        responses:
          200:
            description: List of available models with loading status
        """
        try:
            models_info = {}
            
            for model_name in ['mimic', 'sepsiexp']:
                try:
                    info = model_loader.model_info(model_name)
                    models_info[model_name] = info
                except Exception as e:
                    models_info[model_name] = {
                        'name': model_name,
                        'loaded': False,
                        'error': str(e)
                    }
            
            return jsonify({
                'status': 'success',
                'models': models_info
            }), 200
            
        except Exception as e:
            logger.error(f"Error in list_available_models: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(stats_bp)
