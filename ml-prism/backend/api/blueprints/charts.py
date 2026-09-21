"""
Charts Blueprint for PRISM API

Handles chart generation and retrieval endpoints.
"""

from flask import Blueprint, request, jsonify, send_file
import os
import logging

from api.auth import requires_auth

logger = logging.getLogger(__name__)

charts_bp = Blueprint('charts', __name__, url_prefix='/api/v1/charts')


def register_charts_routes(app, db, chart_generator):
    """
    Register chart-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        chart_generator: ChartGenerator instance
    """

    @charts_bp.route('/<filename>', methods=['GET'])
    @requires_auth
    def get_chart(filename):
        """
        Retrieve a generated chart image.
        
        ---
        parameters:
          - name: filename
            in: path
            required: true
            type: string
            description: Chart filename (e.g., risk_distribution.png)
        responses:
          200:
            description: Chart image file
          404:
            description: Chart not found
        """
        try:
            # Security: canonicalize path to prevent directory traversal (SEC-7)
            filename = os.path.basename(filename)
            if not filename:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid filename'
                }), 400
            
            chart_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'api_data', 'charts')
            chart_path = os.path.join(chart_dir, filename)
            
            if not os.path.exists(chart_path):
                # Try to find matching file (handle partial matches)
                matches = [
                    f for f in os.listdir(chart_dir)
                    if filename.replace('.png', '').replace('.jpg', '').replace('.pdf', '') in f
                ]
                if matches:
                    chart_path = os.path.join(chart_dir, matches[0])
                else:
                    return jsonify({
                        'status': 'error',
                        'message': f'Chart not found: {filename}'
                    }), 404
            
            # Determine mimetype based on extension
            mimetype = 'image/png'
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                mimetype = 'image/jpeg'
            elif filename.endswith('.pdf'):
                mimetype = 'application/pdf'
            
            return send_file(
                chart_path,
                mimetype=mimetype,
                as_attachment=False
            )
            
        except Exception as e:
            logger.error(f"Error in get_chart: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @charts_bp.route('/generate/risk-distribution', methods=['POST'])
    @requires_auth
    def generate_risk_distribution():
        """
        Generate a risk distribution chart.
        
        ---
        definitions:
          GenerateChartRequest:
            type: object
            properties:
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
              title:
                type: string
                description: Custom chart title
        responses:
          200:
            description: Chart generated successfully
        """
        try:
            payload = request.get_json(silent=True) or {}
            db_name = payload.get('db_name', 'mimic').lower()
            title = payload.get('title', f'Risk Distribution - {db_name.upper()}')
            
            # Get recent scores
            records = db.get_recent_scores(db_name=db_name, limit=1000)
            
            if not records:
                return jsonify({
                    'status': 'error',
                    'message': 'No data available for chart generation'
                }), 404
            
            # Generate chart
            chart_path = chart_generator.generate_risk_distribution_chart(
                records=records,
                title=title,
                db_name=db_name
            )
            
            return jsonify({
                'status': 'success',
                'chart_path': chart_path,
                'filename': os.path.basename(chart_path)
            }), 200
            
        except Exception as e:
            logger.error(f"Error in generate_risk_distribution: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @charts_bp.route('/generate/score-distribution', methods=['POST'])
    @requires_auth
    def generate_score_distribution():
        """
        Generate a score distribution histogram.
        
        ---
        definitions:
          GenerateChartRequest:
            type: object
            properties:
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
              bins:
                type: integer
                default: 30
                description: Number of histogram bins
              title:
                type: string
                description: Custom chart title
        responses:
          200:
            description: Chart generated successfully
        """
        try:
            payload = request.get_json(silent=True) or {}
            db_name = payload.get('db_name', 'mimic').lower()
            bins = payload.get('bins', 30)
            title = payload.get('title', f'Score Distribution - {db_name.upper()}')
            
            # Get recent scores
            records = db.get_recent_scores(db_name=db_name, limit=1000)
            
            if not records:
                return jsonify({
                    'status': 'error',
                    'message': 'No data available for chart generation'
                }), 404
            
            # Generate chart
            chart_path = chart_generator.generate_score_distribution_chart(
                records=records,
                title=title,
                db_name=db_name,
                bins=bins
            )
            
            return jsonify({
                'status': 'success',
                'chart_path': chart_path,
                'filename': os.path.basename(chart_path)
            }), 200
            
        except Exception as e:
            logger.error(f"Error in generate_score_distribution: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @charts_bp.route('/generate/combined', methods=['POST'])
    @requires_auth
    def generate_combined_chart():
        """
        Generate a combined chart with risk distribution and score histogram.
        
        ---
        definitions:
          GenerateCombinedChartRequest:
            type: object
            properties:
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
              title:
                type: string
                description: Custom chart title
        responses:
          200:
            description: Combined chart generated successfully
        """
        try:
            payload = request.get_json(silent=True) or {}
            db_name = payload.get('db_name', 'mimic').lower()
            title = payload.get('title', f'PRISM Analysis - {db_name.upper()}')
            
            # Get recent scores
            records = db.get_recent_scores(db_name=db_name, limit=1000)
            
            if not records:
                return jsonify({
                    'status': 'error',
                    'message': 'No data available for chart generation'
                }), 404
            
            # Generate combined chart
            chart_path = chart_generator.generate_combined_chart(
                records=records,
                title=title,
                db_name=db_name
            )
            
            return jsonify({
                'status': 'success',
                'chart_path': chart_path,
                'filename': os.path.basename(chart_path)
            }), 200
            
        except Exception as e:
            logger.error(f"Error in generate_combined_chart: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(charts_bp)
