"""
Patients Blueprint for PRISM API

Handles patient listing, history, statistics, submissions, and sequence prediction endpoints.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import logging

from api.auth import requires_auth

logger = logging.getLogger(__name__)

patients_bp = Blueprint('patients', __name__, url_prefix='/api/v1/patients')


def register_patients_routes(app, db, ml_scorer, model_loader):
    """
    Register patient-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
        model_loader: ModelLoader instance
    """

    @patients_bp.route('', methods=['GET'])
    @requires_auth
    def get_patients():
        """
        List all patients with pagination and search.
        
        ---
        parameters:
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
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
          - name: search
            in: query
            type: string
            description: Search in patient IDs
        responses:
          200:
            description: List of patients with pagination info
        """
        try:
            page = request.args.get('page', 1, type=int)
            page_size = request.args.get('page_size', 50, type=int)
            db_name = request.args.get('db_name', 'mimic').lower()
            search = request.args.get('search')
            
            result = db.list_patients(
                db_name=db_name,
                page=page,
                page_size=page_size,
                search=search
            )
            
            return jsonify({
                'status': 'success',
                'patients': result.get('patients', []),
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': result.get('total_patients', 0),
                    'pages': result.get('total_pages', 0)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_patients: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @patients_bp.route('/<sample_id>/history', methods=['GET'])
    @requires_auth
    def get_patient_history(sample_id: str):
        """
        Get the complete history of predictions for a patient.
        
        ---
        parameters:
          - name: sample_id
            in: path
            required: true
            type: string
            description: Patient sample identifier
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
          - name: limit
            in: query
            type: integer
            default: 100
            description: Maximum number of records to return
        responses:
          200:
            description: Patient prediction history
          404:
            description: Patient not found
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            limit = request.args.get('limit', 100, type=int)
            
            history = db.get_patient_history(
                db_name=db_name,
                sample_id=sample_id,
                limit=limit
            )
            
            if not history:
                return jsonify({
                    'status': 'error',
                    'message': f'No history found for patient: {sample_id}'
                }), 404
            
            return jsonify({
                'status': 'success',
                'sample_id': sample_id,
                'history': history
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_patient_history: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @patients_bp.route('/<sample_id>/statistics', methods=['GET'])
    @requires_auth
    def get_patient_statistics(sample_id: str):
        """
        Get aggregated statistics for a patient.
        
        ---
        parameters:
          - name: sample_id
            in: path
            required: true
            type: string
            description: Patient sample identifier
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
        responses:
          200:
            description: Patient statistics
          404:
            description: Patient not found
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            
            stats = db.get_patient_statistics(
                db_name=db_name,
                sample_id=sample_id
            )
            
            if not stats:
                return jsonify({
                    'status': 'error',
                    'message': f'No statistics found for patient: {sample_id}'
                }), 404
            
            return jsonify({
                'status': 'success',
                'sample_id': sample_id,
                'statistics': stats
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_patient_statistics: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @patients_bp.route('/<sample_id>/submissions', methods=['GET'])
    @requires_auth
    def get_patient_submissions(sample_id: str):
        """
        Get all submissions for a specific patient.
        
        ---
        parameters:
          - name: sample_id
            in: path
            required: true
            type: string
            description: Patient sample identifier
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
        responses:
          200:
            description: List of submissions for the patient
          404:
            description: No submissions found
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            
            submissions = db.get_patient_submissions(
                db_name=db_name,
                sample_id=sample_id
            )
            
            if not submissions:
                return jsonify({
                    'status': 'error',
                    'message': f'No submissions found for patient: {sample_id}'
                }), 404
            
            return jsonify({
                'status': 'success',
                'sample_id': sample_id,
                'submissions': submissions
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_patient_submissions: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @patients_bp.route('/<sample_id>/predict-sequence', methods=['GET'])
    @requires_auth
    def predict_patient_sequence(sample_id: str):
        """
        Run sequence prediction on a patient's time-series data using LSTM model.
        
        ---
        parameters:
          - name: sample_id
            in: path
            required: true
            type: string
            description: Patient sample identifier
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
          - name: target_len
            in: query
            type: integer
            default: 24
            description: Target sequence length for sliding window
          - name: stride
            in: query
            type: integer
            default: 6
            description: Stride for sliding window
        responses:
          200:
            description: Sequence prediction results
          404:
            description: Patient not found or insufficient data
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            target_len = request.args.get('target_len', 24, type=int)
            stride = request.args.get('stride', 6, type=int)
            
            # Get patient history
            history = db.get_patient_history(
                db_name=db_name,
                sample_id=sample_id,
                limit=1000
            )
            
            if not history or len(history) < target_len:
                return jsonify({
                    'status': 'error',
                    'message': f'Insufficient data for sequence prediction. Need at least {target_len} records, found {len(history) if history else 0}'
                }), 404
            
            # Sort by timestamp
            history.sort(key=lambda x: x.get('timestamp', ''))
            
            # Prepare records for sequence prediction
            records = []
            for h in history:
                record = {
                    'sample_id': sample_id,
                    'timestamp': h.get('timestamp', ''),
                }
                # Add all clinical features
                for key, value in h.items():
                    if key not in ['sample_id', 'timestamp', '_id', 'score', 'class', 'updated_at']:
                        record[key] = value
                records.append(record)
            
            # Run sequence prediction
            result = model_loader.predict_sequence(
                records=records,
                model_name=db_name,
                target_len=target_len,
                stride=stride
            )
            
            return jsonify({
                'status': 'success',
                'sample_id': sample_id,
                'windows': result.get('windows', []),
                'valid_window': result.get('valid_window'),
                'summary': result.get('summary', {}),
                'metadata': {
                    'total_records': len(records),
                    'target_len': target_len,
                    'stride': stride,
                    'model': db_name
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error in predict_patient_sequence: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(patients_bp)
