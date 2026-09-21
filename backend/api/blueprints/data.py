"""
Data Blueprint for PRISM API

Handles data submission, retrieval, and update endpoints.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import logging

from api.auth import requires_auth
from api.validators import validate_submission
from api.field_mappings import extract_sample_id, normalize_record_ids, get_sample_id_field
from api.blueprints.shared import process_records_batch

logger = logging.getLogger(__name__)

data_bp = Blueprint('data', __name__, url_prefix='/api/v1/data')


def register_data_routes(app, db, ml_scorer):
    """
    Register data-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
    """

    @data_bp.route('', methods=['POST'])
    @requires_auth
    def submit_data():
        """
        Submit patient data for ML prediction.
        
        ---
        definitions:
          SubmitDataRequest:
            type: object
            required:
              - data
            properties:
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
              data:
                type: array
                items:
                  type: object
                  description: Patient record with clinical features
        responses:
          200:
            description: Data submitted and predictions generated successfully
          400:
            description: Validation error
          401:
            description: Unauthorized - invalid or missing JWT token
        """
        try:
            payload = request.get_json(silent=True)
            
            if not payload:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid JSON payload'
                }), 400
            
            # Validate submission
            validation_result = validate_submission(payload)
            if not validation_result.get('valid', False):
                return jsonify({
                    'status': 'error',
                    'message': 'Validation failed',
                    'errors': validation_result.get('metadata_errors', []) + validation_result.get('data_errors', [])
                }), 400
            
            db_name = (payload.get('db_name') or payload.get('db', 'mimic')).lower()
            data_records = payload.get('data', [])
            
            if not data_records:
                return jsonify({
                    'status': 'error',
                    'message': 'No data records provided'
                }), 400
            
            # Generate submission ID
            submission_id = f"SUB_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{len(data_records)}"
            
            # Store submission metadata
            db.store_submission(
                submission_id=submission_id,
                db_name=db_name,
                payload=payload,
                status='completed'
            )
            
            # Process each record — store, score, update
            results, errors = process_records_batch(
                records=data_records,
                db_name=db_name,
                submission_id=submission_id,
                db=db,
                ml_scorer=ml_scorer,
            )
            
            return jsonify({
                'status': 'success',
                'submission_id': submission_id,
                'db_name': db_name,
                'total_records': len(data_records),
                'processed': len(results),
                'errors': len(errors),
                'results': results,
                'error_details': errors if errors else None
            }), 200
            
        except Exception as e:
            logger.error(f"Error in submit_data: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @data_bp.route('/<sample_id>', methods=['GET'])
    @requires_auth
    def get_sample(sample_id):
        """
        Retrieve a specific patient record by sample ID.
        
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
          - name: timestamp
            in: query
            type: string
            description: Optional timestamp to retrieve specific version
        responses:
          200:
            description: Patient record retrieved successfully
          404:
            description: Record not found
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            timestamp = request.args.get('timestamp')
            
            record = db.get_record(db_name=db_name, sample_id=sample_id, timestamp=timestamp)
            
            if not record:
                return jsonify({
                    'status': 'error',
                    'message': f'Record not found for sample_id: {sample_id}'
                }), 404
            
            return jsonify({
                'status': 'success',
                'record': record
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_sample: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @data_bp.route('/<sample_id>', methods=['PATCH'])
    @requires_auth
    def update_sample(sample_id):
        """
        Update an existing patient record.
        
        ---
        parameters:
          - name: sample_id
            in: path
            required: true
            type: string
            description: Patient sample identifier
        definitions:
          UpdateSampleRequest:
            type: object
            properties:
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
              updates:
                type: object
                description: Fields to update
        responses:
          200:
            description: Record updated successfully
          404:
            description: Record not found
        """
        try:
            payload = request.get_json(silent=True)
            
            if not payload:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid JSON payload'
                }), 400
            
            db_name = (payload.get('db_name') or payload.get('db', 'mimic')).lower()
            updates = payload.get('updates', {})
            
            if not updates:
                return jsonify({
                    'status': 'error',
                    'message': 'No updates provided'
                }), 400
            
            # Get existing record
            existing = db.get_record(db_name=db_name, sample_id=sample_id)
            
            if not existing:
                return jsonify({
                    'status': 'error',
                    'message': f'Record not found for sample_id: {sample_id}'
                }), 404
            
            # Apply updates
            updated_record = db.update_record(
                db_name=db_name,
                sample_id=sample_id,
                updates=updates
            )
            
            # Recalculate prediction if clinical data changed
            recalculation = None
            clinical_fields_changed = any(
                key not in ['sample_id', 'timestamp', 'updated_at', 'score', 'class']
                for key in updates.keys()
            )
            
            if clinical_fields_changed:
                try:
                    prediction = ml_scorer.predict(db_name=db_name, record=updated_record)
                    db.update_record_score(
                        db_name=db_name,
                        record_id=updated_record.get('_id'),
                        score=prediction['score'],
                        class_label=prediction['class']
                    )
                    recalculation = {
                        'score': prediction['score'],
                        'class': prediction['class']
                    }
                except Exception as e:
                    logger.warning(f"Could not recalculate prediction: {str(e)}")
            
            return jsonify({
                'status': 'success',
                'record': updated_record,
                'recalculation': recalculation
            }), 200
            
        except Exception as e:
            logger.error(f"Error in update_sample: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @data_bp.route('/patient/<patient_id>', methods=['GET'])
    @requires_auth
    def get_sample_by_patient_id(patient_id):
        """
        Retrieve patient records by patient ID.
        
        ---
        parameters:
          - name: patient_id
            in: path
            required: true
            type: string
            description: Patient identifier
          - name: db_name
            in: query
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
          - name: data_timestep
            in: query
            type: string
            description: Optional timestep filter
        responses:
          200:
            description: Patient record(s) retrieved successfully
          404:
            description: No records found for patient
        """
        try:
            db_name = request.args.get('db_name', 'mimic').lower()
            data_timestep = request.args.get('data_timestep')
            
            record = db.get_record_by_patient_id(
                db_name=db_name,
                patient_id=patient_id,
                data_timestep=data_timestep
            )
            
            if not record:
                return jsonify({
                    'status': 'error',
                    'message': f'No records found for patient_id: {patient_id}'
                }), 404
            
            return jsonify({
                'status': 'success',
                'record': record
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_sample_by_patient_id: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(data_bp)
