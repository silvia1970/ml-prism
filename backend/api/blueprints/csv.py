"""
CSV Blueprint for PRISM API

Handles CSV upload, template download, and prediction endpoints.
"""

from flask import Blueprint, request, jsonify, send_file
from datetime import datetime, timezone
import os
import logging
import tempfile

import pandas as pd

from api.auth import requires_auth
from api.csv_handler import CSVHandler, process_csv_upload
from api.field_mappings import normalize_record_ids, extract_sample_id
from api.blueprints.shared import process_records_batch

logger = logging.getLogger(__name__)

csv_bp = Blueprint('csv', __name__, url_prefix='/api/v1')


def register_csv_routes(app, db, ml_scorer, inference_engine):
    """
    Register CSV-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
        inference_engine: InferenceEngine instance
    """

    @csv_bp.route('/upload-csv', methods=['POST'])
    @requires_auth
    def upload_csv():
        """
        Upload a CSV file for batch prediction.
        
        ---
        definitions:
          UploadCSVRequest:
            type: object
            properties:
              file:
                type: string
                format: binary
                description: CSV file with patient data
              db_name:
                type: string
                enum: [mimic, sepsiexp]
                description: Dataset type
        responses:
          200:
            description: CSV processed and predictions generated successfully
          400:
            description: Validation or processing error
          401:
            description: Unauthorized - invalid or missing JWT token
        """
        try:
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
            
            db_name = request.form.get('db_name', 'mimic').lower()
            
            # Validate file
            is_valid, error_msg = CSVHandler.validate_file(file)
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'message': error_msg
                }), 400
            
            # Generate submission ID
            submission_id = f"CSV_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            
            # Process CSV — parse and validate (signature: file, db_type, submission_id)
            success, result = process_csv_upload(
                file=file,
                db_type=db_name,
                submission_id=submission_id
            )
            
            if not success:
                return jsonify(result), 400
            
            # result['payload'] contains the validated submission data
            payload = result.get('payload', {})
            data_records = payload.get('data', [])
            
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
                'error_details': errors if errors else None,
                'filename': file.filename
            }), 200
            
        except Exception as e:
            logger.error(f"Error in upload_csv: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @csv_bp.route('/csv-template/<db_name>', methods=['GET'])
    @requires_auth
    def download_csv_template(db_name):
        """
        Download a CSV template for the specified dataset.
        
        ---
        parameters:
          - name: db_name
            in: path
            required: true
            type: string
            enum: [mimic, sepsiexp]
            description: Dataset type
        responses:
          200:
            description: CSV template file
          400:
            description: Invalid dataset name
        """
        try:
            if db_name.lower() not in ['mimic', 'sepsiexp']:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid dataset: {db_name}. Use "mimic" or "sepsiexp"'
                }), 400
            
            # Generate CSV template
            csv_content = CSVHandler.create_csv_template(db_name.lower())
            
            # Create temporary file
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, f'prism_template_{db_name}.csv')
            
            with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
                f.write(csv_content)
            
            def cleanup():
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            
            response = send_file(
                tmp_path,
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'prism_template_{db_name}.csv'
            )
            response.call_on_close(cleanup)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in download_csv_template: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @csv_bp.route('/predict-csv', methods=['POST'])
    @requires_auth
    def predict_from_csv_upload():
        """
        Upload a CSV file for prediction without storing records.
        
        ---
        definitions:
          PredictCSVRequest:
            type: object
            properties:
              file:
                type: string
                format: binary
                description: CSV file with patient data
              model_name:
                type: string
                enum: [mimic, sepsisexp]
                description: Model to use for prediction
        responses:
          200:
            description: Predictions generated successfully
          400:
            description: Validation or processing error
        """
        try:
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'message': 'No file provided'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
            
            model_name = request.form.get('model_name', 'mimic').lower()
            
            # Read CSV data
            csv_data = pd.read_csv(file)
            
            # Get prediction
            result = inference_engine.predict_from_csv(
                csv_data=csv_data,
                model_name=model_name
            )
            
            return jsonify(result), 200 if result.get('status') == 'success' else 400
            
        except Exception as e:
            logger.error(f"Error in predict_from_csv_upload: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(csv_bp)
