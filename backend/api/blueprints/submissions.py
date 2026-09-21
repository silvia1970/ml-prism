"""
Submissions Blueprint for PRISM API

Handles submission listing, details, deletion, and recalculation endpoints.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
import logging

from api.auth import requires_auth

logger = logging.getLogger(__name__)

submissions_bp = Blueprint('submissions', __name__, url_prefix='/api/v1/submissions')


def register_submissions_routes(app, db, ml_scorer):
    """
    Register submission-related routes with the Flask app.
    
    Args:
        app: Flask application instance
        db: Database instance
        ml_scorer: MLScorer instance
    """

    @submissions_bp.route('', methods=['GET'])
    @requires_auth
    def get_submissions():
        """
        List all submissions with pagination.
        
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
            enum: [mimic, sepsiexp, all]
            description: Filter by dataset
          - name: search
            in: query
            type: string
            description: Search in submission IDs
        responses:
          200:
            description: List of submissions
        """
        try:
            page = request.args.get('page', 1, type=int)
            page_size = request.args.get('page_size', 50, type=int)
            db_name = request.args.get('db_name', 'all').lower()
            search = request.args.get('search')
            
            result = db.get_submissions(
                page=page,
                page_size=page_size,
                db_filter=db_name,
                submission_id=search
            )
            
            return jsonify({
                'status': 'success',
                'submissions': result.get('submissions', []),
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total': result.get('total_submissions', 0),
                    'pages': result.get('page_total', 0)
                }
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_submissions: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @submissions_bp.route('/<submission_id>', methods=['GET'])
    @requires_auth
    def get_submission_details(submission_id):
        """
        Get details for a specific submission.
        
        ---
        parameters:
          - name: submission_id
            in: path
            required: true
            type: string
            description: Submission identifier
        responses:
          200:
            description: Submission details
          404:
            description: Submission not found
        """
        try:
            submission = db.find_submission(submission_id)
            
            if not submission:
                return jsonify({
                    'status': 'error',
                    'message': f'Submission not found: {submission_id}'
                }), 404
            
            # Get records for this submission
            db_name = submission.get('db_name', 'mimic')
            records = db.get_submission_records(db_name=db_name, submission_id=submission_id)
            
            return jsonify({
                'status': 'success',
                'submission': submission,
                'records': records
            }), 200
            
        except Exception as e:
            logger.error(f"Error in get_submission_details: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @submissions_bp.route('/<submission_id>', methods=['DELETE'])
    @requires_auth
    def delete_submission(submission_id):
        """
        Delete a submission and its associated records.
        
        ---
        parameters:
          - name: submission_id
            in: path
            required: true
            type: string
            description: Submission identifier
        responses:
          200:
            description: Submission deleted successfully
          404:
            description: Submission not found
        """
        try:
            submission = db.find_submission(submission_id)
            
            if not submission:
                return jsonify({
                    'status': 'error',
                    'message': f'Submission not found: {submission_id}'
                }), 404
            
            db_name = submission.get('db_name', 'mimic')

            # Delegate all deletion logic to the database layer
            deleted_count = db.delete_submission(submission_id=submission_id, db_name=db_name)
            
            return jsonify({
                'status': 'success',
                'message': f'Deleted submission {submission_id} and {deleted_count} records'
            }), 200
            
        except Exception as e:
            logger.error(f"Error in delete_submission: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    @submissions_bp.route('/<submission_id>/recalculate', methods=['POST'])
    @requires_auth
    def recalculate_submission(submission_id):
        """
        Recalculate predictions for all records in a submission.
        
        ---
        parameters:
          - name: submission_id
            in: path
            required: true
            type: string
            description: Submission identifier
        responses:
          200:
            description: Predictions recalculated successfully
          404:
            description: Submission not found
        """
        try:
            submission = db.find_submission(submission_id)
            
            if not submission:
                return jsonify({
                    'status': 'error',
                    'message': f'Submission not found: {submission_id}'
                }), 404
            
            db_name = submission.get('db_name', 'mimic')
            records = db.get_submission_records(db_name=db_name, submission_id=submission_id)
            
            results = []
            errors = []
            
            for record in records:
                try:
                    prediction = ml_scorer.predict(db_name=db_name, record=record)
                    db.update_record_score(
                        db_name=db_name,
                        record_id=record.get('_id'),
                        score=prediction['score'],
                        class_label=prediction['class']
                    )
                    results.append({
                        'sample_id': record.get('sample_id'),
                        'score': prediction['score'],
                        'class': prediction['class']
                    })
                except Exception as e:
                    errors.append({
                        'sample_id': record.get('sample_id'),
                        'error': str(e)
                    })
            
            return jsonify({
                'status': 'success',
                'submission_id': submission_id,
                'recalculated': len(results),
                'errors': len(errors),
                'results': results,
                'error_details': errors if errors else None
            }), 200
            
        except Exception as e:
            logger.error(f"Error in recalculate_submission: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

    app.register_blueprint(submissions_bp)
