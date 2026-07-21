"""API Blueprint routes for PRISM."""
from flask import Blueprint, request, jsonify
from datetime import datetime, timezone
from src.api.auth import requires_auth
from src.utils.field_mappings import MIMIC_FEATURES, SEPSIEXP_FEATURES

data_bp = Blueprint('data', __name__, url_prefix='/api/v1/data')
csv_bp = Blueprint('csv', __name__, url_prefix='/api/v1')
submissions_bp = Blueprint('submissions', __name__, url_prefix='/api/v1/submissions')
patients_bp = Blueprint('patients', __name__, url_prefix='/api/v1/patients')
charts_bp = Blueprint('charts', __name__, url_prefix='/api/v1/charts')
stats_bp = Blueprint('stats', __name__, url_prefix='/api/v1')


def register_all_routes(app, db, ml_scorer, model_loader, inference_engine, chart_generator):

    @data_bp.route('', methods=['POST'])
    @requires_auth
    def submit_data():
        payload = request.get_json()
        if not payload or 'data' not in payload:
            return jsonify({'status': 'error', 'message': 'Missing data field'}), 400
        db_name = payload.get('db_name', 'sepsiexp')
        data = payload['data']
        results = []
        for record in data:
            score_result = ml_scorer.predict(record, db_name)
            db.save_record(db_name, record)
            results.append(score_result)
        return jsonify({'status': 'success', 'results': results, 'count': len(results)}), 200

    @data_bp.route('', methods=['GET'])
    @requires_auth
    def get_data():
        db_name = request.args.get('db_name', 'sepsiexp')
        sample_id = request.args.get('sample_id')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        records = db.get_records(db_name, sample_id, limit, offset)
        return jsonify({'status': 'success', 'records': records, 'count': len(records)}), 200

    @data_bp.route('/<int:record_id>', methods=['PUT'])
    @requires_auth
    def update_data(record_id):
        payload = request.get_json()
        db_name = payload.get('db_name', 'sepsiexp')
        score = payload.get('score')
        class_label = payload.get('class')
        if score is not None:
            db.update_score(db_name, record_id, score, class_label)
        return jsonify({'status': 'success', 'message': 'Record updated'}), 200

    @csv_bp.route('/predict-csv', methods=['POST'])
    @requires_auth
    def predict_csv():
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
        file = request.files['file']
        db_name = request.form.get('db_name', 'sepsiexp')
        target_len = int(request.form.get('target_len', 24))
        stride = int(request.form.get('stride', 6))
        try:
            import pandas as pd
            csv_data = pd.read_csv(file)
            result = inference_engine.predict_from_csv(csv_data, db_name, target_len, stride)
            return jsonify({'status': 'success', **result}), 200
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @csv_bp.route('/csv-template', methods=['GET'])
    @requires_auth
    def csv_template():
        db_name = request.args.get('db_name', 'sepsiexp')
        features = MIMIC_FEATURES if db_name == 'mimic' else SEPSIEXP_FEATURES
        from io import StringIO
        import pandas as pd
        df = pd.DataFrame(columns=features)
        csv_str = df.to_csv(index=False)
        return csv_str, 200, {'Content-Type': 'text/csv',
                              'Content-Disposition': f'attachment; filename={db_name}_template.csv'}

    @submissions_bp.route('', methods=['GET'])
    @requires_auth
    def get_submissions():
        db_name = request.args.get('db_name')
        submission_id = request.args.get('submission_id')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        submissions = db.get_submissions(db_name, submission_id, date_from, date_to, limit, offset)
        return jsonify({'status': 'success', 'submissions': submissions, 'count': len(submissions)}), 200

    @patients_bp.route('/<sample_id>', methods=['GET'])
    @requires_auth
    def get_patient(sample_id):
        db_name = request.args.get('db_name', 'sepsiexp')
        records = db.get_records(db_name, sample_id)
        return jsonify({'status': 'success', 'sample_id': sample_id, 'records': records}), 200

    @patients_bp.route('/<sample_id>/sequence', methods=['POST'])
    @requires_auth
    def patient_sequence(sample_id):
        payload = request.get_json()
        db_name = payload.get('db_name', 'sepsiexp')
        records = payload.get('records', [])
        target_len = int(payload.get('target_len', 24))
        stride = int(payload.get('stride', 6))
        result = model_loader.predict_sequence(records, db_name, target_len, stride)
        return jsonify({'status': 'success', 'sample_id': sample_id, **result}), 200

    @charts_bp.route('/score-distribution', methods=['GET'])
    @requires_auth
    def score_distribution():
        db_name = request.args.get('db_name', 'sepsiexp')
        records = db.get_records(db_name, limit=1000)
        scores = [r['score'] for r in records if r.get('score') is not None]
        if not scores:
            return jsonify({'status': 'error', 'message': 'No scores available'}), 404
        chart = chart_generator.generate_score_distribution(scores)
        return jsonify({'status': 'success', 'chart': chart}), 200

    @charts_bp.route('/timeline', methods=['GET'])
    @requires_auth
    def timeline():
        db_name = request.args.get('db_name', 'sepsiexp')
        sample_id = request.args.get('sample_id')
        if not sample_id:
            return jsonify({'status': 'error', 'message': 'sample_id required'}), 400
        records = db.get_records(db_name, sample_id)
        scores = [r['score'] for r in records if r.get('score') is not None]
        if not scores:
            return jsonify({'status': 'error', 'message': 'No scores available'}), 404
        chart = chart_generator.generate_timeline_chart([], scores)
        return jsonify({'status': 'success', 'chart': chart}), 200

    @stats_bp.route('/stats', methods=['GET'])
    @requires_auth
    def get_stats():
        db_name = request.args.get('db_name', 'sepsiexp')
        count = db.get_record_count(db_name)
        return jsonify({'status': 'success', 'db_name': db_name, 'record_count': count}), 200

    @stats_bp.route('/models', methods=['GET'])
    @requires_auth
    def get_models():
        models = inference_engine.list_models()
        return jsonify({'status': 'success', 'models': models}), 200

    @stats_bp.route('/fields/ranges', methods=['GET'])
    @requires_auth
    def get_field_ranges():
        db_name = request.args.get('db_name', 'sepsiexp')
        features = MIMIC_FEATURES if db_name == 'mimic' else SEPSIEXP_FEATURES
        ranges = {f: {'min': 0.0, 'max': 100.0} for f in features}
        return jsonify({'status': 'success', 'db_name': db_name, 'ranges': ranges}), 200

    # Register all blueprints
    app.register_blueprint(data_bp)
    app.register_blueprint(csv_bp)
    app.register_blueprint(submissions_bp)
    app.register_blueprint(patients_bp)
    app.register_blueprint(charts_bp)
    app.register_blueprint(stats_bp)