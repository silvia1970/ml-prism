#!/usr/bin/env python3
"""
Comprehensive test suite for all PRISM API endpoints.
Uses Flask test client for isolated testing without a running server.
Run: cd ml-prism && AUTH0_ENABLED=false python -m pytest tests/backend/test_all_apis.py -v
"""

import os
os.environ['AUTH0_ENABLED'] = 'false'

import sys
import json
import uuid
import tempfile
import csv
from datetime import datetime, timezone
from io import BytesIO

import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from api.app import app
from api.validators import validate_submission, SCHEMAS, validate_record
from api.database import Database


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        yield client


@pytest.fixture
def temp_db():
    data_dir = tempfile.mkdtemp()
    db = Database(data_dir=data_dir)
    yield db, data_dir


@pytest.fixture
def valid_mimic_record():
    return {
        "sample_id": f"MIM-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meanbp": 91.0, "resprate": 18.0, "heartrate": 88.0,
        "spo2_pulsoxy": 97.0, "tempc": 37.1, "cardiacoutput": 5.5,
        "o2flow": 2.0, "fio2": 0.40,
        "albumin": 3.5, "bands": 5.0, "bicarbonate": 24.0,
        "bilirubin": 0.8, "creatinine": 1.1, "chloride": 102.0,
        "glucose": 105.0, "hemoglobin": 13.5, "lactate": 1.8,
        "platelet": 245.0, "potassium": 4.1, "ptt": 35.0,
        "inr": 1.2, "sodium": 138.0, "wbc": 9.2,
        "creatinekinase": 150.0, "ck_mb": 8.0,
        "fibrinogen": 350.0, "ldh": 220.0, "magnesium": 2.1,
        "calcium_free": 1.15, "po2_bloodgas": 95.0,
        "ph_bloodgas": 7.38, "pco2_bloodgas": 42.0,
        "so2_bloodgas": 96.0, "troponin_t": 0.08
    }


@pytest.fixture
def valid_sepsiexp_record():
    return {
        "sample_id": f"SEP-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "age": 65.0,
        "heartrate": 88.0, "svri": 1800.0, "meanbp": 91.0,
        "hearttimevolume": 5.5, "oxygensaturation": 97.0,
        "deltatemp": 0.5, "oxygenationsaturation": 75.0,
        "lactate": 1.8, "creatinine": 1.1, "bilirubin": 0.8,
        "sodium": 138.0, "potassium": 4.1, "hemoglobin": 13.5,
        "chloride": 102.0, "leukocytes": 9.2, "bicarbonate": 24.0,
        "pancreaticlipase": 150.0, "bun": 22.0, "pct": 2.5,
        "buncreatinineratio": 20.0, "ast": 45.0, "crp": 80.0,
        "respiratoryminutevolume": 8.0, "fio2": 0.40,
        "arterialph": 7.38, "pa_o2": 95.0
    }


@pytest.fixture
def valid_mimic_payload(valid_mimic_record):
    return {
        "db": "mimic",
        "submission_id": f"TEST-MIMIC-{uuid.uuid4().hex[:8]}",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "client_version": "test-1.0.0",
        "schema_version": "3.0",
        "meta": {"source": "test_suite", "rows": 1},
        "data": [valid_mimic_record]
    }


@pytest.fixture
def valid_sepsiexp_payload(valid_sepsiexp_record):
    return {
        "db": "sepsiexp",
        "submission_id": f"TEST-SEP-{uuid.uuid4().hex[:8]}",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "client_version": "test-1.0.0",
        "schema_version": "3.0",
        "meta": {"source": "test_suite", "rows": 1},
        "data": [valid_sepsiexp_record]
    }


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    def test_health_root(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data

    def test_health_api_v1(self, client):
        resp = client.get('/api/v1/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'healthy'


# =============================================================================
# Data Submission Tests (POST /api/v1/data)
# =============================================================================

class TestDataSubmission:
    def test_submit_valid_mimic(self, client, valid_mimic_payload):
        resp = client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['db_name'] == 'mimic'
        assert data['processed'] == 1
        assert len(data['results']) == 1
        result = data['results'][0]
        assert 'score' in result
        assert 'class' in result
        assert result['class'] in ['low_risk', 'moderate_risk', 'high_risk']
        assert 0.0 <= result['score'] <= 1.0

    def test_submit_valid_sepsiexp(self, client, valid_sepsiexp_payload):
        resp = client.post('/api/v1/data', json=valid_sepsiexp_payload, content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['db_name'] == 'sepsiexp'
        assert data['processed'] == 1

    def test_submit_empty_body(self, client):
        resp = client.post('/api/v1/data', content_type='application/json')
        assert resp.status_code == 400

    def test_submit_missing_db(self, client):
        payload = {
            "submission_id": "TEST-001",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": []
        }
        resp = client.post('/api/v1/data', json=payload, content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['status'] == 'error'

    def test_submit_invalid_db(self, client):
        payload = {
            "db": "invalid_db",
            "submission_id": "TEST-002",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": [{"sample_id": "X"}]
        }
        resp = client.post('/api/v1/data', json=payload, content_type='application/json')
        assert resp.status_code == 400

    def test_submit_missing_required_fields(self, client):
        payload = {
            "db": "mimic",
            "submission_id": "TEST-003",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": [{"sample_id": "INCOMPLETE-001"}]
        }
        resp = client.post('/api/v1/data', json=payload, content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['status'] == 'error'

    def test_submit_multiple_records(self, client, valid_mimic_record):
        records = [
            {**valid_mimic_record, "sample_id": f"MULTI-{i}"}
            for i in range(3)
        ]
        payload = {
            "db": "mimic",
            "submission_id": f"TEST-MULTI-{uuid.uuid4().hex[:8]}",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": records
        }
        resp = client.post('/api/v1/data', json=payload, content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['processed'] == 3
        assert len(data['results']) == 3

    def test_submit_invalid_field_type(self, client):
        payload = {
            "db": "mimic",
            "submission_id": "TEST-TYPE",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": [{
                "sample_id": "TYPE-ERR",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "meanbp": "not_a_number",
            }]
        }
        resp = client.post('/api/v1/data', json=payload, content_type='application/json')
        assert resp.status_code == 400


# =============================================================================
# CSV Upload Tests (POST /api/v1/upload-csv)
# =============================================================================

class TestCSVUpload:
    def test_upload_no_file(self, client):
        resp = client.post('/api/v1/upload-csv', data={'db_name': 'mimic'})
        assert resp.status_code == 400

    def test_upload_invalid_db(self, client):
        csv_data = b"sample_id,meanbp\nTEST-1,90"
        resp = client.post('/api/v1/upload-csv',
                           data={
                               'file': (BytesIO(csv_data), 'test.csv'),
                               'db_name': 'invalid_db'
                           },
                           content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_upload_valid_mimic_csv(self, client):
        headers = ['sample_id', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                    'tempc', 'cardiacoutput', 'o2flow', 'fio2', 'albumin', 'bands',
                    'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
                    'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt', 'inr',
                    'sodium', 'wbc', 'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh',
                    'magnesium', 'calcium_free', 'po2_bloodgas', 'ph_bloodgas',
                    'pco2_bloodgas', 'so2_bloodgas', 'troponin_t']
        row = ['CSV-TEST-001'] + [str(v) for v in [
            91, 18, 88, 97, 37.1, 5.5, 2, 0.40, 3.5, 5, 24, 0.8, 1.1, 102, 105,
            13.5, 1.8, 245, 4.1, 35, 1.2, 138, 9.2, 150, 8, 350, 220, 2.1, 1.15,
            95, 7.38, 42, 96, 0.08
        ]]
        csv_content = '\n'.join([','.join(headers), ','.join(row)])
        resp = client.post('/api/v1/upload-csv',
                           data={
                               'file': (BytesIO(csv_content.encode()), 'test_mimic.csv'),
                               'db_name': 'mimic'
                           },
                           content_type='multipart/form-data')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['processed'] >= 1


# =============================================================================
# CSV Template Tests (GET /api/v1/csv-template/<db_name>)
# =============================================================================

class TestCSVTemplate:
    def test_download_mimic_template(self, client):
        resp = client.get('/api/v1/csv-template/mimic')
        assert resp.status_code == 200
        assert 'csv' in resp.content_type or 'text' in resp.content_type

    def test_download_sepsiexp_template(self, client):
        resp = client.get('/api/v1/csv-template/sepsiexp')
        assert resp.status_code == 200
        assert 'csv' in resp.content_type or 'text' in resp.content_type

    def test_download_invalid_db_template(self, client):
        resp = client.get('/api/v1/csv-template/invalid_db')
        assert resp.status_code == 400


# =============================================================================
# Data Retrieval Tests (GET /api/v1/data/<sample_id>)
# =============================================================================

class TestDataRetrieval:
    def test_get_existing_sample(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        resp = client.get(f'/api/v1/data/{sample_id}?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'record' in data

    def test_get_nonexistent_sample(self, client):
        resp = client.get('/api/v1/data/NONEXISTENT-001?db_name=mimic')
        assert resp.status_code == 404
        data = resp.get_json()
        assert data['status'] == 'error'


# =============================================================================
# Data Update Tests (PATCH /api/v1/data/<sample_id>)
# =============================================================================

class TestDataUpdate:
    def test_update_existing_sample(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        updates_payload = {"db_name": "mimic", "updates": {"heartrate": 75.0, "lactate": 1.5}}
        resp = client.patch(f'/api/v1/data/{sample_id}?db_name=mimic',
                            json=updates_payload, content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'record' in data

    def test_update_nonexistent_sample(self, client):
        resp = client.patch('/api/v1/data/NONEXISTENT?db_name=mimic',
                            json={"db_name": "mimic", "updates": {"heartrate": 75.0}}, content_type='application/json')
        assert resp.status_code == 404

    def test_update_empty_body(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        resp = client.patch(f'/api/v1/data/{sample_id}?db_name=mimic',
                            content_type='application/json')
        assert resp.status_code == 400


# =============================================================================
# Statistics Tests (GET /api/v1/stats)
# =============================================================================

class TestStatistics:
    def test_get_stats(self, client):
        resp = client.get('/api/v1/stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'counts' in data


# =============================================================================
# Submissions Tests
# =============================================================================

class TestSubmissions:
    def test_get_submissions_list(self, client):
        resp = client.get('/api/v1/submissions')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'submissions' in data
        assert 'pagination' in data

    def test_get_submissions_with_pagination(self, client):
        resp = client.get('/api/v1/submissions?page=1&page_size=10')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['pagination']['page'] == 1
        assert data['pagination']['page_size'] == 10

    def test_get_submissions_filter_db(self, client):
        resp = client.get('/api/v1/submissions?db=mimic')
        assert resp.status_code == 200

    def test_get_submission_details(self, client, valid_mimic_payload):
        resp_submit = client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        assert resp_submit.status_code == 200
        submit_data = resp_submit.get_json()
        submission_id = submit_data['submission_id']
        resp = client.get(f'/api/v1/submissions/{submission_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['submission']['submission_id'] == submission_id

    def test_get_nonexistent_submission(self, client):
        resp = client.get('/api/v1/submissions/NONEXISTENT-SUBMISSION')
        assert resp.status_code == 404

    def test_delete_submission(self, client, valid_mimic_payload):
        resp_submit = client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        assert resp_submit.status_code == 200
        submission_id = resp_submit.get_json()['submission_id']
        resp = client.delete(f'/api/v1/submissions/{submission_id}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'

    def test_recalculate_submission(self, client, valid_mimic_payload):
        resp_submit = client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        assert resp_submit.status_code == 200
        submission_id = resp_submit.get_json()['submission_id']
        resp = client.post(f'/api/v1/submissions/{submission_id}/recalculate',
                          json={}, content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'


# =============================================================================
# Scores Tests (GET /api/v1/scores)
# =============================================================================

class TestScores:
    def test_get_scores_mimic(self, client):
        resp = client.get('/api/v1/scores?db_name=mimic&limit=10')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'scores' in data

    def test_get_scores_sepsiexp(self, client):
        resp = client.get('/api/v1/scores?db_name=sepsiexp&limit=10')
        assert resp.status_code == 200


# =============================================================================
# Patients Tests
# =============================================================================

class TestPatients:
    def test_list_patients(self, client):
        resp = client.get('/api/v1/patients?db_name=mimic&page=1&page_size=10')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'

    def test_patient_history(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        resp = client.get(f'/api/v1/patients/{sample_id}/history?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert data['sample_id'] == sample_id

    def test_patient_statistics(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        resp = client.get(f'/api/v1/patients/{sample_id}/statistics?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'

    def test_patient_submissions(self, client, valid_mimic_payload):
        client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        resp = client.get(f'/api/v1/patients/{sample_id}/submissions?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'


# =============================================================================
# Records Tests (GET /api/v1/records)
# =============================================================================

class TestRecords:
    def test_list_records(self, client):
        resp = client.get('/api/v1/records?db_name=mimic&page=1&page_size=10')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'


# =============================================================================
# Critical Statistics Tests (GET /api/v1/critical-statistics)
# =============================================================================

class TestCriticalStatistics:
    def test_get_critical_statistics(self, client):
        resp = client.get('/api/v1/critical-statistics?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'


# =============================================================================
# Field Ranges Tests (GET /api/v1/fields/ranges)
# =============================================================================

class TestFieldRanges:
    def test_get_mimic_field_ranges(self, client):
        resp = client.get('/api/v1/fields/ranges?db_name=mimic')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'fields' in data

    def test_get_sepsiexp_field_ranges(self, client):
        resp = client.get('/api/v1/fields/ranges?db_name=sepsiexp')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'


# =============================================================================
# Models Tests (GET /api/v1/models)
# =============================================================================

class TestModels:
    @pytest.mark.skip(reason="torch model loading may fail in test environment")
    def test_list_models(self, client):
        resp = client.get('/api/v1/models')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'
        assert 'models' in data


# =============================================================================
# Error Handler Tests
# =============================================================================

class TestErrorHandlers:
    def test_404_handler(self, client):
        resp = client.get('/api/v1/nonexistent-route')
        assert resp.status_code == 404
        data = resp.get_json()
        assert data['error_type'] == 'NotFound'

    def test_method_not_allowed(self, client):
        resp = client.delete('/health')
        assert resp.status_code == 405


# =============================================================================
# Validation Module Tests (Unit Tests)
# =============================================================================

class TestValidators:
    def test_validate_valid_mimic_submission(self, valid_mimic_payload):
        result = validate_submission(valid_mimic_payload)
        assert result['valid'] is True

    def test_validate_valid_sepsiexp_submission(self, valid_sepsiexp_payload):
        result = validate_submission(valid_sepsiexp_payload)
        assert result['valid'] is True

    def test_validate_missing_db(self):
        result = validate_submission({"data": []})
        assert result['valid'] is False

    def test_validate_invalid_db(self):
        result = validate_submission({"db": "invalid", "data": []})
        assert result['valid'] is False

    def test_validate_empty_data(self):
        result = validate_submission({
            "db": "mimic",
            "submission_id": "TEST",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": []
        })
        assert result['valid'] is False

    def test_validate_optional_alt_field_sepsiexp(self):
        payload = {
            "db": "sepsiexp",
            "submission_id": "TEST-ALT",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "data": [{
                "sample_id": "ALT-TEST",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "age": 65.0,
                "heartrate": 88.0, "svri": 1800.0, "meanbp": 91.0,
                "hearttimevolume": 5.5, "oxygensaturation": 97.0,
                "deltatemp": 0.5, "oxygenationsaturation": 75.0,
                "lactate": 1.8, "creatinine": 1.1, "bilirubin": 0.8,
                "sodium": 138.0, "potassium": 4.1, "hemoglobin": 13.5,
                "chloride": 102.0, "leukocytes": 9.2, "bicarbonate": 24.0,
                "pancreaticlipase": 150.0, "bun": 22.0, "pct": 2.5,
                "buncreatinineratio": 20.0, "ast": 45.0, "crp": 80.0,
                "respiratoryminutevolume": 8.0, "fio2": 0.40,
                "arterialph": 7.38, "pa_o2": 95.0
            }]
        }
        result = validate_submission(payload)
        assert result['valid'] is True

    def test_schemas_have_correct_field_counts(self):
        assert len(SCHEMAS['mimic']['required_fields']) == 34
        assert len(SCHEMAS['sepsiexp']['required_fields']) == 27

    def test_validate_record_invalid_type(self):
        record = {
            "sample_id": "TYPE-TEST",
            "meanbp": "not_a_number"
        }
        errors = validate_record(record, SCHEMAS['mimic'], 0)
        assert len(errors) > 0


# =============================================================================
# Integration Tests (Full Flow)
# =============================================================================

class TestIntegration:
    def test_full_mimic_workflow(self, client, valid_mimic_payload):
        resp = client.post('/api/v1/data', json=valid_mimic_payload, content_type='application/json')
        assert resp.status_code == 200
        submit_data = resp.get_json()
        sample_id = valid_mimic_payload['data'][0]['sample_id']
        submission_id = submit_data['submission_id']

        resp = client.get(f'/api/v1/data/{sample_id}?db_name=mimic')
        assert resp.status_code == 200

        resp = client.patch(f'/api/v1/data/{sample_id}?db_name=mimic',
                            json={"db_name": "mimic", "updates": {"heartrate": 70.0}}, content_type='application/json')
        assert resp.status_code == 200

        resp = client.get('/api/v1/stats')
        assert resp.status_code == 200

        resp = client.get(f'/api/v1/submissions/{submission_id}')
        assert resp.status_code == 200

    def test_full_sepsiexp_workflow(self, client, valid_sepsiexp_payload):
        resp = client.post('/api/v1/data', json=valid_sepsiexp_payload, content_type='application/json')
        assert resp.status_code == 200
        submit_data = resp.get_json()
        sample_id = valid_sepsiexp_payload['data'][0]['sample_id']
        submission_id = submit_data['submission_id']

        resp = client.get(f'/api/v1/data/{sample_id}?db_name=sepsiexp')
        assert resp.status_code == 200

        resp = client.get(f'/api/v1/patients/{sample_id}/history?db_name=sepsiexp')
        assert resp.status_code == 200

    def test_csv_upload_and_retrieve(self, client):
        headers = ['sample_id', 'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy',
                    'tempc', 'cardiacoutput', 'o2flow', 'fio2', 'albumin', 'bands',
                    'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
                    'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt', 'inr',
                    'sodium', 'wbc', 'creatinekinase', 'ck_mb', 'fibrinogen', 'ldh',
                    'magnesium', 'calcium_free', 'po2_bloodgas', 'ph_bloodgas',
                    'pco2_bloodgas', 'so2_bloodgas', 'troponin_t']
        row = ['INTEGRATION-CSV-001'] + [str(v) for v in [
            91, 18, 88, 97, 37.1, 5.5, 2, 0.40, 3.5, 5, 24, 0.8, 1.1, 102, 105,
            13.5, 1.8, 245, 4.1, 35, 1.2, 138, 9.2, 150, 8, 350, 220, 2.1, 1.15,
            95, 7.38, 42, 96, 0.08
        ]]
        csv_content = '\n'.join([','.join(headers), ','.join(row)])

        resp = client.post('/api/v1/upload-csv',
                           data={
                               'file': (BytesIO(csv_content.encode()), 'integration_test.csv'),
                               'db_name': 'mimic'
                           },
                           content_type='multipart/form-data')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'success'

        resp = client.get('/api/v1/data/INTEGRATION-CSV-001?db_name=mimic')
        assert resp.status_code == 200