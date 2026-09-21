"""
Data Validation Module for PRISM API
Validates submissions against MIMIC and SepsisExp schemas
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime
import re


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


# Schema definitions based on PyTorch model requirements
# MIMIC: 34 features required by model mimic_24-6_None.pth
# SEPSIEXP: 29 features required by model sepsisexp_24-6_None_29features.pth
SCHEMAS = {
    'mimic': {
        'db': 'mimic',
        'version': '2.0',
        'required_fields': [
            'meanbp', 'resprate', 'heartrate', 'spo2_pulsoxy', 'tempc',
            'cardiacoutput', 'o2flow', 'fio2', 'albumin', 'bands',
            'bicarbonate', 'bilirubin', 'creatinine', 'chloride', 'glucose',
            'hemoglobin', 'lactate', 'platelet', 'potassium', 'ptt',
            'inr', 'sodium', 'wbc', 'creatinekinase', 'ck_mb',
            'fibrinogen', 'ldh', 'magnesium', 'calcium_free', 'po2_bloodgas',
            'ph_bloodgas', 'pco2_bloodgas', 'so2_bloodgas', 'troponin_t'
        ],
        'variables': {
            # Vitals (required)
            'meanbp': {'type': 'float', 'required': True},
            'resprate': {'type': 'float', 'required': True},
            'heartrate': {'type': 'float', 'required': True},
            'spo2_pulsoxy': {'type': 'float', 'required': True},
            'tempc': {'type': 'float', 'required': True},
            'cardiacoutput': {'type': 'float', 'required': True},
            
            # Respiratory (required)
            'o2flow': {'type': 'float', 'required': True},
            'fio2': {'type': 'float', 'required': True},
            
            # Lab Results (all required)
            'albumin': {'type': 'float', 'required': True},
            'bands': {'type': 'float', 'required': True},
            'bicarbonate': {'type': 'float', 'required': True},
            'bilirubin': {'type': 'float', 'required': True},
            'creatinine': {'type': 'float', 'required': True},
            'glucose': {'type': 'float', 'required': True},
            'hemoglobin': {'type': 'float', 'required': True},
            'inr': {'type': 'float', 'required': True},
            'potassium': {'type': 'float', 'required': True},
            'ptt': {'type': 'float', 'required': True},
            'sodium': {'type': 'float', 'required': True},
            'wbc': {'type': 'float', 'required': True},
            'creatinekinase': {'type': 'float', 'required': True},
            'ck_mb': {'type': 'float', 'required': True},
            'fibrinogen': {'type': 'float', 'required': True},
            'ldh': {'type': 'float', 'required': True},
            'magnesium': {'type': 'float', 'required': True},
            'calcium_free': {'type': 'float', 'required': True},
            'chloride': {'type': 'float', 'required': True},
            'lactate': {'type': 'float', 'required': True},
            'platelet': {'type': 'float', 'required': True},
            'troponin_t': {'type': 'float', 'required': True},
            
            # Blood Gas (all required)
            'po2_bloodgas': {'type': 'float', 'required': True},
            'ph_bloodgas': {'type': 'float', 'required': True},
            'pco2_bloodgas': {'type': 'float', 'required': True},
            'so2_bloodgas': {'type': 'float', 'required': True},
        }
    },
    'sepsiexp': {
        'db': 'sepsiexp',
        'version': '2.0',
        'required_fields': [
            # Patient Info
            'age',
            # Vitals (7 fields)
            'heartrate', 'svri', 'meanbp', 'hearttimevolume',
            'oxygensaturation', 'deltatemp', 'oxygenationsaturation',
            # Lab Results (15 required; 'alt' is optional)
            'lactate', 'creatinine', 'bilirubin', 'sodium', 'potassium',
            'hemoglobin', 'chloride', 'leukocytes', 'bicarbonate',
            'pancreaticlipase', 'bun', 'pct',
            'buncreatinineratio', 'ast', 'crp',
            # Respiratory (2 fields)
            'respiratoryminutevolume', 'fio2',
            # Blood Gas (2 fields)
            'arterialph', 'pa_o2'
        ],
        'variables': {
            # Patient Info (required)
            'age': {'type': 'float', 'required': True},

            # Vitals (all required - 7 fields)
            'heartrate': {'type': 'float', 'required': True},
            'svri': {'type': 'float', 'required': True},
            'meanbp': {'type': 'float', 'required': True},
            'hearttimevolume': {'type': 'float', 'required': True},
            'oxygensaturation': {'type': 'float', 'required': True},
            'deltatemp': {'type': 'float', 'required': True},
            'oxygenationsaturation': {'type': 'float', 'required': True},

            # Lab Results (15 required + alt optional)
            'lactate': {'type': 'float', 'required': True},
            'creatinine': {'type': 'float', 'required': True},
            'bilirubin': {'type': 'float', 'required': True},
            'sodium': {'type': 'float', 'required': True},
            'potassium': {'type': 'float', 'required': True},
            'hemoglobin': {'type': 'float', 'required': True},
            'chloride': {'type': 'float', 'required': True},
            'leukocytes': {'type': 'float', 'required': True},
            'bicarbonate': {'type': 'float', 'required': True},
            'pancreaticlipase': {'type': 'float', 'required': True},
            'bun': {'type': 'float', 'required': True},
            'pct': {'type': 'float', 'required': True},
            'alt': {'type': 'float', 'required': False},  # stored but not used by model
            'buncreatinineratio': {'type': 'float', 'required': True},
            'ast': {'type': 'float', 'required': True},
            'crp': {'type': 'float', 'required': True},

            # Respiratory (all required - 2 fields)
            'respiratoryminutevolume': {'type': 'float', 'required': True},
            'fio2': {'type': 'float', 'required': True},

            # Blood Gas (all required - 2 fields)
            'arterialph': {'type': 'float', 'required': True},
            'pa_o2': {'type': 'float', 'required': True},
        }
    }
}


def validate_timestamp(timestamp_str: str) -> Tuple[bool, str]:
    """
    Validate ISO 8601 timestamp format
    
    Args:
        timestamp_str: Timestamp string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return True, ""
    except (ValueError, AttributeError):
        return False, f"Invalid timestamp format. Expected ISO 8601 format, got: {timestamp_str}"


def validate_field_type(field_name: str, value: Any, schema: Dict) -> Tuple[bool, str]:
    """
    Validate field type and constraints
    
    Args:
        field_name: Name of the field
        value: Value to validate
        schema: Field schema definition
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if value is None:
        return True, ""  # None values are acceptable for non-required fields
    
    expected_type = schema.get('type')
    
    # Type validation
    if expected_type == 'string':
        if not isinstance(value, str):
            return False, f"Expected type 'string', but received '{type(value).__name__}'. Value: {value}"
    
    elif expected_type == 'integer':
        if not isinstance(value, int) or isinstance(value, bool):
            return False, f"Expected type 'integer', but received '{type(value).__name__}'. Value: {value}"
    
    elif expected_type == 'float':
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False, f"Expected type 'float', but received '{type(value).__name__}'. Value: {value}"
        value = float(value)
    
    elif expected_type == 'categorical':
        if value not in schema.get('enum', []):
            return False, f"Value '{value}' is not in allowed values: {schema.get('enum')}"
    
    return True, ""


def validate_record(record: Dict, db_schema: Dict, record_index: int = 0) -> List[Dict]:
    """
    Validate a single data record
    
    Args:
        record: Data record to validate
        db_schema: Schema definition for the database
        record_index: Index of the record in the submission
        
    Returns:
        List of error dictionaries
    """
    errors = []
    sample_id = record.get('sample_id', 'N/A')
    
    # Check required fields (skip fields marked as not required in schema)
    for required_field in db_schema['required_fields']:
        # Check if the field is actually required in the variables schema
        field_schema = db_schema['variables'].get(required_field, {})
        if not field_schema.get('required', True):
            continue  # Skip optional fields
        if required_field not in record or record[required_field] is None:
            errors.append({
                'record_index': record_index,
                'sample_id': sample_id,
                'field': required_field,
                'code': 'missing_required_field',
                'description': f"The '{required_field}' is a required field."
            })
    
    # Validate timestamp if present
    if 'timestamp' in record:
        is_valid, error_msg = validate_timestamp(record['timestamp'])
        if not is_valid:
            errors.append({
                'record_index': record_index,
                'sample_id': sample_id,
                'field': 'timestamp',
                'code': 'invalid_format',
                'description': error_msg
            })
    
    # Validate each field in the record
    for field_name, value in record.items():
        if field_name in ['timestamp', 'sample_id']:
            continue
        
        if field_name in db_schema['variables']:
            field_schema = db_schema['variables'][field_name]
            is_valid, error_msg = validate_field_type(field_name, value, field_schema)
            
            if not is_valid:
                errors.append({
                    'record_index': record_index,
                    'sample_id': sample_id,
                    'field': field_name,
                    'code': 'invalid_type',
                    'description': error_msg
                })
    
    return errors


def validate_submission(payload: Dict) -> Dict:
    """
    Validate entire submission payload
    
    Args:
        payload: Complete submission payload
        
    Returns:
        Dictionary with validation results
    """
    metadata_errors = []
    data_errors = []
    
    # Validate metadata - accept both 'db' and 'db_name' for backward compatibility
    db_value = payload.get('db_name') or payload.get('db')
    if not db_value:
        metadata_errors.append({
            'field': 'db_name',
            'code': 'missing_required_field',
            'description': "The 'db_name' (or 'db') field is required to identify the database schema."
        })
        return {
            'valid': False,
            'message': 'Missing required database identifier.',
            'metadata_errors': metadata_errors
        }
    
    db_name = db_value.lower()
    if db_name not in SCHEMAS:
        metadata_errors.append({
            'field': 'db',
            'code': 'invalid_value',
            'description': f"Database '{db_name}' is not supported. Supported databases: {list(SCHEMAS.keys())}"
        })
        return {
            'valid': False,
            'message': f"Unsupported database: {db_name}",
            'metadata_errors': metadata_errors
        }
    
    if 'submission_id' not in payload:
        metadata_errors.append({
            'field': 'submission_id',
            'code': 'missing_required_field',
            'description': "The 'submission_id' field is required for tracking purposes."
        })
    
    if 'submitted_at' not in payload:
        metadata_errors.append({
            'field': 'submitted_at',
            'code': 'missing_required_field',
            'description': "The 'submitted_at' field is required for tracking purposes."
        })
    else:
        is_valid, error_msg = validate_timestamp(payload['submitted_at'])
        if not is_valid:
            metadata_errors.append({
                'field': 'submitted_at',
                'code': 'invalid_format',
                'description': error_msg
            })
    
    if 'data' not in payload or not isinstance(payload['data'], list):
        metadata_errors.append({
            'field': 'data',
            'code': 'missing_required_field',
            'description': "The 'data' field is required and must be an array of records."
        })
        return {
            'valid': False,
            'message': 'Missing or invalid data array.',
            'metadata_errors': metadata_errors
        }
    
    if len(payload['data']) == 0:
        metadata_errors.append({
            'field': 'data',
            'code': 'empty_array',
            'description': "The 'data' array must contain at least one record."
        })
    
    # Validate each data record
    db_schema = SCHEMAS[db_name]
    for idx, record in enumerate(payload['data']):
        record_errors = validate_record(record, db_schema, idx)
        data_errors.extend(record_errors)
    
    # Determine if validation passed
    has_errors = len(metadata_errors) > 0 or len(data_errors) > 0
    
    if has_errors:
        return {
            'valid': False,
            'message': f"The submission failed due to validation errors in {'metadata and ' if metadata_errors else ''}{len(data_errors)} data record(s).",
            'metadata_errors': metadata_errors,
            'data_errors': data_errors
        }
    
    return {
        'valid': True,
        'message': 'Validation successful'
    }
