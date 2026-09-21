"""
CSV File Handler for PRISM API
Processes CSV uploads and converts to validated patient records
"""

import csv
import io
from typing import Dict, List, Any, Tuple
from datetime import datetime, timezone
from werkzeug.datastructures import FileStorage
import logging

from api.validators import validate_submission
from api.field_mappings import (
    get_sample_id_field, 
    extract_sample_id, 
    normalize_record_ids
)

logger = logging.getLogger(__name__)


class CSVProcessingError(Exception):
    """Custom exception for CSV processing errors"""
    pass


class CSVHandler:
    """
    Handles CSV file upload, parsing, and validation
    """
    
    # Maximum file size (100 MB - increased for large datasets)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum number of records per file (100,000 for large clinical datasets)
    MAX_RECORDS = 100_000
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.csv', '.txt'}
    
    # Field name mappings: CSV column name -> Schema field name
    # Maps CSV names (with special chars) to schema names (normalized with underscores)
    FIELD_MAPPINGS = {
        # SepsisExp mappings (CSV uses special chars, schema uses underscores)
        'delta-temperature': 'delta_temperature',
        'bun/creatinine_ratio': 'bun_creatinine_ratio',
        'c-reactive_protein': 'c_reactive_protein',
        'partial_pressure_art._o2': 'partial_pressure_art_o2',
    }
    
    @staticmethod
    def validate_file(file: FileStorage) -> Tuple[bool, str]:
        """
        Validate uploaded file before processing
        
        Args:
            file: Uploaded file from Flask request
            
        Returns:
            (is_valid, error_message)
        """
        # Check if file exists
        if not file or not file.filename:
            return False, "No file provided"
        
        # Check file extension
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in CSVHandler.ALLOWED_EXTENSIONS):
            return False, f"Invalid file type. Allowed: {', '.join(CSVHandler.ALLOWED_EXTENSIONS)}"
        
        # Check file size (read into memory to check)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > CSVHandler.MAX_FILE_SIZE:
            size_mb = CSVHandler.MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size: {size_mb} MB"
        
        if file_size == 0:
            return False, "File is empty"
        
        return True, ""
    
    @staticmethod
    def parse_csv(
        file: FileStorage,
        db_type: str,
        submission_id: str = None
    ) -> Dict[str, Any]:
        """
        Parse CSV file and prepare submission payload
        
        Expected CSV format:
        - First row: column headers (field names)
        - Subsequent rows: patient data
        - Must include 'sample_id' column
        - Column names must match MIMIC or SepsisExp schema
        
        Args:
            file: Uploaded CSV file
            db_type: 'mimic' or 'sepsiexp'
            submission_id: Optional submission ID (auto-generated if not provided)
            
        Returns:
            Dict with parsed data ready for validation
            
        Raises:
            CSVProcessingError: If CSV parsing fails
        """
        try:
            # Read file content
            file.seek(0)
            content = file.read().decode('utf-8-sig')  # Handle BOM
            
            # Parse CSV
            csv_reader = csv.DictReader(io.StringIO(content))
            
            # Check for required columns
            if not csv_reader.fieldnames:
                raise CSVProcessingError("CSV file has no headers")
            
            # Normalize field names (lowercase, strip whitespace)
            fieldnames = [name.strip().lower() for name in csv_reader.fieldnames]
            
            # Determine which ID field to look for based on db_type
            # MIMIC uses 'icustay_id', SepsisExp uses 'id', but 'sample_id' is also accepted
            native_id_field = get_sample_id_field(db_type)
            has_id_field = (
                native_id_field in fieldnames or 
                'sample_id' in fieldnames
            )
            
            if not has_id_field:
                expected_field = f"'{native_id_field}' or 'sample_id'"
                raise CSVProcessingError(f"CSV must contain {expected_field} column")
            
            # Parse all rows
            records = []
            row_number = 1  # Start from 1 (after header)
            
            for row_data in csv_reader:
                row_number += 1
                
                # Check max records limit
                if len(records) >= CSVHandler.MAX_RECORDS:
                    raise CSVProcessingError(
                        f"Too many records. Maximum: {CSVHandler.MAX_RECORDS}"
                    )
                
                # Normalize row data
                record = {}
                for original_field, value in row_data.items():
                    field = original_field.strip().lower()
                    
                    # Apply field name mapping (CSV name -> Schema name)
                    field = CSVHandler.FIELD_MAPPINGS.get(field, field)
                    
                    # Skip empty values
                    if value is None or value.strip() == '':
                        continue
                    
                    # Convert value to appropriate type
                    record[field] = CSVHandler._convert_value(field, value.strip())
                
                # Add timestamp if not present
                if 'timestamp' not in record:
                    record['timestamp'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                # Normalize ID fields (ensure both native and sample_id are present)
                record = normalize_record_ids(record, db_type)
                
                # Validate that we have a sample_id (after normalization)
                sample_id = extract_sample_id(record, db_type)
                if not sample_id:
                    native_id_field = get_sample_id_field(db_type)
                    raise CSVProcessingError(
                        f"Row {row_number}: Missing '{native_id_field}' or 'sample_id'"
                    )
                
                records.append(record)
            
            # Check we got some records
            if not records:
                raise CSVProcessingError("CSV file contains no data rows")
            
            # Generate submission ID if not provided
            if not submission_id:
                submission_id = f"CSV-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Build submission payload
            payload = {
                'db': db_type.lower(),
                'submission_id': submission_id,
                'submitted_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'meta': {
                    'source': 'csv_upload',
                    'filename': file.filename,
                    'rows': len(records),
                    'upload_timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                },
                'data': records
            }
            
            logger.info(
                f"CSV parsed successfully: {len(records)} records from {file.filename}"
            )
            
            return payload
            
        except csv.Error as e:
            raise CSVProcessingError(f"CSV parsing error: {str(e)}")
        except UnicodeDecodeError:
            raise CSVProcessingError("Invalid file encoding. Use UTF-8")
        except Exception as e:
            logger.error(f"Unexpected error parsing CSV: {e}", exc_info=True)
            raise CSVProcessingError(f"Failed to parse CSV: {str(e)}")
    
    @staticmethod
    def _convert_value(field: str, value: str) -> Any:
        """
        Convert string value to appropriate Python type based on field name
        
        Args:
            field: Field name (lowercase)
            value: String value from CSV
            
        Returns:
            Converted value (int, float, or str)
        """
        # Keep as string for these fields
        string_fields = {'sample_id', 'sex', 'timestamp', 'submission_id'}
        if field in string_fields:
            return value
        
        # Try to convert to number
        try:
            # Check if it looks like an integer
            if '.' not in value and 'e' not in value.lower():
                return int(value)
            else:
                return float(value)
        except ValueError:
            # Keep as string if conversion fails
            return value
    
    @staticmethod
    def validate_csv_data(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate CSV data using existing PRISM validators
        
        Args:
            payload: Parsed CSV payload
            
        Returns:
            Validation result from validators.validate_submission()
        """
        return validate_submission(payload)
    
    @staticmethod
    def create_csv_template(db_type: str) -> str:
        """
        Generate CSV template file path for download (using validators.py schema field names)
        
        Args:
            db_type: 'mimic' or 'sepsiexp'
            
        Returns:
            Path to temporary CSV template file
        """
        import tempfile
        import os
        from api.validators import SCHEMAS
        
        schema = SCHEMAS.get(db_type.lower())
        if not schema:
            raise ValueError(f"Invalid database type: {db_type}")
        
        if db_type.lower() == 'mimic':
            # MIMIC: 34 features from validators.py + icustay_id (native ID) + chart_time (native timestamp)
            headers = ['icustay_id', 'chart_time'] + schema['required_fields']
            example_row = [
                '200001', '2025-12-11T15:00:00Z',  # icustay_id, chart_time
                '91', '18', '88', '97', '37.1', '5.5',  # meanbp, resprate, heartrate, spo2, tempc, cardiacoutput
                '2', '40',  # o2flow, fio2
                '3.5', '5', '24', '0.8', '1.1', '102', '105',  # albumin, bands, bicarbonate, bilirubin, creatinine, chloride, glucose
                '13.5', '1.8', '245', '4.1', '35',  # hemoglobin, lactate, platelets, potassium, ptt
                '1.2', '138', '9.2', '150', '8',  # inr, sodium, wbc, creatinekinase, ck_mb
                '350', '220', '2.1', '1.15', '95',  # fibrinogen, ldh, magnesium, calcium_free, po2_bloodgas
                '7.38', '42', '96', '0.08'  # ph_bloodgas, pco2_bloodgas, so2_bloodgas, troponin_t
            ]
        else:  # sepsiexp
            # SEPSIEXP: 28 features from validators.py (age + 27 clinical) + id (native ID) + timestamp
            headers = ['id', 'timestep', 'timestamp'] + schema['required_fields']
            example_row = [
                '1', '0', '2025-12-11T15:10:00Z',  # id, timestep, timestamp
                '65',  # age
                '88', '1800', '91', '5.5', '97', '0.5', '75',  # heart_rate, svri, mean_bp, heart_time_volume, oxygen_saturation, delta_temperature, mixed_venous_oxygen_saturation
                '1.8', '1.1', '0.8', '138', '4.1', '13.5', '102', '9.2', '24', '150', '22', '2.5',  # lactate, creatinine, bilirubin, sodium, potassium, hemoglobin, chloride, leukocytes, bicarbonate, pancreatic_lipase, blood_urea_nitrogen, procalcitonin
                '35', '20', '180', '45',  # alanine_transaminase, bun_creatinine_ratio, aspartate_transaminase, c_reactive_protein
                '8.5', '40',  # respiratory_minute_volume, fraction_of_inspired_o2
                '7.38', '95'  # arterial_ph, partial_pressure_art_o2
            ]
        
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.csv', prefix=f'{db_type}_template_')
        
        try:
            with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow(example_row)
            
            logger.info(f"Created template file: {temp_path}")
            return temp_path
            
        except Exception as e:
            # Clean up on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise CSVProcessingError(f"Failed to create template: {str(e)}")


def process_csv_upload(
    file: FileStorage,
    db_type: str,
    submission_id: str = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Complete CSV processing pipeline
    
    Steps:
    1. Validate file
    2. Parse CSV
    3. Validate data
    4. Return result
    
    Args:
        file: Uploaded file
        db_type: 'mimic' or 'sepsiexp'
        submission_id: Optional submission ID
        
    Returns:
        (success, result_dict)
        - If success: result_dict contains parsed and validated data
        - If failure: result_dict contains error details
    """
    try:
        # Step 1: Validate file
        is_valid, error_msg = CSVHandler.validate_file(file)
        if not is_valid:
            return False, {
                'status': 'failure',
                'error_type': 'FileValidationError',
                'message': error_msg
            }
        
        # Step 2: Parse CSV
        try:
            payload = CSVHandler.parse_csv(file, db_type, submission_id)
        except CSVProcessingError as e:
            return False, {
                'status': 'failure',
                'error_type': 'CSVParsingError',
                'message': str(e)
            }
        
        # Step 3: Validate data
        validation_result = CSVHandler.validate_csv_data(payload)
        
        if not validation_result['valid']:
            return False, {
                'status': 'failure',
                'error_type': 'ValidationError',
                'message': 'CSV data failed validation',
                'validation_errors': validation_result
            }
        
        # Success - return parsed and validated payload
        return True, {
            'status': 'success',
            'payload': payload,
            'records_count': len(payload['data']),
            'filename': file.filename
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in CSV processing: {e}", exc_info=True)
        return False, {
            'status': 'failure',
            'error_type': 'InternalError',
            'message': f'Unexpected error: {str(e)}'
        }
