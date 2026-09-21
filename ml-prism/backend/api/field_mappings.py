"""
Field Mappings between JSON Schema and PyTorch Model Features
Maps user-friendly JSON field names to internal PyTorch feature names
"""

# MIMIC: JSON Schema names → PyTorch model names
MIMIC_JSON_TO_PYTORCH = {
    # MIMIC Primary Identifiers
    # icustay_id: Primary identifier for MIMIC records (maps to sample_id in API layer)
    # chart_time: Timestamp field from MIMIC (maps to timestamp in API layer)
    'icustay_id': 'icustay_id',
    'sample_id': 'icustay_id',  # Alias: API uses sample_id, maps to icustay_id
    'chart_time': 'chart_time',
    'timestamp': 'chart_time',  # Alias: API uses timestamp, maps to chart_time
    'label': 'label',  # Target variable (not used for prediction)
    
    # Patient info (not used by model but may be in data)
    'age': 'age',  # Not used by PyTorch model
    'sex': 'sex',  # Not used by PyTorch model
    'height': 'height',  # Not used by PyTorch model
    'weight': 'weight',  # Not used by PyTorch model
    
    # Vitals - with aliases
    'meanbp': 'meanbp',
    'resprate': 'resprate',
    'heartrate': 'heartrate',
    'spo2_pulsoxy': 'spo2_pulsoxy',
    'spo2': 'spo2_pulsoxy',  # Alias
    'tempc': 'tempc',
    'cardiacoutput': 'cardiacoutput',
    
    # NOT USED BY MODEL (34 features)
    'sysbp': None,  
    'diabp': None,
    'gcseye': None,
    'gcsverbal': None,
    'gcsmotor': None,
    'baseexcess': None,
    'aniongap': None,
    
    # Respiratory
    'o2flow': 'o2flow',
    'fio2': 'fio2',
    
    # Lab Results - with aliases
    'albumin': 'albumin',
    'bands': 'bands',
    'bicarbonate': 'bicarbonate',
    'bilirubin': 'bilirubin',
    'creatinine': 'creatinine',
    'glucose': 'glucose',
    'hematocrit': 'hematocrit',  # Not in PyTorch model
    'hemoglobin': 'hemoglobin',
    'hgb': 'hemoglobin',  # Alias
    'inr': 'inr',
    'potassium': 'potassium',
    'ptt': 'ptt',
    'bun': None,  # Not used directly
    'sodium': 'sodium',
    'wbc': 'wbc',
    'creatinekinase': 'creatinekinase',
    'ck_mb': 'ck_mb',
    'fibrinogen': 'fibrinogen',
    'ldh': 'ldh',
    'magnesium': 'magnesium',
    'calcium_free': 'calcium_free',
    'calcium': 'calcium_free',  # Alias
    
    # Blood Gas - with aliases
    'po2_bloodgas': 'po2_bloodgas',
    'pao2': 'po2_bloodgas',  # Alias
    'ph_bloodgas': 'ph_bloodgas',
    'ph': 'ph_bloodgas',  # Alias
    'pco2_bloodgas': 'pco2_bloodgas',
    'paco2': 'pco2_bloodgas',  # Alias
    'so2_bloodgas': 'so2_bloodgas',
    
    # Additional fields
    'chloride': 'chloride',
    'lactate': 'lactate',
    'platelet': 'platelet',
    'troponin_t': 'troponin_t',
}

# MIMIC: PyTorch model names → JSON Schema names (reverse mapping)
MIMIC_PYTORCH_TO_JSON = {v: k for k, v in MIMIC_JSON_TO_PYTORCH.items() if v is not None}

# SepsisExp: JSON Schema names → PyTorch model names
# Schema JSON uses normalized names with underscores
# PyTorch model uses original dataset names with hyphens and special chars
SEPSIEXP_JSON_TO_PYTORCH = {
    # SepsisExp Primary Identifiers
    # id: Primary identifier for SepsisExp records (maps to sample_id in API layer)
    # timestep: Timestep number from SepsisExp data
    'id': 'id',
    'sample_id': 'id',  # Alias: API uses sample_id, maps to id
    'timestep': 'timestep',
    'timestamp': 'timestamp',  # API timestamp (separate from data timestep)
    'severity': 'severity',  # Metadata field
    'sepsis': 'sepsis',  # Target variable (0 or 1)
    
    # Patient Info
    'age': 'age',

    # Vitals — DataFlow schema names → PyTorch feature names
    'heartrate': 'heart_rate',                      # DataFlow → model
    'svri': 'svri',
    'meanbp': 'mean_bp',                            # DataFlow → model
    'hearttimevolume': 'heart_time_volume',          # DataFlow → model
    'oxygensaturation': 'oxygen_saturation',         # DataFlow → model
    'deltatemp': 'delta-temperature',                # DataFlow → model (hyphen)
    'oxygenationsaturation': 'mixed_venous_oxygen_saturation',  # DataFlow → model

    # Backward-compat aliases (old snake_case names still accepted)
    'heart_rate': 'heart_rate',
    'mean_bp': 'mean_bp',
    'heart_time_volume': 'heart_time_volume',
    'oxygen_saturation': 'oxygen_saturation',
    'delta_temperature': 'delta-temperature',
    'mixed_venous_oxygen_saturation': 'mixed_venous_oxygen_saturation',

    # Lab Results — DataFlow names → PyTorch names
    'lactate': 'lactate',
    'creatinine': 'creatinine',
    'bilirubin': 'bilirubin',
    'sodium': 'sodium',
    'potassium': 'potassium',
    'hemoglobin': 'hemoglobin',
    'chloride': 'chloride',
    'leukocytes': 'leukocytes',
    'bicarbonate': 'bicarbonate',
    'pancreaticlipase': 'pancreatic_lipase',         # DataFlow → model
    'bun': 'blood_urea_nitrogen',                    # DataFlow → model
    'pct': 'procalcitonin',                          # DataFlow → model
    'alt': None,                                     # stored but NOT used by model
    'buncreatinineratio': 'bun/creatinine_ratio',    # DataFlow → model (slash)
    'ast': 'aspartate_transaminase',                 # DataFlow → model
    'crp': 'c-reactive_protein',                    # DataFlow → model (hyphen)

    # Backward-compat aliases for lab fields
    'pancreatic_lipase': 'pancreatic_lipase',
    'blood_urea_nitrogen': 'blood_urea_nitrogen',
    'procalcitonin': 'procalcitonin',
    'alanine_transaminase': None,
    'bun_creatinine_ratio': 'bun/creatinine_ratio',
    'aspartate_transaminase': 'aspartate_transaminase',
    'c_reactive_protein': 'c-reactive_protein',

    # Respiratory — DataFlow names → PyTorch names
    'respiratoryminutevolume': 'respiratory_minute_volume',  # DataFlow → model
    'fio2': 'fraction_of_inspired_o2',               # DataFlow → model

    # Backward-compat aliases
    'respiratory_minute_volume': 'respiratory_minute_volume',
    'fraction_of_inspired_o2': 'fraction_of_inspired_o2',

    # Blood Gas — DataFlow names → PyTorch names
    'arterialph': 'arterial_ph',                     # DataFlow → model
    'pa_o2': 'partial_pressure_art._o2',             # DataFlow → model (dot)

    # Backward-compat aliases
    'arterial_ph': 'arterial_ph',
    'partial_pressure_art_o2': 'partial_pressure_art._o2',
}

# SepsisExp: PyTorch model names → JSON Schema names (reverse mapping)
SEPSIEXP_PYTORCH_TO_JSON = {v: k for k, v in SEPSIEXP_JSON_TO_PYTORCH.items()}

# Additional PyTorch features not in JSON schema (need to be added or have default values)
SEPSIEXP_MISSING_IN_JSON = []


def map_json_to_pytorch(data: dict, db_type: str) -> dict:
    """
    Convert JSON schema field names to PyTorch model field names
    
    Args:
        data: Dictionary with JSON schema field names
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Dictionary with PyTorch model field names (excluding None mappings)
    """
    if db_type.lower() == 'mimic':
        mapping = MIMIC_JSON_TO_PYTORCH
    else:
        mapping = SEPSIEXP_JSON_TO_PYTORCH
    
    result = {}
    for json_key, value in data.items():
        pytorch_key = mapping.get(json_key, json_key)  # Use original if no mapping
        # Skip fields that map to None (not used by model)
        if pytorch_key is not None:
            result[pytorch_key] = value
    
    return result


def map_pytorch_to_json(data: dict, db_type: str) -> dict:
    """
    Convert PyTorch model field names to JSON schema field names
    
    Args:
        data: Dictionary with PyTorch model field names
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Dictionary with JSON schema field names
    """
    if db_type.lower() == 'mimic':
        mapping = MIMIC_PYTORCH_TO_JSON
    else:
        mapping = SEPSIEXP_PYTORCH_TO_JSON
    
    result = {}
    for pytorch_key, value in data.items():
        json_key = mapping.get(pytorch_key, pytorch_key)  # Use original if no mapping
        result[json_key] = value
    
    return result


def get_sample_id_field(db_type: str) -> str:
    """
    Get the native sample ID field name for a database type.
    
    - MIMIC uses 'icustay_id' as the primary identifier
    - SepsisExp uses 'id' as the primary identifier
    
    Args:
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Native field name for sample ID
    """
    if db_type.lower() == 'mimic':
        return 'icustay_id'
    else:
        return 'id'


def get_timestamp_field(db_type: str) -> str:
    """
    Get the native timestamp field name for a database type.
    
    - MIMIC uses 'chart_time' as the timestamp
    - SepsisExp uses 'timestamp' as the timestamp
    
    Args:
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Native field name for timestamp
    """
    if db_type.lower() == 'mimic':
        return 'chart_time'
    else:
        return 'timestamp'


def extract_sample_id(record: dict, db_type: str) -> str:
    """
    Extract the sample ID from a record, checking both native and API field names.
    
    Priority for MIMIC: icustay_id > sample_id
    Priority for SepsisExp: id > sample_id
    
    Args:
        record: Data record dictionary
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Sample ID value or None if not found
    """
    if db_type.lower() == 'mimic':
        return record.get('icustay_id') or record.get('sample_id')
    else:
        return record.get('id') or record.get('sample_id')


def normalize_record_ids(record: dict, db_type: str) -> dict:
    """
    Normalize a record to ensure both native and API ID fields are present.
    
    For MIMIC: Copies icustay_id to sample_id (or vice versa)
    For SepsisExp: Copies id to sample_id (or vice versa)
    
    Args:
        record: Data record dictionary
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Record with normalized ID fields
    """
    result = record.copy()
    
    if db_type.lower() == 'mimic':
        # Ensure both icustay_id and sample_id exist
        if 'icustay_id' in result and 'sample_id' not in result:
            result['sample_id'] = result['icustay_id']
        elif 'sample_id' in result and 'icustay_id' not in result:
            result['icustay_id'] = result['sample_id']
        # Also handle timestamp
        if 'chart_time' in result and 'timestamp' not in result:
            result['timestamp'] = result['chart_time']
        elif 'timestamp' in result and 'chart_time' not in result:
            result['chart_time'] = result['timestamp']
    else:
        # SepsisExp: Ensure both id and sample_id exist
        if 'id' in result and 'sample_id' not in result:
            result['sample_id'] = result['id']
        elif 'sample_id' in result and 'id' not in result:
            result['id'] = result['sample_id']
    
    return result
