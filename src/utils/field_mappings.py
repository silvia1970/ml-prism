"""
Feature mappings between JSON API schema and PyTorch model features.
"""

# MIMIC: JSON Schema names -> PyTorch model names
MIMIC_JSON_TO_PYTORCH = {
    'icustay_id': 'icustay_id',
    'sample_id': 'icustay_id',
    'chart_time': 'chart_time',
    'timestamp': 'chart_time',
    'label': 'label',
    'age': 'age',
    'sex': 'sex',
    'height': 'height',
    'weight': 'weight',
    'meanbp': 'meanbp',
    'resprate': 'resprate',
    'heartrate': 'heartrate',
    'spo2_pulsoxy': 'spo2_pulsoxy',
    'spo2': 'spo2_pulsoxy',
    'tempc': 'tempc',
    'cardiacoutput': 'cardiacoutput',
    'sysbp': None,
    'diabp': None,
    'gcseye': None,
    'gcsverbal': None,
    'gcsmotor': None,
    'baseexcess': None,
    'aniongap': None,
    'o2flow': 'o2flow',
    'fio2': 'fio2',
    'albumin': 'albumin',
    'bands': 'bands',
    'bicarbonate': 'bicarbonate',
    'bilirubin': 'bilirubin',
    'creatinine': 'creatinine',
    'glucose': 'glucose',
    'hemoglobin': 'hemoglobin',
    'hgb': 'hemoglobin',
    'inr': 'inr',
    'potassium': 'potassium',
    'ptt': 'ptt',
    'sodium': 'sodium',
    'wbc': 'wbc',
    'creatinekinase': 'creatinekinase',
    'ck_mb': 'ck_mb',
    'fibrinogen': 'fibrinogen',
    'ldh': 'ldh',
    'magnesium': 'magnesium',
    'calcium_free': 'calcium_free',
    'calcium': 'calcium_free',
    'po2_bloodgas': 'po2_bloodgas',
    'pao2': 'po2_bloodgas',
    'ph_bloodgas': 'ph_bloodgas',
    'ph': 'ph_bloodgas',
    'pco2_bloodgas': 'pco2_bloodgas',
    'paco2': 'pco2_bloodgas',
    'so2_bloodgas': 'so2_bloodgas',
    'chloride': 'chloride',
    'lactate': 'lactate',
    'platelet': 'platelet',
    'troponin_t': 'troponin_t',
}

MIMIC_PYTORCH_TO_JSON = {v: k for k, v in MIMIC_JSON_TO_PYTORCH.items() if v is not None}

# SepsisExp: JSON Schema names -> PyTorch model names
SEPSIEXP_JSON_TO_PYTORCH = {
    'id': 'id',
    'sample_id': 'id',
    'timestep': 'timestep',
    'timestamp': 'timestamp',
    'severity': 'severity',
    'sepsis': 'sepsis',
    'age': 'age',
    'heartrate': 'heart_rate',
    'heart_rate': 'heart_rate',
    'svri': 'svri',
    'meanbp': 'mean_bp',
    'mean_bp': 'mean_bp',
    'hearttimevolume': 'heart_time_volume',
    'heart_time_volume': 'heart_time_volume',
    'oxygensaturation': 'oxygen_saturation',
    'oxygen_saturation': 'oxygen_saturation',
    'deltatemp': 'delta-temperature',
    'delta_temperature': 'delta-temperature',
    'oxygenationsaturation': 'mixed_venous_oxygen_saturation',
    'mixed_venous_oxygen_saturation': 'mixed_venous_oxygen_saturation',
    'lactate': 'lactate',
    'creatinine': 'creatinine',
    'bilirubin': 'bilirubin',
    'sodium': 'sodium',
    'potassium': 'potassium',
    'hemoglobin': 'hemoglobin',
    'chloride': 'chloride',
    'leukocytes': 'leukocytes',
    'bicarbonate': 'bicarbonate',
    'pancreaticlipase': 'pancreatic_lipase',
    'pancreatic_lipase': 'pancreatic_lipase',
    'bun': 'blood_urea_nitrogen',
    'blood_urea_nitrogen': 'blood_urea_nitrogen',
    'pct': 'procalcitonin',
    'procalcitonin': 'procalcitonin',
    'alt': None,
    'alanine_transaminase': None,
    'buncreatinineratio': 'bun/creatinine_ratio',
    'bun_creatinine_ratio': 'bun/creatinine_ratio',
    'ast': 'aspartate_transaminase',
    'aspartate_transaminase': 'aspartate_transaminase',
    'crp': 'c-reactive_protein',
    'c_reactive_protein': 'c-reactive_protein',
    'respiratoryminutevolume': 'respiratory_minute_volume',
    'respiratory_minute_volume': 'respiratory_minute_volume',
    'fio2': 'fraction_of_inspired_o2',
    'fraction_of_inspired_o2': 'fraction_of_inspired_o2',
    'arterialph': 'arterial_ph',
    'arterial_ph': 'arterial_ph',
    'pa_o2': 'partial_pressure_art._o2',
    'partial_pressure_art_o2': 'partial_pressure_art._o2',
}

SEPSIEXP_PYTORCH_TO_JSON = {v: k for k, v in SEPSIEXP_JSON_TO_PYTORCH.items() if v is not None}

# Feature lists
MIMIC_FEATURES = [
    "meanbp", "resprate", "heartrate", "spo2_pulsoxy", "tempc",
    "cardiacoutput", "o2flow", "fio2", "albumin", "bands",
    "bicarbonate", "bilirubin", "creatinine", "chloride", "glucose",
    "hemoglobin", "lactate", "platelet", "potassium", "ptt",
    "inr", "sodium", "wbc", "creatinekinase", "ck_mb",
    "fibrinogen", "ldh", "magnesium", "calcium_free", "po2_bloodgas",
    "ph_bloodgas", "pco2_bloodgas", "so2_bloodgas", "troponin_t"
]

SEPSIEXP_FEATURES = [
    'age', 'heart_rate', 'svri', 'mean_bp', 'heart_time_volume',
    'oxygen_saturation', 'delta-temperature', 'mixed_venous_oxygen_saturation',
    'lactate', 'creatinine', 'bilirubin', 'sodium', 'potassium',
    'hemoglobin', 'chloride', 'leukocytes', 'bicarbonate',
    'pancreatic_lipase', 'blood_urea_nitrogen', 'procalcitonin',
    'bun/creatinine_ratio', 'aspartate_transaminase',
    'c-reactive_protein', 'respiratory_minute_volume',
    'fraction_of_inspired_o2', 'arterial_ph', 'partial_pressure_art._o2'
]


def map_json_to_pytorch(data: dict, db_type: str) -> dict:
    """Convert JSON schema field names to PyTorch model field names."""
    mapping = MIMIC_JSON_TO_PYTORCH if db_type.lower() == 'mimic' else SEPSIEXP_JSON_TO_PYTORCH
    result = {}
    for json_key, value in data.items():
        pytorch_key = mapping.get(json_key, json_key)
        if pytorch_key is not None:
            result[pytorch_key] = value
    return result


def map_pytorch_to_json(data: dict, db_type: str) -> dict:
    """Convert PyTorch model field names to JSON schema field names."""
    mapping = MIMIC_PYTORCH_TO_JSON if db_type.lower() == 'mimic' else SEPSIEXP_PYTORCH_TO_JSON
    return {mapping.get(k, k): v for k, v in data.items()}


def get_sample_id_field(db_type: str) -> str:
    """Get the native sample ID field name for a database type."""
    return 'icustay_id' if db_type.lower() == 'mimic' else 'id'


def get_timestamp_field(db_type: str) -> str:
    """Get the native timestamp field name for a database type."""
    return 'chart_time' if db_type.lower() == 'mimic' else 'timestamp'


def extract_sample_id(record: dict, db_type: str):
    """Extract the sample ID from a record."""
    if db_type.lower() == 'mimic':
        return record.get('icustay_id') or record.get('sample_id')
    return record.get('id') or record.get('sample_id')


def normalize_record_ids(record: dict, db_type: str) -> dict:
    """Normalize a record to ensure both native and API ID fields are present."""
    result = record.copy()
    if db_type.lower() == 'mimic':
        if 'icustay_id' in result and 'sample_id' not in result:
            result['sample_id'] = result['icustay_id']
        elif 'sample_id' in result and 'icustay_id' not in result:
            result['icustay_id'] = result['sample_id']
        if 'chart_time' in result and 'timestamp' not in result:
            result['timestamp'] = result['chart_time']
        elif 'timestamp' in result and 'chart_time' not in result:
            result['chart_time'] = result['timestamp']
    else:
        if 'id' in result and 'sample_id' not in result:
            result['sample_id'] = result['id']
        elif 'sample_id' in result and 'id' not in result:
            result['id'] = result['sample_id']
    return result