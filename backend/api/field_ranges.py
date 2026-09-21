"""
Field Value Ranges for PRISM ML
Defines minimum and maximum acceptable values for clinical features
Based on medical reference ranges and training data distributions
"""

# MIMIC Field Ranges (34 features used by model)
MIMIC_RANGES = {
    # Demographics
    'age': {'min': 0, 'max': 120, 'unit': 'anni', 'label': 'Età'},
    
    # Vitals
    'meanbp': {'min': 30, 'max': 180, 'unit': 'mmHg', 'label': 'Pressione Arteriosa Media'},
    'resprate': {'min': 4, 'max': 60, 'unit': 'breaths/min', 'label': 'Frequenza Respiratoria'},
    'heartrate': {'min': 20, 'max': 250, 'unit': 'bpm', 'label': 'Frequenza Cardiaca'},
    'spo2_pulsoxy': {'min': 50, 'max': 100, 'unit': '%', 'label': 'Saturazione Ossigeno'},
    'tempc': {'min': 30, 'max': 43, 'unit': '°C', 'label': 'Temperatura Corporea'},
    'cardiacoutput': {'min': 1, 'max': 20, 'unit': 'L/min', 'label': 'Gittata Cardiaca'},
    
    # Respiratory
    'o2flow': {'min': 0, 'max': 100, 'unit': 'L/min', 'label': 'Flusso Ossigeno'},
    'fio2': {'min': 0.21, 'max': 1.0, 'unit': 'fraction', 'label': 'Frazione Ossigeno Inspirato'},
    
    # Lab Results
    'albumin': {'min': 1.0, 'max': 6.0, 'unit': 'g/dL', 'label': 'Albumina'},
    'bands': {'min': 0, 'max': 50, 'unit': '%', 'label': 'Neutrofili Immaturi'},
    'bicarbonate': {'min': 5, 'max': 50, 'unit': 'mEq/L', 'label': 'Bicarbonato'},
    'bilirubin': {'min': 0.1, 'max': 50, 'unit': 'mg/dL', 'label': 'Bilirubina'},
    'creatinine': {'min': 0.1, 'max': 25, 'unit': 'mg/dL', 'label': 'Creatinina'},
    'chloride': {'min': 70, 'max': 130, 'unit': 'mEq/L', 'label': 'Cloruro'},
    'glucose': {'min': 20, 'max': 1000, 'unit': 'mg/dL', 'label': 'Glucosio'},
    'hemoglobin': {'min': 3, 'max': 20, 'unit': 'g/dL', 'label': 'Emoglobina'},
    'hgb': {'min': 3, 'max': 20, 'unit': 'g/dL', 'label': 'Emoglobina'},
    'inr': {'min': 0.5, 'max': 10, 'unit': 'ratio', 'label': 'INR'},
    'potassium': {'min': 2.0, 'max': 8.0, 'unit': 'mEq/L', 'label': 'Potassio'},
    'ptt': {'min': 15, 'max': 150, 'unit': 'seconds', 'label': 'PTT'},
    'sodium': {'min': 100, 'max': 180, 'unit': 'mEq/L', 'label': 'Sodio'},
    'wbc': {'min': 0.1, 'max': 100, 'unit': 'K/µL', 'label': 'Globuli Bianchi'},
    'creatinekinase': {'min': 10, 'max': 10000, 'unit': 'U/L', 'label': 'Creatina Chinasi'},
    'ck_mb': {'min': 0, 'max': 500, 'unit': 'ng/mL', 'label': 'CK-MB'},
    'fibrinogen': {'min': 50, 'max': 1000, 'unit': 'mg/dL', 'label': 'Fibrinogeno'},
    'ldh': {'min': 50, 'max': 5000, 'unit': 'U/L', 'label': 'Lattato Deidrogenasi'},
    'magnesium': {'min': 0.5, 'max': 5.0, 'unit': 'mg/dL', 'label': 'Magnesio'},
    'calcium_free': {'min': 0.5, 'max': 3.0, 'unit': 'mmol/L', 'label': 'Calcio Ionizzato'},
    'calcium': {'min': 0.5, 'max': 3.0, 'unit': 'mmol/L', 'label': 'Calcio Ionizzato'},
    
    # Blood Gas
    'po2_bloodgas': {'min': 20, 'max': 500, 'unit': 'mmHg', 'label': 'PO2 Emogasanalisi'},
    'pao2': {'min': 20, 'max': 500, 'unit': 'mmHg', 'label': 'PaO2'},
    'ph_bloodgas': {'min': 6.5, 'max': 8.0, 'unit': 'pH', 'label': 'pH Emogasanalisi'},
    'ph': {'min': 6.5, 'max': 8.0, 'unit': 'pH', 'label': 'pH'},
    'pco2_bloodgas': {'min': 10, 'max': 150, 'unit': 'mmHg', 'label': 'PCO2 Emogasanalisi'},
    'paco2': {'min': 10, 'max': 150, 'unit': 'mmHg', 'label': 'PaCO2'},
    'so2_bloodgas': {'min': 50, 'max': 100, 'unit': '%', 'label': 'SO2 Emogasanalisi'},
    
    # Additional
    'lactate': {'min': 0.1, 'max': 30, 'unit': 'mmol/L', 'label': 'Lattato'},
    'platelet': {'min': 10, 'max': 1000, 'unit': 'K/µL', 'label': 'Piastrine'},
    'troponin_t': {'min': 0, 'max': 100, 'unit': 'ng/mL', 'label': 'Troponina T'},
}

# SepsisExp Field Ranges (29 features: age + timestamp + 27 clinical features)
SEPSIEXP_RANGES = {
    # Metadata
    'age': {'min': 0, 'max': 120, 'unit': 'anni', 'label': 'Età'},
    
    # Vitals
    'heartrate': {'min': 20, 'max': 250, 'unit': 'bpm', 'label': 'Frequenza Cardiaca'},
    'svri': {'min': 100, 'max': 5000, 'unit': 'dyn·s/cm⁵/m²', 'label': 'Indice Resistenza Vascolare Sistemica'},
    'meanbp': {'min': 30, 'max': 180, 'unit': 'mmHg', 'label': 'Pressione Arteriosa Media'},
    'hearttimevolume': {'min': 1, 'max': 20, 'unit': 'L/min', 'label': 'Volume Cardiaco per Tempo'},
    'oxygensaturation': {'min': 50, 'max': 100, 'unit': '%', 'label': 'Saturazione Ossigeno'},
    'deltatemp': {'min': -5, 'max': 10, 'unit': '°C', 'label': 'Delta Temperatura'},
    'oxygenationsaturation': {'min': 40, 'max': 100, 'unit': '%', 'label': 'Saturazione Ossigenazione Venosa Mista'},
    
    # Lab Results
    'lactate': {'min': 0.1, 'max': 30, 'unit': 'mmol/L', 'label': 'Lattato'},
    'creatinine': {'min': 0.1, 'max': 25, 'unit': 'mg/dL', 'label': 'Creatinina'},
    'bilirubin': {'min': 0.1, 'max': 50, 'unit': 'mg/dL', 'label': 'Bilirubina'},
    'sodium': {'min': 100, 'max': 180, 'unit': 'mEq/L', 'label': 'Sodio'},
    'potassium': {'min': 2.0, 'max': 8.0, 'unit': 'mEq/L', 'label': 'Potassio'},
    'hemoglobin': {'min': 3, 'max': 20, 'unit': 'g/dL', 'label': 'Emoglobina'},
    'chloride': {'min': 70, 'max': 130, 'unit': 'mEq/L', 'label': 'Cloruro'},
    'leukocytes': {'min': 0.1, 'max': 100, 'unit': 'K/µL', 'label': 'Leucociti'},
    'bicarbonate': {'min': 5, 'max': 50, 'unit': 'mEq/L', 'label': 'Bicarbonato'},
    'pancreaticlipase': {'min': 0, 'max': 5000, 'unit': 'U/L', 'label': 'Lipasi Pancreatica'},
    'bun': {'min': 2, 'max': 200, 'unit': 'mg/dL', 'label': 'Azotemia'},
    'pct': {'min': 0, 'max': 200, 'unit': 'ng/mL', 'label': 'Procalcitonina'},
    'alt': {'min': 5, 'max': 5000, 'unit': 'U/L', 'label': 'Alanina Transaminasi'},
    'buncreatinineratio': {'min': 3, 'max': 100, 'unit': 'ratio', 'label': 'Rapporto BUN/Creatinina'},
    'ast': {'min': 5, 'max': 5000, 'unit': 'U/L', 'label': 'Aspartato Transaminasi'},
    'crp': {'min': 0, 'max': 500, 'unit': 'mg/L', 'label': 'Proteina C-Reattiva'},
    
    # Respiratory
    'respiratoryminutevolume': {'min': 1, 'max': 50, 'unit': 'L/min', 'label': 'Volume Minuto Respiratorio'},
    'fio2': {'min': 0.21, 'max': 1.0, 'unit': 'fraction', 'label': 'Frazione Ossigeno Inspirato'},
    
    # Blood Gas
    'arterialph': {'min': 6.5, 'max': 8.0, 'unit': 'pH', 'label': 'pH Arterioso'},
    'pa_o2': {'min': 20, 'max': 500, 'unit': 'mmHg', 'label': 'Pressione Parziale O2 Arteriosa'},
}


def get_field_range(field_name: str, db_type: str) -> dict:
    """
    Get validation range for a field
    
    Args:
        field_name: Field name
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Dictionary with min, max, unit, label or None if not found
    """
    ranges = MIMIC_RANGES if db_type.lower() == 'mimic' else SEPSIEXP_RANGES
    return ranges.get(field_name)


def validate_field_value(field_name: str, value: float, db_type: str) -> tuple[bool, str]:
    """
    Validate if a field value is within acceptable range
    
    Args:
        field_name: Field name
        value: Field value
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    field_range = get_field_range(field_name, db_type)
    
    if not field_range:
        return True, ""  # No range defined, accept any value
    
    if value < field_range['min']:
        return False, f"{field_range['label']} troppo basso: {value} < {field_range['min']} {field_range['unit']}"
    
    if value > field_range['max']:
        return False, f"{field_range['label']} troppo alto: {value} > {field_range['max']} {field_range['unit']}"
    
    return True, ""


def get_all_fields_with_ranges(db_type: str) -> dict:
    """
    Get all fields with their ranges for a database type
    
    Args:
        db_type: 'mimic' or 'sepsiexp'
        
    Returns:
        Dictionary of field_name -> range_info
    """
    return MIMIC_RANGES.copy() if db_type.lower() == 'mimic' else SEPSIEXP_RANGES.copy()
