"""
Shared blueprint utilities.

Contains helper functions used by multiple blueprints to avoid
duplicated store→score→update loops (DUPL-6).
"""

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def process_records_batch(
    records: List[Dict],
    db_name: str,
    submission_id: str,
    db,
    ml_scorer,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Store, score, and persist score for a batch of patient records.

    Shared by the data and CSV blueprints.

    Returns:
        (results, errors) where each element is a list of dicts.
    """
    from api.field_mappings import normalize_record_ids, extract_sample_id

    results: List[Dict] = []
    errors: List[Dict] = []

    for idx, record in enumerate(records):
        try:
            record = normalize_record_ids(record, db_name)
            sample_id = extract_sample_id(record, db_name)

            stored = db.store_record(
                db_name=db_name,
                submission_id=submission_id,
                record=record,
            )
            record_id = stored.get('_id')

            prediction = ml_scorer.predict(db_name=db_name, record=record)

            db.update_record_score(
                db_name=db_name,
                record_id=record_id,
                score=prediction['score'],
                class_label=prediction['class'],
            )

            results.append({
                'sample_id': sample_id,
                'record_id': record_id,
                'score': prediction['score'],
                'class': prediction['class'],
                'explanations': prediction.get('explanations', {}),
            })
        except Exception as exc:
            logger.error(f"Error processing record {idx}: {exc}")
            errors.append({'index': idx, 'error': str(exc)})

    return results, errors