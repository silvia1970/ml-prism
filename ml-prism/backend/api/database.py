"""
Database Module for PRISM API

Handles data persistence and retrieval using JSON files and SQLite.
Uses WAL mode for better concurrency and context managers for safe connections.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import threading

# Import field mapping utilities for sample_id handling
from api.field_mappings import (
    extract_sample_id,
    normalize_record_ids,
    get_sample_id_field
)

# MinIO object storage (optional — transparent when disabled)
from api import minio_storage

logger = logging.getLogger(__name__)


class Database:
    """Database handler for PRISM API"""
    
    def __init__(self, data_dir: str = 'api_data'):
        """
        Initialize database
        
        Args:
            data_dir: Directory for storing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Separate directories for each database
        self.mimic_dir = self.data_dir / 'mimic'
        self.sepsiexp_dir = self.data_dir / 'sepsiexp'
        self.submissions_dir = self.data_dir / 'submissions'
        
        for dir_path in [self.mimic_dir, self.sepsiexp_dir, self.submissions_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # SQLite for metadata and fast queries
        # Separate DB files: mimic.db and sepsiexp.db (plus legacy prism.db)
        self.db_path = self.data_dir / 'prism.db'          # legacy / cross-DB fallback
        self.mimic_db_path = self.data_dir / 'mimic.db'    # MIMIC-only records & submissions
        self.sepsiexp_db_path = self.data_dir / 'sepsiexp.db'  # SepsisExp-only
        self._init_sqlite()
        self._lock = threading.Lock()
    
    def _init_sqlite(self):
        """Initialize SQLite database schema for all DB files."""
        for db_path in [self.db_path, self.mimic_db_path, self.sepsiexp_db_path]:
            self._init_single_db(db_path)

    def _init_single_db(self, db_path):
        """Create tables, indexes and enable WAL mode in a single SQLite file."""
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Enable WAL journal mode for better read/write concurrency
            cursor.execute('PRAGMA journal_mode=WAL')

            # Submissions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS submissions (
                    submission_id TEXT PRIMARY KEY,
                    db_name TEXT NOT NULL,
                    client_version TEXT,
                    submitted_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score_calculated INTEGER DEFAULT 0,
                    records_count INTEGER DEFAULT 0,
                    source TEXT,
                    original_filename TEXT,
                    completion_time_ms INTEGER,
                    failure_reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    db_name TEXT NOT NULL,
                    submission_id TEXT,
                    timestamp TEXT,
                    score REAL,
                    class TEXT,
                    file_path TEXT,
                    patient_id TEXT,
                    data_timestep TEXT,
                    severity TEXT,
                    sepsis TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (submission_id) REFERENCES submissions(submission_id)
                )
            ''')

            # Migrate: add original_filename column if missing (backward compat)
            try:
                cursor.execute('ALTER TABLE submissions ADD COLUMN original_filename TEXT')
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sample_id ON records(sample_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_db_name ON records(db_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON records(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON records(patient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submission_id ON submissions(submission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submitted_at ON submissions(submitted_at)')

            conn.commit()

    def _get_db_path(self, db_name: str):
        """Return the SQLite path for a given database type."""
        if db_name == 'mimic':
            return self.mimic_db_path
        elif db_name == 'sepsiexp':
            return self.sepsiexp_db_path
        return self.db_path  # legacy fallback

    def _get_db_dir(self, db_name: str) -> Path:
        """Get directory path for database"""
        return self.mimic_dir if db_name == 'mimic' else self.sepsiexp_dir
    
    def _save_record_to_file(self, db_name: str, sample_id: str, record: Dict) -> str:
        """
        Save record to JSON file
        
        Args:
            db_name: Database name
            sample_id: Sample identifier
            record: Record data
            
        Returns:
            File path where record was saved
        """
        db_dir = self._get_db_dir(db_name)
        timestamp = record.get('timestamp', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
        
        # Sanitise sample_id: allow only alphanumerics, hyphens and underscores (SEC-4 path traversal)
        safe_sample_id = ''.join(c for c in sample_id if c.isalnum() or c in ('-', '_'))
        if not safe_sample_id:
            safe_sample_id = 'unknown'

        # Create filename from safe sample_id and timestamp
        filename = f"{safe_sample_id}_{timestamp.replace(':', '-')}.json"
        file_path = db_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        # Mirror to MinIO if enabled
        minio_key = f"{db_name}/{filename}"
        minio_storage.put_json(minio_key, record)
        
        return str(file_path)
    
    def store_record(self, db_name: str, submission_id: str, record: Dict) -> Dict:
        """
        Store a single record
        
        Args:
            db_name: Database name (mimic or sepsiexp)
            submission_id: Submission identifier
            record: Record data
            
        Returns:
            Stored record with metadata
            
        Note:
            For MIMIC, uses 'icustay_id' as the sample identifier.
            For SepsisExp, uses 'id' as the sample identifier.
            Both are normalized to 'sample_id' for internal storage.
        """
        with self._lock:
            # Normalize record to have both native ID and sample_id
            record = normalize_record_ids(record, db_name)
            
            # Extract sample_id (handles both native and API field names)
            sample_id = extract_sample_id(record, db_name)
            if not sample_id:
                native_field = get_sample_id_field(db_name)
                raise ValueError(f"Record missing '{native_field}' or 'sample_id'")
            
            timestamp = record.get('timestamp', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
            
            # Extract metadata fields for SepsisExp (if present)
            patient_id = record.get('id')  # SepsisExp 'id' field
            data_timestep = record.get('timestep')  # SepsisExp 'timestep' field
            severity = record.get('severity')  # SepsisExp 'severity' field
            sepsis = record.get('sepsis')  # SepsisExp 'sepsis' field
            
            # Save to file
            file_path = self._save_record_to_file(db_name, sample_id, record)
            
            # Save metadata to SQLite
            with sqlite3.connect(self._get_db_path(db_name)) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO records (sample_id, db_name, submission_id, timestamp, file_path,
                                       patient_id, data_timestep, severity, sepsis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (sample_id, db_name, submission_id, timestamp, file_path,
                      patient_id, data_timestep, severity, sepsis))

                record_id = cursor.lastrowid
                conn.commit()
            
            return {**record, 'submission_id': submission_id, '_id': record_id}
    
    def get_record(self, db_name: str, sample_id: str, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve a record by sample_id and optional timestamp
        
        Args:
            db_name: Database name
            sample_id: Sample identifier
            timestamp: Optional timestamp filter
            
        Returns:
            Record data or None if not found
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        if timestamp:
            cursor.execute('''
                SELECT file_path, submission_id FROM records
                WHERE sample_id = ? AND db_name = ? AND timestamp = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (sample_id, db_name, timestamp))
        else:
            cursor.execute('''
                SELECT file_path, submission_id FROM records
                WHERE sample_id = ? AND db_name = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (sample_id, db_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        file_path, submission_id = result
        
        # Load from file, fallback to MinIO
        try:
            with open(file_path, 'r') as f:
                record = json.load(f)
            record['submission_id'] = submission_id
            return record
        except FileNotFoundError:
            # Try MinIO fallback
            minio_key = '/'.join(Path(file_path).parts[-2:])
            minio_data = minio_storage.get_json(minio_key)
            if minio_data:
                minio_data['submission_id'] = submission_id
                return minio_data
            return None
    
    def get_record_by_patient_id(self, db_name: str, patient_id: str, data_timestep: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve a record by patient_id and optional timestep (for SepsisExp)
        
        Args:
            db_name: Database name
            patient_id: Patient identifier (from 'id' field in SepsisExp data)
            data_timestep: Optional timestep filter (from 'timestep' field in SepsisExp data)
            
        Returns:
            Record data or None if not found
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        if data_timestep:
            cursor.execute('''
                SELECT file_path, submission_id FROM records
                WHERE patient_id = ? AND db_name = ? AND data_timestep = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (patient_id, db_name, data_timestep))
        else:
            cursor.execute('''
                SELECT file_path, submission_id FROM records
                WHERE patient_id = ? AND db_name = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (patient_id, db_name))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        file_path, submission_id = result
        
        # Load from file, fallback to MinIO
        try:
            with open(file_path, 'r') as f:
                record = json.load(f)
            record['submission_id'] = submission_id
            return record
        except FileNotFoundError:
            minio_key = '/'.join(Path(file_path).parts[-2:])
            minio_data = minio_storage.get_json(minio_key)
            if minio_data:
                minio_data['submission_id'] = submission_id
                return minio_data
            return None
    
    def update_record(self, db_name: str, sample_id: str, updates: Dict) -> Dict:
        """
        Update a record
        
        Args:
            db_name: Database name
            sample_id: Sample identifier
            updates: Fields to update
            
        Returns:
            Updated record
        """
        with self._lock:
            # Get existing record
            existing = self.get_record(db_name, sample_id)
            
            if not existing:
                raise ValueError(f"Record not found: {sample_id}")
            
            # Merge updates
            updated_record = {**existing, **updates}
            updated_record['updated_at'] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            
            # Save updated record
            file_path = self._save_record_to_file(db_name, sample_id, updated_record)
            
            # Update SQLite metadata
            conn = sqlite3.connect(self._get_db_path(db_name))
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE records SET file_path = ?
                WHERE sample_id = ? AND db_name = ?
            ''', (file_path, sample_id, db_name))
            
            conn.commit()
            conn.close()
            
            return updated_record
    
    def get_submission_records(self, db_name: str, submission_id: str) -> List[Dict]:
        """
        Get all records for a specific submission
        
        Args:
            db_name: Database name
            submission_id: Submission identifier
            
        Returns:
            List of records for the submission
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT file_path, sample_id, timestamp, score, class
            FROM records
            WHERE submission_id = ? AND db_name = ?
            ORDER BY timestamp
        ''', (submission_id, db_name))
        
        results = cursor.fetchall()
        conn.close()
        
        records = []
        for file_path, sample_id, timestamp, score, class_label in results:
            try:
                with open(file_path, 'r') as f:
                    record = json.load(f)
            except (FileNotFoundError, TypeError):
                # Fallback to MinIO
                minio_key = '/'.join(Path(file_path).parts[-2:]) if file_path else ''
                record = minio_storage.get_json(minio_key) if minio_key else None
                if record is None:
                    continue
            record['submission_id'] = submission_id
            record['sample_id'] = sample_id
            record['timestamp'] = timestamp
            record['score'] = score
            record['class'] = class_label
            records.append(record)
        
        return records
    
    def update_record_score(self, db_name: str, record_id: int, score: float, class_label: str):
        """
        Update the score and class for a specific record by its unique ID

        Args:
            db_name: Database name
            record_id: Unique record identifier (from database)
            score: Prediction score
            class_label: Prediction class
        """
        with sqlite3.connect(self._get_db_path(db_name)) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE records SET score = ?, class = ?
                WHERE id = ? AND db_name = ?
            ''', (score, class_label, record_id, db_name))

            conn.commit()
    
    def update_record_score_by_sample(self, db_name: str, sample_id: str, score: float, 
                                       class_label: str, timestamp: str = None):
        """
        Update the score and class for a record by sample_id (and optional timestamp)
        Used for record editing when record_id is not available.
        
        Args:
            db_name: Database name
            sample_id: Sample identifier
            score: Prediction score
            class_label: Prediction class
            timestamp: Optional timestamp to identify specific record
        """
        with sqlite3.connect(self._get_db_path(db_name)) as conn:
            cursor = conn.cursor()

            if timestamp:
                cursor.execute('''
                    UPDATE records SET score = ?, class = ?
                    WHERE sample_id = ? AND db_name = ? AND timestamp = ?
                ''', (score, class_label, sample_id, db_name, timestamp))
            else:
                # Update the most recent record for this sample_id
                cursor.execute('''
                    UPDATE records SET score = ?, class = ?
                    WHERE id = (
                        SELECT id FROM records
                        WHERE sample_id = ? AND db_name = ?
                        ORDER BY created_at DESC LIMIT 1
                    )
                ''', (score, class_label, sample_id, db_name))

            conn.commit()
    
    def store_submission(self, submission_id: str, db_name: str, payload: Dict, 
                        status: str, records_count: int = 0, failure_reason: str = None):
        """
        Store submission metadata
        
        Args:
            submission_id: Submission identifier
            db_name: Database name
            payload: Original submission payload
            status: Submission status (success/failure/error)
            records_count: Number of records processed
            failure_reason: Optional failure reason
        """
        with self._lock:
            # Save full payload to file
            submission_file = self.submissions_dir / f"{submission_id}.json"
            with open(submission_file, 'w') as f:
                json.dump(payload, f, indent=2)
            
            # Mirror submission to MinIO
            minio_storage.put_json(f"submissions/{submission_id}.json", payload)
            
            # Save metadata to SQLite
            with sqlite3.connect(self._get_db_path(db_name)) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO submissions
                    (submission_id, db_name, client_version, submitted_at, status,
                     score_calculated, records_count, source, original_filename, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    submission_id,
                    db_name,
                    payload.get('client_version', ''),
                    payload.get('submitted_at', datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')),
                    status,
                    1 if status == 'success' else 0,
                    records_count,
                    payload.get('meta', {}).get('source', ''),
                    payload.get('meta', {}).get('filename') or payload.get('filename'),
                    failure_reason
                ))

                conn.commit()

    def delete_submission(self, submission_id: str, db_name: str) -> int:
        """
        Delete a submission and all its associated records from SQLite and JSON files.

        Returns:
            Number of records deleted.
        """
        import os as _os

        db_dir = self._get_db_dir(db_name)

        # Gather record file paths before deletion
        with sqlite3.connect(self._get_db_path(db_name)) as conn:
            cur = conn.cursor()
            cur.execute(
                'SELECT file_path FROM records WHERE submission_id = ? AND db_name = ?',
                (submission_id, db_name)
            )
            file_paths = [row[0] for row in cur.fetchall()]
            # Delete records
            cur.execute(
                'DELETE FROM records WHERE submission_id = ? AND db_name = ?',
                (submission_id, db_name)
            )
            deleted_count = cur.rowcount
            # Delete submission metadata
            cur.execute(
                'DELETE FROM submissions WHERE submission_id = ?',
                (submission_id,)
            )
            conn.commit()

        # Remove JSON files and MinIO mirrors
        for file_path in file_paths:
            try:
                if file_path and _os.path.exists(file_path):
                    _os.remove(file_path)
                    filename = Path(file_path).name
                    minio_storage.delete_object(f"{db_name}/{filename}")
            except Exception as exc:
                logger.warning(f"Could not remove record file {file_path}: {exc}")

        # Remove submission JSON file
        submission_file = self.submissions_dir / f"{submission_id}.json"
        try:
            if submission_file.exists():
                submission_file.unlink()
                minio_storage.delete_object(f"submissions/{submission_id}.json")
        except Exception as exc:
            logger.warning(f"Could not remove submission file {submission_file}: {exc}")

        return deleted_count

    def find_submission(self, submission_id: str) -> Optional[Dict]:
        """
        Cerca una submission per ID su tutti i database (mimic.db, sepsiexp.db, prism.db).

        Returns:
            Dict con i campi della submission (incluso 'submission_id') oppure None se non trovata.
        """
        query = '''
            SELECT db_name, status, submitted_at, records_count, source,
                   original_filename, client_version, completion_time_ms, failure_reason
            FROM submissions WHERE submission_id = ?
        '''
        query_ilike = '''
            SELECT submission_id,
                   db_name, status, submitted_at, records_count, source,
                   original_filename, client_version, completion_time_ms, failure_reason
            FROM submissions WHERE LOWER(submission_id) = LOWER(?)
        '''
        for db_path in [self.mimic_db_path, self.sepsiexp_db_path, self.db_path]:
            try:
                with sqlite3.connect(db_path) as conn:
                    cur = conn.cursor()
                    cur.execute(query, (submission_id,))
                    row = cur.fetchone()
                    if row:
                        return {
                            'submission_id': submission_id,
                            'db_name': row[0],
                            'status': row[1],
                            'submitted_at': row[2],
                            'records_count': row[3],
                            'source': row[4],
                            'original_filename': row[5],
                            'client_version': row[6],
                            'completion_time_ms': row[7],
                            'failure_reason': row[8],
                        }
                    # prova case-insensitive
                    cur.execute(query_ilike, (submission_id,))
                    row = cur.fetchone()
                    if row:
                        return {
                            'submission_id': row[0],
                            'db_name': row[1],
                            'status': row[2],
                            'submitted_at': row[3],
                            'records_count': row[4],
                            'source': row[5],
                            'original_filename': row[6],
                            'client_version': row[7],
                            'completion_time_ms': row[8],
                            'failure_reason': row[9],
                        }
            except Exception:
                pass
        return None

    def get_submissions(self, page: int = 1, page_size: int = 50,
                       db_filter: str = 'all',
                       submission_id: str = None,
                       sample_id: str = None,
                       date_from: str = None,
                       date_to: str = None) -> Dict:
        """
        Get paginated list of submissions.
        When db_filter='all', both mimic.db and sepsiexp.db are queried and merged.
        Supports filtering by submission_id, sample_id, and date range.
        """
        def _fetch(db_path, db_name_filter=None):
            c = sqlite3.connect(db_path)
            cur = c.cursor()

            # Build WHERE clauses dynamically
            conditions = []
            params = []

            if db_name_filter:
                conditions.append('s.db_name = ?')
                params.append(db_name_filter)

            if submission_id:
                conditions.append('s.submission_id LIKE ?')
                params.append(f'%{submission_id}%')

            if date_from:
                conditions.append('s.submitted_at >= ?')
                params.append(date_from)

            if date_to:
                # Include the entire end day
                conditions.append('s.submitted_at < ?')
                params.append(date_to + 'T23:59:59.999999')

            if sample_id:
                # Join with records table to filter by sample_id
                join_clause = 'INNER JOIN records r ON s.submission_id = r.submission_id'
                if conditions:
                    where_clause = 'WHERE ' + ' AND '.join(conditions) + ' AND r.sample_id LIKE ?'
                else:
                    where_clause = 'WHERE r.sample_id LIKE ?'
                params.append(f'%{sample_id}%')
                cur.execute(f'''
                    SELECT DISTINCT s.submission_id, s.db_name, s.client_version, s.submitted_at, s.status,
                           s.score_calculated, s.records_count, s.source, s.original_filename,
                           s.completion_time_ms, s.failure_reason
                    FROM submissions s {join_clause}
                    {where_clause}
                    ORDER BY s.submitted_at DESC
                ''', params)
            else:
                where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''
                cur.execute(f'''
                    SELECT submission_id, db_name, client_version, submitted_at, status,
                           score_calculated, records_count, source, original_filename,
                           completion_time_ms, failure_reason
                    FROM submissions s
                    {where_clause}
                    ORDER BY submitted_at DESC
                ''', params)

            rows = cur.fetchall()
            c.close()
            return rows

        if db_filter == 'all':
            rows = _fetch(self.mimic_db_path) + _fetch(self.sepsiexp_db_path)
            # Also include any legacy records from prism.db not yet migrated
            rows += _fetch(self.db_path)
            rows.sort(key=lambda r: r[3] or '', reverse=True)
        else:
            rows = _fetch(self._get_db_path(db_filter), db_filter)

        # Deduplicate by submission_id (legacy prism.db may overlap)
        seen = set()
        unique_rows = []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                unique_rows.append(r)
        rows = unique_rows

        total_submissions = len(rows)
        offset = (page - 1) * page_size
        page_rows = rows[offset:offset + page_size]

        submissions = []
        for row in page_rows:
            submission = {
                'submission_id': row[0],
                'db': row[1],
                'client_version': row[2],
                'submitted_at': row[3],
                'status': row[4],
                'score_calculated': bool(row[5]),
                'records_count': row[6],
                'source': row[7],
                'file_info': {
                    'original_filename': row[8] or None
                }
            }
            if row[9]:
                submission['completion_time_ms'] = row[9]
            if row[10]:
                submission['failure_reason'] = row[10]
            submissions.append(submission)

        return {
            'status': 'success',
            'total_submissions': total_submissions,
            'page_size': page_size,
            'page_current': page,
            'page_total': (total_submissions + page_size - 1) // page_size if total_submissions else 1,
            'submissions': submissions
        }
    
    def get_stats(self, db_name: str) -> Dict:
        """
        Get statistics for a database
        
        Args:
            db_name: Database name
            
        Returns:
            Dictionary with counts
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc)
        today_str = now.strftime('%Y-%m-%d')  # Format: 2025-12-11
        week_ago = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        month_ago = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        year_ago = (now - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Total entries
        cursor.execute('SELECT COUNT(*) FROM records WHERE db_name = ?', (db_name,))
        total = cursor.fetchone()[0]
        
        # Today (using DATE() function to extract date from timestamp)
        cursor.execute('''
            SELECT COUNT(*) FROM records 
            WHERE db_name = ? AND DATE(created_at) = ?
        ''', (db_name, today_str))
        today_count = cursor.fetchone()[0]
        
        # This week
        cursor.execute('''
            SELECT COUNT(*) FROM records 
            WHERE db_name = ? AND DATE(created_at) >= ?
        ''', (db_name, week_ago))
        week_count = cursor.fetchone()[0]
        
        # This month
        cursor.execute('''
            SELECT COUNT(*) FROM records 
            WHERE db_name = ? AND DATE(created_at) >= ?
        ''', (db_name, month_ago))
        month_count = cursor.fetchone()[0]
        
        # This year
        cursor.execute('''
            SELECT COUNT(*) FROM records 
            WHERE db_name = ? AND DATE(created_at) >= ?
        ''', (db_name, year_ago))
        year_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_entries': total,
            'submitted_today': today_count,
            'submitted_this_week': week_count,
            'submitted_this_month': month_count,
            'submitted_this_year': year_count
        }
    
    def get_recent_scores(self, db_name: str, limit: int = 100) -> List[Dict]:
        """
        Get recent prediction scores for charting
        
        Args:
            db_name: Database name
            limit: Maximum number of records to return
            
        Returns:
            List of score records
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, submission_id, sample_id, score, class
            FROM records
            WHERE db_name = ? AND score IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (db_name, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        scores = []
        for row in rows:
            scores.append({
                'timestamp': row[0],
                'submission_id': row[1],
                'sample_id': row[2],
                'score': row[3],
                'class': row[4]
            })
        
        return scores
    
    def get_patient_history(self, db_name: str, sample_id: str, 
                           start_date: Optional[str] = None, 
                           end_date: Optional[str] = None,
                           limit: int = 10000) -> List[Dict]:
        """
        Get complete history for a specific patient
        
        Args:
            db_name: Database name
            sample_id: Patient sample identifier
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            limit: Maximum number of records to return (default 10000)
            
        Returns:
            List of patient records with scores and predictions
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        query = '''
            SELECT id, timestamp, submission_id, score, class, file_path, created_at
            FROM records
            WHERE sample_id = ? AND db_name = ?
        '''
        params = [sample_id, db_name]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            record_id, timestamp, submission_id, score, class_label, file_path, created_at = row
            
            # Load full record from file
            record_data = {}
            if file_path:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            record_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load record file {file_path}: {e}")
                else:
                    # Fallback to MinIO
                    minio_key = '/'.join(Path(file_path).parts[-2:])
                    minio_data = minio_storage.get_json(minio_key)
                    if minio_data:
                        record_data = minio_data
            
            history.append({
                'record_id': record_id,
                'timestamp': timestamp,
                'submission_id': submission_id,
                'score': score,
                'class': class_label,
                'created_at': created_at,
                'data': record_data
            })
        
        return history
    
    def get_patient_submissions(self, db_name: str, sample_id: str) -> List[Dict]:
        """
        Get submission statistics for a specific patient
        
        Args:
            db_name: Database name
            sample_id: Patient sample identifier
            
        Returns:
            List of submissions with counts
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT submission_id, 
                   COUNT(*) as total,
                   SUM(CASE WHEN class = 'high_risk' OR class = 'moderate_risk' THEN 1 ELSE 0 END) as positives,
                   SUM(CASE WHEN class = 'low_risk' THEN 1 ELSE 0 END) as negatives,
                   MIN(created_at) as first_created,
                   AVG(score) as avg_score
            FROM records
            WHERE sample_id = ? AND db_name = ?
            GROUP BY submission_id
            ORDER BY first_created DESC
        ''', (sample_id, db_name))
        
        rows = cursor.fetchall()
        conn.close()
        
        submissions = []
        for row in rows:
            submissions.append({
                'submission_id': row[0],
                'total': row[1],
                'positives': row[2] or 0,
                'negatives': row[3] or 0,
                'created_at': row[4],
                'avg_score': float(row[5]) if row[5] else 0
            })
        
        return submissions
    
    def get_patient_statistics(self, db_name: str, sample_id: str) -> Dict:
        """
        Get statistical summary for a specific patient
        
        Args:
            db_name: Database name
            sample_id: Patient sample identifier
            
        Returns:
            Dictionary with patient statistics
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        # Count total records
        cursor.execute('''
            SELECT COUNT(*) FROM records
            WHERE sample_id = ? AND db_name = ?
        ''', (sample_id, db_name))
        total_records = cursor.fetchone()[0]
        
        if total_records == 0:
            conn.close()
            return {
                'sample_id': sample_id,
                'total_records': 0,
                'has_data': False
            }
        
        # Get score statistics
        cursor.execute('''
            SELECT MIN(score), MAX(score), AVG(score), 
                   SUM(CASE WHEN class IN ('high_risk', 'moderate_risk') THEN 1 ELSE 0 END) as positive_count,
                   SUM(CASE WHEN class = 'low_risk' THEN 1 ELSE 0 END) as negative_count,
                   MIN(timestamp), MAX(timestamp)
            FROM records
            WHERE sample_id = ? AND db_name = ? AND score IS NOT NULL
        ''', (sample_id, db_name))
        
        stats_row = cursor.fetchone()
        conn.close()
        
        min_score, max_score, avg_score, positive_count, negative_count, first_timestamp, last_timestamp = stats_row
        
        return {
            'sample_id': sample_id,
            'db_name': db_name,
            'total_records': total_records,
            'has_data': True,
            'score_statistics': {
                'min': float(min_score) if min_score is not None else None,
                'max': float(max_score) if max_score is not None else None,
                'average': float(avg_score) if avg_score is not None else None
            },
            'risk_distribution': {
                'positive': positive_count or 0,
                'negative': negative_count or 0
            },
            'temporal_range': {
                'first_record': first_timestamp,
                'last_record': last_timestamp
            }
        }
    
    def list_patients(self, db_name: str, page: int = 1, page_size: int = 50, search: Optional[str] = None) -> Dict:
        """
        List all unique patients in database
        
        Args:
            db_name: Database name
            page: Page number (1-indexed)
            page_size: Records per page
            search: Optional search filter for sample_id (partial match)
            
        Returns:
            Dictionary with patient list and pagination info
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_clause = "db_name = ?"
        params = [db_name]
        if search:
            where_clause += " AND sample_id LIKE ?"
            params.append(f"%{search}%")
        
        # Count unique patients
        cursor.execute(f'''
            SELECT COUNT(DISTINCT sample_id)
            FROM records
            WHERE {where_clause}
        ''', params)
        total_patients = cursor.fetchone()[0]
        
        # Get patient list with record counts
        offset = (page - 1) * page_size
        cursor.execute(f'''
            SELECT sample_id, COUNT(*) as record_count, 
                   MAX(timestamp) as last_updated, 
                   MAX(score) as max_score
            FROM records
            WHERE {where_clause}
            GROUP BY sample_id
            ORDER BY MAX(created_at) DESC
            LIMIT ? OFFSET ?
        ''', params + [page_size, offset])
        
        rows = cursor.fetchall()
        conn.close()
        
        patients = []
        for row in rows:
            patients.append({
                'sample_id': row[0],
                'record_count': row[1],
                'last_updated': row[2],
                'max_score': row[3]
            })
        
        return {
            'db_name': db_name,
            'total_patients': total_patients,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_patients + page_size - 1) // page_size,
            'patients': patients
        }

    def list_all_records(self, db_name: str, page: int = 1, page_size: int = 50,
                          sample_id_filter: Optional[str] = None) -> Dict:
        """
        List all individual records (not aggregated by sample_id)
        
        Args:
            db_name: Database name ('mimic' or 'sepsiexp')
            page: Page number (1-indexed)
            page_size: Records per page
            sample_id_filter: Optional filter by sample_id (partial match)
            
        Returns:
            Dictionary with records list, pagination info, and clinical data
        """
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        # Build query with optional filter
        count_query = 'SELECT COUNT(*) FROM records WHERE db_name = ?'
        params = [db_name]
        
        if sample_id_filter:
            count_query += ' AND sample_id LIKE ?'
            params.append(f'%{sample_id_filter}%')
        
        cursor.execute(count_query, params)
        total_records = cursor.fetchone()[0]
        
        # Get records with pagination
        offset = (page - 1) * page_size
        select_query = '''
            SELECT id, sample_id, timestamp, submission_id, score, class, file_path, created_at
            FROM records
            WHERE db_name = ?
        '''
        params = [db_name]
        
        if sample_id_filter:
            select_query += ' AND sample_id LIKE ?'
            params.append(f'%{sample_id_filter}%')
        
        select_query += ' ORDER BY created_at DESC, timestamp DESC LIMIT ? OFFSET ?'
        params.extend([page_size, offset])
        
        cursor.execute(select_query, params)
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            record_id, sample_id, timestamp, submission_id, score, class_label, file_path, created_at = row
            
            # Load clinical data from JSON file
            clinical_data = {}
            if file_path:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            clinical_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not load record file {file_path}: {e}")
                else:
                    minio_key = '/'.join(Path(file_path).parts[-2:])
                    minio_data = minio_storage.get_json(minio_key)
                    if minio_data:
                        clinical_data = minio_data
            
            records.append({
                'record_id': record_id,
                'sample_id': sample_id,
                'timestamp': timestamp,
                'submission_id': submission_id,
                'score': float(score) if score is not None else None,
                'class': class_label,
                'created_at': created_at,
                'clinical_data': clinical_data
            })
        
        return {
            'db_name': db_name,
            'total_records': total_records,
            'page': page,
            'page_size': page_size,
            'total_pages': (total_records + page_size - 1) // page_size if total_records > 0 else 1,
            'records': records
        }

    def get_critical_statistics(self, db_name: str, sample_id_filter: Optional[str] = None) -> Dict:
        """
        Calculate statistics for critical clinical values across all records
        
        Args:
            db_name: Database name ('mimic' or 'sepsiexp')
            sample_id_filter: Optional filter by sample_id
            
        Returns:
            Dictionary with min/max/avg for each critical field
        """
        from api.field_ranges import MIMIC_RANGES, SEPSIEXP_RANGES
        
        # Define critical fields for each database
        if db_name.lower() == 'mimic':
            critical_fields = [
                'meanbp', 'heartrate', 'resprate', 'spo2_pulsoxy', 'tempc',
                'lactate', 'creatinine', 'wbc', 'hemoglobin', 'platelet',
                'bilirubin', 'glucose', 'potassium', 'sodium', 'ph'
            ]
            field_ranges = MIMIC_RANGES
        else:
            critical_fields = [
                'meanbp', 'heartrate', 'oxygensaturation', 'lactate',
                'creatinine', 'bilirubin', 'leukocytes', 'hemoglobin',
                'pct', 'crp', 'potassium', 'sodium', 'arterialph'
            ]
            field_ranges = SEPSIEXP_RANGES
        
        conn = sqlite3.connect(self._get_db_path(db_name))
        cursor = conn.cursor()
        
        # Get all file paths
        query = 'SELECT file_path FROM records WHERE db_name = ?'
        params = [db_name]
        
        if sample_id_filter:
            query += ' AND sample_id LIKE ?'
            params.append(f'%{sample_id_filter}%')
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Collect values for each critical field
        field_values = {field: [] for field in critical_fields}
        
        for (file_path,) in rows:
            data = None
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                except Exception:
                    pass

            # Fallback: try MinIO if local file missing
            if data is None and file_path:
                try:
                    parts = Path(file_path).parts
                    minio_key = '/'.join(parts[-2:]) if len(parts) >= 2 else Path(file_path).name
                    data = minio_storage.get_json(minio_key)
                except Exception:
                    pass

            if data:
                for field in critical_fields:
                    if field in data and data[field] is not None:
                        try:
                            val = float(data[field])
                            field_values[field].append(val)
                        except (ValueError, TypeError):
                            pass
        
        # Calculate statistics
        statistics = {}
        for field in critical_fields:
            values = field_values[field]
            field_info = field_ranges.get(field, {})
            
            if values:
                statistics[field] = {
                    'label': field_info.get('label', field),
                    'unit': field_info.get('unit', ''),
                    'count': len(values),
                    'min': round(min(values), 2),
                    'max': round(max(values), 2),
                    'avg': round(sum(values) / len(values), 2),
                    'normal_min': field_info.get('min'),
                    'normal_max': field_info.get('max')
                }
            else:
                statistics[field] = {
                    'label': field_info.get('label', field),
                    'unit': field_info.get('unit', ''),
                    'count': 0,
                    'min': None,
                    'max': None,
                    'avg': None,
                    'normal_min': field_info.get('min'),
                    'normal_max': field_info.get('max')
                }
        
        return {
            'db_name': db_name,
            'total_records_analyzed': len(rows),
            'critical_statistics': statistics
        }
