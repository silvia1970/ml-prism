"""Database Module for PRISM API."""
import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
import threading

from src.utils.field_mappings import extract_sample_id, normalize_record_ids, get_sample_id_field

logger = logging.getLogger(__name__)


class Database:
    """Database handler for PRISM API."""

    def __init__(self, data_dir: str = 'api_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.mimic_dir = self.data_dir / 'mimic'
        self.sepsiexp_dir = self.data_dir / 'sepsiexp'
        self.submissions_dir = self.data_dir / 'submissions'
        for d in [self.mimic_dir, self.sepsiexp_dir, self.submissions_dir]:
            d.mkdir(exist_ok=True)
        self.db_path = self.data_dir / 'prism.db'
        self.mimic_db_path = self.data_dir / 'mimic.db'
        self.sepsiexp_db_path = self.data_dir / 'sepsiexp.db'
        self._init_sqlite()
        self._lock = threading.Lock()

    def _init_sqlite(self):
        for db_path in [self.db_path, self.mimic_db_path, self.sepsiexp_db_path]:
            self._init_single_db(db_path)

    def _init_single_db(self, db_path):
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('PRAGMA journal_mode=WAL')
            cursor.execute('''CREATE TABLE IF NOT EXISTS submissions (
                submission_id TEXT PRIMARY KEY, db_name TEXT NOT NULL, client_version TEXT,
                submitted_at TEXT NOT NULL, status TEXT NOT NULL, score_calculated INTEGER DEFAULT 0,
                records_count INTEGER DEFAULT 0, source TEXT, original_filename TEXT,
                completion_time_ms INTEGER, failure_reason TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL, db_name TEXT NOT NULL,
                submission_id TEXT, timestamp TEXT, score REAL, class TEXT, file_path TEXT,
                data TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_sample_id ON records(sample_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_db_name ON records(db_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_records_submission_id ON records(submission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_submissions_db_name ON submissions(db_name)')
            conn.commit()

    def _get_db(self, db_name: str):
        if db_name == 'mimic':
            return self.mimic_db_path
        elif db_name == 'sepsiexp':
            return self.sepsiexp_db_path
        return self.db_path

    def save_record(self, db_name: str, record: Dict, submission_id: str = None) -> int:
        with self._lock:
            db_path = self._get_db(db_name)
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                sample_id = extract_sample_id(record, db_name)
                cursor.execute(
                    'INSERT INTO records (sample_id, db_name, submission_id, timestamp, data) VALUES (?, ?, ?, ?, ?)',
                    (sample_id, db_name, submission_id,
                     record.get('timestamp') or datetime.now(timezone.utc).isoformat(),
                     json.dumps(record)))
                conn.commit()
                return cursor.lastrowid

    def save_batch(self, db_name: str, records: List[Dict], submission_id: str = None) -> int:
        count = 0
        for record in records:
            self.save_record(db_name, record, submission_id)
            count += 1
        return count

    def get_records(self, db_name: str, sample_id: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        db_path = self._get_db(db_name)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if sample_id:
                cursor.execute(
                    'SELECT * FROM records WHERE sample_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
                    (sample_id, limit, offset))
            else:
                cursor.execute('SELECT * FROM records ORDER BY created_at DESC LIMIT ? OFFSET ?', (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def update_score(self, db_name: str, record_id: int, score: float, class_label: str):
        with self._lock:
            db_path = self._get_db(db_name)
            with sqlite3.connect(db_path) as conn:
                conn.execute('UPDATE records SET score = ?, class = ? WHERE id = ?', (score, class_label, record_id))
                conn.commit()

    def save_submission(self, submission: Dict):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''INSERT OR REPLACE INTO submissions
                    (submission_id, db_name, client_version, submitted_at, status,
                     records_count, source, original_filename, completion_time_ms, failure_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (submission.get('submission_id'), submission.get('db_name'),
                     submission.get('client_version'), submission.get('submitted_at'),
                     submission.get('status', 'pending'), submission.get('records_count', 0),
                     submission.get('source'), submission.get('original_filename'),
                     submission.get('completion_time_ms'), submission.get('failure_reason')))
                conn.commit()

    def get_submissions(self, db_name: str = None, submission_id: str = None,
                         date_from: str = None, date_to: str = None,
                         limit: int = 100, offset: int = 0) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = 'SELECT * FROM submissions WHERE 1=1'
            params = []
            if db_name:
                query += ' AND db_name = ?'
                params.append(db_name)
            if submission_id:
                query += ' AND submission_id = ?'
                params.append(submission_id)
            if date_from:
                query += ' AND submitted_at >= ?'
                params.append(date_from)
            if date_to:
                query += ' AND submitted_at <= ?'
                params.append(date_to)
            query += ' ORDER BY submitted_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_record_count(self, db_name: str) -> int:
        db_path = self._get_db(db_name)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM records')
            return cursor.fetchone()['count']