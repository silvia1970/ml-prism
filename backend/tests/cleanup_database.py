#!/usr/bin/env python3
"""
Database Cleanup Script for PRISM API
Removes all data and resets the database to a clean state
"""

import os
import sqlite3
import shutil
from pathlib import Path


def cleanup_database(data_dir='api_data'):
    """
    Clean up all database files and reset to initial state
    
    Args:
        data_dir: Directory containing database files
    """
    data_path = Path(data_dir)
    
    print("🧹 Starting PRISM Database Cleanup...")
    print(f"📁 Data directory: {data_path.absolute()}")
    
    if not data_path.exists():
        print("✅ Data directory doesn't exist. Nothing to clean.")
        return
    
    # Backup SQLite database if it exists
    db_file = data_path / 'prism.db'
    if db_file.exists():
        backup_file = data_path / f'prism_backup_{os.getpid()}.db'
        print(f"💾 Backing up SQLite database to: {backup_file}")
        shutil.copy2(db_file, backup_file)
    
    # Close any open SQLite connections
    try:
        conn = sqlite3.connect(db_file)
        conn.close()
    except:
        pass
    
    # Remove all JSON files in subdirectories
    subdirs = ['mimic', 'sepsiexp', 'submissions', 'charts']
    
    for subdir in subdirs:
        subdir_path = data_path / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.rglob('*')))
            print(f"🗑️  Removing {file_count} files from {subdir}/")
            shutil.rmtree(subdir_path)
            subdir_path.mkdir(exist_ok=True)
            print(f"✅ {subdir}/ cleaned and recreated")
    
    # Reset SQLite database
    if db_file.exists():
        print("🔄 Resetting SQLite database...")
        os.remove(db_file)
        
        # Recreate with empty schema
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
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
                submission_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                score REAL,
                class TEXT,
                file_path TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (submission_id) REFERENCES submissions(submission_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_records_sample_id 
            ON records(sample_id, db_name)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_records_submission_id 
            ON records(submission_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_submissions_db_name 
            ON submissions(db_name)
        ''')
        
        conn.commit()
        conn.close()
        
        print("✅ SQLite database reset with clean schema")
    
    print("\n🎉 Database cleanup completed successfully!")
    print(f"📊 Backup available at: {backup_file if db_file.exists() else 'N/A'}")
    print("\n📋 Summary:")
    print("  - All submission data removed")
    print("  - All record data removed")
    print("  - All charts removed")
    print("  - Database schema recreated")
    print("  - Ready for new data!")


def confirm_cleanup():
    """Ask for user confirmation before cleaning"""
    print("\n⚠️  WARNING: This will delete ALL data in the PRISM database!")
    print("This action cannot be undone (except from backup).\n")
    
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        return True
    else:
        print("❌ Cleanup cancelled.")
        return False


if __name__ == '__main__':
    import sys
    
    # Check for --force flag to skip confirmation
    force = '--force' in sys.argv or '-f' in sys.argv
    
    if force or confirm_cleanup():
        cleanup_database()
    else:
        sys.exit(0)
