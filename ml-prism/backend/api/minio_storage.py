"""
MinIO Storage Adapter for PRISM API
Provides transparent S3-compatible object storage for JSON records,
submissions and charts — replacing (or sitting alongside) local filesystem.
"""

import os
import io
import json
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy-loaded minio client (thread-safe via lock: THREAD-1)
_minio_client = None
_minio_lock = threading.Lock()

# Cache MINIO_ENABLED at import time to avoid repeated env-var reads (SMELL-5)
_MINIO_ENABLED: bool = os.environ.get('MINIO_ENABLED', 'false').lower() in ('true', '1', 'yes')


def _is_enabled() -> bool:
    return _MINIO_ENABLED


def _get_client():
    """Return a thread-safe singleton Minio client (lazy init)."""
    global _minio_client
    # Fast path (no lock) once initialised
    if _minio_client is not None:
        return _minio_client

    with _minio_lock:
        # Re-check inside lock (double-checked locking)
        if _minio_client is not None:
            return _minio_client

        try:
            from minio import Minio
        except ImportError:
            logger.warning("minio package not installed — MinIO storage disabled")
            return None

        endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
        access_key = os.environ.get('MINIO_ACCESS_KEY')
        secret_key = os.environ.get('MINIO_SECRET_KEY')
        if not access_key or not secret_key:
            logger.warning("MINIO_ACCESS_KEY / MINIO_SECRET_KEY not set — MinIO storage disabled (SEC-6)")
            return None
        if access_key == 'minioadmin' or secret_key == 'minioadmin':
            logger.warning("MinIO is using default credentials 'minioadmin'. Replace them before production use.")
        secure = os.environ.get('MINIO_SECURE', 'false').lower() in ('true', '1', 'yes')

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)

        bucket = _bucket_name()
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            logger.info("Created MinIO bucket: %s", bucket)

        _minio_client = client
        return _minio_client


def _bucket_name() -> str:
    return os.environ.get('MINIO_BUCKET', 'prism-data')


# -------------------------------------------------------------------
#  Public API
# -------------------------------------------------------------------

def is_available() -> bool:
    """Return True when MinIO is enabled *and* reachable."""
    if not _is_enabled():
        return False
    try:
        client = _get_client()
        return client is not None
    except Exception as exc:
        logger.warning("MinIO not reachable: %s", exc)
        return False


def put_json(object_name: str, data: dict) -> bool:
    """
    Upload a JSON document to MinIO.

    Args:
        object_name: key inside the bucket, e.g. ``mimic/200001_2026-01-12.json``
        data: serialisable dict
    Returns:
        True on success, False otherwise
    """
    if not _is_enabled():
        return False
    try:
        client = _get_client()
        payload = json.dumps(data, indent=2).encode('utf-8')
        client.put_object(
            _bucket_name(),
            object_name,
            io.BytesIO(payload),
            length=len(payload),
            content_type='application/json',
        )
        return True
    except Exception as exc:
        logger.error("MinIO put_json(%s) failed: %s", object_name, exc)
        return False


def get_json(object_name: str) -> Optional[dict]:
    """
    Download and parse a JSON document from MinIO.
    Returns None when the object does not exist or on error.
    """
    if not _is_enabled():
        return None
    try:
        client = _get_client()
        response = client.get_object(_bucket_name(), object_name)
        data = json.loads(response.read())
        response.close()
        response.release_conn()
        return data
    except Exception as exc:
        logger.debug("MinIO get_json(%s) failed: %s", object_name, exc)
        return None


def put_file(object_name: str, file_path: str, content_type: str = 'application/octet-stream') -> bool:
    """Upload any local file to MinIO."""
    if not _is_enabled():
        return False
    try:
        client = _get_client()
        client.fput_object(_bucket_name(), object_name, file_path, content_type=content_type)
        return True
    except Exception as exc:
        logger.error("MinIO put_file(%s) failed: %s", object_name, exc)
        return False


def get_file_bytes(object_name: str) -> Optional[bytes]:
    """Download raw bytes from MinIO.  Returns None on error."""
    if not _is_enabled():
        return None
    try:
        client = _get_client()
        response = client.get_object(_bucket_name(), object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except Exception as exc:
        logger.debug("MinIO get_file_bytes(%s) failed: %s", object_name, exc)
        return None


def delete_object(object_name: str) -> bool:
    """Delete an object from MinIO."""
    if not _is_enabled():
        return False
    try:
        client = _get_client()
        client.remove_object(_bucket_name(), object_name)
        return True
    except Exception as exc:
        logger.error("MinIO delete_object(%s) failed: %s", object_name, exc)
        return False


def list_objects(prefix: str = '', recursive: bool = True):
    """
    Yield object names under *prefix*.
    Returns an empty iterator when MinIO is disabled.
    """
    if not _is_enabled():
        return
    try:
        client = _get_client()
        for obj in client.list_objects(_bucket_name(), prefix=prefix, recursive=recursive):
            yield obj.object_name
    except Exception as exc:
        logger.error("MinIO list_objects(%s) failed: %s", prefix, exc)
