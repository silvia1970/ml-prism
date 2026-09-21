#!/bin/sh
# ============================================================
#  MinIO Init — create bucket and seed existing api_data
# ============================================================
set -e

echo "=== MinIO Init: configuring alias ==="
mc alias set prism "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}"

echo "=== Creating bucket: ${MINIO_BUCKET} ==="
mc mb --ignore-existing "prism/${MINIO_BUCKET}"

# Seed existing api_data if the seed directory contains files
if [ -d "/seed_data" ] && [ "$(ls -A /seed_data 2>/dev/null)" ]; then
    echo "=== Seeding existing api_data into MinIO ==="

    # Mirror each subdirectory preserving the structure:
    #   prism-data/mimic/...
    #   prism-data/sepsiexp/...
    #   prism-data/submissions/...
    #   prism-data/charts/...
    for subdir in mimic sepsiexp submissions charts; do
        if [ -d "/seed_data/${subdir}" ] && [ "$(ls -A /seed_data/${subdir} 2>/dev/null)" ]; then
            echo "  -> Uploading ${subdir}/ ..."
            mc mirror --overwrite "/seed_data/${subdir}/" "prism/${MINIO_BUCKET}/${subdir}/"
        fi
    done

    # Upload SQLite db files to the bucket root (for backup / portability)
    for dbfile in /seed_data/*.db; do
        if [ -f "${dbfile}" ]; then
            fname=$(basename "${dbfile}")
            echo "  -> Uploading ${fname}"
            mc cp "${dbfile}" "prism/${MINIO_BUCKET}/${fname}"
        fi
    done

    echo "=== Seed complete ==="
else
    echo "=== No seed data found, bucket is empty ==="
fi

# Copy SQLite databases into the backend shared volume
# so the Flask app can query them directly
if [ -d "/backend_data" ]; then
    echo "=== Copying SQLite databases to backend volume ==="
    # Create subdirectories
    mkdir -p /backend_data/mimic /backend_data/sepsiexp /backend_data/submissions /backend_data/charts
    for dbfile in /seed_data/*.db; do
        if [ -f "${dbfile}" ]; then
            fname=$(basename "${dbfile}")
            # Only copy if missing (don't overwrite existing data)
            if [ ! -f "/backend_data/${fname}" ]; then
                echo "  -> Seeding ${fname} to backend volume"
                cp "${dbfile}" "/backend_data/${fname}"
            else
                echo "  -> ${fname} already exists in backend volume, skipping"
            fi
        fi
    done
fi

echo "=== MinIO init finished ==="
