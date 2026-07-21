#!/bin/bash
# MinIO initialization script
set -e

echo "Creating MinIO bucket: ${MINIO_BUCKET:-prism-data}"

mc alias set local ${MINIO_ENDPOINT} ${MINIO_ACCESS_KEY} ${MINIO_SECRET_KEY}

mc mb --ignore-existing local/${MINIO_BUCKET:-prism-data}

if [ -d /seed_data ]; then
    echo "Seeding data from /seed_data..."
    mc cp --recursive /seed_data/ local/${MINIO_BUCKET:-prism-data}/
fi

echo "MinIO initialization complete."