#!/bin/bash

# Create necessary directories
echo "Creating required directories..."
mkdir -p /app/documents
mkdir -p /app/logs
echo "Created required directories"

# Start Celery worker with detailed logging
echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=INFO --uid=1000 --gid=1000 > /app/logs/celery.log 2>&1 &
CELERY_PID=$!

# Function to cleanup background processes
cleanup() {
    echo "Shutting down services..."
    kill $CELERY_PID
    exit 0
}

# Set up signal handling
trap cleanup SIGTERM SIGINT SIGQUIT

# Start FastAPI application in foreground
echo "Starting FastAPI application..."
PORT="${PORT:-8000}"  # Use Railway's PORT env var or default to 8000
exec uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info
