#!/bin/bash

# Create necessary directories
echo "Creating required directories..."
mkdir -p /app/documents
mkdir -p /app/logs
echo "Created required directories"

# Start Redis
echo "Starting Redis..."
redis-server /app/redis.conf &
REDIS_PID=$!

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
until redis-cli ping > /dev/null 2>&1; do
  echo "Redis not ready - waiting..."
  sleep 1
done
echo "Redis is ready!"

# Start Celery worker with detailed logging
echo "Starting Celery worker..."
celery -A celery_app worker --loglevel=INFO --uid=1000 --gid=1000 > /app/logs/celery.log 2>&1 &
CELERY_PID=$!

# Start FastAPI application
echo "Starting FastAPI application..."
uvicorn main:app --host 0.0.0.0 --port 8000 --log-level debug > /app/logs/fastapi.log 2>&1 &
FASTAPI_PID=$!

# Function to forward signals to child processes
cleanup() {
    echo "Shutting down services..."
    kill $REDIS_PID
    kill $CELERY_PID
    kill $FASTAPI_PID
    exit 0
}

# Set up signal handling
trap cleanup SIGTERM SIGINT SIGQUIT

# Monitor logs
echo "Monitoring services..."
tail -f /app/logs/*.log &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
