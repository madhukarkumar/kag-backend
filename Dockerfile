# Use Python 3.12 as base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set up app directory and required subdirectories
WORKDIR /app
RUN mkdir -p /app/documents /app/logs /app/config && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Ensure proper permissions
RUN chmod +x start_services.sh && \
    chmod 644 redis.conf && \
    chmod -R 777 /app/documents /app/logs

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Start services using the script
CMD ["./start_services.sh"]
