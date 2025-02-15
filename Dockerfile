# Use Python 3.12 as base image
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/documents /app/logs && \
    chmod -R 755 /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Make scripts executable
RUN chmod +x start_services.sh

# Expose the API port
EXPOSE 8000

# Start services using the script
ENTRYPOINT ["./start_services.sh"]
