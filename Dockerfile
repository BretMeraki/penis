# Use an official Python runtime as a parent image
# Choose a version compatible with your application (e.g., 3.11, 3.12)
# Using slim-bullseye for a smaller image size
FROM python:3.11-slim-bullseye

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app:/app/forest_app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - curl: needed to download the proxy
# - ca-certificates: needed for SSL connections by the proxy
# - graphviz: needed for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates graphviz && \
    rm -rf /var/lib/apt/lists/*

# Download and install the Cloud SQL Auth Proxy
# Check for the latest version and correct architecture (amd64) if needed
# See: https://github.com/GoogleCloudPlatform/cloud-sql-proxy/releases
RUN curl -o /usr/local/bin/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.3/cloud-sql-proxy.linux.amd64 && \
    chmod +x /usr/local/bin/cloud-sql-proxy

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
# First, create the forest_app directory structure
RUN mkdir -p /app/forest_app
COPY forest_app/ /app/forest_app/

# Copy Alembic configuration and migrations
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic/

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on (should match the port in entrypoint.sh)
EXPOSE 8000

# Run the entrypoint script when the container launches
# This script will start the proxy and then the application
ENTRYPOINT ["/app/entrypoint.sh"]
