# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app:/app/forest_app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates graphviz && \
    rm -rf /var/lib/apt/lists/*

# Download and install the Cloud SQL Auth Proxy
RUN curl -o /usr/local/bin/cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.3/cloud-sql-proxy.linux.amd64 && \
    chmod +x /usr/local/bin/cloud-sql-proxy

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . /app/

# Ensure critical directories exist and have __init__.py files
RUN mkdir -p /app/forest_app/snapshot /app/forest_app/core /app/alembic
RUN for dir in /app/forest_app /app/forest_app/snapshot /app/forest_app/core /app/alembic; do \
    if [ ! -f "$dir/__init__.py" ]; then \
      echo "# Package initialization" > "$dir/__init__.py"; \
    fi \
    done

# Create necessary directories
RUN mkdir -p /app/forest_app

# Copy Alembic configuration and migrations if they exist
COPY alembic.ini /app/alembic.ini
COPY alembic/ /app/alembic/

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose the port the app runs on
EXPOSE 8000

# Run the entrypoint script when the container launches
ENTRYPOINT ["/app/entrypoint.sh"]
