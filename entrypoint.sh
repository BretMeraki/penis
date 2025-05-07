#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Path where the Service Account key file will be written inside the container
KEY_FILE_PATH="/tmp/gcp_key.json"
# Your Cloud SQL Instance Connection Name (Ensure this is correct)
INSTANCE_CONNECTION_NAME="winged-verbena-457705-p3:us-central1:forestapp"
# --- End Configuration ---

# Ensure Python path is set correctly and create any missing __init__.py files
export PYTHONPATH=/app:/app/forest_app:.

# Create directories first, then create __init__.py files
mkdir -p /app/forest_app /app/forest_app/snapshot /app/forest_app/core /app/alembic

# Now create __init__.py files in each directory
for dir in /app/forest_app /app/forest_app/snapshot /app/forest_app/core /app/alembic; do
  if [ ! -f "$dir/__init__.py" ]; then
    echo "# Package initialization" > "$dir/__init__.py"
    echo "Created missing __init__.py in $dir"
  fi
done

# Check if GCP_SA_KEY environment variable is set
if [ -z "$GCP_SA_KEY" ]; then
  echo "Warning: GCP_SA_KEY environment variable is not set. Will proceed without Cloud SQL proxy."
  USE_PROXY=false
else
  USE_PROXY=true
  # Write the content of the GCP_SA_KEY secret to the key file
  echo "$GCP_SA_KEY" > "$KEY_FILE_PATH"
  echo "GCP Service Account key file created at $KEY_FILE_PATH"

  # Set the standard Google Cloud environment variable for authentication
  export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE_PATH"
  echo "GOOGLE_APPLICATION_CREDENTIALS environment variable set."
fi

# Only start the proxy if we have credentials
if [ "$USE_PROXY" = true ]; then
  # Start the Cloud SQL Auth Proxy in the background
  echo "Starting Cloud SQL Auth Proxy for $INSTANCE_CONNECTION_NAME..."
  /usr/local/bin/cloud-sql-proxy "$INSTANCE_CONNECTION_NAME" & 
  PROXY_PID=$! # Store the proxy's process ID

  # Wait for the proxy to establish the connection tunnel
  echo "Waiting for proxy to initialize..."
  sleep 5

  # Check if proxy started successfully
  if ! kill -0 $PROXY_PID > /dev/null 2>&1; then
    echo "Error: Cloud SQL Auth Proxy failed to start."
    [ -f "$KEY_FILE_PATH" ] && rm "$KEY_FILE_PATH" # Clean up key file if it exists
    exit 1
  fi
  echo "Cloud SQL Auth Proxy started successfully (PID: $PROXY_PID)."
fi

# Run database migrations if alembic configuration exists
if [ -f "/app/alembic.ini" ] && [ -d "/app/alembic" ]; then
  echo "Running database migrations..."
  cd /app
  # Print Python paths for debugging
  echo "PYTHONPATH: $PYTHONPATH"
  
  # Try running migrations with better error handling
  if ! alembic upgrade head; then
    echo "Warning: Database migrations failed. Application may not work correctly."
    # Continue anyway to allow manual troubleshooting
  else
    echo "Database migrations completed successfully."
  fi
fi

# Create a basic FastAPI app if it doesn't exist
if [ ! -f "/app/forest_app/main.py" ]; then
  echo "Creating basic FastAPI app..."
  mkdir -p /app/forest_app
  cat > /app/forest_app/main.py << 'EOL'
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Forest app is running"}
EOL
fi

# Start the application
cd /app
echo "Starting FastAPI application on port 8000..."
exec uvicorn forest_app.main:app --host 0.0.0.0 --port 8000
