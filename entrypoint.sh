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

# Create __init__.py files in case they're missing (makes modules importable)
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
  if ! python -m alembic upgrade head; then
    echo "Warning: Database migrations failed. Application may not work correctly."
    # Continue anyway to allow manual troubleshooting
  else
    echo "Database migrations completed successfully."
  fi
fi

# Determine the main application file to run
FASTAPI_APP="forest_app.core.main:app"
if [ -f "/app/forest_app/main.py" ]; then
  FASTAPI_APP="forest_app.main:app"
elif [ -f "/app/main.py" ]; then
  FASTAPI_APP="main:app"
fi

# Determine if we have a Streamlit app
if [ -f "/app/streamlit_app.py" ]; then
  HAS_STREAMLIT=true
else
  HAS_STREAMLIT=false
fi

# Start the application(s)
cd /app

# Start both apps if we have Streamlit
if [ "$HAS_STREAMLIT" = true ]; then
  echo "Starting FastAPI on port 8000 and Streamlit on port 8501..."
  uvicorn $FASTAPI_APP --host 0.0.0.0 --port 8000 & 
  FASTAPI_PID=$!
  
  streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0 &
  STREAMLIT_PID=$!
  
  # Wait for both processes
  wait $FASTAPI_PID $STREAMLIT_PID
else
  # Just start FastAPI
  echo "Starting FastAPI application on port 8000..."
  exec uvicorn $FASTAPI_APP --host 0.0.0.0 --port 8000
fi

# --- Cleanup (Optional: Usually not reached with exec) ---
[ "$USE_PROXY" = true ] && [ -f "$KEY_FILE_PATH" ] && rm "$KEY_FILE_PATH"
# echo "Application server exited. Cleaning up proxy..."
# kill $PROXY_PID
# wait $PROXY_PID # Wait for proxy to terminate
# rm "$KEY_FILE_PATH" # Remove the key file
# echo "Cleanup complete."
