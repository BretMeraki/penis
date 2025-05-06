#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Path where the Service Account key file will be written inside the container
KEY_FILE_PATH="/tmp/gcp_key.json"
# Your Cloud SQL Instance Connection Name (Ensure this is correct)
INSTANCE_CONNECTION_NAME="winged-verbena-457705-p3:us-central1:forestapp"
# --- End Configuration ---

# Set Python path to include both /app and /app/forest_app
export PYTHONPATH=/app:/app/forest_app

# Check if GCP_SA_KEY environment variable is set (Koyeb injects secrets this way)
if [ -z "$GCP_SA_KEY" ]; then
  echo "Error: GCP_SA_KEY environment variable is not set. Cannot authenticate proxy."
  exit 1
fi

# Write the content of the GCP_SA_KEY secret to the key file
# This makes the credentials available to the proxy
echo "$GCP_SA_KEY" > "$KEY_FILE_PATH"
echo "GCP Service Account key file created at $KEY_FILE_PATH"

# Set the standard Google Cloud environment variable for authentication
export GOOGLE_APPLICATION_CREDENTIALS="$KEY_FILE_PATH"
echo "GOOGLE_APPLICATION_CREDENTIALS environment variable set."

# Start the Cloud SQL Auth Proxy in the background
# It will listen on localhost (127.0.0.1) port 5432 by default for PostgreSQL
echo "Starting Cloud SQL Auth Proxy for $INSTANCE_CONNECTION_NAME..."
# Ensure the path to cloud-sql-proxy is correct for your container image
# Common paths: /usr/local/bin/cloud-sql-proxy, /cloud_sql_proxy, /cloud-sql-proxy
# Verify the correct path if startup fails. Using a likely path here:
/usr/local/bin/cloud-sql-proxy "$INSTANCE_CONNECTION_NAME" & # Runs in background
PROXY_PID=$! # Store the proxy's process ID

# Wait a moment for the proxy to establish the connection tunnel
# Adjust sleep time if needed, but 3-5 seconds is often sufficient
echo "Waiting for proxy to initialize..."
sleep 5

# Check if proxy started successfully (basic check)
if ! kill -0 $PROXY_PID > /dev/null 2>&1; then
    echo "Error: Cloud SQL Auth Proxy failed to start."
    rm "$KEY_FILE_PATH" # Clean up key file
    exit 1
fi
echo "Cloud SQL Auth Proxy started successfully (PID: $PROXY_PID)."

# Run database migrations
echo "Running database migrations..."
cd /app
alembic upgrade head
echo "Database migrations finished."

# Start the FastAPI application
echo "Starting FastAPI application on port 8000..."
cd /app/forest_app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# --- Cleanup (Optional: Usually not reached due to exec) ---
# echo "Application server exited. Cleaning up proxy..."
# kill $PROXY_PID
# wait $PROXY_PID # Wait for proxy to terminate
# rm "$KEY_FILE_PATH" # Remove the key file
# echo "Cleanup complete."
