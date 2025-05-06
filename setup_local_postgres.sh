#!/bin/bash

set -e

# Configuration
DB_USER="postgres"
DB_PASSWORD="woozyBoi0!"
DB_NAME="postgres"
DB_PORT="5432"
CONTAINER_NAME="local-postgres"

echo "🔄 Stopping and removing any existing Postgres container..."
docker rm -f $CONTAINER_NAME 2>/dev/null || true

echo "🚀 Starting a new Postgres container..."
docker run --name $CONTAINER_NAME -e POSTGRES_PASSWORD=$DB_PASSWORD -e POSTGRES_USER=$DB_USER -e POSTGRES_DB=$DB_NAME -p $DB_PORT:5432 -d postgres

echo "⏳ Waiting for Postgres to be ready..."
sleep 5

echo "✅ Postgres is running on 127.0.0.1:$DB_PORT"
echo "   Username: $DB_USER"
echo "   Password: $DB_PASSWORD"
echo "   Database: $DB_NAME"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "DATABASE_URL=postgresql+psycopg2://$DB_USER:$DB_PASSWORD@127.0.0.1:$DB_PORT/$DB_NAME" > .env
    echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env
    echo "✅ .env file created with DATABASE_URL and SECRET_KEY"
else
    echo "⚠️  .env file already exists. Please ensure it contains:"
    echo "DATABASE_URL=postgresql+psycopg2://$DB_USER:$DB_PASSWORD@127.0.0.1:$DB_PORT/$DB_NAME"
    echo "SECRET_KEY=your-very-secret-key"
fi

echo "📦 Installing psycopg2-binary if needed..."
pip install psycopg2-binary

echo "🎉 Local Postgres setup complete! You can now run your FastAPI app." 