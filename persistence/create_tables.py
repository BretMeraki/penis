# create_tables.py
from forest_app.persistence import init_db  # Make sure your path/import works
import logging  # Optional: add logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Attempting to create database tables...")
try:
    init_db()
    logger.info("Database tables should be created successfully (check for forest.db).")
except Exception as e:
    logger.exception("Error creating database tables: %s", e)
