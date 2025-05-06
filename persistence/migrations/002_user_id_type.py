"""Migration script to convert user_id column from String to Integer."""

from sqlalchemy import text
from logging import getLogger

logger = getLogger(__name__)

def migrate_user_id_column(conn):
    """Converts user_id column in memory_snapshots table from String to Integer."""
    try:
        logger.info("Starting user_id column type migration...")
        # Add temporary integer column
        conn.execute(text("ALTER TABLE memory_snapshots ADD COLUMN user_id_int INTEGER"))
        logger.info("Added user_id_int column")

        # Convert string values to integers 
        conn.execute(text("UPDATE memory_snapshots SET user_id_int = CAST(user_id AS INTEGER)"))
        logger.info("Migrated user_id values to integer format")

        # Drop original string column and rename new one
        conn.execute(text("ALTER TABLE memory_snapshots DROP COLUMN user_id"))
        conn.execute(text("ALTER TABLE memory_snapshots RENAME COLUMN user_id_int TO user_id"))
        logger.info("Renamed column and dropped old user_id")

        # Create index on the new integer column
        conn.execute(text("CREATE INDEX ix_memory_snapshots_user_id ON memory_snapshots (user_id)"))
        logger.info("Created index on user_id column")

        return True
    except Exception as e:
        logger.error("Failed to migrate user_id column: %s", e, exc_info=True)
        return False