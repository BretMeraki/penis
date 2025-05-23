# alembic/env.py

import os  # Import the os module
import sys  # Import sys module
import importlib.util  # Import for dynamic imports

# Hardcode project root for Docker reliability
sys.path.insert(0, '/app')
sys.path.insert(0, '.')
print("PYTHONPATH:", sys.path)  # For debugging

from logging.config import fileConfig

# Import necessary SQLAlchemy components
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy import create_engine

# Import Alembic context
from alembic import context

# --- DIRECT IMPORT FIX ---
# Try to load the models module directly using importlib
try:
    # Try standard import first
    from sqlalchemy.orm import DeclarativeBase
    
    # Define a fallback Base class if import fails
    class Base(DeclarativeBase):
        pass
    
    # Try to find the models.py file directly
    model_paths = [
        '/app/forest_app/snapshot/models.py',  # Docker path
        os.path.join(os.getcwd(), 'forest_app', 'snapshot', 'models.py')  # Local path
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Found models at: {model_path}")
            # Load the module directly
            spec = importlib.util.spec_from_file_location("models", model_path)
            models_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(models_module)
            Base = getattr(models_module, 'Base')  # Get the Base class from the loaded module
            break
    print(f"Using Base class: {Base}")

except Exception as e:
    print(f"Error importing models: {e}")
    raise
# --- END DIRECT IMPORT FIX ---

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- MODIFY THIS ---
# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata
# -----------------

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # Get the URL from the main Alembic configuration section in alembic.ini
    # This will resolve the %(DB_CONNECTION_STRING)s variable if needed for offline mode
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata, # Pass the metadata for offline checks
        literal_binds=True, # Render SQL statements with literal values
        dialect_opts={"paramstyle": "named"}, # Dialect-specific options
    )

    # Run migrations within a transaction block
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # --- REVERTED TO EXPLICIT ENV VAR READING ---
    # Get the database URL directly from the environment variable
    db_url = os.environ.get('DB_CONNECTION_STRING')
    if not db_url:
        raise ValueError("DB_CONNECTION_STRING environment variable not set for Alembic online migration")

    # Create the SQLAlchemy engine directly using the fetched URL
    # Pass poolclass=pool.NullPool to avoid hanging connections during migrations
    connectable = create_engine(db_url, poolclass=pool.NullPool)
    # --------------------------------------------

    # Establish connection using the configured engine
    with connectable.connect() as connection:
        # Configure the Alembic context with the connection and metadata
        context.configure(
            connection=connection,
            target_metadata=target_metadata # Pass the metadata for comparison
        )

        # Run migrations within a transaction block
        with context.begin_transaction():
            context.run_migrations()


# --- Run the appropriate migration function based on context mode ---
# Checks if Alembic is invoked in offline mode (e.g., alembic upgrade --sql)
# or online mode (e.g., alembic upgrade head)
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
