# forest_app/main.py (MODIFIED: DI Wiring moved before router inclusion)

import logging
from forest_app.core.logging_tracking import log_once_per_session
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Callable, Any # Added Any

# Set up rotating error log at the very top
logfile = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'error.log'))
rotating_handler = RotatingFileHandler(logfile, maxBytes=1_000_000, backupCount=3)
rotating_handler.setLevel(logging.WARNING)
rotating_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
root_logger = logging.getLogger()
if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == logfile for h in root_logger.handlers):
    root_logger.addHandler(rotating_handler)

# --- Explicitly add /app to sys.path ---
# This helps resolve module imports in some deployment environments
APP_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT_DIR not in sys.path:
    sys.path.insert(0, APP_ROOT_DIR)
    sys.path.insert(0, os.path.join(APP_ROOT_DIR, 'forest_app'))
# --- End sys.path modification ---

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- Core, Persistence & Feature Flag Imports ---
from forest_app.core.security import initialize_security_dependencies
from forest_app.snapshot.database import init_db
from forest_app.snapshot.repository import get_user_by_email
from forest_app.snapshot.models import UserModel

# --- Import Feature Flags and Checker ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    feature_flags_available = True
except ImportError as ff_import_err:
    log_once_per_session('error',"Failed to import Feature Flags components: %s", ff_import_err)
    feature_flags_available = False
    class Feature: pass
    def is_enabled(feature: Any) -> bool: return False

# --- Initialize Container ---
from forest_app.core.containers import Container
container = Container()

# --- Wire Container ---
try:
    container.wire(modules=Container.wiring_config.modules)
    logger.info("DI Container wiring applied successfully.")
except Exception as wire_err:
    logger.critical(f"CRITICAL: Failed to apply DI container wiring: {wire_err}", exc_info=True)
    sys.exit(f"CRITICAL: DI Container wiring failed: {wire_err}")

# --- Sentry Integration Imports ---
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

# --------------------------------------------------------------------------
# Sentry Integration
# --------------------------------------------------------------------------
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    try:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
            integrations=[SqlalchemyIntegration(), sentry_logging],
            environment=os.getenv("APP_ENV", "development"),
            release=os.getenv("APP_RELEASE_VERSION", "unknown"),
        )
        logger.info("Sentry SDK initialized successfully.")
    except Exception as sentry_init_e:
        logger.exception("Failed to initialize Sentry SDK: %s", sentry_init_e)
else:
    log_once_per_session('warning',"SENTRY_DSN environment variable not found. Sentry integration skipped.")

# --------------------------------------------------------------------------
# Database Initialization
# --------------------------------------------------------------------------
try:
    logger.info("Attempting to initialize database tables via init_db()...")
    init_db()
    logger.info("Database initialization check complete.")
except Exception as db_init_e:
    logger.exception("CRITICAL Error during database initialization: %s", db_init_e)
    sys.exit(f"CRITICAL: Database initialization failed: {db_init_e}")

# --------------------------------------------------------------------------
# Security Dependency Initialization
# --------------------------------------------------------------------------
logger.info("Initializing security dependencies...")
try:
    if not hasattr(UserModel, '__annotations__') or 'email' not in UserModel.__annotations__:
         logger.critical("UserModel may be incomplete or a dummy class. Security init might fail.")
    initialize_security_dependencies(get_user_by_email, UserModel)
    logger.info("Security dependencies initialized successfully.")
except TypeError as sec_init_err:
     logger.exception(f"CRITICAL: Failed security init - check function signature in core.security: {sec_init_err}")
     sys.exit(f"CRITICAL: Security dependency initialization failed (TypeError): {sec_init_err}")
except Exception as sec_init_gen_err:
     logger.exception(f"CRITICAL: Unexpected error during security init: {sec_init_gen_err}")
     sys.exit(f"CRITICAL: Security dependency initialization failed unexpectedly: {sec_init_gen_err}")

# --------------------------------------------------------------------------
# FastAPI Application Instance Creation
# --------------------------------------------------------------------------
logger.info("Creating FastAPI application instance...")
app = FastAPI(
    title="Forest OS API",
    version="1.23",
    description="API for interacting with the Forest OS personal growth assistant.",
)

# --- Store container on app.state ---
app.state.container = container
logger.info("DI Container instance stored in app.state.")

# --- Configure CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Import and Include Routers ---
# Import routers after container is wired and stored in app.state
try:
    from forest_app.routers import core, auth, onboarding, users, hta
    from forest_app.snapshot import routers as snapshots
    
    # Include routers with error handling
    routers = [
        (auth.router, "/auth", ["auth"]),
        (core.router, "/core", ["core"]),
        (onboarding.router, "/onboarding", ["onboarding"]),
        (users.router, "/users", ["users"]),
        (hta.router, "/hta", ["hta"]),
        (snapshots.router, "/snapshots", ["snapshots"])
    ]
    
    for router, prefix, tags in routers:
        try:
            app.include_router(router, prefix=prefix, tags=tags)
            logger.debug(f"Successfully included router at {prefix}")
        except Exception as router_err:
            log_once_per_session('error',f"Failed to include router at {prefix}: {router_err}")
            raise
            
    logger.info("All routers included successfully.")
except ImportError as import_err:
    logger.critical(f"Failed to import routers: {import_err}")
    sys.exit(f"CRITICAL: Router imports failed: {import_err}")
except Exception as router_err:
    logger.critical(f"Unexpected error during router setup: {router_err}")
    sys.exit(f"CRITICAL: Router setup failed: {router_err}")

# --------------------------------------------------------------------------
# Startup Event (Feature Flag Logging Only) ### UPDATED COMMENT ###
# --------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event executing...")

    # --- Feature Flag Logging (Kept in startup event) ---
    logger.info("--- Verifying Feature Flag Status (from settings) ---")
    if feature_flags_available and hasattr(Feature, '__members__'):
        for feature in Feature:
            try:
                # Use the imported is_enabled function
                status = is_enabled(feature)
                logger.info(f"Feature: {feature.name:<35} Status: {'ENABLED' if status else 'DISABLED'}")
            except Exception as e:
                log_once_per_session('error',f"Error checking status for feature {feature.name}: {e}")
    elif not feature_flags_available:
        log_once_per_session('error',"Feature flags module failed import, cannot check status.")
    else:
         log_once_per_session('warning',"Feature enum has no members defined?")
    logger.info("-----------------------------------------------------")
    # --- END Feature Flag Logging ---

    logger.info("Startup event complete.")

    # Set up global rotating error log for the entire app
    from forest_app.core.logging_tracking import setup_global_rotating_error_log
    setup_global_rotating_error_log()

# --------------------------------------------------------------------------
# Shutdown Event
# --------------------------------------------------------------------------
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown event executing...")
    # Add cleanup logic here if needed
    logger.info("Shutdown event complete.")


# --------------------------------------------------------------------------
# Root Endpoint
# --------------------------------------------------------------------------
@app.get("/", tags=["Status"], include_in_schema=False)
async def read_root():
    """ Basic status endpoint """
    return {"message": f"Welcome to the Forest OS API (Version {app.version})"}

# --------------------------------------------------------------------------
# Local Development Run Hook
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn development server directly via __main__...")
    reload_flag = os.getenv("APP_ENV", "development") == "development" and \
                 os.getenv("UVICORN_RELOAD", "True").lower() in ("true", "1")

    # Make sure to pass the app object correctly
    # Use "forest_app.core.main:app" if running from outside the directory
    # Use "main:app" if running from within the forest_app directory
    uvicorn.run(
        "forest_app.core.main:app", # Changed for direct run
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=reload_flag,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info").lower(),
    )
