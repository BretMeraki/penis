# forest_app/modules/logging_tracking.py

import logging
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler

# --- Persistence Imports ---
# Import repositories for logging events
from forest_app.snapshot.repository import (
    TaskEventLogRepository,
    ReflectionEventLogRepository,
)

# --- HTA Imports (Direct) ---
# Removed try...except ImportError block. Relying on direct import.
# Ensure this import path is correct for your project structure.
from forest_app.hta_tree.hta_tree import HTANode
# ─────────────────────────────

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Log-Once-Per-Session Utility ---
_logged_once_set = set()
def log_once_per_session(level: str, msg: str):
    """
    Log a message only once per process/session, regardless of how many times it's called.
    Usage: log_once_per_session('warning', 'Some warning message')
    """
    key = f"{level}:{msg}"
    if key in _logged_once_set:
        return
    _logged_once_set.add(key)
    if level.lower() == 'warning':
        logger.warning(msg)
    elif level.lower() == 'error':
        logger.error(msg)
    elif level.lower() == 'info':
        logger.info(msg)
    else:
        logger.log(logging.getLevelName(level.upper()), msg)

def setup_global_rotating_error_log(logfile: str = 'error.log', max_bytes: int = 1_000_000, backup_count: int = 3):
    root_logger = logging.getLogger()
    # Only add if not already present
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', None) == logfile for h in root_logger.handlers):
        handler = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        root_logger.addHandler(handler)

class TaskFootprintLogger:
    """
    Logs detailed task events with context directly to the database.
    Requires a database session to operate.
    """

    def __init__(self, db: Session):
        """Initializes the logger with a database session."""
        if not db:
             # Add check for valid DB session
             logger.error("TaskFootprintLogger requires a valid database session.")
             raise ValueError("Database session is required for TaskFootprintLogger.")
        self.repo = TaskEventLogRepository(db)

    def log_task_event(
        self,
        task_id: str,
        event_type: str, # e.g., 'generated', 'completed', 'failed', 'skipped'
        snapshot: Dict[str, Any], # Current snapshot state dictionary
        hta_node: Optional[HTANode] = None, # Optional linked HTA node object
        event_metadata: Optional[Dict[str, Any]] = None, # Additional structured data
    ):
        """
        Logs a task event with context extracted from the snapshot.
        """
        if not task_id or not event_type:
             logger.error("Task ID and event type are required for logging task event.")
             return

        # Safely extract context from snapshot, providing defaults or None
        capacity = snapshot.get("capacity")
        shadow_score = snapshot.get("shadow_score")
        # Example of accessing nested state safely
        active_seed_data = snapshot.get("component_state", {}).get("seed_manager", {}).get("seeds", {})
        # Assuming single active seed for simplicity; adjust if multiple are possible
        active_seed_name = None
        if active_seed_data and isinstance(active_seed_data, dict):
             active_seed_obj = next(iter(active_seed_data.values()), None) # Get first seed dict
             if active_seed_obj and isinstance(active_seed_obj, dict):
                  active_seed_name = active_seed_obj.get("seed_name")

        active_archetype_data = snapshot.get("component_state", {}).get("archetype_manager", {}).get("active_archetype", {})
        active_archetype_name = active_archetype_data.get("name") if isinstance(active_archetype_data, dict) else None

        # Prepare data dictionary for the database log entry
        log_data = {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            # Use getattr for safe access to hta_node.id
            "linked_hta_node_id": getattr(hta_node, "id", None) if hta_node else None,
            "capacity_at_event": capacity,
            "shadow_score_at_event": shadow_score,
            "active_seed_name": active_seed_name,
            "active_archetype_name": active_archetype_name,
            "event_metadata": event_metadata or {}, # Ensure metadata is always a dict
        }

        try:
            # Attempt to create the log entry in the database
            created_log = self.repo.create_log(log_data)
            if created_log:
                 logger.debug("Logged task event for task %s (Type: %s)", task_id, event_type)
                 # Log HTA linking separately if provided
                 if hta_node and hasattr(hta_node, 'id'):
                     logger.info(
                         "Task event '%s' linked to HTA node ID '%s'.",
                         task_id,
                         hta_node.id,
                     )
            else:
                 # This case might indicate an issue in repo.create_log if it should always return a model
                 logger.warning("Task event log repo did not return a created log object for task %s.", task_id)

        except Exception as e:
            # Catch and log any exceptions during database interaction
            logger.error("Failed to log task event for task %s: %s", task_id, e, exc_info=True) # Add exc_info for traceback


class ReflectionLogLogger:
    """
    Logs detailed reflection events with context directly to the database.
    Requires a database session to operate.
    """

    def __init__(self, db: Session):
        """Initializes the logger with a database session."""
        if not db:
             logger.error("ReflectionLogLogger requires a valid database session.")
             raise ValueError("Database session is required for ReflectionLogLogger.")
        self.repo = ReflectionEventLogRepository(db)

    def log_reflection_event(
        self,
        reflection_id: str, # Unique ID for the reflection event/session
        event_type: str, # e.g., 'started', 'processed', 'sentiment_analyzed'
        snapshot: Dict[str, Any], # Current snapshot state dictionary
        hta_node: Optional[HTANode] = None, # Optional linked HTA node object
        event_metadata: Optional[Dict[str, Any]] = None, # Additional structured data
    ):
        """
        Logs a reflection event with context extracted from the snapshot.
        """
        if not reflection_id or not event_type:
             logger.error("Reflection ID and event type are required for logging reflection event.")
             return

        # Safely extract context from snapshot
        # Example: Drill down safely for nested data
        sentiment_score = snapshot.get("component_state", {}).get("metrics_engine", {}).get("last_sentiment")
        capacity = snapshot.get("capacity")
        shadow_score = snapshot.get("shadow_score")

        active_seed_data = snapshot.get("component_state", {}).get("seed_manager", {}).get("seeds", {})
        active_seed_name = None
        if active_seed_data and isinstance(active_seed_data, dict):
             active_seed_obj = next(iter(active_seed_data.values()), None)
             if active_seed_obj and isinstance(active_seed_obj, dict):
                  active_seed_name = active_seed_obj.get("seed_name")

        active_archetype_data = snapshot.get("component_state", {}).get("archetype_manager", {}).get("active_archetype", {})
        active_archetype_name = active_archetype_data.get("name") if isinstance(active_archetype_data, dict) else None


        # Prepare data dictionary for the database log entry
        log_data = {
            "reflection_id": reflection_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            # Use getattr for safe access to hta_node.id
            "linked_hta_node_id": getattr(hta_node, "id", None) if hta_node else None,
            "sentiment_score_at_event": sentiment_score, # Rename field for clarity
            "capacity_at_event": capacity,
            "shadow_score_at_event": shadow_score,
            "active_seed_name": active_seed_name,
            "active_archetype_name": active_archetype_name,
            "event_metadata": event_metadata or {}, # Ensure metadata is always a dict
        }

        try:
            # Attempt to create the log entry in the database
            created_log = self.repo.create_log(log_data)
            if created_log:
                logger.debug("Logged reflection event for ID %s (Type: %s)", reflection_id, event_type)
                if hta_node and hasattr(hta_node, 'id'):
                    logger.info(
                        "Reflection event '%s' linked to HTA node ID '%s'.",
                        reflection_id,
                        hta_node.id,
                    )
            else:
                 logger.warning("Reflection event log repo did not return a created log object for reflection %s.", reflection_id)

        except Exception as e:
            # Catch and log any exceptions during database interaction
            logger.error("Failed to log reflection event for %s: %s", reflection_id, e, exc_info=True) # Add exc_info
