# forest_app/persistence/repository.py (Refactored - Commits Removed & Model Names Corrected, flag_modified Added, get_latest_snapshot_model Added)

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union # <-- Add Union here

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
# --- ADD THIS IMPORT ---
from sqlalchemy.orm.attributes import flag_modified
# --- END IMPORT ---
# Cast/String might still be needed if other parts of your app use them,
# but removed from user_id logic here. Keeping import for now.
from sqlalchemy import cast, Integer, String


# --- Models ---
try:
    # ** CHANGE 1: Corrected model names in import **
    from forest_app.snapshot.models import MemorySnapshotModel, TaskFootprintModel, ReflectionLogModel, UserModel
except ImportError:
    # This block should ideally NOT be reached now if PYTHONPATH and model names are correct
    logging.critical("Failed to import ORM models from forest_app.snapshot.models")
    # Define dummy classes only if absolutely necessary for testing parts in isolation
    class MemorySnapshotModel: pass
    class TaskFootprintModel: pass # Renamed dummy class for consistency
    class ReflectionLogModel: pass # Renamed dummy class for consistency
    class UserModel: pass

# --- Logging ---
logger = logging.getLogger(__name__)

# === User Repository Logic ===

def get_user_by_email(db: Session, email: str) -> Optional[UserModel]:
    """Retrieves a user by their email address."""
    if not isinstance(db, Session):
        logger.error("get_user_by_email called with invalid db session type: %s", type(db))
        return None
    try:
        # Check if UserModel was actually imported or if it's the dummy class
        if not hasattr(UserModel, 'email'):
            logger.error("UserModel (potentially dummy) does not have 'email' attribute for query.")
            return None
        return db.query(UserModel).filter(UserModel.email == email).first()
    except SQLAlchemyError as e:
        logger.error("Database error retrieving user by email %s: %s", email, e, exc_info=True)
        # Propagate the error for the endpoint handler to manage the transaction
        raise
    except Exception as e:
        logger.error("Unexpected error retrieving user by email %s: %s", email, e, exc_info=True)
        # Propagate the error
        raise

def create_user(db: Session, user_data: Dict[str, Any]) -> Optional[UserModel]:
    """
    Creates a new user in the database.
    Expects user_data dictionary to contain 'email', 'hashed_password',
    and optionally 'full_name', 'is_active'.
    !! Commits the transaction immediately. !! (Keep commit here as user creation is usually atomic)
    """
    if not isinstance(db, Session):
        logger.error("create_user called with invalid db session type: %s", type(db))
        return None

    required_fields = ["email", "hashed_password"]
    missing_fields = [field for field in required_fields if field not in user_data or user_data[field] is None]
    if missing_fields:
        logger.error(f"Missing required fields for user creation: {missing_fields}")
        return None

    try:
        # Check if UserModel was actually imported or if it's the dummy class
        if not hasattr(UserModel, 'email'): # Check for an expected attribute
              logger.error("UserModel (potentially dummy) cannot be instantiated for user creation.")
              return None
        db_user = UserModel(
            email=user_data["email"],
            hashed_password=user_data["hashed_password"],
            is_active=user_data.get("is_active", True)
        )

        db.add(db_user)
        db.commit() # Commit user creation
        db.refresh(db_user)
        logger.info("Successfully created user with email: %s (ID: %d)", db_user.email, db_user.id)
        return db_user
    except TypeError as e: # Catch TypeError specifically, e.g., if UserModel() takes no args
        logger.error("TypeError during UserModel instantiation (likely import issue): %s", e, exc_info=True)
        db.rollback()
        return None # Cannot create user if model is wrong
    except SQLAlchemyError as e:
        db.rollback() # Rollback on error during create_user commit
        # Use lower() for case-insensitive checking of common unique constraint error messages
        err_str = str(e).lower()
        if "uniqueviolation" in err_str or "duplicate key value violates unique constraint" in err_str:
            logger.warning("Attempted to create user with existing email (DB constraint): %s", user_data.get("email"))
            return None # Return None on unique constraint violation
        else:
            logger.error("Database error creating user %s: %s", user_data.get("email"), e, exc_info=True)
            # Re-raise other SQLAlchemy errors to be handled by the endpoint
            raise
    except Exception as e:
        db.rollback() # Rollback on unexpected error
        logger.error("Unexpected error creating user %s: %s", user_data.get("email"), e, exc_info=True)
        # Re-raise unexpected errors
        raise


# === MemorySnapshotRepository ===

class MemorySnapshotRepository:
    """Repository for memory snapshots."""

    def __init__(self, db: Session):
        self.db = db

    def get_snapshot(self, snapshot_id: int) -> Optional[MemorySnapshotModel]:
        """Get snapshot by ID."""
        return self.db.query(MemorySnapshotModel).filter(MemorySnapshotModel.id == snapshot_id).first()

    def get_all_snapshots(self) -> List[MemorySnapshotModel]:
        """Get all snapshots."""
        return self.db.query(MemorySnapshotModel).all()

    def create_snapshot(self, snapshot: MemorySnapshotModel) -> MemorySnapshotModel:
        """Create a new snapshot."""
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def update_snapshot(self, snapshot: MemorySnapshotModel) -> MemorySnapshotModel:
        """Update an existing snapshot."""
        self.db.commit()
        self.db.refresh(snapshot)
        return snapshot

    def delete_snapshot(self, snapshot_id: int) -> None:
        """Delete a snapshot."""
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot:
            self.db.delete(snapshot)
            self.db.commit()

    def create_snapshot(
        self, user_id: int, snapshot_data: dict, codename: Optional[str] = None
    ) -> Optional[MemorySnapshotModel]:
        """
        Creates a new MemorySnapshot model instance and adds it to the session.
        **Does NOT commit the transaction.**
        """
        if not isinstance(user_id, int):
            logger.error("User ID must be an integer to create a snapshot.")
            # Raise TypeError for the endpoint to handle
            raise TypeError("User ID must be an integer to create a snapshot.")

        now = datetime.utcnow()
        try:
            model = MemorySnapshotModel(
                user_id=user_id,
                snapshot_data=snapshot_data,
                codename=codename,
                created_at=now,
                # 'updated_at' is typically set by the database or on update operations
            )
        except TypeError as e:
            logger.error("TypeError during MemorySnapshotModel instantiation (likely import issue): %s", e, exc_info=True)
            raise # Re-raise to signal failure

        try:
            self.db.add(model)
            # --- ADDED flag_modified FOR CREATE ---
            # Flag mutable field on add, ensuring changes are detected even if the instance
            # isn't modified further before flush/commit.
            flag_modified(model, "snapshot_data")
            # --- END ADDED ---
            logger.info("Added new snapshot object for user ID %d (codename: '%s') to session.",
                        user_id, codename)
            return model
        except SQLAlchemyError as e:
            logger.error("Database error preparing snapshot model for user ID %d: %s", user_id, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error preparing snapshot model for user ID %d: %s", user_id, e, exc_info=True)
            raise

    def get_latest_snapshot(self, user_id: int) -> Optional[MemorySnapshotModel]:
        """Retrieves the latest MemorySnapshot for the specified user."""
        if not isinstance(user_id, int):
            logger.error("User ID must be an integer to get latest snapshot.")
            # Raising error as this indicates a programming issue upstream
            raise TypeError("User ID must be an integer.")
        try:
            # Determine the field to order by (prefer updated_at if it exists)
            order_by_field = 'updated_at' if hasattr(MemorySnapshotModel, 'updated_at') else 'created_at'
            if not all(hasattr(MemorySnapshotModel, attr) for attr in ['user_id', order_by_field]):
                logger.error("MemorySnapshotModel missing required attributes (user_id, %s) for query.", order_by_field)
                raise AttributeError(f"MemorySnapshotModel missing required query attributes (user_id, {order_by_field}).")

            snapshot = (
                self.db.query(MemorySnapshotModel)
                .filter(MemorySnapshotModel.user_id == user_id)
                .order_by(getattr(MemorySnapshotModel, order_by_field).desc()) # Order by available timestamp
                .first()
            )

            # Basic validation on the result (optional but good practice)
            if snapshot and (not hasattr(snapshot, 'user_id') or snapshot.user_id != user_id):
                 logger.error("Data integrity issue: Retrieved snapshot user_id (%s) mismatch or missing for requested user_id %d",
                              getattr(snapshot, 'user_id', 'MISSING'), user_id)
                 # Return None or raise an error depending on desired behavior for corrupted data
                 return None

            return snapshot

        except SQLAlchemyError as e:
            logger.error("Database error retrieving latest snapshot for user ID %d: %s", user_id, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error retrieving latest snapshot for user ID %d: %s", user_id, e, exc_info=True)
            raise

    def update_snapshot(
        self, snapshot_model: MemorySnapshotModel, new_data: dict, codename: Optional[str] = None
    ) -> Optional[MemorySnapshotModel]:
        """
        Updates attributes of an existing MemorySnapshot model instance within the session.
        **Does NOT commit the transaction.** Uses flag_modified for JSONB changes.
        """
        # Check if it's a valid model instance (and not the dummy class)
        if not snapshot_model or not isinstance(snapshot_model, MemorySnapshotModel) or not hasattr(snapshot_model, 'id'):
            logger.warning("Attempted to update a non-existent or invalid snapshot model.")
            return None

        try:
            # Update attributes on the existing model instance
            snapshot_model.snapshot_data = new_data # Assign the new dictionary

            # --- ADDED THIS LINE ---
            # Explicitly mark the snapshot_data field as modified
            # Crucial for JSON/JSONB fields where SQLAlchemy might not detect internal changes.
            flag_modified(snapshot_model, "snapshot_data")
            # --- END ADDED LINE ---

            if hasattr(snapshot_model, 'updated_at'):
                 snapshot_model.updated_at = datetime.utcnow()
            if codename is not None: # Allow updating codename
                 snapshot_model.codename = codename

            log_user_id = snapshot_model.user_id
            logger.info("Prepared update (flagged modified) for snapshot id %s for user ID %d (codename: '%s') in session.",
                         snapshot_model.id, log_user_id, snapshot_model.codename)
            return snapshot_model
        except AttributeError as e:
             logger.error("AttributeError updating snapshot id %s (likely invalid model due to import or data issue): %s",
                          getattr(snapshot_model, 'id', 'N/A'), e, exc_info=True)
             raise # Propagate error indicating model issue
        except SQLAlchemyError as e:
            logger.error("Database error preparing update for snapshot id %s: %s", snapshot_model.id, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error preparing update for snapshot id %s: %s", snapshot_model.id, e, exc_info=True)
            raise

    def list_snapshots(self, user_id: int, limit: int = 100) -> List[MemorySnapshotModel]:
        """Lists snapshots for a specific user, ordered by most recently created/updated first."""
        if not isinstance(user_id, int):
            logger.error("User ID must be an integer to list snapshots.")
            return []
        try:
            # Determine the field to order by
            order_by_field = 'updated_at' if hasattr(MemorySnapshotModel, 'updated_at') else 'created_at'
            if not all(hasattr(MemorySnapshotModel, attr) for attr in ['user_id', order_by_field]):
                logger.error("MemorySnapshotModel missing required attributes (user_id, %s) for query.", order_by_field)
                return []

            query = (
                self.db.query(MemorySnapshotModel)
                .filter(MemorySnapshotModel.user_id == user_id)
                .order_by(getattr(MemorySnapshotModel, order_by_field).desc()) # Order by available timestamp
            )

            if limit > 0: # Apply limit if positive
                query = query.limit(limit)

            results = query.all()
            return results
        except SQLAlchemyError as e:
            logger.error("Database error listing snapshots for user ID %d: %s", user_id, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error listing snapshots for user ID %d: %s", user_id, e, exc_info=True)
            raise

    def get_snapshot_by_id(self, snapshot_id: int, user_id: int) -> Optional[MemorySnapshotModel]:
        """Retrieves a specific snapshot by its ID, ensuring it belongs to the user."""
        if not isinstance(user_id, int):
            logger.error("User ID must be an integer to get snapshot by ID.")
            raise TypeError("User ID must be an integer.")
        if not isinstance(snapshot_id, int):
            logger.error("Snapshot ID must be an integer.")
            raise TypeError("Snapshot ID must be an integer.")

        try:
            if not all(hasattr(MemorySnapshotModel, attr) for attr in ['id', 'user_id']):
                logger.error("MemorySnapshotModel missing required attributes for query (id, user_id).")
                raise AttributeError("MemorySnapshotModel missing required query attributes.")

            snapshot = (
                self.db.query(MemorySnapshotModel)
                .filter(
                    MemorySnapshotModel.id == snapshot_id,
                    MemorySnapshotModel.user_id == user_id # Ensure correct user
                )
                .first()
            )
            return snapshot
        except SQLAlchemyError as e:
            logger.error("Database error getting snapshot id %s for user ID %d: %s", snapshot_id, user_id, e, exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error getting snapshot id %s for user ID %d: %s", snapshot_id, user_id, e, exc_info=True)
            raise

    def delete_snapshot_by_id(self, snapshot_id: int, user_id: int) -> bool:
        """
        Deletes a specific snapshot by its ID, ensuring it belongs to the user.
        !! Commits the transaction immediately. !! (Keep commit here as delete is usually atomic)
        """
        if not isinstance(user_id, int):
            logger.error("User ID must be an integer to delete snapshot by ID.")
            return False
        if not isinstance(snapshot_id, int):
            logger.error("Snapshot ID must be an integer to delete snapshot.")
            return False
        try:
            # Use the method above which includes user ownership check
            snapshot_to_delete = self.get_snapshot_by_id(snapshot_id, user_id)

            if snapshot_to_delete: # Check if a valid model was returned and belongs to user
                self.db.delete(snapshot_to_delete)
                self.db.commit() # Commit deletion
                logger.info("Deleted snapshot id %s for user ID %d", snapshot_id, user_id)
                return True
            else:
                logger.warning("Snapshot id %s not found or not owned by user ID %d for deletion.", snapshot_id, user_id)
                return False
        except SQLAlchemyError as e:
            self.db.rollback() # Rollback on error during deletion
            logger.error("Database error deleting snapshot id %s: %s", snapshot_id, e, exc_info=True)
            # Consider raising instead of returning False if caller needs to know about DB errors
            return False
        except Exception as e:
            self.db.rollback() # Rollback on unexpected error
            logger.error("Unexpected error deleting snapshot id %s: %s", snapshot_id, e, exc_info=True)
            # Consider raising
            return False

    def save_snapshot(self, user_id: int, snapshot_data: dict, codename: Optional[str] = None) -> Optional[MemorySnapshotModel]:
        """
        Saves (updates or creates) a MemorySnapshot for the user. If a snapshot exists, update it; otherwise, create a new one.
        """
        # Try to get the latest snapshot for the user
        latest = self.get_latest_snapshot(user_id)
        if latest:
            # Update the existing snapshot
            return self.update_snapshot(latest, snapshot_data, codename=codename)
        else:
            # Create a new snapshot
            return self.create_snapshot(user_id, snapshot_data, codename=codename)

# === Standalone Helper Function for Snapshot Retrieval ===

# --- ADDED THIS FUNCTION ---
def get_latest_snapshot_model(user_id: int, db: Session) -> Optional[MemorySnapshotModel]:
    """
    Retrieves the latest MemorySnapshotModel for a given user ID using the provided Session.
    This is a standalone function to be used by services/endpoints.
    Note: This function is SYNCHRONOUS as standard SQLAlchemy sessions are synchronous.
          Calls from async FastAPI endpoints will run this in a thread pool. Remove 'await'
          from calls to this function in your endpoint code (e.g., core.py).
    """
    if not isinstance(user_id, int):
        logger.error("User ID must be an integer to get latest snapshot model.")
        raise TypeError("User ID must be an integer.")
    if not isinstance(db, Session):
        logger.error("Invalid database session provided to get_latest_snapshot_model.")
        raise TypeError("db must be a SQLAlchemy Session.")

    try:
        # Determine the field to order by (prefer updated_at if it exists)
        order_by_field = 'updated_at' if hasattr(MemorySnapshotModel, 'updated_at') else 'created_at'
        if not all(hasattr(MemorySnapshotModel, attr) for attr in ['user_id', order_by_field]):
            logger.error("MemorySnapshotModel missing required attributes (user_id, %s) for query.", order_by_field)
            raise AttributeError(f"MemorySnapshotModel missing required query attributes (user_id, {order_by_field}).")

        snapshot = (
            db.query(MemorySnapshotModel)
            .filter(MemorySnapshotModel.user_id == user_id)
            .order_by(getattr(MemorySnapshotModel, order_by_field).desc())
            .first()
        )

        # Optional validation
        if snapshot and (not hasattr(snapshot, 'user_id') or snapshot.user_id != user_id):
            logger.error("Data integrity issue: Retrieved snapshot (helper fn) user_id (%s) mismatch or missing for requested user_id %d",
                         getattr(snapshot, 'user_id', 'MISSING'), user_id)
            return None

        return snapshot

    except SQLAlchemyError as e:
        logger.error("Database error retrieving latest snapshot model for user ID %d: %s", user_id, e, exc_info=True)
        raise # Propagate DB errors
    except Exception as e:
        logger.error("Unexpected error retrieving latest snapshot model for user ID %d: %s", user_id, e, exc_info=True)
        raise # Propagate other errors

# --- END ADDED FUNCTION ---


# === TaskEventLogRepository -> Renamed conceptually, using TaskFootprintModel ===
# Consider renaming class TaskEventLogRepository -> TaskFootprintRepository later
class TaskEventLogRepository:
    # ** CHANGE 2: Updated docstring **
    """Repository for managing TaskFootprintModel persistence."""
    def __init__(self, db: Session):
        if not isinstance(db, Session): raise TypeError("db must be a SQLAlchemy Session")
        # Check attribute that exists on TaskFootprintModel
        if not hasattr(TaskFootprintModel, 'task_id'):
             raise ImportError("TaskFootprintModel appears to be incompletely imported.")
        self.db = db

    # ** CHANGE 4: Updated type hint and model instantiation **
    def create_log(self, log_data: Dict[str, Any]) -> Optional[TaskFootprintModel]:
        """Creates and commits a TaskFootprint log entry."""
        required = ["user_id", "task_id", "event_type"] # Define required fields for clarity
        if any(field not in log_data for field in required):
            logger.warning(f"Missing required fields for TaskFootprintModel: needed {required}, got {list(log_data.keys())}")
            return None

        log_data.setdefault('timestamp', datetime.utcnow()) # Ensure timestamp exists
        try:
            # Use the correct model name here
            log_entry = TaskFootprintModel(**log_data)
        except TypeError as e:
            logger.error("TypeError during TaskFootprintModel instantiation (check log_data keys match model fields): %s | Data: %s", e, log_data, exc_info=True)
            return None # Cannot create if model is wrong or data mismatch

        try:
            self.db.add(log_entry)
            self.db.commit() # Keep commit for logs (often treated as atomic writes)
            self.db.refresh(log_entry)
            return log_entry
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("DB error creating task footprint log: %s", e, exc_info=True)
            # Return None, let caller decide how to handle logging failures
            return None
        except Exception as e:
            self.db.rollback()
            logger.error("Unexpected error creating task footprint log: %s", e, exc_info=True)
            # Return None or raise depending on policy
            return None

    # Using TaskFootprintModel for querying
    def get_logs_for_task(self, task_id: str) -> List[TaskFootprintModel]:
        """Retrieves all TaskFootprint logs for a given task_id, ordered by timestamp."""
        try:
            # Use correct model name and check attributes
            if not all(hasattr(TaskFootprintModel, attr) for attr in ['task_id', 'timestamp']):
                logger.error("TaskFootprintModel missing required attributes for query.")
                return []
            # Query using the correct model name
            return self.db.query(TaskFootprintModel).filter(TaskFootprintModel.task_id == task_id).order_by(TaskFootprintModel.timestamp.asc()).all()
        except SQLAlchemyError as e:
            logger.error("DB error getting task footprint logs for task %s: %s", task_id, e, exc_info=True)
            raise # Propagate errors
        except Exception as e:
            logger.error("Unexpected error getting task footprint logs for task %s: %s", task_id, e, exc_info=True)
            raise # Propagate errors


# === ReflectionEventLogRepository -> Renamed conceptually, using ReflectionLogModel ===
# Consider renaming class ReflectionEventLogRepository -> ReflectionLogRepository later
class ReflectionEventLogRepository:
    # ** CHANGE 3: Updated docstring **
    """Repository for managing ReflectionLogModel persistence."""
    def __init__(self, db: Session):
        if not isinstance(db, Session): raise TypeError("db must be a SQLAlchemy Session")
        # ** CHANGE 6: Updated validation check **
        if not hasattr(ReflectionLogModel, 'user_id'): # Check attribute that exists
             raise ImportError("ReflectionLogModel appears to be incompletely imported.")
        self.db = db

    # ** CHANGE 5: Updated type hint and model instantiation **
    def create_log(self, log_data: Dict[str, Any]) -> Optional[ReflectionLogModel]:
        """Creates and commits a ReflectionLog entry."""
        # Assuming 'user_id' and 'reflection_text' are required for ReflectionLogModel
        required = ["user_id", "reflection_text"] # Define required fields
        if any(field not in log_data for field in required):
            logger.warning(f"Missing required fields for ReflectionLogModel: needed {required}, got {list(log_data.keys())}")
            return None

        log_data.setdefault('timestamp', datetime.utcnow()) # Ensure timestamp exists
        try:
            # Use the correct model name here
            log_entry = ReflectionLogModel(**log_data)
        except TypeError as e:
            logger.error("TypeError during ReflectionLogModel instantiation (check log_data keys match model fields): %s | Data: %s", e, log_data, exc_info=True)
            return None # Cannot create if model is wrong or data mismatch

        try:
            self.db.add(log_entry)
            self.db.commit() # Keep commit for logs
            self.db.refresh(log_entry)
            return log_entry
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error("DB error creating reflection log: %s", e, exc_info=True)
            return None
        except Exception as e:
            self.db.rollback()
            logger.error("Unexpected error creating reflection log: %s", e, exc_info=True)
            return None

    # Using ReflectionLogModel for querying
    def get_logs_for_reflection(self, reflection_id: Union[str, int]) -> List[ReflectionLogModel]:
        """
        Retrieves ReflectionLog entries based on an identifier (adjust field as needed).
        NOTE: Assumes 'reflection_id' exists on the model. Modify if the identifier is different.
        """
        logger.warning("get_logs_for_reflection filtering by 'reflection_id'. Verify this field exists on ReflectionLogModel.")
        # Determine the type of the ID if necessary for filtering
        query_attr = 'reflection_id' # Change this if ReflectionLogModel uses a different identifier field name

        try:
            # Check attributes required for the intended query
            if not all(hasattr(ReflectionLogModel, attr) for attr in [query_attr, 'timestamp']):
                 logger.error("ReflectionLogModel missing required attributes for query ('%s', 'timestamp').", query_attr)
                 return []
            # Query using the correct model name and attribute
            return self.db.query(ReflectionLogModel).filter(getattr(ReflectionLogModel, query_attr) == reflection_id).order_by(ReflectionLogModel.timestamp.asc()).all()
        except SQLAlchemyError as e:
            logger.error("DB error getting reflection logs for ID %s: %s", reflection_id, e, exc_info=True)
            raise # Propagate errors
        except Exception as e:
            logger.error("Unexpected error getting reflection logs for ID %s: %s", reflection_id, e, exc_info=True)
            raise # Propagate errors
