# forest_app/modules/trail_manager.py

import logging
import hashlib # Moved import here
from datetime import datetime, timezone # Added timezone
from typing import List, Dict, Any, Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("trail_manager_init")
    logger.warning("Feature flags module not found in trail_manager. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        TRAIL_MANAGER = "FEATURE_ENABLE_TRAIL_MANAGER" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrailEvent:
    """
    Represents a single event in a trail. Does not check flags itself,
    relies on Trail/TrailManager to control its creation/addition.
    """
    def __init__(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None, # Use Optional
        object_class: Optional[str] = None, # Use Optional
    ):
        self.event_type = event_type
        self.description = description
        self.metadata = metadata.copy() if metadata else {}
        if object_class:
            self.metadata["object_class"] = object_class
        self.timestamp = datetime.now(timezone.utc).isoformat() # Use timezone aware UTC

    def to_dict(self) -> dict:
        # No flag check needed for simple data serialization
        return {
            "event_type": self.event_type,
            "description": self.description,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrailEvent":
        # No flag check needed for simple data deserialization
        if not isinstance(data, dict):
             # Handle invalid data gracefully
             logger.warning("Invalid data type for TrailEvent.from_dict: %s", type(data))
             return cls(event_type="error", description="Invalid event data")

        # Safely get metadata and object_class
        metadata = data.get("metadata", {})
        object_class = metadata.get("object_class") if isinstance(metadata, dict) else None

        event = cls(
            event_type=data.get("event_type", "unknown"),
            description=data.get("description", ""),
            metadata=metadata,
            object_class=object_class, # Pass explicitly, already extracted
        )
        # Load timestamp or default to now() if missing/invalid
        # Could add validation here if needed
        event.timestamp = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        return event


class Trail:
    """
    Represents an entire trail in the Forest system.
    Event addition/update respects the TRAIL_MANAGER feature flag.
    """
    def __init__(self, trail_id: str, trail_type: str, description: str):
        self.trail_id = trail_id
        self.trail_type = trail_type
        self.description = description
        self.events: List[TrailEvent] = []
        now_ts = datetime.now(timezone.utc).isoformat()
        self.created_at = now_ts
        self.updated_at = now_ts

    def add_event(self, event: TrailEvent):
        """Adds an event if the TRAIL_MANAGER feature is enabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
             logger.debug("Skipping Trail.add_event: TRAIL_MANAGER feature disabled.")
             return # Do not add event or update timestamp
        # --- End Check ---

        if isinstance(event, TrailEvent):
            self.events.append(event)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            logger.info("Added event to trail '%s': type=%s", self.trail_id, event.event_type)
        else:
             logger.warning("Attempted to add non-TrailEvent object to trail '%s'.", self.trail_id)


    def update_event(self, index: int, new_event: TrailEvent):
        """Updates an event if the TRAIL_MANAGER feature is enabled."""
         # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
             logger.debug("Skipping Trail.update_event: TRAIL_MANAGER feature disabled.")
             return # Do not update event or timestamp
        # --- End Check ---

        if not isinstance(new_event, TrailEvent):
             logger.warning("Attempted to update trail '%s' with non-TrailEvent object.", self.trail_id)
             return

        if 0 <= index < len(self.events):
            self.events[index] = new_event
            self.updated_at = datetime.now(timezone.utc).isoformat()
            logger.info("Updated event at index %d for trail '%s'.", index, self.trail_id)
        else:
            logger.warning("Attempted to update nonexistent event index %d for trail '%s'.", index, self.trail_id)

    def to_dict(self) -> dict:
        # No flag check needed here, manager controls overall serialization
        return {
            "trail_id": self.trail_id,
            "trail_type": self.trail_type,
            "description": self.description,
            "events": [event.to_dict() for event in self.events],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trail":
         # No flag check needed here, manager controls overall deserialization
        if not isinstance(data, dict):
             logger.error("Invalid data type for Trail.from_dict: %s", type(data))
             # Return a default/empty trail or raise error? Returning default:
             return cls(trail_id="error_invalid_data", trail_type="error", description="Invalid trail data")

        trail = cls(
            trail_id=data.get("trail_id", f"unknown_{datetime.now(timezone.utc).toordinal()}"), # Default ID
            trail_type=data.get("trail_type", "unknown"),
            description=data.get("description", ""),
        )
        now_ts = datetime.now(timezone.utc).isoformat()
        trail.created_at = data.get("created_at", now_ts)
        trail.updated_at = data.get("updated_at", trail.created_at) # Default to created_at if missing

        events_data = data.get("events", [])
        if isinstance(events_data, list):
             trail.events = [TrailEvent.from_dict(event_data) for event_data in events_data]
        else:
             logger.warning("Invalid 'events' format in trail data for trail '%s'. Events not loaded.", trail.trail_id)
             trail.events = [] # Ensure events is a list

        return trail


class TrailManager:
    """
    Manages trails (journey paths). Respects the TRAIL_MANAGER feature flag.
    """

    def __init__(self):
        self.trails: Dict[str, Trail] = {}  # Maps trail_id to Trail objects.
        logger.info("TrailManager initialized.")

    def create_trail(self, trail_type: str, description: str) -> Optional[Trail]: # Return Optional
        """
        Creates a new trail. Returns None if TRAIL_MANAGER feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
            logger.debug("Skipping create_trail: TRAIL_MANAGER feature disabled.")
            return None
        # --- End Check ---

        # Use a more robust way to generate ID if needed, MD5 is fast but has collisions
        # Consider uuid library: import uuid; trail_id = str(uuid.uuid4())[:8]
        timestamp_str = datetime.now(timezone.utc).isoformat()
        trail_id_base = f"{description}-{timestamp_str}".encode("utf-8")
        trail_id = hashlib.md5(trail_id_base).hexdigest()[:8]

        # Ensure basic input validity
        if not trail_type or not description:
             logger.warning("Cannot create trail with empty type or description.")
             return None

        trail = Trail(trail_id=trail_id, trail_type=trail_type, description=description)
        self.trails[trail_id] = trail
        logger.info("Created new trail '%s' of type '%s'.", trail_id, trail_type)
        return trail

    # --- Helper to reduce repetition in add_* methods ---
    def _add_event_to_trail(self, trail_id: str, event_type: str, description: str,
                            metadata: Optional[Dict[str, Any]], object_class: Optional[str]) -> bool:
        """Internal helper to add a generic event, checking feature flag."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
            logger.debug("Skipping _add_event_to_trail for type '%s': TRAIL_MANAGER feature disabled.", event_type)
            return False
        # --- End Check ---

        trail = self.trails.get(trail_id)
        if trail:
            event = TrailEvent(
                event_type=event_type,
                description=description,
                metadata=metadata,
                object_class=object_class,
            )
            # Trail.add_event also checks flag, but check here prevents unnecessary TrailEvent creation
            trail.add_event(event)
            return True # Indicates attempt was made (success depends on trail.add_event)
        else:
            logger.warning("Attempted to add %s event to unknown trail '%s'.", event_type, trail_id)
            return False

    def add_bench(self, trail_id: str, bench_description: str, metadata: Optional[Dict[str, Any]] = None, object_class: Optional[str] = None) -> bool:
        """Adds a bench event. Returns False if feature disabled or trail not found."""
        return self._add_event_to_trail(trail_id, "bench", bench_description, metadata, object_class)

    def add_lightning_event(self, trail_id: str, event_description: str, metadata: Optional[Dict[str, Any]] = None, object_class: Optional[str] = None) -> bool:
        """Adds a lightning event. Returns False if feature disabled or trail not found."""
        return self._add_event_to_trail(trail_id, "lightning", event_description, metadata, object_class)

    def add_wonder_event(self, trail_id: str, event_description: str, metadata: Optional[Dict[str, Any]] = None, object_class: Optional[str] = None) -> bool:
        """Adds a wonder event. Returns False if feature disabled or trail not found."""
        return self._add_event_to_trail(trail_id, "wonder", event_description, metadata, object_class)

    def add_wild_path(self, trail_id: str, path_description: str, metadata: Optional[Dict[str, Any]] = None, object_class: Optional[str] = None) -> bool:
        """Adds a wild path event. Returns False if feature disabled or trail not found."""
        return self._add_event_to_trail(trail_id, "wild_path", path_description, metadata, object_class)
    # -------------------------------------------------------

    def get_trail_summary(self, trail_id: str) -> dict:
        """
        Returns a summary dictionary. Returns empty dict if feature disabled or trail not found.
        """
         # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
            logger.debug("Skipping get_trail_summary: TRAIL_MANAGER feature disabled.")
            return {}
        # --- End Check ---

        trail = self.trails.get(trail_id)
        if trail:
            summary = trail.to_dict()
            logger.debug("Retrieved summary for trail '%s'.", trail_id) # Changed to debug
            return summary
        else:
            logger.warning("No trail found with id '%s' for get_trail_summary.", trail_id)
            return {}

    def to_dict(self) -> dict:
        """
        Serializes all trails. Returns empty dict if TRAIL_MANAGER feature disabled.
        """
         # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
            logger.debug("Skipping TrailManager serialization: TRAIL_MANAGER feature disabled.")
            return {} # Return empty dict representing no state
        # --- End Check ---

        logger.debug("Serializing TrailManager state (%d trails).", len(self.trails))
        # Structure as {"trails": {...}} to match update_from_dict expectation
        return {"trails": {trail_id: trail.to_dict() for trail_id, trail in self.trails.items()}}

    def update_from_dict(self, data: dict):
        """
        Rehydrates the TrailManager. Clears state if TRAIL_MANAGER feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.TRAIL_MANAGER):
            logger.debug("Clearing state via update_from_dict: TRAIL_MANAGER feature disabled.")
            self.trails = {} # Clear existing trails
            return
        # --- End Check ---

        # Feature enabled, proceed with loading
        if not isinstance(data, dict):
             logger.warning("Invalid data type passed to TrailManager.update_from_dict: %s. State not updated.", type(data))
             self.trails = {} # Reset if data is invalid
             return

        # Expecting data in the format {"trails": {trail_id: trail_data, ...}}
        trails_data = data.get("trails", {})
        if not isinstance(trails_data, dict):
             logger.warning("Invalid 'trails' format in data: Expected dict, got %s. State not updated.", type(trails_data))
             self.trails = {} # Reset if trails data is invalid
             return

        loaded_trails = {}
        loaded_count = 0
        skipped_count = 0
        for trail_id, tdata in trails_data.items():
            try:
                if isinstance(tdata, dict):
                     loaded_trails[trail_id] = Trail.from_dict(tdata)
                     loaded_count += 1
                else:
                     logger.warning("Skipping invalid trail data for id '%s': type %s", trail_id, type(tdata))
                     skipped_count += 1
            except Exception as e:
                 logger.exception("Error loading trail '%s' from dict: %s", trail_id, e)
                 skipped_count += 1

        self.trails = loaded_trails # Assign fully loaded dictionary
        logger.info("TrailManager state updated from dict. Loaded %d trails, skipped %d.", loaded_count, skipped_count)

    def __str__(self):
        # status = "ENABLED" if is_enabled(Feature.TRAIL_MANAGER) else "DISABLED"
        # return f"TrailManager (Status: {status}, Trails: {len(self.trails)})"
        # Simpler version:
        return f"TrailManager (Trails: {len(self.trails)})"


# Make classes available for import if this file is treated as a module index
__all__ = ["TrailEvent", "Trail", "TrailManager"]
