# Rewritten snapshot module (e.g., forest_app/modules/snapshot_utils.py or similar)
import json
import logging
import os
import sys # For stderr
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Dict, Any, List # Added typing

# --- Import Feature Flags ---
# Need to check flags to determine if data *should* be present
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    # Use the real is_enabled if available
except ImportError:
    # Fallback if feature flags module isn't available
    print("ERROR: Snapshot module could not import feature flags. Cannot reliably check features.", file=sys.stderr)
    # Define minimal dummy Feature enum and is_enabled
    class Feature:
        XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"
        SHADOW_ANALYSIS = "FEATURE_ENABLE_SHADOW_ANALYSIS" # Assumption
        METRICS_SPECIFIC = "FEATURE_ENABLE_METRICS_SPECIFIC" # Assumption for capacity/magnitude
        DEVELOPMENT_INDEX = "FEATURE_ENABLE_DEVELOPMENT_INDEX"
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES" # For last_ritual_mode?
        SEED_MANAGER = "FEATURE_ENABLE_SEED_MANAGER" # Assuming seed manager might have its own flag or relates to another feature
        # Add others if needed for snapshot fields
        # If a field doesn't have a flag, we assume it's core or always enabled if present

    def is_enabled(feature) -> bool: return False # Assume all off if flags missing

# Placeholder for missing data due to disabled features or errors
FEATURE_DISABLED_SENTINEL = "FEATURE_DISABLED"
DATA_UNAVAILABLE_SENTINEL = "UNAVAILABLE" # If feature enabled but data access failed

logger = logging.getLogger(__name__)
# Configure logger (basic example, adjust as needed)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.INFO)


def load_snapshot_config():
    """
    Load snapshot configuration (e.g., fields to include) from an external JSON file.
    If the file is not found or an error occurs, return default configuration.
    NOTE: Current builder uses explicit logic; config might be for future tuning.
    """
    # Use path relative to this file or absolute path as appropriate
    # Corrected path assumption relative to a potential app root
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "snapshot_config.json")
    # Or use settings: from forest_app.config.settings import settings; config_path = settings.SNAPSHOT_CONFIG_PATH

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info("Loaded snapshot configuration from %s", config_path)
            return config
    except FileNotFoundError:
        logger.warning(
            "Snapshot config file not found at %s. Using default field list.",
            config_path,
        )
    except Exception as e:
        logger.warning(
            "Error loading snapshot config from %s: %s. Using default field list.",
            config_path, e,
        )
    # Default config if load fails
    return {
        "fields": [
            "xp", "shadow_score", "capacity", "magnitude",
            "current_seed", "top_tags", "development_indexes",
            "last_ritual_mode", "timestamp"
        ]
    }


class CallbackTrigger:
    """
    Monitors the number of interactions and triggers a snapshot build once the counter
    reaches a preset frequency.
    Uses CompressedSnapshotBuilder to create the snapshot.
    """

    def __init__(self, frequency: int = 5):
        if frequency <= 0:
            logger.warning("Snapshot trigger frequency must be positive, defaulting to 5.")
            frequency = 5
        self.counter = 0
        self.frequency = frequency
        self.last_snapshot: Optional[Dict[str, Any]] = None
        # Instantiate the builder here or receive it via DI if needed elsewhere
        self.builder = CompressedSnapshotBuilder()
        logger.info("CallbackTrigger initialized with frequency %d", self.frequency)

    def register_interaction(self, full_snapshot: Any) -> Optional[Dict[str, Any]]:
        """Increments counter and triggers snapshot build if frequency is met."""
        self.counter += 1
        logger.debug("Interaction registered, counter at %d/%d", self.counter, self.frequency)
        if self.counter >= self.frequency:
            logger.info("Snapshot frequency reached. Triggering build.")
            self.counter = 0 # Reset counter
            self.last_snapshot = self.builder.build(full_snapshot)
            # logger.info("Snapshot triggered: %s", self.last_snapshot) # Builder logs this
            return self.last_snapshot
        return None # Return None if snapshot wasn't triggered

    def force_trigger(self, full_snapshot: Any) -> Dict[str, Any]:
        """Forces a snapshot build immediately."""
        logger.info("Forcing snapshot build.")
        self.counter = 0 # Reset counter
        self.last_snapshot = self.builder.build(full_snapshot)
        # logger.info("Forced snapshot: %s", self.last_snapshot) # Builder logs this
        return self.last_snapshot

    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Returns the most recently built snapshot."""
        return self.last_snapshot


class SnapshotRotatingSaver:
    """
    Maintains a rolling backup of compressed snapshots in memory using a deque.
    Includes basic JSON export/load functionality.
    """

    def __init__(self, max_snapshots: int = 10):
        if max_snapshots <= 0:
             logger.warning("Max snapshots must be positive, defaulting to 10.")
             max_snapshots = 10
        self.snapshots: deque = deque(maxlen=max_snapshots)
        logger.info("SnapshotRotatingSaver initialized with max_snapshots %d", max_snapshots)


    def store_snapshot(self, snapshot: Dict[str, Any]):
        """Stores a snapshot dictionary with a timestamp."""
        if not isinstance(snapshot, dict):
             logger.error("Attempted to store non-dict snapshot: %s", type(snapshot))
             return
        # Use UTC time
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "snapshot": snapshot}
        self.snapshots.append(record)
        logger.info("Snapshot stored at %s (Deque size: %d)", record["timestamp"], len(self.snapshots))

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent snapshot record (dict with 'timestamp' and 'snapshot')."""
        return self.snapshots[-1] if self.snapshots else None

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns all stored snapshot records as a list."""
        return list(self.snapshots)

    def export_to_json(self, filepath: str):
        """Exports all stored snapshots to a JSON file."""
        logger.info("Attempting to export snapshots to %s", filepath)
        try:
            # Ensure directory exists if filepath includes directories
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(self.get_all(), f, indent=2)
            logger.info("Snapshots successfully exported to %s", filepath)
        except (IOError, OSError, json.JSONDecodeError) as e:
            logger.error("Error during snapshot export to %s: %s", filepath, e, exc_info=True)
        except Exception as e:
            logger.error("Unexpected error during snapshot export to %s: %s", filepath, e, exc_info=True)


    def load_from_json(self, filepath: str):
        """Loads snapshots from a JSON file, replacing current contents."""
        logger.info("Attempting to load snapshots from %s", filepath)
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                 raise ValueError("Loaded data is not a list.")
            # Re-initialize deque with loaded data and original maxlen
            self.snapshots = deque(data, maxlen=self.snapshots.maxlen)
            logger.info("Snapshots successfully loaded from %s (Deque size: %d)", filepath, len(self.snapshots))
        except FileNotFoundError:
             logger.error("Snapshot file not found during load: %s", filepath)
        except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
            logger.error("Error during snapshot load from %s: %s", filepath, e, exc_info=True)
        except Exception as e:
             logger.error("Unexpected error during snapshot load from %s: %s", filepath, e, exc_info=True)


class GPTMemorySync:
    """
    Packages a compressed snapshot into a succinct context string for LLM prompts,
    handling potentially missing data gracefully.
    """

    def __init__(self):
        self.last_processed_snapshot: Optional[Dict[str, Any]] = None

    def _format_value(self, value: Any, default: str = DATA_UNAVAILABLE_SENTINEL) -> str:
        """Helper to format values, handling None and specific sentinels."""
        if value is None or value == FEATURE_DISABLED_SENTINEL:
            return default
        if isinstance(value, list):
             # Filter out sentinels from lists before joining
             filtered_list = [str(item) for item in value if item not in (None, FEATURE_DISABLED_SENTINEL, DATA_UNAVAILABLE_SENTINEL)]
             return ', '.join(filtered_list) if filtered_list else default
        if isinstance(value, dict):
             # Filter out sentinels from dicts before dumping
             filtered_dict = {k: v for k, v in value.items() if v not in (None, FEATURE_DISABLED_SENTINEL, DATA_UNAVAILABLE_SENTINEL)}
             return json.dumps(filtered_dict) if filtered_dict else default
        return str(value)

    def inject_into_context(self, compressed_snapshot: Optional[Dict[str, Any]]) -> str:
        """Creates the context string for the LLM."""
        if not compressed_snapshot:
            logger.warning("GPTMemorySync received empty snapshot for context injection.")
            return "No memory state available."

        self.last_processed_snapshot = compressed_snapshot

        # Format each field, providing a default if data is missing/disabled
        context_lines = [
            f"XP: {self._format_value(compressed_snapshot.get('xp'))}",
            f"Shadow Score: {self._format_value(compressed_snapshot.get('shadow_score'))}",
            f"Capacity: {self._format_value(compressed_snapshot.get('capacity'))}",
            f"Magnitude: {self._format_value(compressed_snapshot.get('magnitude'))}",
            f"Current Seed: {self._format_value(compressed_snapshot.get('current_seed'))}",
            f"Top Tags: {self._format_value(compressed_snapshot.get('top_tags'))}",
            f"Development Indexes: {self._format_value(compressed_snapshot.get('development_indexes'))}",
            f"Last Ritual Mode: {self._format_value(compressed_snapshot.get('last_ritual_mode'))}"
        ]

        # Filter out lines where data was unavailable even after formatting (optional)
        # context_lines = [line for line in context_lines if DATA_UNAVAILABLE_SENTINEL not in line]

        context_string = "\n".join(context_lines)
        logger.info("Injecting snapshot context for LLM:\n%s", context_string)
        return context_string


class CompressedSnapshotBuilder:
    """
    Builds a compressed snapshot dictionary from a full snapshot object,
    respecting feature flags to handle potentially missing data.
    """

    def __init__(self):
        # Load config - currently used to inform which fields *might* exist
        self.config = load_snapshot_config()
        self.fields_to_include = self.config.get("fields", []) # Get list of expected fields
        logger.info("CompressedSnapshotBuilder initialized. Expected fields: %s", self.fields_to_include)

    def _safe_get(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get an attribute, returning default if attribute is missing."""
        return getattr(obj, attr, default)

    def _get_feature_data(self, feature: Feature, snapshot: Any, attr_name: str, extraction_func: Optional[callable] = None) -> Any:
        """
        Helper to get data point based on feature flag status.
        Handles attribute access and optional extraction functions.
        """
        if is_enabled(feature):
            value = self._safe_get(snapshot, attr_name, None)
            if value is None:
                 # Attribute missing even though feature is enabled
                 logger.warning("Attribute '%s' not found on snapshot despite feature '%s' being enabled.", attr_name, feature.name)
                 return DATA_UNAVAILABLE_SENTINEL
            if extraction_func:
                try:
                    return extraction_func(value)
                except Exception as e:
                    logger.error("Error during extraction function for feature '%s' attribute '%s': %s", feature.name, attr_name, e, exc_info=True)
                    return DATA_UNAVAILABLE_SENTINEL # Data exists but couldn't be processed
            return value
        else:
            return FEATURE_DISABLED_SENTINEL # Feature is disabled

    def build(self, full_snapshot: Any) -> Dict[str, Any]:
        """Builds the compressed dictionary, checking feature flags."""
        if full_snapshot is None:
             logger.error("CompressedSnapshotBuilder received None as full_snapshot.")
             # Return a minimal dict indicating failure or empty state
             return {"error": "No snapshot data provided", "timestamp": datetime.now(timezone.utc).isoformat()}

        compressed = {}

        # --- XP ---
        compressed["xp"] = self._get_feature_data(
             feature=Feature.XP_MASTERY, snapshot=full_snapshot, attr_name="xp"
        )

        # --- Shadow Score --- (Assuming SHADOW_ANALYSIS flag)
        compressed["shadow_score"] = self._get_feature_data(
             feature=Feature.SHADOW_ANALYSIS, snapshot=full_snapshot, attr_name="shadow_score"
        )

        # --- Capacity & Magnitude --- (Assuming METRICS_SPECIFIC flag)
        compressed["capacity"] = self._get_feature_data(
             feature=Feature.METRICS_SPECIFIC, snapshot=full_snapshot, attr_name="capacity"
        )
        compressed["magnitude"] = self._get_feature_data(
             feature=Feature.METRICS_SPECIFIC, snapshot=full_snapshot, attr_name="magnitude"
        )

        # --- Current Seed --- (Assuming SEED_MANAGER or related flag)
        def extract_seed(seed_manager_obj):
             # Safely call to_dict and get first element if possible
             if hasattr(seed_manager_obj, 'to_dict'):
                 seed_list = seed_manager_obj.to_dict()
                 return seed_list[0] if seed_list else {"name": "None", "status": "inactive"}
             logger.warning("Seed manager object missing 'to_dict' method.")
             return {"name": "Error", "status": "unknown"}

        compressed["current_seed"] = self._get_feature_data(
             feature=Feature.SEED_MANAGER, # Or whichever feature enables this
             snapshot=full_snapshot,
             attr_name="seed_manager",
             extraction_func=extract_seed
        )

        # --- Top Tags --- (Assuming Pattern ID or similar feature enables active_tags)
        def extract_top_tags(active_tags_dict):
             if isinstance(active_tags_dict, dict):
                 sorted_tags = sorted(active_tags_dict.items(), key=lambda x: x[1], reverse=True)
                 return [tag for tag, _ in sorted_tags[:3]]
             logger.warning("active_tags attribute was not a dictionary.")
             return []

        compressed["top_tags"] = self._get_feature_data(
             feature=Feature.PATTERN_ID, # Or flag related to tags
             snapshot=full_snapshot,
             attr_name="active_tags",
             extraction_func=extract_top_tags
        )

        # --- Development Indexes ---
        def extract_dev_index(dev_index_obj):
             if hasattr(dev_index_obj, 'to_dict'):
                 return dev_index_obj.to_dict()
             logger.warning("Development index object missing 'to_dict' method.")
             return {}

        compressed["development_indexes"] = self._get_feature_data(
             feature=Feature.DEVELOPMENT_INDEX,
             snapshot=full_snapshot,
             attr_name="dev_index",
             extraction_func=extract_dev_index
        )

        # --- Last Ritual Mode --- (Assuming NARRATIVE_MODES flag)
        compressed["last_ritual_mode"] = self._get_feature_data(
             feature=Feature.NARRATIVE_MODES, # Or flag related to rituals/modes
             snapshot=full_snapshot,
             attr_name="last_ritual_mode"
        )

        # --- Timestamp --- (Always add)
        compressed["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Optionally filter based on loaded config fields, though explicit checks are safer
        # filtered_compressed = {k: v for k, v in compressed.items() if k in self.fields_to_include}
        # logger.info("Compressed snapshot built: %s", filtered_compressed)
        # return filtered_compressed

        logger.info("Compressed snapshot built: %s", compressed)
        return compressed


class SnapshotFlowController:
    """
    Coordinates snapshot operations using helper classes.
    Receives a full snapshot object when interactions occur.
    Handles interaction -> trigger -> builder -> saver -> memory sync flow.
    """

    def __init__(self, frequency: int = 5, max_snapshots: int = 10):
        """Initializes coordinator with trigger, saver, and sync components."""
        logger.info("Initializing SnapshotFlowController...")
        self.trigger = CallbackTrigger(frequency=frequency)
        self.saver = SnapshotRotatingSaver(max_snapshots=max_snapshots)
        self.memory_sync = GPTMemorySync()
        # Consider loading snapshots from file on init if persistence is desired
        # self.saver.load_from_json("path/to/snapshots.json")

    def register_user_submission(self, full_snapshot: Any) -> Dict[str, Any]:
        """
        Registers an interaction (providing the full system state) and handles
        snapshot creation, storage, and context preparation if triggered.

        Args:
            full_snapshot: The complete snapshot object representing system state.

        Returns:
            A dictionary containing:
                'synced': bool indicating if a new snapshot was created and synced.
                'context_injection': The string context for LLM, or None.
                'compressed_snapshot': The created compressed snapshot dict, or None.
        """
        logger.debug("SnapshotFlowController registering user submission.")

        # Trigger potentially builds and returns a new compressed snapshot
        new_compressed_snapshot = self.trigger.register_interaction(full_snapshot)

        if new_compressed_snapshot:
            logger.info("New snapshot generated by trigger.")
            # Store the newly created snapshot
            self.saver.store_snapshot(new_compressed_snapshot)

            # Prepare context string using the new snapshot
            context_string = self.memory_sync.inject_into_context(new_compressed_snapshot)

            return {
                "synced": True,
                "context_injection": context_string,
                "compressed_snapshot": new_compressed_snapshot,
            }
        else:
            # Snapshot frequency not met
            logger.debug("Snapshot frequency not met, no new snapshot generated.")
            return {"synced": False, "context_injection": None, "compressed_snapshot": None}

    def get_latest_context(self) -> str:
        """
        Gets the LLM context string based on the most recently stored snapshot.
        """
        latest_record = self.saver.get_latest()
        if latest_record and 'snapshot' in latest_record:
            logger.info("Providing LLM context from latest stored snapshot.")
            # Use the snapshot part of the record
            return self.memory_sync.inject_into_context(latest_record["snapshot"])
        else:
            logger.warning("No recent snapshot available in saver to generate context.")
            # Provide default message via memory_sync
            return self.memory_sync.inject_into_context(None)

    def force_snapshot(self, full_snapshot: Any) -> Dict[str, Any]:
         """Forces snapshot creation, storage, and context generation."""
         logger.info("SnapshotFlowController forcing snapshot.")
         # Force trigger to build
         forced_snapshot = self.trigger.force_trigger(full_snapshot)
         # Store it
         self.saver.store_snapshot(forced_snapshot)
         # Prepare context
         context_string = self.memory_sync.inject_into_context(forced_snapshot)
         return {
             "synced": True, # Considered synced as it was just forced
             "context_injection": context_string,
             "compressed_snapshot": forced_snapshot,
         }

    # --- Optional: Add methods for saving/loading state ---
    def save_state_to_json(self, filepath: str):
         """Exports the rotating snapshots to JSON."""
         self.saver.export_to_json(filepath)

    def load_state_from_json(self, filepath: str):
         """Loads rotating snapshots from JSON."""
         self.saver.load_from_json(filepath)
