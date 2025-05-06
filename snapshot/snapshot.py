# forest_app/core/snapshot.py (MODIFIED FOR BATCH TRACKING)
import json
import logging
from datetime import datetime, timezone # Use timezone-aware
# --- Ensure necessary typing imports ---
from typing import Dict, List, Any, Optional, TYPE_CHECKING, cast

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Can uncomment for verbose debug

# --- Import Feature enum and is_enabled with better error handling ---
feature_flags_available = False
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    feature_flags_available = True
    logger.info("Feature flags module loaded successfully")
except ImportError as e:
    logger.warning("Feature flags module not found. Feature flag recording in snapshot will be disabled: %s", e)
    class Feature:
        __members__ = {}
    def is_enabled(feature: Any) -> bool: return False
except Exception as e:
    logger.error("Unexpected error loading feature flags: %s", e)
    class Feature:
        __members__ = {}
    def is_enabled(feature: Any) -> bool: return False

# --- ADDED: Import Field from Pydantic if needed ---
# If you transition this class to Pydantic, you'll use Field
# from pydantic import Field, BaseModel
# For now, we'll add attributes directly

class MemorySnapshot:
    """Serializable container for user journey state (including feature flags, batch tracking)."""

    def __init__(self) -> None:
        # ---- Core progress & wellbeing gauges ----
        self.shadow_score: float = 0.50
        self.capacity: float = 0.50
        self.magnitude: float = 5.00
        self.resistance: float = 0.00
        self.relationship_index: float = 0.50

        # ---- Narrative scaffolding ----
        self.story_beats: List[Dict[str, Any]] = []
        self.totems: List[Dict[str, Any]] = []

        # ---- Desire & pairing caches ----
        self.wants_cache: Dict[str, float] = {}
        self.partner_profiles: Dict[str, Dict[str, Any]] = {}

        # ---- Engagement maintenance ----
        self.withering_level: float = 0.00

        # ---- Activation & core pathing ----
        self.activated_state: Dict[str, Any] = {
            "activated": False, "mode": None, "goal_set": False,
        }
        self.core_state: Dict[str, Any] = {} # Holds HTA Tree under 'hta_tree' key
        self.decor_state: Dict[str, Any] = {}

        # ---- Path & deadlines ----
        self.current_path: str = "structured"
        self.estimated_completion_date: Optional[str] = None

        # ---- Logs / context ----
        self.reflection_context: Dict[str, Any] = {
            "themes": [], "recent_insight": "", "current_priority": "",
        }
        self.reflection_log: List[Dict[str, Any]] = []
        self.task_backlog: List[Dict[str, Any]] = []
        self.task_footprints: List[Dict[str, Any]] = []

        # ---- Conversation History ----
        self.conversation_history: List[Dict[str, str]] = []

        # --- Feature flag state ---
        self.feature_flags: Dict[str, bool] = {}

        # --- Batch Tracking ---
        self.current_frontier_batch_ids: List[str] = []
        # --- MODIFIED: Added field for accumulating reflections ---
        self.current_batch_reflections: List[str] = []
        # --- END MODIFIED ---

        # ---- Component state stubs ----
        # Stores serializable state from various engines/managers
        self.component_state: Dict[str, Any] = {
            "sentiment_engine_calibration": {}, "metrics_engine": {},
            "seed_manager": {}, "archetype_manager": {}, "dev_index": {},
            "memory_system": {}, "xp_mastery": {}, "pattern_engine_config": {},
            "emotional_integrity_index": {}, "desire_engine": {},
            "resistance_engine": {}, "reward_index": {},
            "last_issued_task_id": None, "last_activity_ts": None,
            # Removed direct engine instances from __init__ as they should be managed via DI
            # and their state loaded/saved via component_state
        }

        # ---- Misc meta ----
        self.template_metadata: Dict[str, Any] = {}
        self.last_ritual_mode: str = "Trail"
        self.timestamp: str = datetime.now(timezone.utc).isoformat() # Use timezone aware

    def record_feature_flags(self) -> None:
        """Record current feature flag states."""
        feature_flags_available = True
        try:
            from forest_app.core.feature_flags import Feature, is_enabled
        except ImportError:
            feature_flags_available = False
        
        self.feature_flags = {}  # Clear previous state first
        
        if not feature_flags_available:
            logger.warning("Feature flags module not available, skipping flag recording")
            return
        
        if not hasattr(Feature, '__members__'):
            logger.error("Feature enum has no __members__ attribute")
            return
        
        for feature_name, feature_enum in Feature.__members__.items():
            try:
                status = is_enabled(feature_enum)
                self.feature_flags[feature_name] = status
                logger.debug("Recorded feature %s status: %s", feature_name, status)
            except Exception as e:
                logger.error("Error recording feature %s status: %s", feature_name, e)
                self.feature_flags[feature_name] = False  # Safe default
        
        logger.debug("Recorded %d feature flags", len(self.feature_flags))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise entire snapshot to a dict (JSON‑safe)."""
        # Ensure timestamp is current at serialization time
        self.timestamp = datetime.now(timezone.utc).isoformat()

        data = {
            # Core gauges
            "shadow_score": self.shadow_score, "capacity": self.capacity,
            "magnitude": self.magnitude, "resistance": self.resistance,
            "relationship_index": self.relationship_index,
            # Narrative
            "story_beats": self.story_beats, "totems": self.totems,
            # Desire / pairing
            "wants_cache": self.wants_cache, "partner_profiles": self.partner_profiles,
            # Engagement
            "withering_level": self.withering_level,
            # Activation / state
            "activated_state": self.activated_state, "core_state": self.core_state,
            "decor_state": self.decor_state,
            # Path & deadlines
            "current_path": self.current_path,
            "estimated_completion_date": self.estimated_completion_date,
            # Logs
            "reflection_context": self.reflection_context,
            "reflection_log": self.reflection_log,
            "task_backlog": self.task_backlog,
            "task_footprints": self.task_footprints,
            # Conversation History
            "conversation_history": self.conversation_history,
            # Feature flags
            "feature_flags": self.feature_flags,
            # --- MODIFIED: Batch Tracking Serialization ---
            "current_frontier_batch_ids": self.current_frontier_batch_ids,
            "current_batch_reflections": self.current_batch_reflections, # <-- Added
            # --- END MODIFIED ---
            # Component states
            "component_state": self.component_state,
            # Misc
            "template_metadata": self.template_metadata,
            "last_ritual_mode": self.last_ritual_mode,
            "timestamp": self.timestamp,
        }
        return data # Return the constructed dictionary

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Rehydrate snapshot from dict, preserving unknown fields defensively."""
        if not isinstance(data, dict):
            logger.error("Invalid data passed to update_from_dict: expected dict, got %s", type(data))
            return

        # --- MODIFIED: Added batch lists to attributes list ---
        attributes_to_load = [
            "shadow_score", "capacity", "magnitude", "resistance",
            "relationship_index",  # Removed hardware_config as it wasn't in __init__
            "activated_state", "core_state", "decor_state", "reflection_context",
            "reflection_log", "task_backlog", "task_footprints",
            "story_beats", "totems", "wants_cache", "partner_profiles",
            "withering_level", "current_path", "estimated_completion_date",
            "template_metadata", "last_ritual_mode", "timestamp",
            "conversation_history", "feature_flags",
            "current_frontier_batch_ids",  # <-- Added
            "current_batch_reflections",  # <-- Added
            # Removed hardware_config from here too
        ]
        # --- END MODIFIED ---

        for attr in attributes_to_load:
            if attr in data:
                value = data[attr]
                # Default expectation is list, adjust based on attr name
                expected_type = list
                default_value = []
                if attr in ["core_state", "feature_flags", "component_state", "activated_state", "decor_state", "reflection_context", "wants_cache", "partner_profiles", "template_metadata"]:  # Removed hardware_config
                    expected_type = dict
                    default_value = {}
                elif attr in ["current_path", "estimated_completion_date", "last_ritual_mode", "timestamp"]:
                    expected_type = str
                    default_value = "" if attr != "current_path" else "structured"
                elif attr in ["shadow_score", "capacity", "magnitude", "resistance", "relationship_index", "withering_level"]:
                    expected_type = float
                    default_value = 0.0
                # --- MODIFIED: Explicit check for the new list ---
                elif attr in ["current_batch_reflections", "current_frontier_batch_ids"]:
                    expected_type = list
                    default_value = []  # Should be list of strings
                # --- END MODIFIED ---

                if isinstance(value, expected_type):
                    setattr(self, attr, value)
                # Handle None for types that support it or reset to default
                elif value is None and expected_type in [str, list, dict]:
                    setattr(self, attr, None if expected_type is str else default_value)
                # --- MODIFIED: Enhanced type conversion handling ---
                elif expected_type is float and isinstance(value, (int, str)):
                    try:
                        converted_value = float(value)
                        logger.debug("Converting %s value '%s' to float: %f", type(value).__name__, value, converted_value)
                        setattr(self, attr, converted_value)
                    except (ValueError, TypeError) as e:
                        logger.warning("Failed to convert '%s' to float for '%s': %s", value, attr, e)
                        setattr(self, attr, default_value)
                elif expected_type is str and not isinstance(value, str):
                    try:
                        converted_value = str(value)
                        logger.debug("Converting %s to string for '%s'", type(value).__name__, attr)
                        setattr(self, attr, converted_value)
                    except Exception as e:
                        logger.warning("Failed to convert value to string for '%s': %s", attr, e)
                        setattr(self, attr, default_value)
                # --- MODIFIED: Enhanced list/dict conversion handling ---
                elif expected_type is list and isinstance(value, str):
                    try:
                        # Try to parse JSON string to list
                        converted_value = json.loads(value)
                        if isinstance(converted_value, list):
                            logger.debug("Converted JSON string to list for '%s'", attr)
                            setattr(self, attr, converted_value)
                        else:
                            logger.warning("JSON string did not parse to list for '%s'", attr)
                            setattr(self, attr, default_value)
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON string to list for '%s': %s", attr, e)
                        setattr(self, attr, default_value)
                elif expected_type is dict and isinstance(value, str):
                    try:
                        # Try to parse JSON string to dict
                        converted_value = json.loads(value)
                        if isinstance(converted_value, dict):
                            logger.debug("Converted JSON string to dict for '%s'", attr)
                            setattr(self, attr, converted_value)
                        else:
                            logger.warning("JSON string did not parse to dict for '%s'", attr)
                            setattr(self, attr, default_value)
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON string to dict for '%s': %s", attr, e)
                        setattr(self, attr, default_value)
                # --- END MODIFIED ---
                else:
                    logger.warning(
                        "Unexpected type for '%s': expected %s, got %s. Using default.",
                        attr, expected_type.__name__, type(value).__name__
                    )
                    setattr(self, attr, default_value)
            elif attr in [  # Ensure list/dict types default correctly if missing
                "conversation_history", "feature_flags", "core_state", "component_state",
                "task_backlog", "reflection_log", "task_footprints", "story_beats", "totems",
                "current_frontier_batch_ids", "current_batch_reflections"  # <-- Added batch lists here
            ]:
                # Use getattr with default to safely check/set default
                if getattr(self, attr, None) is None:
                    default_value = [] if 'list' in str(self.__annotations__.get(attr, '')).lower() or 'List' in str(self.__annotations__.get(attr, '')) else {}
                    logger.debug("Attribute '%s' missing in loaded data, setting default.", attr)
                    setattr(self, attr, default_value)

        # Ensure type consistency *after* loading attempt
        # --- MODIFIED: Added checks for batch tracking list types ---
        if not isinstance(getattr(self, 'conversation_history', []), list): self.conversation_history = []
        if not isinstance(getattr(self, 'core_state', {}), dict): self.core_state = {}
        if not isinstance(getattr(self, 'feature_flags', {}), dict): self.feature_flags = {}
        if not isinstance(getattr(self, 'component_state', {}), dict): self.component_state = {}
        if not isinstance(getattr(self, 'current_frontier_batch_ids', []), list):
            logger.warning("Post-load current_frontier_batch_ids is not a list (%s), resetting.", type(getattr(self, 'current_frontier_batch_ids', None)))
            self.current_frontier_batch_ids = []
        if not isinstance(getattr(self, 'current_batch_reflections', []), list):
            logger.warning("Post-load current_batch_reflections is not a list (%s), resetting.", type(getattr(self, 'current_batch_reflections', None)))
            self.current_batch_reflections = []
        # --- END MODIFIED ---

        # Component_state blob loading remains unchanged
        loaded_cs = data.get("component_state")
        if isinstance(loaded_cs, dict):
            self.component_state = loaded_cs
        elif loaded_cs is not None:
            logger.warning("Loaded component_state is not a dict (%s), ignoring.", type(loaded_cs))
            if not hasattr(self, 'component_state') or not isinstance(self.component_state, dict): self.component_state = {}
        else:
             if not hasattr(self, 'component_state') or not isinstance(self.component_state, dict): self.component_state = {}


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySnapshot":
        """Creates a new MemorySnapshot instance from dictionary data."""
        # [Logging remains largely unchanged, ensure sensitive data isn't logged excessively]
        snap = cls()
        if isinstance(data, dict):
            snap.update_from_dict(data)
            # Log state *after* update_from_dict has run
            logger.debug("FROM_DICT: Value of instance.core_state['hta_tree'] AFTER update: %s",
                         snap.core_state.get('hta_tree', 'MISSING_POST_ASSIGNMENT'))
            logger.debug("FROM_DICT: Loaded feature flags AFTER update: %s", snap.feature_flags)
            # --- ADDED: Log batch state ---
            logger.debug("FROM_DICT: Loaded batch IDs AFTER update: %s", snap.current_frontier_batch_ids)
            logger.debug("FROM_DICT: Loaded batch reflections count AFTER update: %s", len(snap.current_batch_reflections))
            # --- END ADDED ---
        else:
            logger.error("Invalid data passed to MemorySnapshot.from_dict: expected dict, got %s. Returning default snapshot.", type(data))

        return snap

    def __str__(self) -> str:
        """Provides a string representation, robust against serialization errors."""
        try:
            # Use a limited set of keys for basic string representation
            repr_dict = {
                "shadow_score": round(getattr(self, 'shadow_score', 0.0), 2),
                "capacity": round(getattr(self, 'capacity', 0.0), 2),
                "magnitude": round(getattr(self, 'magnitude', 0.0), 1),
                "feature_flags_count": len(getattr(self, 'feature_flags', {})),
                "batch_ids_count": len(getattr(self, 'current_frontier_batch_ids', [])), # <-- Modified
                "batch_refl_count": len(getattr(self, 'current_batch_reflections', [])), # <-- Added
                "timestamp": getattr(self, 'timestamp', 'N/A')
            }
            return f"<Snapshot {json.dumps(repr_dict, default=str)} ...>"
        except Exception as exc:
            logger.error("Snapshot __str__ error: %s", exc)
            return f"<Snapshot ts={getattr(self, 'timestamp', 'N/A')} (error rendering)>"
