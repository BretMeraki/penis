# forest_app/modules/archetype.py

import json
import logging
from typing import List, Dict, Optional, Any # Added Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Import Feature Flags ---
# Assuming feature_flags.py is accessible from this module's path
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger.warning("Feature flags module not found in archetype. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        ARCHETYPES = "FEATURE_ENABLE_ARCHETYPES" # Define the specific flag used here
    def is_enabled(feature: Any) -> bool: # Dummy function - default to True or False
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, depending on desired fallback

# --- Import Constants ---
from forest_app.config.constants import (
    ARCHETYPE_ACTIVATION_THRESHOLD,
    ARCHETYPE_DOMINANCE_FACTOR,
    DEFAULT_ARCHETYPE_WEIGHT,
    ARCHETYPE_CONTEXT_FACTOR_CAPACITY,
    ARCHETYPE_CONTEXT_FACTOR_SHADOW,
    LOW_CAPACITY_THRESHOLD,
    HIGH_SHADOW_THRESHOLD,
    DEFAULT_SNAPSHOT_CAPACITY,
    DEFAULT_SNAPSHOT_SHADOW,
)


class Archetype:
    """
    Represents a single archetype with defined traits and dynamic context parameters.
    Weight adjustment respects the ARCHETYPES feature flag.
    """
    def __init__(
        self,
        name: str,
        core_trait: str,
        emotional_priority: str,
        shadow_expression: str,
        transformation_style: str,
        tag_bias: List[str],
        default_weight: float = DEFAULT_ARCHETYPE_WEIGHT,
        context_factors: Optional[Dict[str, float]] = None,
    ):
        self.name = name
        self.core_trait = core_trait
        self.emotional_priority = emotional_priority
        self.shadow_expression = shadow_expression
        self.transformation_style = transformation_style
        self.tag_bias = tag_bias
        self.default_weight = default_weight
        self.context_factors = context_factors or {}
        self.current_weight = default_weight

    def to_dict(self) -> dict:
        """Serializes Archetype object to a dictionary."""
        # No flag check needed here, just serializes current state
        return {
            "name": self.name,
            "core_trait": self.core_trait,
            "emotional_priority": self.emotional_priority,
            "shadow_expression": self.shadow_expression,
            "transformation_style": self.transformation_style,
            "tag_bias": self.tag_bias,
            "default_weight": self.default_weight,
            "context_factors": self.context_factors,
            "current_weight": self.current_weight,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Archetype":
        """Creates an Archetype object from a dictionary."""
        # No flag check needed here, just deserializes
        arch = cls(
            name=data.get("name", "Unknown Archetype"),
            core_trait=data.get("core_trait", ""),
            emotional_priority=data.get("emotional_priority", ""),
            shadow_expression=data.get("shadow_expression", ""),
            transformation_style=data.get("transformation_style", ""),
            tag_bias=data.get("tag_bias", []),
            default_weight=data.get("default_weight", DEFAULT_ARCHETYPE_WEIGHT),
            context_factors=data.get("context_factors"),
        )
        # Also load current_weight if present, otherwise it stays default
        arch.current_weight = data.get("current_weight", arch.default_weight)
        return arch

    def adjust_weight(self, context: Dict[str, float]) -> None:
        """
        Dynamically adjusts current_weight based on capacity and shadow_score from context.
        Resets to default weight if ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            if self.current_weight != self.default_weight:
                 logger.debug("Resetting archetype '%s' weight to default: ARCHETYPES feature disabled.", self.name)
                 self.current_weight = self.default_weight
            return # Skip adjustments if feature is off
        # --- End Check ---

        capacity = context.get("capacity", DEFAULT_SNAPSHOT_CAPACITY)
        shadow = context.get("shadow_score", DEFAULT_SNAPSHOT_SHADOW)
        new_weight = self.default_weight

        if capacity < LOW_CAPACITY_THRESHOLD and "caretaker" in self.name.lower():
            capacity_factor = self.context_factors.get("capacity", ARCHETYPE_CONTEXT_FACTOR_CAPACITY)
            new_weight += capacity_factor

        if shadow > HIGH_SHADOW_THRESHOLD and "healer" in self.name.lower():
            shadow_factor = self.context_factors.get("shadow", ARCHETYPE_CONTEXT_FACTOR_SHADOW)
            new_weight += shadow_factor

        adjusted_weight = max(0.0, new_weight)
        if adjusted_weight != self.current_weight:
             self.current_weight = adjusted_weight
             logger.debug(
                 "Archetype '%s' adjusted weight to %.2f based on context %s",
                 self.name, self.current_weight, {"capacity": capacity, "shadow_score": shadow}
             )
        # No logging if weight didn't change

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ArchetypeManager:
    """
    Manages a collection of archetypes, dynamically selecting and blending them
    based on the MemorySnapshot state. Respects the ARCHETYPES feature flag.
    """
    def __init__(self):
        self.archetypes: List[Archetype] = [] # Stores all loaded archetype definitions
        self.active_archetypes: Dict[str, Archetype] = {} # Stores currently active archetypes

    def _clear_state(self):
        """Helper to reset the manager's state."""
        self.archetypes = []
        self.active_archetypes = {}
        logger.debug("ArchetypeManager state cleared.")

    def load_archetypes(self, archetype_list: List[dict]):
        """
        Load archetypes from a list of dicts. Does nothing if ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            logger.debug("Skipping load_archetypes: ARCHETYPES feature disabled.")
            self._clear_state() # Ensure state is empty if feature is off
            return
        # --- End Check ---

        self._clear_state() # Clear previous before loading
        valid_archetypes = 0
        for item in archetype_list:
            try:
                if isinstance(item, dict) and "name" in item:
                    arch = Archetype.from_dict(item)
                    self.archetypes.append(arch)
                    valid_archetypes += 1
                else:
                    logger.warning("Skipping invalid archetype data during load (missing name or not dict): %s", item)
            except Exception as e:
                logger.exception("Error loading archetype from data: %s. Data: %s", e, item)

        # Initialize active_archetypes with all loaded archetypes; weights will be updated later
        # No need for flag check here as we wouldn't reach this point if the flag was off
        self.active_archetypes = {arch.name: arch for arch in self.archetypes}
        logger.info("Loaded %d valid archetypes.", valid_archetypes)

    def set_active_archetype(self, name: str) -> bool:
        """
        Forcefully replace the active set with a single archetype matching `name`.
        Does nothing if ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            logger.warning("Cannot set_active_archetype: ARCHETYPES feature disabled.")
            return False
        # --- End Check ---

        found_archetype = None
        for arch in self.archetypes: # Search in loaded definitions
            if arch.name.lower() == name.lower():
                found_archetype = arch
                break

        if found_archetype:
            self.active_archetypes = {found_archetype.name: found_archetype}
            logger.info("Active archetype forcefully set to '%s'. Dynamic weights bypassed.", found_archetype.name)
            return True
        else:
            logger.warning("Archetype '%s' not found for set_active_archetype.", name)
            return False

    def update_active_archetypes(self, snapshot: dict):
        """
        Recomputes weights and selects active archetypes based on snapshot context.
        Clears active archetypes if ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            if self.active_archetypes: # Only clear if not already empty
                 logger.debug("Clearing active archetypes: ARCHETYPES feature disabled.")
                 self.active_archetypes = {}
            return
        # --- End Check ---

        context = {
            "capacity": snapshot.get("capacity", DEFAULT_SNAPSHOT_CAPACITY),
            "shadow_score": snapshot.get("shadow_score", DEFAULT_SNAPSHOT_SHADOW),
        }

        if not self.archetypes:
            logger.debug("No archetypes loaded. Cannot update active archetypes.") # Changed to debug
            self.active_archetypes = {}
            return

        for arch in self.archetypes:
            try:
                arch.adjust_weight(context)
            except Exception as e:
                logger.exception("Error adjusting weight for archetype '%s': %s", arch.name, e)

        filtered_archetypes = {
            arch.name: arch
            for arch in self.archetypes
            if arch.current_weight >= ARCHETYPE_ACTIVATION_THRESHOLD
        }

        if filtered_archetypes:
            self.active_archetypes = filtered_archetypes
        elif self.archetypes:
            try:
                top_archetype = max(self.archetypes, key=lambda a: a.current_weight)
                self.active_archetypes = {top_archetype.name: top_archetype}
                logger.info("No archetypes met activation threshold. Falling back to single highest: '%s' (Weight: %.2f)",
                            top_archetype.name, top_archetype.current_weight)
            except ValueError:
                logger.warning("Cannot determine top archetype: no archetypes loaded.")
                self.active_archetypes = {}
        else:
             self.active_archetypes = {}

        logger.info("Active archetypes after update: %s", list(self.active_archetypes.keys()))

    def get_influence(self) -> dict:
        """
        Blend or pick dominant archetype influence. Returns neutral defaults
        if ARCHETYPES feature is disabled or no archetypes are active.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            logger.debug("Returning neutral influence: ARCHETYPES feature disabled.")
            return {"transformation_style": "neutral", "tag_bias": []}
        # --- End Check ---

        if not self.active_archetypes:
            logger.debug("Returning neutral influence: No active archetypes.")
            return {"transformation_style": "neutral", "tag_bias": []}

        sorted_active = sorted(
            self.active_archetypes.values(),
            key=lambda a: getattr(a, 'current_weight', 0.0),
            reverse=True
        )

        if len(sorted_active) > 1 and \
           getattr(sorted_active[0], 'current_weight', 0.0) >= \
           ARCHETYPE_DOMINANCE_FACTOR * getattr(sorted_active[1], 'current_weight', 0.0):
            dominant_arch = sorted_active[0]
            logger.debug("Dominant archetype influence selected: '%s'", dominant_arch.name)
            return {
                "transformation_style": dominant_arch.transformation_style,
                "tag_bias": list(dominant_arch.tag_bias),
            }
        else:
            blended_style = " / ".join(
                f"{a.transformation_style} ({getattr(a, 'current_weight', 0.0):.2f})"
                for a in sorted_active
            )
            blended_tags = []
            seen_tags = set()
            for a in sorted_active:
                for tag in getattr(a, 'tag_bias', []):
                    if tag not in seen_tags:
                        blended_tags.append(tag)
                        seen_tags.add(tag)
            logger.debug("Blending influence from active archetypes: %s", list(self.active_archetypes.keys()))
            return {"transformation_style": blended_style, "tag_bias": blended_tags}

    def to_dict(self) -> dict:
        """
        Serializes the ArchetypeManager state. Returns empty dict if
        ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            logger.debug("Skipping ArchetypeManager serialization: ARCHETYPES feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing ArchetypeManager state.")
        return {
            "archetypes": [a.to_dict() for a in self.archetypes],
            "active_archetypes": {name: arch.to_dict() for name, arch in self.active_archetypes.items()},
        }

    def update_from_dict(self, data: dict):
        """
        Loads the ArchetypeManager state. Clears state and ignores data if
        ARCHETYPES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.ARCHETYPES):
            logger.debug("Clearing state via update_from_dict: ARCHETYPES feature disabled.")
            self._clear_state() # Reset to empty if feature is off
            return
        # --- End Check ---

        if not isinstance(data, dict):
            logger.error("Invalid data passed to ArchetypeManager.update_from_dict: Expected dict, got %s", type(data))
            self._clear_state() # Reset if data is invalid
            return

        # Feature is enabled, proceed with loading
        self._clear_state() # Clear before loading new state

        # Load definitions
        archetype_defs_data = data.get("archetypes", [])
        if isinstance(archetype_defs_data, list):
            for arch_data in archetype_defs_data:
                try:
                    if isinstance(arch_data, dict):
                        loaded_arch = Archetype.from_dict(arch_data)
                        # Restore current_weight from saved data
                        loaded_arch.current_weight = arch_data.get("current_weight", loaded_arch.default_weight)
                        self.archetypes.append(loaded_arch)
                    else:
                        logger.warning("Skipping non-dict item in archetypes list during load: %s", type(arch_data))
                except Exception as e:
                    logger.exception("Error loading archetype definition from dict: %s. Data: %s", e, arch_data)
        else:
            logger.warning("Archetypes data in update_from_dict is not a list. Definitions not loaded.")

        # Load active archetypes (use definitions loaded above)
        active_archetypes_data = data.get("active_archetypes", {})
        if isinstance(active_archetypes_data, dict):
            # Build a lookup of loaded definitions by name
            definitions_lookup = {arch.name: arch for arch in self.archetypes}
            for name, arch_data in active_archetypes_data.items():
                 # Find the corresponding loaded definition
                 if name in definitions_lookup:
                      # Use the instance from self.archetypes
                      active_arch_instance = definitions_lookup[name]
                      # Update its current_weight from the saved active data (important!)
                      if isinstance(arch_data, dict):
                           active_arch_instance.current_weight = arch_data.get("current_weight", active_arch_instance.default_weight)
                      # Add the instance (from self.archetypes) to the active dict
                      self.active_archetypes[name] = active_arch_instance
                 else:
                      logger.warning("Active archetype '%s' found in data, but no matching definition loaded. Skipping.", name)
                      # Optionally try to load from arch_data itself as fallback?
                      # try:
                      #    if isinstance(arch_data, dict):
                      #         fallback_arch = Archetype.from_dict(arch_data)
                      #         self.active_archetypes[name] = fallback_arch
                      # except Exception as e: logger.exception(...)

        else:
            logger.warning("Active archetypes data in update_from_dict is not a dict. Active set not loaded.")

        logger.debug("ArchetypeManager state updated from dict. Loaded %d definitions, %d active.",
                     len(self.archetypes), len(self.active_archetypes))

    def __str__(self):
        # status = "ENABLED" if is_enabled(Feature.ARCHETYPES) else "DISABLED"
        # active_names = list(self.active_archetypes.keys())
        # return f"ArchetypeManager (Status: {status}, Active: {active_names})\nDefinitions: {len(self.archetypes)}"
        # Simpler version for now:
        return json.dumps(self.to_dict(), indent=2)


# Make classes available for import
__all__ = ["Archetype", "ArchetypeManager"]
