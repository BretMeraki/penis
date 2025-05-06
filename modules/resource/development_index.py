# forest_app/modules/development_index.py
# =====================================================================
#  FullDevelopmentIndex – 0‑to‑1 gauges of positive personal capacity
# =====================================================================
from __future__ import annotations

import json
import logging
from typing import Dict, List, Any # Added Any for Dict type hints

# --- Import Feature Flags ---
# Assuming feature_flags.py is accessible from this module's path
# Adjust the import path if necessary
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Provide fallbacks if feature flags aren't available
    logger.warning("Feature flags module not found in development_index. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        DEVELOPMENT_INDEX = "FEATURE_ENABLE_DEVELOPMENT_INDEX" # Define the specific flag used here
    def is_enabled(feature: Any) -> bool: # Dummy function - default to True or False based on desired fallback
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, depending on whether the feature should work if flags are broken

# --- Import Constants ---
from forest_app.config.constants import (
    DEVELOPMENT_INDEX_KEYS,
    DEFAULT_DEVELOPMENT_INDEX_VALUE,
    MIN_DEVELOPMENT_INDEX_VALUE,
    MAX_DEVELOPMENT_INDEX_VALUE,
    POSITIVE_REFLECTION_HINTS,
    BASELINE_NUDGE_KEYS,
    BASELINE_REFLECTION_NUDGE_AMOUNT,
    TASK_EFFECT_BASE_BOOST,
)

logger = logging.getLogger(__name__)
# Consider setting level via central config, but INFO is reasonable for module-level logs
logger.setLevel(logging.INFO)


# Helper function using constants for default clamp range
def _clamp(
    val: float,
    lo: float = MIN_DEVELOPMENT_INDEX_VALUE,
    hi: float = MAX_DEVELOPMENT_INDEX_VALUE
) -> float:
    """Clamps a value within the defined min/max range for the development index."""
    try:
        return max(float(lo), min(float(hi), float(val)))
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for clamping: {val}. Error: {e}. Returning default lower bound.")
        return float(lo)


class FullDevelopmentIndex:
    """
    Mutable container of {dev_key: float}, tracking development indexes.
    Respects the DEVELOPMENT_INDEX feature flag.
    • All values live within [MIN_DEVELOPMENT_INDEX_VALUE, MAX_DEVELOPMENT_INDEX_VALUE].
    • Provides helpers for baseline loading and task‑driven boosts using constants.
    """

    # ---------------------------------------------------------------- #
    def __init__(self) -> None:
        """Initializes all development indexes to the default constant value."""
        # Initialize state regardless of feature flag status
        self._reset_state()
        logger.debug("FullDevelopmentIndex initialized with default values.")

    # ---------------------------------------------------------------- #
    #  Internal State Reset Helper
    # ---------------------------------------------------------------- #
    def _reset_state(self) -> None:
        """Resets all indexes to their default value."""
        self.indexes: Dict[str, float] = {
            k: DEFAULT_DEVELOPMENT_INDEX_VALUE for k in DEVELOPMENT_INDEX_KEYS
        }
        logger.debug("Development indexes reset to default values.")

    # ---------------------------------------------------------------- #
    #  Public APIs (respecting feature flag)
    # ---------------------------------------------------------------- #
    def baseline_from_reflection(self, reflection: str) -> None:
        """
        Nudges happiness-adjacent gauges up slightly if reflection contains
        positive hint words (defined in constants). Does nothing if
        DEVELOPMENT_INDEX feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DEVELOPMENT_INDEX):
            logger.debug("Skipping baseline_from_reflection: DEVELOPMENT_INDEX feature disabled.")
            return
        # --- End Check ---

        if not isinstance(reflection, str) or not reflection:
            return # Skip if reflection is invalid

        low_reflection = reflection.lower()
        if any(hint.lower() in low_reflection for hint in POSITIVE_REFLECTION_HINTS):
            logger.debug("Positive reflection hints detected.")
            for key in BASELINE_NUDGE_KEYS:
                if key in self.indexes:
                    original_value = self.indexes[key]
                    new_value = _clamp(original_value + BASELINE_REFLECTION_NUDGE_AMOUNT)
                    if new_value > original_value: # Only update if changed
                         self.indexes[key] = new_value
                         logger.debug("Nudged index '%s' by %.4f -> %.4f", key, BASELINE_REFLECTION_NUDGE_AMOUNT, new_value)

    def dynamic_adjustment(self, deltas: Dict[str, float]) -> None:
        """
        Applies arbitrary external tweaks (e.g., from Metrics engine) to specified indexes.
        Does nothing if DEVELOPMENT_INDEX feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DEVELOPMENT_INDEX):
            logger.debug("Skipping dynamic_adjustment: DEVELOPMENT_INDEX feature disabled.")
            return
        # --- End Check ---

        if not isinstance(deltas, dict):
            logger.warning("dynamic_adjustment received non-dict deltas: %s", type(deltas))
            return

        for key, delta_value in deltas.items():
            if key in self.indexes:
                try:
                    new_value = _clamp(self.indexes[key] + float(delta_value))
                    if new_value != self.indexes[key]:
                        logger.debug("Dynamically adjusting index '%s': %.4f -> %.4f (delta: %.4f)",
                                     key, self.indexes[key], new_value, float(delta_value))
                        self.indexes[key] = new_value
                except (ValueError, TypeError) as e:
                    logger.error("Invalid delta value for key '%s': %s. Error: %s", key, delta_value, e)
            else:
                logger.warning("Attempted dynamic adjustment for unknown index key: %s", key)

    def apply_task_effect(
        self,
        relevant_indexes: List[str], # List of index keys this task affects
        tier_mult: float,            # Multiplier based on task tier (e.g., 1.0, 1.5, 2.0)
        momentum: float,             # Overall user momentum (0-1)
    ) -> None:
        """
        Boosts each relevant development index based on task tier and user momentum.
        Uses TASK_EFFECT_BASE_BOOST constant. Does nothing if
        DEVELOPMENT_INDEX feature is disabled.

        boost = TASK_EFFECT_BASE_BOOST * tier_mult * momentum
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DEVELOPMENT_INDEX):
            logger.debug("Skipping apply_task_effect: DEVELOPMENT_INDEX feature disabled.")
            return
        # --- End Check ---

        if not relevant_indexes or not isinstance(relevant_indexes, list):
            logger.debug("apply_task_effect called with no relevant indexes.")
            return

        try:
            valid_tier_mult = float(tier_mult)
            valid_momentum = _clamp(float(momentum), lo=0.0, hi=1.0) # Ensure momentum is 0-1 specifically
        except (ValueError, TypeError) as e:
            logger.error("Invalid tier_mult or momentum for apply_task_effect: %s. Aborting.", e)
            return

        boost = TASK_EFFECT_BASE_BOOST * valid_tier_mult * valid_momentum
        if boost <= 0: # No boost to apply (use <= to catch negative cases too)
            logger.debug("Calculated task effect boost is zero or negative.")
            return

        applied_keys = []
        for key in relevant_indexes:
            if key in self.indexes:
                original_value = self.indexes[key]
                new_value = _clamp(original_value + boost)
                if new_value > original_value: # Only log and update if there was an actual positive change
                    self.indexes[key] = new_value
                    applied_keys.append(key)
            else:
                logger.warning("Task effect specified relevant index '%s' which does not exist.", key)

        if applied_keys:
            logger.info(
                "Dev-indexes boosted %s by %.4f (Base: %.4f, TierMult: %.2f, Momentum: %.2f)",
                applied_keys, boost, TASK_EFFECT_BASE_BOOST, valid_tier_mult, valid_momentum,
            )

    # ---------------------------------------------------------------- #
    #  Persistence helpers (respecting feature flag)
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]: # Changed type hint slightly
        """
        Serializes the current index values to a dictionary.
        Returns an empty dictionary if DEVELOPMENT_INDEX feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DEVELOPMENT_INDEX):
            logger.debug("Skipping to_dict serialization: DEVELOPMENT_INDEX feature disabled.")
            return {} # Return empty dict as specified
        # --- End Check ---

        # Feature enabled, return state
        logger.debug("Serializing Development Index state.")
        return {"indexes": dict(self.indexes)} # Return a copy

    def update_from_dict(self, data: Dict[str, Any]) -> None: # Changed type hint slightly
        """
        Loads index values from a dictionary, ensuring keys are valid and values are clamped.
        Resets state to default if DEVELOPMENT_INDEX feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DEVELOPMENT_INDEX):
            logger.debug("Resetting state via update_from_dict: DEVELOPMENT_INDEX feature disabled.")
            self._reset_state() # Reset to defaults if feature is off
            return
        # --- End Check ---

        # Feature enabled, proceed with loading
        if not isinstance(data, dict):
            logger.error("Invalid data passed to update_from_dict: expected dict, got %s. State not updated.", type(data))
            return

        indexes_data = data.get("indexes")
        if isinstance(indexes_data, dict):
            loaded_count = 0
            # Reset state before loading valid values from dict
            # This handles cases where the saved dict might be missing keys
            current_state = {k: DEFAULT_DEVELOPMENT_INDEX_VALUE for k in DEVELOPMENT_INDEX_KEYS}

            for key, value in indexes_data.items():
                if key in DEVELOPMENT_INDEX_KEYS:
                    try:
                        current_state[key] = _clamp(float(value)) # Load and clamp into temporary dict
                        loaded_count += 1
                    except (ValueError, TypeError) as e:
                        logger.error("Invalid value for index '%s' during load: %s. Error: %s. Using default.", key, value, e)
                        # current_state[key] remains default from initialization above
                else:
                    logger.warning("Ignoring unknown index key '%s' during load from dict.", key)

            self.indexes = current_state # Assign the fully processed state
            logger.debug("FullDevelopmentIndex updated from dict. Loaded/Validated %d indexes.", loaded_count)

        elif indexes_data is not None:
            logger.warning("Indexes data in update_from_dict is not a dict (%s). State not updated.", type(indexes_data))
        else:
             # Input dict exists, but is missing the 'indexes' key. Reset to default.
             logger.warning("Input data for update_from_dict missing 'indexes' key. Resetting state.")
             self._reset_state()


    # ---------------------------------------------------------------- #
    #  Debug convenience
    # ---------------------------------------------------------------- #
    def __str__(self) -> str:  # noqa: Dunder
        """Returns a JSON string representation of the current index values."""
        # Consider adding feature status to string representation if helpful
        # status = "ENABLED" if is_enabled(Feature.DEVELOPMENT_INDEX) else "DISABLED"
        # return f"DevelopmentIndex (Status: {status})\n{json.dumps(self.indexes, indent=2, default=str)}"
        return json.dumps(self.indexes, indent=2, default=str)
