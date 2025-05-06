# forest_app/modules/reward_index.py

import logging
from typing import Any, Dict # Added Any, Dict

# --- Import Feature Flags ---
try:
    # Assumes feature_flags.py is accessible
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if feature flags module isn't found
    logger = logging.getLogger(__name__) # Ensure logger defined for warning
    logger.warning("Feature flags module not found in reward_index. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        REWARDS = "FEATURE_ENABLE_REWARDS" # Define the specific flag used here
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, based on desired fallback behavior

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Helper for clamping ---
def _clamp01(value: Any, default: float = 0.5) -> float:
    """Clamps a value to 0.0-1.0, returning default if input is invalid."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (ValueError, TypeError):
        logger.warning("Invalid value '%s' for clamping, using default %.2f", value, default, exc_info=True)
        return default

class RewardIndex:
    """
    Tracks the reward-related state, which influences offering generation.
    Respects the REWARDS feature flag for state persistence.

    Attributes:
        - readiness: A float (0.0 to 1.0) representing the current readiness for a reward.
        - generosity: A float (0.0 to 1.0) indicating the evolving generosity level.
        - desire_signal: A float (0.0 to 1.0) capturing the user's expressed desire for reward/intervention.
    """
    # Define default values as class constants for clarity
    DEFAULT_READINESS = 0.5
    DEFAULT_GENEROSITY = 0.5
    DEFAULT_DESIRE = 0.5

    def __init__(self):
        self._reset_state() # Initialize using reset method
        logger.debug("RewardIndex initialized.")

    def _reset_state(self):
        """Resets the state attributes to their default values."""
        self.readiness = self.DEFAULT_READINESS
        self.generosity = self.DEFAULT_GENEROSITY
        self.desire_signal = self.DEFAULT_DESIRE
        logger.debug("RewardIndex state reset to defaults.")

    def to_dict(self) -> Dict[str, float]:
        """
        Serializes state for persistence. Returns empty dict if
        REWARDS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.REWARDS):
            logger.debug("Skipping RewardIndex serialization: REWARDS feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing RewardIndex state.")
        return {
            "readiness": self.readiness,
            "generosity": self.generosity,
            "desire_signal": self.desire_signal,
        }

    def update_from_dict(self, data: Dict[str, Any]):
        """
        Loads state from snapshot data. Resets state if REWARDS feature is disabled
        or if data is invalid. Clamps loaded values between 0.0 and 1.0.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.REWARDS):
            logger.debug("Resetting state via update_from_dict: REWARDS feature disabled.")
            self._reset_state()
            return
        # --- End Check ---

        if not isinstance(data, dict):
            logger.warning("Invalid data format for update_from_dict: %s. Resetting state.", type(data))
            self._reset_state()
            return

        # Load values, using clamp helper for validation and range enforcement
        self.readiness = _clamp01(data.get("readiness"), default=self.DEFAULT_READINESS)
        self.generosity = _clamp01(data.get("generosity"), default=self.DEFAULT_GENEROSITY)
        self.desire_signal = _clamp01(data.get("desire_signal"), default=self.DEFAULT_DESIRE)
        logger.debug("RewardIndex state updated from dict.")

    # --- Public methods to potentially update state would go here ---
    # Example (If needed later, remember to add feature flag checks):
    # def update_readiness(self, change: float):
    #     if not is_enabled(Feature.REWARDS):
    #         logger.debug("Skipping update_readiness: REWARDS feature disabled.")
    #         return
    #     self.readiness = _clamp01(self.readiness + change)
    #     logger.info("Updated readiness to %.2f", self.readiness)
    # --------------------------------------------------------------
