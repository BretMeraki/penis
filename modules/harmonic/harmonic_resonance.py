# forest_app/modules/harmonic_resonance.py

import logging
from datetime import datetime, timezone, timedelta # Added timedelta for potential use
from typing import Optional, Dict, Any # Added Optional, Dict, Any

# --- Import Feature Flags ---
try:
    # Assumes feature_flags.py is accessible
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if feature flags module isn't found
    logger = logging.getLogger("hr_engine_init") # Specific logger for init issues
    logger.warning("Feature flags module not found in harmonic_resonance. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        HARMONIC_RESONANCE = "FEATURE_ENABLE_HARMONIC_RESONANCE" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, based on desired fallback behavior

# --- Import Constants ---
try:
    from forest_app.config.constants import (
        RESONANCE_WEIGHT_CAPACITY,
        RESONANCE_WEIGHT_SHADOW,
        RESONANCE_WEIGHT_MAGNITUDE,
        DEFAULT_SNAPSHOT_CAPACITY,
        DEFAULT_SNAPSHOT_SHADOW,
        DEFAULT_SNAPSHOT_MAGNITUDE,
        MAGNITUDE_MIN_VALUE,
        MAGNITUDE_MAX_VALUE,
        MAX_SHADOW_SCORE,
        MIN_RESONANCE_SCORE,
        MAX_RESONANCE_SCORE,
        RESONANCE_THRESHOLD_RENEWAL,
        RESONANCE_THRESHOLD_RESILIENCE,
        RESONANCE_THRESHOLD_REFLECTION,
        DEFAULT_SCORE_PRECISION,
    )
except ImportError:
     # Fallback constants if import fails (adjust values as needed)
     logging.getLogger("hr_engine_init").critical("Failed to import constants for HarmonicResonanceEngine. Using fallback defaults.")
     RESONANCE_WEIGHT_CAPACITY = 0.4
     RESONANCE_WEIGHT_SHADOW = 0.4
     RESONANCE_WEIGHT_MAGNITUDE = 0.2
     DEFAULT_SNAPSHOT_CAPACITY = 0.5
     DEFAULT_SNAPSHOT_SHADOW = 0.5
     DEFAULT_SNAPSHOT_MAGNITUDE = 5.0
     MAGNITUDE_MIN_VALUE = 1.0
     MAGNITUDE_MAX_VALUE = 10.0
     MAX_SHADOW_SCORE = 1.0
     MIN_RESONANCE_SCORE = 0.0
     MAX_RESONANCE_SCORE = 1.0
     RESONANCE_THRESHOLD_RENEWAL = 0.8
     RESONANCE_THRESHOLD_RESILIENCE = 0.6
     RESONANCE_THRESHOLD_REFLECTION = 0.4
     DEFAULT_SCORE_PRECISION = 3


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HarmonicResonanceEngine:
    """
    Computes an internal resonance score reflecting system balance using constants.
    Respects the HARMONIC_RESONANCE feature flag.

    Utilizes core metrics (capacity, shadow score, magnitude) weighted by constants.
    The resonance score (normalized) determines a dominant theme based on constant thresholds.
    """
    # Default configuration using constants (defined safely above)
    DEFAULT_CONFIG = {
        "capacity_weight": RESONANCE_WEIGHT_CAPACITY,
        "shadow_weight": RESONANCE_WEIGHT_SHADOW,
        "magnitude_weight": RESONANCE_WEIGHT_MAGNITUDE,
    }
    # Default output when feature is disabled or calculation fails
    DEFAULT_OUTPUT = {"theme": "Neutral", "resonance_score": 0.0}


    def __init__(self, config: Optional[Dict[str, float]] = None): # Use Optional
        """Initializes with default weights sourced from constants."""
        self.config = self.DEFAULT_CONFIG.copy() # Start with defaults
        if isinstance(config, dict):
            self.config.update(config) # Override with provided config

        self.last_computed: Optional[str] = None
        logger.info("HarmonicResonanceEngine initialized.")


    def _reset_state(self):
         """Resets config to default and clears last computed time."""
         self.config = self.DEFAULT_CONFIG.copy()
         self.last_computed = None
         logger.debug("HarmonicResonanceEngine state reset.")


    def compute_resonance(self, snapshot: Dict[str, Any]) -> Dict[str, Any]: # Use Any for snapshot flexibility
        """
        Computes a composite resonance score based on key metrics from the snapshot.
        Returns default neutral output if HARMONIC_RESONANCE feature is disabled.

        Returns:
            A dictionary with keys "theme" and "resonance_score".
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.HARMONIC_RESONANCE):
            logger.debug("Skipping compute_resonance: HARMONIC_RESONANCE feature disabled. Returning default.")
            return self.DEFAULT_OUTPUT.copy()
        # --- End Check ---

        # Feature is enabled, proceed with calculation
        try:
            capacity = float(snapshot.get("capacity", DEFAULT_SNAPSHOT_CAPACITY))
            shadow = float(snapshot.get("shadow_score", DEFAULT_SNAPSHOT_SHADOW))
            magnitude = max(MAGNITUDE_MIN_VALUE, min(MAGNITUDE_MAX_VALUE,
                              float(snapshot.get("magnitude", DEFAULT_SNAPSHOT_MAGNITUDE))))
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid snapshot value type for resonance calc: {e}. Using defaults.")
            # Return default output on error during input processing
            return self.DEFAULT_OUTPUT.copy()

        magnitude_range = MAGNITUDE_MAX_VALUE - MAGNITUDE_MIN_VALUE
        if magnitude_range <= 0:
            normalized_magnitude = 0.5
            logger.warning("Magnitude range is zero or negative. Using default normalized magnitude.")
        else:
            normalized_magnitude = (MAGNITUDE_MAX_VALUE - magnitude) / magnitude_range
        normalized_magnitude = max(0.0, min(1.0, normalized_magnitude))

        # Use self.config safely, assuming keys exist from __init__
        capacity_component = self.config.get("capacity_weight", 0.0) * capacity
        shadow_component = self.config.get("shadow_weight", 0.0) * (MAX_SHADOW_SCORE - shadow)
        magnitude_component = self.config.get("magnitude_weight", 0.0) * normalized_magnitude

        resonance_score = capacity_component + shadow_component + magnitude_component
        resonance_score = round(
            max(MIN_RESONANCE_SCORE, min(MAX_RESONANCE_SCORE, resonance_score)),
            DEFAULT_SCORE_PRECISION
        )
        self.last_computed = datetime.now(timezone.utc).isoformat()

        if resonance_score >= RESONANCE_THRESHOLD_RENEWAL:
            theme = "Renewal"
        elif resonance_score >= RESONANCE_THRESHOLD_RESILIENCE:
            theme = "Resilience"
        elif resonance_score >= RESONANCE_THRESHOLD_REFLECTION:
            theme = "Reflection"
        else:
            theme = "Reset"

        logger.info("Computed resonance: score=%.*f, theme=%s",
                    DEFAULT_SCORE_PRECISION, resonance_score, theme)
        return {"theme": theme, "resonance_score": resonance_score}


    def to_dict(self) -> dict:
        """
        Serializes the engine's state. Returns empty dict if
        HARMONIC_RESONANCE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.HARMONIC_RESONANCE):
            logger.debug("Skipping HarmonicResonanceEngine serialization: HARMONIC_RESONANCE feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing HarmonicResonanceEngine state.")
        return {"config": self.config.copy(), "last_computed": self.last_computed}


    def update_from_dict(self, data: dict):
        """
        Updates the engine's state. Resets state if HARMONIC_RESONANCE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.HARMONIC_RESONANCE):
            logger.debug("Resetting state via update_from_dict: HARMONIC_RESONANCE feature disabled.")
            self._reset_state()
            return
        # --- End Check ---

        # Feature is enabled, proceed with loading
        if isinstance(data, dict):
            # Update config if present and valid
            loaded_config = data.get("config")
            if isinstance(loaded_config, dict):
                 # You might want to validate keys/values here if needed
                 self.config.update(loaded_config)
            elif loaded_config is not None:
                 logger.warning("Invalid 'config' type in data: %s. Config not updated.", type(loaded_config))

            # Update last_computed timestamp if present and valid
            loaded_ts = data.get("last_computed")
            if isinstance(loaded_ts, str):
                 # Basic validation - check if it looks like an ISO timestamp?
                 # More robust parsing could be added if strict format is required.
                 self.last_computed = loaded_ts
            elif loaded_ts is not None:
                 logger.warning("Invalid 'last_computed' type in data: %s.", type(loaded_ts))
                 self.last_computed = None # Reset if invalid type
            else:
                 self.last_computed = None # Reset if key is missing

            logger.debug("HarmonicResonanceEngine state updated from dict.")
        else:
            logger.warning("Invalid data type passed to HarmonicResonanceEngine.update_from_dict: %s. Resetting state.", type(data))
            self._reset_state() # Reset if overall data structure is wrong
