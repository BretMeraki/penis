# forest_app/modules/resistance_engine.py

import logging
from typing import Any # Added Any for feature flag fallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Import Feature Flags ---
# Assuming feature_flags.py is accessible from this module's path
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger.warning("Feature flags module not found in resistance_engine. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        RESISTANCE_ENGINE = "FEATURE_ENABLE_RESISTANCE_ENGINE" # Define the specific flag used here
    def is_enabled(feature: Any) -> bool: # Dummy function - default to True or False
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, choose appropriate fallback

def clamp01(x: float) -> float:
    """Clamp a float to the 0.0–1.0 range."""
    return max(0.0, min(1.0, x))


class ResistanceEngine:
    """
    Computes the 'resistance' value for a task, on a scale from 0.0 (very easy)
    to 1.0 (very difficult), based on core metrics. Respects the
    RESISTANCE_ENGINE feature flag.

    Formula (§8):
        R = clamp₀₋₁(
            0.4
          + 0.5 * σ
          - 0.3 * c
          - 0.2 * μ
          + 0.05 * (M - 5)
        )
    """

    @staticmethod
    def compute(
        shadow_score: float,
        capacity: float,
        momentum: float,
        magnitude: float
    ) -> float:
        """
        Calculate resistance based on the core metrics. Returns 0.0 if
        RESISTANCE_ENGINE feature is disabled.

        Args:
            shadow_score: float in [0.0, 1.0]
            capacity:     float in [0.0, 1.0]
            momentum:     float in [0.0, 1.0]
            magnitude:    float, expected in [1.0, 10.0]

        Returns:
            resistance R, clamped to [0.0, 1.0] (or 0.0 if feature is disabled)
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RESISTANCE_ENGINE):
            logger.debug("Skipping resistance computation: RESISTANCE_ENGINE feature disabled. Returning 0.0.")
            return 0.0 # Return a default neutral/easy resistance value
        # --- End Check ---

        # Proceed with calculation if feature is enabled
        base = 0.4
        try:
            # Ensure inputs are floats before calculation
            s_score = float(shadow_score)
            cap = float(capacity)
            mom = float(momentum)
            mag = float(magnitude)

            comp = (
                base
                + 0.5 * s_score
                - 0.3 * cap
                - 0.2 * mom
                + 0.05 * (mag - 5.0)
            )
            r = clamp01(comp)
            logger.debug(
                "Computed resistance: base=%.2f +0.5*σ(%.2f) -0.3*c(%.2f) -0.2*μ(%.2f) +0.05*(M-5)(%.2f) = %.2f → R=%.2f",
                base, s_score, cap, mom, (mag - 5.0), comp, r
            )
            return r
        except (ValueError, TypeError) as e:
             logger.error("Invalid input type for resistance computation: %s. Returning default 0.0.", e,
                          exc_info=True) # Add exception info for debugging
             return 0.0 # Return default on calculation error
