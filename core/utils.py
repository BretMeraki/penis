# forest_app/core/utils.py

# --- ADDED Logging ---
import logging
# --- END ADDED Logging ---

from forest_app.config.constants import MAGNITUDE_MIN_VALUE, MAGNITUDE_MAX_VALUE

# --- ADDED Logger instance ---
logger = logging.getLogger(__name__)
# --- END ADDED Logger instance ---


def normalize_magnitude(raw: float) -> float:
    """
    Convert a raw 1–10 magnitude into a 0–1 normalized value for scoring
    without altering the raw magnitude itself.
    """
    span = MAGNITUDE_MAX_VALUE - MAGNITUDE_MIN_VALUE
    if span <= 0:
        return 0.0
    norm = (raw - MAGNITUDE_MIN_VALUE) / span
    return max(0.0, min(1.0, norm))

# --- ADD THIS FUNCTION ---
def clamp01(value: float) -> float:
    """Clamps a value between 0.0 and 1.0."""
    try:
        # Attempt to convert to float first for robustness
        float_value = float(value)
        return max(0.0, min(1.0, float_value))
    except (ValueError, TypeError):
        # Log or handle error if value cannot be converted to float
        # For now, returning a default clamp value (e.g., 0.5)
        # depending on desired behavior for invalid input.
        logger.warning("Could not clamp non-numeric value: %s. Returning 0.5", value, exc_info=True) # Added logging
        return 0.5 # Defaulting to 0.5 on error
# --- END ADDED FUNCTION ---
