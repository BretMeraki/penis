# forest_app/core/harmonic_framework.py

import logging
# --- Import Constants ---
from forest_app.config.constants import (
    # Weights for Silent Scoring (Ensure these sum as intended, e.g., to 1.0)
    WEIGHT_SHADOW_SCORE,
    WEIGHT_CAPACITY,
    WEIGHT_MAGNITUDE,
    # Default snapshot values if keys are missing
    DEFAULT_SNAPSHOT_SHADOW,
    DEFAULT_SNAPSHOT_CAPACITY,
    DEFAULT_SNAPSHOT_MAGNITUDE,
    # Harmonic Routing Thresholds
    HARMONY_THRESHOLD_REFLECTION,
    HARMONY_THRESHOLD_RENEWAL,
    HARMONY_THRESHOLD_RESILIENCE,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SilentScoring:
    """
    Computes the 'silent' internal scores that reflect the underlying state of the system.

    Aggregates weighted scores for key metrics (shadow_score, capacity, magnitude)
    into a composite silent score. XP component has been removed.
    """

    def __init__(self):
        # Weights are now sourced from constants.py
        # Ensure these constants are defined and sum appropriately (e.g., to 1.0)
        self.weights = {
            # "xp": REMOVED
            "shadow_score": WEIGHT_SHADOW_SCORE,
            "capacity": WEIGHT_CAPACITY,
            "magnitude": WEIGHT_MAGNITUDE,
        }
        # Validate weights sum if necessary (optional runtime check)
        # total_weight = sum(self.weights.values())
        # if abs(total_weight - 1.0) > 1e-6: # Allow for floating point inaccuracy
        #     logger.warning(f"SilentScoring weights do not sum to 1.0 (Sum: {total_weight})")


    def compute_detailed_scores(self, snapshot_dict: dict) -> dict:
        """
        Computes detailed silent scores based on selected snapshot fields using constants.
        Returns a dictionary with individual weighted scores.
        XP score calculation removed.
        Uses default constants if snapshot keys are missing.
        """
        # Use constants for default values in .get()
        shadow = snapshot_dict.get("shadow_score", DEFAULT_SNAPSHOT_SHADOW)
        capacity = snapshot_dict.get("capacity", DEFAULT_SNAPSHOT_CAPACITY)
        magnitude = snapshot_dict.get("magnitude", DEFAULT_SNAPSHOT_MAGNITUDE)

        # Ensure values are numeric before multiplication (optional safety)
        try:
             shadow = float(shadow)
             capacity = float(capacity)
             magnitude = float(magnitude)
        except (ValueError, TypeError) as e:
             logger.error(f"Non-numeric value encountered in snapshot for scoring: {e}. Using defaults.")
             shadow = DEFAULT_SNAPSHOT_SHADOW
             capacity = DEFAULT_SNAPSHOT_CAPACITY
             magnitude = DEFAULT_SNAPSHOT_MAGNITUDE

        detailed = {
            # "xp_score": REMOVED
            # Note: Shadow score weight applied directly. If lower is better, consider inverting (1 - shadow) before weighting.
            # Assuming direct weighting for now as per original logic.
            "shadow_component": shadow * self.weights["shadow_score"],
            "capacity_component": capacity * self.weights["capacity"],
            # Note: Magnitude might need normalization if its scale (e.g., 1-10) differs significantly from others (0-1)
            # Assuming direct weighting for now as per original logic.
            "magnitude_component": magnitude * self.weights["magnitude"],
        }
        logger.debug("Computed detailed silent scores: %s", detailed) # Changed to debug level
        return detailed

    def compute_composite_score(self, snapshot_dict: dict) -> float:
        """
        Aggregates detailed scores into a single composite silent score.
        """
        detailed = self.compute_detailed_scores(snapshot_dict)
        # Sum the values from the detailed score dictionary
        composite = sum(detailed.values())
        # Ensure score is float
        composite = float(composite)
        logger.debug("Composite silent score computed: %.4f", composite) # Changed to debug, increased precision
        return composite


class HarmonicRouting:
    """
    Determines the harmonic theme based on the composite silent score using constants.

    This theme (e.g., "Reflection", "Renewal", "Resilience", or "Transcendence")
    informs the overall tone and complexity of the tasks and narrative.
    """

    def __init__(self):
        # Thresholds for different harmonic themes now sourced from constants.py
        # The keys remain the theme names, values are the upper bounds from constants.
        self.theme_thresholds = {
            "Reflection": HARMONY_THRESHOLD_REFLECTION,
            "Renewal": HARMONY_THRESHOLD_RENEWAL,
            "Resilience": HARMONY_THRESHOLD_RESILIENCE,
            # "Transcendence" is implicitly anything above the Resilience threshold
        }
        # Store sorted thresholds for efficient lookup (lowest to highest)
        self._sorted_thresholds = sorted(self.theme_thresholds.items(), key=lambda item: item[1])


    def route_harmony(self, snapshot_dict: dict, detailed_scores: dict = None) -> dict:
        """
        Determines the harmonic theme based on detailed silent scores using constant thresholds.

        If the composite score is low, choose a theme like "Reflection"; as the score increases,
        themes transition through "Renewal", "Resilience", up to "Transcendence."
        Returns a dictionary with keys 'theme' and 'routing_score'.
        """
        # Calculate composite score if detailed scores are provided
        if detailed_scores and isinstance(detailed_scores, dict):
            composite = sum(detailed_scores.values())
            composite = float(composite) # Ensure float
        else:
            # If detailed scores aren't provided, we might need SilentScoring instance
            # or recalculate. For now, assuming detailed_scores should be passed in
            # or handle recalculation if needed. Let's default to 0 if not provided.
            # Alternative: scorer = SilentScoring(); composite = scorer.compute_composite_score(snapshot_dict)
            logger.warning("Detailed scores not provided to route_harmony. Using composite score of 0.0.")
            composite = 0.0


        # Determine theme by comparing score against sorted thresholds
        theme = "Transcendence" # Default for scores above the highest threshold
        for theme_name, threshold_value in self._sorted_thresholds:
            if composite < threshold_value:
                theme = theme_name
                break # Found the correct theme bracket

        routing_info = {"theme": theme, "routing_score": composite}
        logger.info("Harmonic routing determined: %s", routing_info) # Keep as info level
        return routing_info
