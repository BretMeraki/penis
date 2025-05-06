# forest_app/modules/practical_consequence.py

import logging
from datetime import datetime, timezone # Use timezone aware datetime
from typing import Dict, Any, Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("prac_consequence_init")
    logger.warning("Feature flags module not found in practical_consequence. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        # Assuming this flag name, confirm or correct as needed
        PRACTICAL_CONSEQUENCE = "FEATURE_ENABLE_PRACTICAL_CONSEQUENCE"
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Or rely on global config

# --- Default values ---
DEFAULT_CALIBRATION = {
    "base_weight": 1.0,
    "time_weight": 0.25,
    "energy_weight": 0.25,
    "money_weight": 0.20,
    "relational_weight": 0.15,
    "safety_weight": 0.15,
}
DEFAULT_SCORE = 0.5
MIN_SCORE = 0.0 # Define min/max explicitly
MAX_SCORE = 1.0
DEFAULT_LEVEL = "Neutral Impact" # Default level when disabled
DEFAULT_MULTIPLIER = 1.0
DEFAULT_TONE = {"empathy": 1.0, "encouragement": 1.0}


class PracticalConsequenceEngine:
    """
    Computes a practical consequence score reflecting real-world pressures.
    Respects the PRACTICAL_CONSEQUENCE feature flag.
    Incorporates the effect of missed deadlines (only when current path is 'structured').
    """

    def __init__(self, calibration: Optional[Dict[str, float]] = None): # Use Optional
        """Initializes the engine with calibration and default state."""
        self._reset_state(initial_calibration=calibration) # Initialize using helper
        logger.info("PracticalConsequenceEngine initialized.")

    def _reset_state(self, initial_calibration: Optional[Dict[str, float]] = None):
        """Resets calibration, score, and timestamp to defaults."""
        # Deep copy default calibration to avoid modifying the original
        # from copy import deepcopy
        # self.calibration = deepcopy(DEFAULT_CALIBRATION)
        self.calibration = DEFAULT_CALIBRATION.copy() # Simple copy likely sufficient
        if isinstance(initial_calibration, dict):
            self.calibration.update(initial_calibration) # Apply overrides if provided at init

        self.score = DEFAULT_SCORE
        self.last_update: Optional[str] = None # Initialize as None
        logger.debug("PracticalConsequenceEngine state reset.")


    def update_signals_from_reflection(self, reflection: str):
        """
        Updates score based on reflection keywords. Does nothing if feature disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Skipping update_signals_from_reflection: PRACTICAL_CONSEQUENCE feature disabled.")
            return
        # --- End Check ---

        if not isinstance(reflection, str) or not reflection.strip():
             logger.debug("Skipping update_signals_from_reflection: Empty or invalid input.")
             return

        reflection_lower = reflection.lower()
        adjustments = {"time": 0.0, "energy": 0.0, "money": 0.0, "relational": 0.0, "safety": 0.0}
        old_score = self.score

        # Example heuristics (Consider making keywords/weights configurable)
        if "rush" in reflection_lower or "deadline" in reflection_lower: adjustments["time"] += 0.1
        if "delay" in reflection_lower or "waiting" in reflection_lower: adjustments["time"] -= 0.05
        if "tired" in reflection_lower or "exhausted" in reflection_lower: adjustments["energy"] += 0.1
        if "energized" in reflection_lower or "motivated" in reflection_lower: adjustments["energy"] -= 0.05
        if "money" in reflection_lower or "debt" in reflection_lower: adjustments["money"] += 0.1
        if "affluent" in reflection_lower or "wealth" in reflection_lower: adjustments["money"] -= 0.05
        if ("lonely" in reflection_lower or "isolated" in reflection_lower or "argument" in reflection_lower): adjustments["relational"] += 0.1
        if "supported" in reflection_lower or "connected" in reflection_lower: adjustments["relational"] -= 0.05
        if "unsafe" in reflection_lower or "fear" in reflection_lower: adjustments["safety"] += 0.1
        if "secure" in reflection_lower or "protected" in reflection_lower: adjustments["safety"] -= 0.05

        # Safely get weights from calibration config
        time_w = self.calibration.get("time_weight", 0.0)
        energy_w = self.calibration.get("energy_weight", 0.0)
        money_w = self.calibration.get("money_weight", 0.0)
        relational_w = self.calibration.get("relational_weight", 0.0)
        safety_w = self.calibration.get("safety_weight", 0.0)
        base_w = self.calibration.get("base_weight", 1.0)

        total_adjustment = (
            time_w * adjustments["time"] + energy_w * adjustments["energy"] +
            money_w * adjustments["money"] + relational_w * adjustments["relational"] +
            safety_w * adjustments["safety"]
        )

        # Update consequence score, clamped between MIN_SCORE and MAX_SCORE.
        self.score = max(MIN_SCORE, min(MAX_SCORE, self.score + base_w * total_adjustment))

        if self.score != old_score: # Log only if changed
            logger.info(
                "Practical consequence score updated via reflection: %.3f (adjustments: %s)",
                self.score, adjustments,
            )
            self.last_update = datetime.now(timezone.utc).isoformat()
        else:
             logger.debug("Reflection analysis resulted in no change to consequence score.")


    def update_with_deadline_penalties(self, snapshot: dict):
        """
        Increases score for missed deadlines (if path='structured').
        Does nothing if feature disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Skipping update_with_deadline_penalties: PRACTICAL_CONSEQUENCE feature disabled.")
            return
        # --- End Check ---

        # Proceed only if feature enabled
        if isinstance(snapshot, dict) and snapshot.get("current_path") == "structured":
            # This function might be called externally when a deadline is missed.
            # The logic here assumes it *is* called upon a missed deadline.
            penalty = 0.05 # Example penalty - make configurable?
            old_score = self.score
            self.score = min(MAX_SCORE, self.score + penalty) # Ensure clamping

            if self.score != old_score:
                 logger.info("Practical consequence score increased due to missed deadline. New score: %.3f", self.score)
                 self.last_update = datetime.now(timezone.utc).isoformat()
            else:
                 logger.debug("Missed deadline penalty resulted in no change (already at max).")
        # else: (No need for else, just don't apply penalty if path != structured)
             # logger.debug("Deadline penalty skipped: path is not 'structured'.")


    def compute_consequence(self) -> float:
        """Returns current score, or default if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Returning default consequence score: PRACTICAL_CONSEQUENCE feature disabled.")
            return DEFAULT_SCORE
        # --- End Check ---
        # Rounding might be better handled by the consumer, but kept here for now
        return round(self.score, 3) # Increased precision


    def get_consequence_level(self) -> str:
        """Returns consequence level string, or default if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Returning default consequence level: PRACTICAL_CONSEQUENCE feature disabled.")
            return DEFAULT_LEVEL
        # --- End Check ---

        # Use current score if feature enabled
        # Consider making thresholds configurable
        if self.score >= 0.8: return "High Impact"
        elif self.score >= 0.6: return "Moderate Impact"
        elif self.score >= 0.4: return "Low Impact"
        else: return "Minimal Impact"


    def get_task_difficulty_multiplier(self) -> float:
        """Returns difficulty multiplier, or default if feature disabled."""
         # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Returning default difficulty multiplier: PRACTICAL_CONSEQUENCE feature disabled.")
            return DEFAULT_MULTIPLIER
        # --- End Check ---

        # Calculate based on current score if feature enabled
        # Higher consequence -> lower score -> higher multiplier (more difficult)
        multiplier = 1.0 + (MAX_SCORE - self.score) * 0.5 # Example scaling
        return round(max(1.0, multiplier), 2) # Ensure multiplier is at least 1.0


    def get_tone_modifier(self) -> dict:
        """Returns tone modifier dict, or default if feature disabled."""
         # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Returning default tone modifier: PRACTICAL_CONSEQUENCE feature disabled.")
            return DEFAULT_TONE.copy() # Return copy
        # --- End Check ---

        # Determine tone based on score if feature enabled
        # Consider making thresholds/values configurable
        if self.score >= 0.8: tone = {"empathy": 1.2, "encouragement": 0.8}
        elif self.score >= 0.6: tone = {"empathy": 1.1, "encouragement": 0.9}
        elif self.score >= 0.4: tone = {"empathy": 1.0, "encouragement": 1.0}
        else: tone = {"empathy": 0.9, "encouragement": 1.1}
        return tone # Return reference (dict is mutable)


    def to_dict(self) -> dict:
        """Serializes state. Returns {} if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Skipping PracticalConsequenceEngine serialization: PRACTICAL_CONSEQUENCE feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing PracticalConsequenceEngine state.")
        return {
            "calibration": self.calibration.copy(), # Return copy
            "score": self.score,
            "last_update": self.last_update,
        }


    def update_from_dict(self, data: dict):
        """Updates state. Resets if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.PRACTICAL_CONSEQUENCE):
            logger.debug("Resetting state via update_from_dict: PRACTICAL_CONSEQUENCE feature disabled.")
            self._reset_state()
            return
        # --- End Check ---

        # Feature enabled, proceed
        if not isinstance(data, dict):
             logger.warning("Invalid data type passed to update_from_dict: %s. Resetting state.", type(data))
             self._reset_state()
             return

        # Load calibration if present and valid
        loaded_cal = data.get("calibration")
        if isinstance(loaded_cal, dict):
             # Could add validation for keys/values here
             self.calibration.update(loaded_cal)
        elif loaded_cal is not None:
             logger.warning("Invalid 'calibration' type in data: %s. Calibration not updated.", type(loaded_cal))

        # Load score if present and valid, clamp it
        loaded_score = data.get("score", DEFAULT_SCORE)
        try:
             self.score = max(MIN_SCORE, min(MAX_SCORE, float(loaded_score)))
        except (ValueError, TypeError):
             logger.warning("Invalid 'score' value in data: %s. Using default.", loaded_score)
             self.score = DEFAULT_SCORE

        # Load last_update timestamp if present and valid
        loaded_ts = data.get("last_update")
        if isinstance(loaded_ts, str):
             # Could add ISO format validation here
             self.last_update = loaded_ts
        elif loaded_ts is not None:
             logger.warning("Invalid 'last_update' type in data: %s. Timestamp not updated.", type(loaded_ts))
             self.last_update = None # Reset if invalid
        else:
             self.last_update = None # Reset if missing

        logger.debug("PracticalConsequenceEngine state updated from dict.")
