# forest_app/modules/shadow.py
"""
Module for analyzing text input to detect and quantify shadow-related cues.
Respects the SHADOW_ANALYSIS feature flag.
"""

import logging
import re
import math
from datetime import datetime, timezone # Use timezone-aware UTC
from typing import Dict, Any, Optional # Added Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("shadow_init")
    logger.warning("Feature flags module not found in shadow.py. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        SHADOW_ANALYSIS = "FEATURE_ENABLE_SHADOW_ANALYSIS" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Default output when feature is disabled ---
DEFAULT_SHADOW_OUTPUT = {"shadow_score": 0.0, "shadow_tags": {}}

class ShadowEngine:
    """
    ShadowEngine analyzes text input to detect and quantify shadow-related cues.
    Respects the SHADOW_ANALYSIS feature flag.

    Refinements:
      1. Contextual Weighting: Optionally adjusts lexicon weights based on context.
      2. Pattern Matching: Uses regex to capture common shadow-expressive phrases.
      3. Synergy with Sentiment: Can optionally incorporate a sentiment factor from context.
      4. Refined Normalization: Optionally normalizes by text length rather than a fixed divisor.
    """

    def __init__(self):
        """Initializes the engine with default lexicons and state."""
        self._initialize_defaults() # Use helper to set initial state

    def _initialize_defaults(self):
        """Sets or resets the engine's default state."""
        # Base lexicon for single keywords (default weight values)
        self.lexicon = {
            "bitterness": 0.8, "avoid": 0.7, "burnout": 0.9, "rigid": 0.6,
            "shame": 0.7, "resent": 0.8, "self-hate": 0.9, "fearful": 0.6,
            "hopeless": 0.8, "despair": 0.9, "guilt": 0.7,
        }
        self.negations = {"not", "never", "no"}
        # Additional regex patterns for common shadow phrases
        self.pattern_lexicon = {
            r"\bi can'?t seem to\b": 0.3,
            r"\bstuck (in|on)\b": 0.3,
            r"\bwhat'?s the point\b": 0.4,
        }
        # Initialize last_update to current UTC time
        self.last_update = datetime.now(timezone.utc).isoformat()
        logger.debug("ShadowEngine state initialized/reset to defaults.")

    def _sigmoid(self, x: float, k: float = 1.0) -> float:
        """Optional sigmoid normalization function with overflow protection."""
        try:
            # Check for large negative exponent to prevent overflow in exp(-k*x)
            if -k * x > 700: # exp(700) is near max float value
                return 0.0
            return 1 / (1 + math.exp(-k * x))
        except OverflowError:
             logger.error("Overflow in sigmoid calculation for x=%.2f, k=%.2f", x, k)
             # Return 1.0 if -k*x is extremely large positive (approaching infinity)
             # Return 0.0 if -k*x is extremely large negative (approaching -infinity)
             # This depends on how overflow occurred, assume positive saturation
             return 1.0


    def analyze_text(self, text: str, context: Optional[Dict] = None) -> dict: # Use Optional
        """
        Analyzes input text for shadow content. Returns default neutral output
        if SHADOW_ANALYSIS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SHADOW_ANALYSIS):
            logger.debug("Skipping analyze_text: SHADOW_ANALYSIS feature disabled.")
            return DEFAULT_SHADOW_OUTPUT.copy() # Return a copy of default
        # --- End Check ---

        # Feature enabled, proceed with analysis
        if not isinstance(text, str) or not text.strip():
            logger.debug("Received empty or invalid text for shadow analysis.")
            return DEFAULT_SHADOW_OUTPUT.copy()

        context = context or {} # Ensure context is a dict, not None
        text_lower = text.lower()
        total_score = 0.0
        tag_scores = {}
        words = text_lower.split()
        if not words: # Handle case where split results in empty list
             return DEFAULT_SHADOW_OUTPUT.copy()
        skip_next = False

        # Adjust lexicon weights based on context (if provided)
        adjusted_lexicon = self.lexicon.copy()
        if context:
            try: # Add try-except for context processing
                capacity = float(context.get("capacity", 0.5))
                resonance_theme = str(context.get("resonance_theme", "")).lower()

                # Example adjustments (keep these robust)
                if capacity < 0.3:
                    for key in ["burnout", "hopeless", "despair"]:
                        if key in adjusted_lexicon: adjusted_lexicon[key] *= 1.2
                if resonance_theme == "reset":
                    if "rigid" in adjusted_lexicon: adjusted_lexicon["rigid"] *= 0.8
            except (ValueError, TypeError) as ctx_err:
                 logger.warning("Invalid context value type during lexicon adjustment: %s", ctx_err)
                 # Continue with unadjusted lexicon if context is bad

        # Process each word for lexicon-based scoring.
        for i, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
            # Handle simple negation.
            if word in self.negations:
                if i + 1 < len(words):
                    next_word = words[i + 1]
                    if next_word in adjusted_lexicon:
                        score = -adjusted_lexicon[next_word]
                        tag_scores[next_word] = tag_scores.get(next_word, 0) + score
                        total_score += score
                        skip_next = True
                continue # Continue even if negation didn't apply to a lexicon word
            # Check non-negated word
            if word in adjusted_lexicon:
                score = adjusted_lexicon[word]
                tag_scores[word] = tag_scores.get(word, 0) + score
                total_score += score

        # Use regex pattern matching to catch common phrases.
        try:
            for pattern, weight in self.pattern_lexicon.items():
                # Use re.finditer for efficiency if needed, findall is ok for few patterns
                matches = re.findall(pattern, text_lower)
                if matches:
                    # Use pattern itself as tag name, or define cleaner names
                    tag_name = f"pattern:{pattern[:20]}" # Truncate long patterns for tag name
                    increment = weight * len(matches)
                    tag_scores[tag_name] = tag_scores.get(tag_name, 0) + increment
                    total_score += increment
        except re.error as re_err:
             logger.error("Regex error during shadow pattern matching: %s", re_err)
             # Continue analysis without pattern matching if regex fails

        # Optional: Factor in overall negative sentiment from context.
        if "sentiment" in context:
            try:
                # Assume sentiment is already a float score
                sentiment = float(context["sentiment"])
                # Add contribution only if sentiment is negative (score < 0)
                if sentiment < 0:
                     # Scale contribution, e.g., multiply by 0.5
                     sentiment_contribution = abs(sentiment) * 0.5
                     total_score += sentiment_contribution
                     tag_scores["negative_sentiment_factor"] = tag_scores.get("negative_sentiment_factor", 0) + sentiment_contribution
            except (ValueError, TypeError) as sent_err:
                 logger.warning("Invalid sentiment value in context: %s", sent_err)


        # Normalization (handle division by zero)
        num_words = len(words)
        normalized_by_length = abs(total_score) / (num_words + 1) if num_words > 0 else 0.0
        # Alternative fixed scaling - consider making divisor configurable
        normalized_by_fixed = abs(total_score) / 10.0
        # Combine - choose max or average? Max amplifies signals.
        raw_normalized = max(normalized_by_length, normalized_by_fixed)
        normalized_shadow = max(0.0, min(1.0, raw_normalized)) # Clamp final result

        logger.info(
            "Shadow analysis complete. Raw score: %.2f; Normalized score: %.2f; Tags: %s",
            total_score, normalized_shadow, tag_scores,
        )
        return {"shadow_score": round(normalized_shadow, 3), "shadow_tags": tag_scores} # Increased precision


    def update_from_text(self, text: str, context: Optional[Dict] = None) -> float: # Use Optional
        """
        Updates the shadow analysis from text and returns the normalized shadow score.
        Returns 0.0 if SHADOW_ANALYSIS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SHADOW_ANALYSIS):
            logger.debug("Skipping update_from_text: SHADOW_ANALYSIS feature disabled.")
            return 0.0 # Return default score
        # --- End Check ---

        analysis = self.analyze_text(text, context=context)
        # Update timestamp only if analysis ran (which implies feature is enabled)
        self.last_update = datetime.now(timezone.utc).isoformat()
        return analysis.get("shadow_score", 0.0) # Safely get score


    def to_dict(self) -> dict:
        """
        Serializes the engine's configuration and last update timestamp.
        Returns empty dict if SHADOW_ANALYSIS feature is disabled.
        """
         # --- Feature Flag Check ---
        if not is_enabled(Feature.SHADOW_ANALYSIS):
            logger.debug("Skipping ShadowEngine serialization: SHADOW_ANALYSIS feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing ShadowEngine state.")
        # Return copies of mutable objects
        return {
            "lexicon": self.lexicon.copy(),
            "pattern_lexicon": self.pattern_lexicon.copy(), # Assuming keys (regex) are immutable strings
            "negations": list(self.negations), # Convert set to list for JSON
            "last_update": self.last_update
        }


    def update_from_dict(self, data: dict):
        """
        Updates the engine from a dictionary. Resets state if
        SHADOW_ANALYSIS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SHADOW_ANALYSIS):
            logger.debug("Resetting state via update_from_dict: SHADOW_ANALYSIS feature disabled.")
            self._initialize_defaults() # Reset to defaults
            return
        # --- End Check ---

        # Feature enabled, proceed with loading
        if not isinstance(data, dict):
             logger.warning("Invalid data type passed to ShadowEngine.update_from_dict: %s. Resetting state.", type(data))
             self._initialize_defaults() # Reset if data is invalid
             return

        # Load lexicon if present and valid
        loaded_lexicon = data.get("lexicon")
        if isinstance(loaded_lexicon, dict):
            # Potentially validate keys/values before updating
            self.lexicon.update(loaded_lexicon)
        elif loaded_lexicon is not None:
             logger.warning("Invalid 'lexicon' type in data: %s. Lexicon not updated.", type(loaded_lexicon))

        # Load pattern_lexicon if present and valid
        loaded_patterns = data.get("pattern_lexicon")
        if isinstance(loaded_patterns, dict):
             # Potential validation: check if keys are valid regex?
             self.pattern_lexicon.update(loaded_patterns)
        elif loaded_patterns is not None:
             logger.warning("Invalid 'pattern_lexicon' type in data: %s. Patterns not updated.", type(loaded_patterns))

        # Load negations if present and valid (expecting list from JSON)
        loaded_negations = data.get("negations")
        if isinstance(loaded_negations, list):
             self.negations = set(loaded_negations) # Convert back to set
        elif loaded_negations is not None:
             logger.warning("Invalid 'negations' type in data: %s. Negations not updated.", type(loaded_negations))

        # Load last_update if present and valid
        loaded_ts = data.get("last_update")
        if isinstance(loaded_ts, str):
             # Could add ISO format validation here if needed
             self.last_update = loaded_ts
        elif loaded_ts is not None:
             logger.warning("Invalid 'last_update' type in data: %s. Timestamp not updated.", type(loaded_ts))
             # Optionally reset to now() or keep existing if type is wrong
             # self.last_update = datetime.now(timezone.utc).isoformat()
        else:
            # If key is missing, keep existing self.last_update or reset?
            # Keeping existing seems reasonable unless reset is desired on partial load.
            pass

        logger.debug("ShadowEngine state updated from dict.")
