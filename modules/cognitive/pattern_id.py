# forest_app/modules/pattern_id.py
# =============================================================================
# Pattern Identification Engine - Analyzes logs and context for recurring patterns
# =============================================================================

import logging
import re
from collections import Counter, defaultdict
from typing import Optional, Dict, Any, List, Tuple, Set

# --- Feature Flag Import ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if flags cannot be imported
    def is_enabled(feature): return False
    class Feature:
        # Define flags used within this module if needed for internal logic
        PATTERN_ID = "FEATURE_ENABLE_PATTERN_ID" # Self-check

logger = logging.getLogger(__name__)

# --- Default Configuration ---
DEFAULT_CONFIG = {
    "reflection_lookback": 10,  # How many reflection log entries to consider
    "task_lookback": 20,        # How many task completion entries to consider
    "min_keyword_occurrence": 3,# Minimum times a keyword must appear in reflections
    "min_cooccurrence": 2,      # Minimum times keywords must appear together
    "min_task_cycle_occurrence": 3, # Minimum times a task type/tag cycle appears
    "high_shadow_threshold": 0.7, # Threshold for considering shadow score high
    "low_capacity_threshold": 0.3, # Threshold for considering capacity low
    "stop_words": [ # Common words to ignore during keyword analysis
        "a", "an", "the", "in", "on", "at", "to", "for", "of", "it", "is", "was",
        "am", "are", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "having", "do", "does", "did", "doing", "the", "and", "but", "if", "or",
        "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
        "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
        "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
        "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        "feel", "think", "get", "go", "make", "know", "try", "really", "want", "need",
        "like", "day", "time", "work", "going", "still", "even", "much", "bit", "today",
        "yesterday", "week", "task", "tasks", # Added common task-related words
    ]
}

class PatternIdentificationEngine:
    """
    Analyzes reflection logs, task history, and snapshot context to identify
    recurring patterns, themes, and potential triggers.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the engine.

        Args:
            config: Optional dictionary to override default configuration values.
                    Expected keys match DEFAULT_CONFIG.
        """
        # Use provided config or default, ensuring type safety
        if isinstance(config, dict):
             # If config comes from DI's config.provided, it might be a Provider object
             # We need the actual dictionary. Let's assume it's resolved by DI or handle Provider case.
             # For now, assume 'config' is the actual dictionary from settings.
             self.config = {**DEFAULT_CONFIG, **config} # Merge provided config with defaults
             logger.debug("PatternIdentificationEngine config updated from provided dict.")
        else:
             self.config = DEFAULT_CONFIG.copy()
             logger.debug("PatternIdentificationEngine config reset to default.")
        logger.info("PatternIdentificationEngine initialized.")


    def _extract_keywords(self, text: str, stop_words: Set[str]) -> List[str]:
        """Extracts potential keywords from text, removing stop words."""
        if not isinstance(text, str):
            return []
        # Simple regex tokenization, lowercase, remove punctuation, filter stop words
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if word not in stop_words and len(word) > 2] # Ignore short words

    def analyze_patterns(self, snapshot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs pattern analysis based on the provided snapshot data (as dict).

        Args:
            snapshot_data: The current snapshot data as a dictionary.

        Returns:
            A dictionary containing identified patterns and scores.
            Keys might include: 'recurring_keywords', 'keyword_pairs',
            'task_cycles', 'potential_triggers', 'overall_focus_score'.
        """
        if not is_enabled(Feature.PATTERN_ID):
             logger.debug("Pattern ID feature disabled. Skipping analysis.")
             return {"status": "disabled"} # Return minimal info if disabled

        logger.info("Starting pattern analysis.")
        results: Dict[str, Any] = {
            "recurring_keywords": [],
            "keyword_pairs": [],
            "task_cycles": [],
            "potential_triggers": [],
            "overall_focus_score": 0.0, # A score indicating clarity or intensity of focus
            "errors": [] # To track specific analysis failures
        }
        stop_words_set = set(self.config.get("stop_words", []))

        # --- 1. Analyze Reflection Logs ---
        try:
            # *** DICT ACCESS FIX ***
            reflection_log = snapshot_data.get('reflection_log', [])
            if not isinstance(reflection_log, list):
                 logger.warning("Reflection log is not a list, skipping keyword analysis.")
                 raise TypeError("Reflection log format invalid") # Raise to add to errors list

            reflection_texts = " ".join([entry.get("content", "") for entry in reflection_log if isinstance(entry, dict) and entry.get("role") == "user"])
            if reflection_texts:
                keywords = self._extract_keywords(reflection_texts, stop_words_set)
                keyword_counts = Counter(keywords)
                min_occurrence = self.config.get("min_keyword_occurrence", 3)
                recurring_keywords = {kw: count for kw, count in keyword_counts.items() if count >= min_occurrence}
                results["recurring_keywords"] = sorted(recurring_keywords.items(), key=lambda item: item[1], reverse=True)

                # Simple co-occurrence (consider pairs within the same log entry?)
                # This is a basic version, more advanced NLP could be used
                pairs = Counter()
                min_cooccurrence = self.config.get("min_cooccurrence", 2)
                if len(recurring_keywords) > 1:
                     # Consider pairs from the list of recurring keywords found across all reflections
                     recurring_set = set(recurring_keywords.keys())
                     # Check pairs within each reflection entry
                     for entry in reflection_log:
                          if isinstance(entry, dict) and entry.get("role") == "user":
                               entry_keywords = set(self._extract_keywords(entry.get("content", ""), stop_words_set))
                               entry_recurring = list(entry_keywords.intersection(recurring_set))
                               for i in range(len(entry_recurring)):
                                    for j in range(i + 1, len(entry_recurring)):
                                         pair = tuple(sorted((entry_recurring[i], entry_recurring[j])))
                                         pairs[pair] += 1
                results["keyword_pairs"] = {pair: count for pair, count in pairs.items() if count >= min_cooccurrence}

                # Update focus score based on keyword density/recurrence
                if keywords:
                     results["overall_focus_score"] += min(1.0, len(recurring_keywords) / 10.0) # Example scoring

        except Exception as e:
             logger.error(f"Error analyzing reflection log patterns: {e}", exc_info=True)
             results["errors"].append(f"Reflection Analysis Error: {type(e).__name__}")


        # --- 2. Analyze Task History (Task Cycles) ---
        try:
            # *** DICT ACCESS FIX ***
            # Assuming task_footprints store completed task info including type/tags
            task_log = snapshot_data.get('task_footprints', [])
            if not isinstance(task_log, list):
                 logger.warning("Task footprint log is not a list, skipping cycle analysis.")
                 raise TypeError("Task footprint log format invalid")

            task_types = [entry.get("task_type", entry.get("metadata", {}).get("type")) # Look for type/tag
                          for entry in task_log if isinstance(entry, dict) and entry.get("event_type") == "completed"]
            task_types = [t for t in task_types if t] # Filter out None/empty types

            if task_types:
                task_counts = Counter(task_types)
                min_cycle = self.config.get("min_task_cycle_occurrence", 3)
                results["task_cycles"] = {task: count for task, count in task_counts.items() if count >= min_cycle}
                # Update focus score based on task consistency
                if results["task_cycles"]:
                     results["overall_focus_score"] += min(1.0, len(results["task_cycles"]) / 5.0) # Example

        except Exception as e:
             logger.error(f"Error analyzing task log patterns: {e}", exc_info=True)
             results["errors"].append(f"Task Analysis Error: {type(e).__name__}")


        # --- 3. Analyze Context for Triggers ---
        try:
            # *** DICT ACCESS FIX ***
            shadow = snapshot_data.get('shadow_score', 0.5)
            capacity = snapshot_data.get('capacity', 0.5)
            # Add other relevant context checks

            if shadow >= self.config.get("high_shadow_threshold", 0.7):
                results["potential_triggers"].append("high_shadow")
            if capacity <= self.config.get("low_capacity_threshold", 0.3):
                results["potential_triggers"].append("low_capacity")
            if shadow >= self.config.get("high_shadow_threshold", 0.7) and \
               capacity <= self.config.get("low_capacity_threshold", 0.3):
                 results["potential_triggers"].append("low_capacity_high_shadow") # Specific combo trigger

            # Update focus score based on stability (low shadow, high capacity)
            results["overall_focus_score"] += (1.0 - shadow) * 0.2 # Example
            results["overall_focus_score"] += capacity * 0.2 # Example

        except Exception as e:
             logger.error(f"Error analyzing potential triggers: {e}", exc_info=True)
             results["errors"].append(f"Trigger Analysis Error: {type(e).__name__}")

        # Clamp final score
        results["overall_focus_score"] = max(0.0, min(1.0, results["overall_focus_score"]))

        if results["errors"]:
             logger.warning(f"Pattern analysis encountered errors: {results['errors']}")

        logger.info("Pattern analysis complete.")
        # logger.debug(f"Pattern Analysis Results: {results}") # Log detailed results if needed
        return results

