# forest_app/modules/narrative_modes.py

import logging
from typing import Optional, Dict, Any

# --- Import Feature Flags ---
try:
    # Assumes feature_flags.py is accessible
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if feature flags module isn't found
    logger = logging.getLogger("narrative_modes_init") # Specific logger for init issues
    logger.warning("Feature flags module not found in narrative_modes. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, based on desired fallback behavior


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Default Configuration ---
# Having defaults defined outside __init__ can be cleaner
DEFAULT_NARRATIVE_CONFIG = {
    "modes": {
        "default": {
            "description": "Standard poetic, attuned System-Veil language.",
            "style_directive": "Maintain the standard poetic, attuned, System-Veil language.",
            "tone_override": None,
        },
        "instructional": {
            "description": "Clear, step-by-step guidance.",
            "style_directive": "Provide clear, numbered or step-by-step instructions where applicable for the task, while maintaining a supportive tone.",
            "tone_override": "clear",
        },
        "symbolic_open": {
            "description": "Emphasis on metaphor and open-ended questions.",
            "style_directive": "Emphasize metaphor, symbolism, and open-ended questions. Use dreamlike, evocative language suitable for open exploration.",
            "tone_override": "whimsical",
        },
        "gentle_safety": {
            "description": "Exceptionally gentle, reassuring, simple language.",
            "style_directive": "Use exceptionally gentle, reassuring, and simple language. Prioritize safety, validation, and non-demand. Avoid complex metaphors or instructions.",
            "tone_override": "gentle",
        },
        "direct_support": {
            "description": "Direct and clear about support or action needed.",
            "style_directive": "Be direct and clear about the support available or the action needed, while remaining compassionate and avoiding pressure. Focus on clarity and reassurance.",
            "tone_override": "supportive",
        },
        "celebratory": {
            "description": "Joyful, acknowledging progress or positive shifts.",
            "style_directive": "Adopt a joyful, celebratory tone, acknowledging positive shifts or milestones achieved. Use uplifting language.",
            "tone_override": "joyful",
        },
    },
    "triggers": {
        "high_abuse": "gentle_safety",
        "urgent_repair_required": "direct_support",
        "low_capacity_high_shadow": "gentle_safety",
        "open_path": "symbolic_open",
        "task_requires_steps": "instructional",
        "high_consequence": "direct_support",
        "major_milestone_reached": "celebratory",
    },
    "thresholds": {
        "low_capacity": 0.2,
        "high_shadow": 0.8,
        "high_consequence": 0.8,
    },
}

# Define the default output when the feature is disabled
DEFAULT_MODE_OUTPUT = {
    "mode": "default",
    "style_directive": DEFAULT_NARRATIVE_CONFIG["modes"]["default"]["style_directive"],
    "tone_override": DEFAULT_NARRATIVE_CONFIG["modes"]["default"]["tone_override"],
}


class NarrativeModesEngine:
    """
    Determines the appropriate narrative mode and style directives for the Arbiter LLM
    based on the current system state and context.
    Respects the NARRATIVE_MODES feature flag.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the engine with configuration.
        """
        # Deep copy default config to avoid modification issues if needed later
        # For simple dicts like this, copy() is usually fine.
        # from copy import deepcopy
        # self.config = deepcopy(DEFAULT_NARRATIVE_CONFIG)
        self.config = DEFAULT_NARRATIVE_CONFIG.copy()
        if isinstance(config, dict):
            # Consider a deep merge function if configs become nested complex objects
            self.config.update(config)
        logger.info("NarrativeModesEngine initialized.")

    def _reset_config(self):
        """Resets configuration to the default."""
        # from copy import deepcopy # if needed
        self.config = DEFAULT_NARRATIVE_CONFIG.copy()
        logger.debug("NarrativeModesEngine configuration reset to default.")

    def determine_narrative_mode(
        self, snapshot_dict: Dict[str, Any], context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the snapshot and context to determine the appropriate narrative mode.
        Returns the 'default' mode if NARRATIVE_MODES feature is disabled.

        Args:
            snapshot_dict: The dictionary representation of the MemorySnapshot.
            context: Optional dictionary containing additional context signals.

        Returns:
            A dictionary containing the selected mode details ('mode', 'style_directive', 'tone_override').
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.NARRATIVE_MODES):
            logger.debug("Skipping narrative mode determination: NARRATIVE_MODES feature disabled. Returning default mode output.")
            return DEFAULT_MODE_OUTPUT.copy() # Return a copy of the default
        # --- End Check ---

        # Feature enabled, proceed with determination logic
        context = context or {}
        threshold_signals = context.get("threshold_signals", {})
        base_task = context.get("base_task", {})

        selected_mode_name = "default"
        trigger_config = self.config.get("triggers", {})
        threshold_values = self.config.get("thresholds", {})
        modes_config = self.config.get("modes", {}) # Get modes config once

        logger.debug("Determining narrative mode. Input Thresholds: %s", threshold_signals)

        # --- Trigger Logic (Order matters) ---
        if threshold_signals.get("high_abuse"):
            selected_mode_name = trigger_config.get("high_abuse", "gentle_safety")
        elif threshold_signals.get("urgent_repair_required"):
            selected_mode_name = trigger_config.get("urgent_repair_required", "direct_support")
        elif snapshot_dict.get("capacity", 0.5) < threshold_values.get("low_capacity", 0.2) and \
             snapshot_dict.get("shadow_score", 0.5) > threshold_values.get("high_shadow", 0.8):
            selected_mode_name = trigger_config.get("low_capacity_high_shadow", "gentle_safety")
        elif snapshot_dict.get("current_path") == "open":
            selected_mode_name = trigger_config.get("open_path", "symbolic_open")
        elif base_task.get("needs_instructional_mode", False):
            selected_mode_name = trigger_config.get("task_requires_steps", "instructional")
        # Add other trigger checks here in priority order...
        # elif context.get("milestone_reached", False):
        #     selected_mode_name = trigger_config.get("major_milestone_reached", "celebratory")

        # --- Get final mode details ---
        # Use selected mode, fall back to default mode from config, then fall back to hardcoded default
        mode_details = modes_config.get(
            selected_mode_name,
            modes_config.get("default", DEFAULT_MODE_OUTPUT) # Use default from config if selected not found
        )

        # Ensure all keys are present in the final output
        final_mode_output = {
            "mode": selected_mode_name,
            "style_directive": mode_details.get("style_directive", DEFAULT_MODE_OUTPUT["style_directive"]),
            "tone_override": mode_details.get("tone_override", DEFAULT_MODE_OUTPUT["tone_override"]),
        }

        if selected_mode_name != "default":
             logger.info("Determined narrative mode: %s -> Details: %s", selected_mode_name, final_mode_output)
        else:
             logger.debug("Determined narrative mode: default -> Details: %s", final_mode_output)

        return final_mode_output


    def to_dict(self) -> dict:
        """
        Serializes the engine's configuration. Returns empty dict if
        NARRATIVE_MODES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.NARRATIVE_MODES):
            logger.debug("Skipping NarrativeModesEngine serialization: NARRATIVE_MODES feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing NarrativeModesEngine config.")
        # Return a copy to prevent modification
        return {"config": self.config.copy()}


    def update_from_dict(self, data: dict):
        """
        Updates the engine's configuration. Resets config if
        NARRATIVE_MODES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.NARRATIVE_MODES):
            logger.debug("Resetting config via update_from_dict: NARRATIVE_MODES feature disabled.")
            self._reset_config()
            return
        # --- End Check ---

        # Feature enabled, proceed with loading
        if isinstance(data, dict) and "config" in data:
            config_update = data.get("config")
            if isinstance(config_update, dict):
                # Perform a simple update. For deep merging, consider importing a utility.
                self.config.update(config_update)
                logger.debug("NarrativeModesEngine config updated from dict.")
            else:
                logger.warning("Invalid format for config update in NarrativeModesEngine: Expected dict, got %s.", type(config_update))
                # Optionally reset to default if format is wrong
                # self._reset_config()
        elif isinstance(data, dict):
             logger.debug("No 'config' key found in data for NarrativeModesEngine update.")
             # Keep existing config if key is missing
        else:
             logger.warning("Invalid data type passed to NarrativeModesEngine.update_from_dict: Expected dict, got %s. Resetting config.", type(data))
             self._reset_config() # Reset if overall data is wrong type
