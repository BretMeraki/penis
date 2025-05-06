# forest_app/core/feature_flags.py
import logging
from enum import Enum
from functools import lru_cache # Assuming you want caching
from typing import Any # Added typing

# Assuming your settings are accessible like this
# Adjust the import if your settings structure is different
try:
    # Ensure this import path matches your project structure
    from forest_app.config.settings import settings
    settings_available = True
except ImportError:
    # Basic config if logger not set yet and settings unavailable
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("feature_flags_init").error(
        "Failed to import settings from forest_app.config.settings. "
        "Feature flags will default to False."
    )
    settings_available = False
    settings = object() # Dummy object to prevent AttributeError later

logger = logging.getLogger(__name__)
# Set level via central config ideally

class Feature(Enum):
    """
    Enumeration of controllable features.
    NOTE: Status comments reflect integration status during our conversation.
          Actual runtime status depends on the 'settings' configuration.
    """
    # --- Features with Flag Integration Implemented ---
    DEVELOPMENT_INDEX = "FEATURE_ENABLE_DEVELOPMENT_INDEX"     # DONE
    ARCHETYPES = "FEATURE_ENABLE_ARCHETYPES"                   # DONE
    SENTIMENT_ANALYSIS = "FEATURE_ENABLE_SENTIMENT_ANALYSIS"   # DONE
    NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES"         # DONE
    PATTERN_ID = "FEATURE_ENABLE_PATTERN_ID"                   # DONE
    SHADOW_ANALYSIS = "FEATURE_ENABLE_SHADOW_ANALYSIS"         # DONE
    TRAIL_MANAGER = "FEATURE_ENABLE_TRAIL_MANAGER"             # DONE
    SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES"           # DONE
    TRIGGER_PHRASES = "FEATURE_ENABLE_TRIGGER_PHRASES"         # DONE
    HARMONIC_RESONANCE = "FEATURE_ENABLE_HARMONIC_RESONANCE"   # DONE
    RESISTANCE_ENGINE = "FEATURE_ENABLE_RESISTANCE_ENGINE"     # DONE
    EMOTIONAL_INTEGRITY = "FEATURE_ENABLE_EMOTIONAL_INTEGRITY" # DONE
    RELATIONAL = "FEATURE_ENABLE_RELATIONAL"                   # DONE
    REWARDS = "FEATURE_ENABLE_REWARDS"                         # DONE (RewardIndex, OfferingRouter)
    FINANCIAL_READINESS = "FEATURE_ENABLE_FINANCIAL_READINESS" # DONE
    PRACTICAL_CONSEQUENCE = "FEATURE_ENABLE_PRACTICAL_CONSEQUENCE" # DONE (Confirm Name/Setting)
    DESIRE_ENGINE = "FEATURE_ENABLE_DESIRE_ENGINE"             # DONE (Handled earlier)
    TASK_RESOURCE_FILTER = "FEATURE_ENABLE_TASK_RESOURCE_FILTER" # <<< ADDED for TaskEngine resource check
    ENABLE_POETIC_ARBITER_VOICE = "FEATURE_ENABLE_POETIC_ARBITER_VOICE" # Arbiter voice control
    ENABLE_HTA_VISUALIZATION = "FEATURE_ENABLE_HTA_VISUALIZATION"    # <<< ADDED for Graphviz display
    ENABLE_WITHERING = "FEATURE_ENABLE_WITHERING"              # <<< ADDED for Orchestrator logic

    # --- Features Deferred (Integration NOT Implemented Yet) ---
    MEMORY_SYSTEM = "FEATURE_ENABLE_MEMORY_SYSTEM"             # DEFERRED (User Hesitation)
    XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"                   # DEFERRED (Keep ON for MVP)

    # --- Features Unclear/Skipped (Integration NOT Implemented) ---
    # HARMONIC_FRAMEWORK = "FEATURE_ENABLE_HARMONIC_FRAMEWORK"   # SKIPPED (Resonance handled instead)
    # SEED_MANAGER = "FEATURE_ENABLE_SEED_MANAGER"             # SKIPPED (Decided Always ON)  <-- Correctly commented out/omitted

    # --- Features Explicitly Skipped (Flags NOT Used in Code Checks) ---
    # LOGGING_TRACKING = "FEATURE_ENABLE_LOGGING_TRACKING"     # SKIPPED (Use logging levels)
    # SNAPSHOT_FLOW = "FEATURE_ENABLE_SNAPSHOT_FLOW"           # SKIPPED (Process always runs)
    # METRICS_SPECIFIC = "FEATURE_ENABLE_METRICS_SPECIFIC"     # SKIPPED (Considered Core)

    # --- Core Functionality (Flags optional, assumed ON) ---
    CORE_ONBOARDING = "FEATURE_ENABLE_CORE_ONBOARDING"         # ASSUMED ON (No checks added)
    CORE_HTA = "FEATURE_ENABLE_CORE_HTA"                       # ASSUMED ON (No checks added)
    CORE_TASK_ENGINE = "FEATURE_ENABLE_CORE_TASK_ENGINE"     # ASSUMED ON (No checks added)


# Cache the result of checking each flag
@lru_cache(maxsize=None)
def is_enabled(feature: Feature) -> bool:
    """
    Checks if a specific feature is enabled based on application settings.
    Defaults to False if settings are unavailable or the flag is missing/invalid.
    """
    if not settings_available:
        return False # Cannot check flags if settings didn't load

    if not isinstance(feature, Feature):
        try:
            feature_repr = repr(feature)
        except Exception:
            feature_repr = str(type(feature))
        logger.error("Invalid argument type provided to is_enabled: %s", feature_repr)
        return False

    flag_name = feature.value # e.g., "FEATURE_ENABLE_DEVELOPMENT_INDEX"
    enabled = getattr(settings, flag_name, False) # Default to False if not found

    # Ensure the value retrieved is actually a boolean
    if not isinstance(enabled, bool):
        # --- Logic for handling non-bool strings ---
        is_truthy_string = isinstance(enabled, str) and enabled.lower() in ['true', '1', 'yes', 'on']
        if is_truthy_string:
            enabled = True
        else:
            if enabled not in [False, None, 0, 'false', '0', 'no', 'off', '']:
                logger.warning(
                    "Setting '%s' for feature '%s' is not a standard boolean "
                    "(value: %s, type: %s). Interpreting as False.",
                    flag_name, feature.name, enabled, type(enabled).__name__
                )
            enabled = False

    # logger.debug("Feature '%s' (%s) checked: %s", feature.name, flag_name, enabled) # Optional: uncomment for verbose logging
    return enabled
