# forest_app/config/constants.py

"""
Centralized configuration of quantitative and qualitative parameters
used throughout Forest OS. (Version: 2025-04-25 - Reconciled with Rationales)
"""

import logging
import os
from typing import List, Dict, Tuple, Final # Use Final for true constants if desired (Python 3.8+)

# =====================================================================
# Environment & Paths
# =====================================================================
# Database connection string (reads from env var DATABASE_URL, falls back to local sqlite)
DB_CONNECTION_STRING: Final[str] = os.getenv(
    "DATABASE_URL",
    "sqlite:///./forest_os_default.db" # Example fallback name
)
# RATIONALE: Allow easy override for different environments via environment variable.

# Path to archetype definitions file (reads from env var, falls back relative to this file)
# Ensure the fallback path 'archetypes.json' is correct relative to constants.py
_DEFAULT_ARCHETYPES_PATH = os.path.join(os.path.dirname(__file__), "archetypes.json")
ARCHETYPES_FILE: Final[str] = os.getenv("ARCHETYPES_FILE", _DEFAULT_ARCHETYPES_PATH)
# RATIONALE: Centralizes path to archetype definitions, allowing override via environment variable.

# =====================================================================
# Core System Settings
# =====================================================================
# Maximum conversation history turns to keep in snapshot
MAX_CONVERSATION_HISTORY: Final[int] = 20
# RATIONALE: Limits memory usage and LLM prompt length for conversation context.

# Default precision for rounding scores/floats for display/logging
DEFAULT_SCORE_PRECISION: Final[int] = 2
# RATIONALE: Standardizes display precision for scores across the application.

# Minimum password length for user registration
MIN_PASSWORD_LENGTH: Final[int] = 8
# RATIONALE: Basic security requirement for password complexity.

# Maximum length for generated snapshot codenames
MAX_CODENAME_LENGTH: Final[int] = 60
# RATIONALE: Prevents excessively long codenames, ensures reasonable length for display/storage.

# =====================================================================
# Status Strings / Enums
# =====================================================================
# Onboarding Statuses
ONBOARDING_STATUS_NEEDS_GOAL: Final[str] = "needs_goal"
ONBOARDING_STATUS_NEEDS_CONTEXT: Final[str] = "needs_context"
ONBOARDING_STATUS_COMPLETED: Final[str] = "completed"
# RATIONALE: Defines the distinct stages of the user onboarding process.

# Seed Statuses
SEED_STATUS_ACTIVE: Final[str] = "active"
SEED_STATUS_COMPLETED: Final[str] = "completed"
SEED_STATUS_EVOLVED: Final[str] = "evolved" # Example status
# SEED_STATUS_PAUSED: Final[str] = "paused" # Example status
# RATIONALE: Defines the lifecycle states for user goals (Seeds).

# Allowed Task Statuses (Used in HTA logic, potentially)
ALLOWED_TASK_STATUSES: Final[Tuple[str, ...]] = (
    "pending", "active", "completed", "skipped", "failed", "pruned"
)
# RATIONALE: Canonical list of valid states for HTA task nodes.

# =====================================================================
# Snapshot Default Values (Used when initializing or if key missing)
# =====================================================================
DEFAULT_SNAPSHOT_SHADOW: Final[float] = 0.5
# RATIONALE: Neutral midpoint for shadow score (0-1 scale), adjusted via reflection.

# --- MODIFIED LINE BELOW ---
DEFAULT_SNAPSHOT_CAPACITY: Final[float] = 0.6 # Changed from 0.5
# --- END MODIFICATION ---
# RATIONALE: Starts users in a balanced resource state (0-1 scale). Adjusted (from 0.5)
#            to ensure default capacity meets or exceeds the cost of 'medium' tasks (assumed 0.6 based on RESOURCE_MAP/logs)
#            to prevent immediate blocking of initial HTA tasks.
DEFAULT_SNAPSHOT_MAGNITUDE: Final[float] = 5.0
# RATIONALE: Midpoint on a 1–10 scale before any real inputs or task effects.
DEFAULT_INITIAL_RELATIONSHIP_INDEX: Final[float] = 0.5 # Keeping original name for this specific one
# RATIONALE: Balanced relational health as a starting point (0-1 scale).

# =====================================================================
# Development Index Constants
# =====================================================================
DEVELOPMENT_INDEX_KEYS: Final[Tuple[str, ...]] = ( # Use Tuple for immutability
    "happiness", "career", "health", "financial", "relationship",
    "executive_functioning", "social_life", "charisma", "entrepreneurship",
    "family_planning", "generational_wealth", "adhd_risk", "odd_risk",
    "homeownership", "dream_location",
)
# RATIONALE: Growth dimensions tracked by the system (from original file).
DEFAULT_DEVELOPMENT_INDEX_VALUE: Final[float] = 0.5
# RATIONALE: Neutral starting point (0-1 scale) for all development indexes.
MIN_DEVELOPMENT_INDEX_VALUE: Final[float] = 0.0
MAX_DEVELOPMENT_INDEX_VALUE: Final[float] = 1.0
# RATIONALE: Defines the valid score range for development indexes.
BASELINE_REFLECTION_NUDGE_AMOUNT: Final[float] = 0.01
# RATIONALE: Small positive adjustment (~1%) to key indexes based on positive reflection hints.
POSITIVE_REFLECTION_HINTS: Final[Tuple[str, ...]] = (
    "grateful", "proud", "excited", "optimistic", "happy", "joyful", "achieved"
)
# RATIONALE: Keywords indicating positive sentiment used to trigger baseline index nudge.
BASELINE_NUDGE_KEYS: Final[Tuple[str, ...]] = ("happiness", "social_life", "charisma")
# RATIONALE: Specific indexes nudged by positive reflection hints.
TASK_EFFECT_BASE_BOOST: Final[float] = 0.02
# RATIONALE: Base increase (~2%) applied to relevant development indexes upon successful task completion, before tier/momentum multipliers.
# TODO: Define Tier Multipliers if used consistently by callers of apply_task_effect
# TIER_MULTIPLIER_BUD: float = 1.0
# TIER_MULTIPLIER_BLOOM: float = 1.5
# TIER_MULTIPLIER_BLOSSOM: float = 2.0

# =====================================================================
# Archetype Constants
# =====================================================================
ARCHETYPE_ACTIVATION_THRESHOLD: Final[float] = 0.8
# RATIONALE: Minimum calculated weight for an archetype to be considered active and influence the system.
ARCHETYPE_DOMINANCE_FACTOR: Final[float] = 1.5
# RATIONALE: If the top archetype's weight is this much times the next highest, its influence is used exclusively.
DEFAULT_ARCHETYPE_WEIGHT: Final[float] = 1.0
# RATIONALE: Baseline weight for archetypes before context adjustments.
ARCHETYPE_CONTEXT_FACTOR_CAPACITY: Final[float] = 0.5
# RATIONALE: Default multiplier modifying archetype weight based on user capacity (if archetype doesn't define its own).
ARCHETYPE_CONTEXT_FACTOR_SHADOW: Final[float] = 0.7
# RATIONALE: Default multiplier modifying archetype weight based on user shadow score (if archetype doesn't define its own).
LOW_CAPACITY_THRESHOLD: Final[float] = 0.4
# RATIONALE: Capacity below this threshold may trigger weight adjustments for specific archetypes (e.g., Caretaker).
HIGH_SHADOW_THRESHOLD: Final[float] = 0.7
# RATIONALE: Shadow score above this threshold may trigger weight adjustments for specific archetypes (e.g., Healer).

# =====================================================================
# Harmonic Framework Constants (Used by SilentScoring / HarmonicRouting)
# =====================================================================
# Weights for components contributing to the composite silent score (Sum = 1.0)
WEIGHT_SHADOW_SCORE: Final[float] = 0.4
WEIGHT_CAPACITY: Final[float] = 0.2
WEIGHT_MAGNITUDE: Final[float] = 0.4
# RATIONALE: Relative importance of core metrics in calculating the silent score for HarmonicRouting.

# Composite score thresholds for determining the HARMONIC theme (in HarmonicRouting)
# Theme is assigned if score is LESS THAN the threshold value
HARMONY_THRESHOLD_REFLECTION: Final[float] = 0.3
HARMONY_THRESHOLD_RENEWAL: Final[float] = 0.6
HARMONY_THRESHOLD_RESILIENCE: Final[float] = 0.8
# RATIONALE: Boundaries defining transitions between harmonic themes based on silent score. "Transcendence" is >= 0.8.

# =====================================================================
# Harmonic Resonance Constants (Used by HarmonicResonanceEngine)
# =====================================================================
# Weights for components contributing to the resonance score (Sum = 1.0)
RESONANCE_WEIGHT_CAPACITY: Final[float] = 0.4
RESONANCE_WEIGHT_SHADOW: Final[float] = 0.4
RESONANCE_WEIGHT_MAGNITUDE: Final[float] = 0.2
# RATIONALE: Relative importance of metrics in calculating resonance score (may differ from silent score weights).

# Assumed Min/Max values for normalizing Magnitude
MAGNITUDE_MIN_VALUE: Final[float] = 1.0
MAGNITUDE_MAX_VALUE: Final[float] = 10.0
# RATIONALE: Defines the expected input range for the magnitude metric for normalization.
MAX_SHADOW_SCORE: Final[float] = 1.0
# RATIONALE: Assumed maximum value for shadow score (0-1 scale), used for inverting the score (MAX - score).

# Min/Max bounds for the calculated resonance score
MIN_RESONANCE_SCORE: Final[float] = 0.0
MAX_RESONANCE_SCORE: Final[float] = 1.0
# RATIONALE: Defines the normalized output range for the resonance score.

# Resonance score thresholds for determining the RESONANCE theme (in HarmonicResonanceEngine)
# Theme is assigned if score is GREATER THAN OR EQUAL TO the threshold value
RESONANCE_THRESHOLD_RENEWAL: Final[float] = 0.75
RESONANCE_THRESHOLD_RESILIENCE: Final[float] = 0.5
RESONANCE_THRESHOLD_REFLECTION: Final[float] = 0.25
# Theme "Reset" is applied if score < RESONANCE_THRESHOLD_REFLECTION
# RATIONALE: Boundaries defining transitions between resonance themes based on resonance score.

# Default theme name used in /command response if none calculated
DEFAULT_RESONANCE_THEME: Final[str] = "default"
# RATIONALE: Fallback theme name if calculation fails or is not applicable.

# TODO: Review Harmonic Framework (HarmonicRouting) vs Harmonic Resonance themes/thresholds.
# They seem conceptually similar but use different names, boundaries, and comparison logic (< vs >=).
# Consider consolidating into a single system.

# =====================================================================
# Emotional Integrity Constants
# =====================================================================
EMOTIONAL_INTEGRITY_BASELINE: Final[float] = 5.0
# RATIONALE: Neutral starting point (0-10 scale) for kindness, respect, consideration scores.
MIN_EMOTIONAL_INTEGRITY_SCORE: Final[float] = 0.0
MAX_EMOTIONAL_INTEGRITY_SCORE: Final[float] = 10.0
# RATIONALE: Defines the valid score range for emotional integrity components.
MIN_EMOTIONAL_INTEGRITY_DELTA: Final[float] = -0.5
MAX_EMOTIONAL_INTEGRITY_DELTA: Final[float] = 0.5
# RATIONALE: Defines the expected output range for LLM-based delta analysis per reflection.
DEFAULT_EMOTIONAL_INTEGRITY_DELTA: Final[float] = 0.0
# RATIONALE: Assumed delta if LLM analysis fails or doesn't provide a value.
EMOTIONAL_INTEGRITY_SCALING_FACTOR: Final[float] = 2.0
# RATIONALE: Multiplier applied to LLM deltas before updating scores, controlling sensitivity. (e.g., max change per reflection = +/- 1.0 point).

# =====================================================================
# Task Engine Constants
# =====================================================================
# Scoring boost applied to tasks that are the last sibling under a parent node
HTA_CHECKPOINT_PROXIMITY_BOOST: Final[float] = 0.15
# RATIONALE: Incentivizes completing sub-goals by boosting the priority of the final task in a sequence.

# Max HTA depth considered for magnitude normalization
HTA_MAX_DEPTH_FOR_MAG_NORM: Final[int] = 5
# RATIONALE: Limits how much deeper tasks contribute disproportionately to magnitude boost calculation.
TASK_MAGNITUDE_DEPTH_WEIGHT: Final[float] = 1.0
# RATIONALE: Controls the impact of normalized HTA depth on task magnitude (Max depth adds this amount).
TASK_DEFAULT_MAGNITUDE: Final[float] = DEFAULT_SNAPSHOT_MAGNITUDE
# RATIONALE: Fallback task magnitude if not determined by tier or other factors.

# Base magnitude associated with different task tiers (example values)
TASK_TIER_BASE_MAGNITUDE: Final[Dict[str, float]] = {
    "Bud": 3.0, "Bloom": 5.0, "Blossom": 7.0,
}
# RATIONALE: Sets a baseline magnitude based on the task's assigned tier/complexity.

# Priority boost applied based on pattern identification insights
PATTERN_PRIORITY_BOOST: Final[float] = 0.1
# RATIONALE: Increases likelihood of tasks related to identified user patterns being selected.

# Default priority for an HTA node if not specified
DEFAULT_NODE_PRIORITY: Final[float] = 0.0
# RATIONALE: Assumed priority value if missing from HTA data.

# Default value if HTA node energy/time estimate key ('low','medium','high') not found in RESOURCE_MAP
DEFAULT_RESOURCE_VALUE: Final[float] = 0.5
# RATIONALE: Fallback resource cost (typically corresponds to 'medium').

# =====================================================================
# Withering / Engagement Constants (Copied from original user file)
# =====================================================================
WITHERING_IDLE_COEFF: Final[Dict[str, float]] = {
    "structured": 0.025,  # ~2.5% per-day decay for structured users.
    "blended":    0.015,   # Gentler decay for blended mode.
    "open":       0.0,     # No decay for freeform mode.
}
# RATIONALE: Rate of engagement decay based on user path and idle time.
WITHERING_OVERDUE_COEFF: Final[Dict[str, float]] = {
    "structured": 0.012,  # 1.2% per-day penalty for missed deadlines.
    "blended":    0.005,   # Reduced pressure in blended.
    # Open mode has no deadlines, so no coefficient needed.
}
# RATIONALE: Rate of engagement decay based on user path and overdue tasks.
WITHERING_DECAY_FACTOR: Final[float] = 0.98
# RATIONALE: Slows down withering accumulation over time, preventing runaway penalties.
WITHERING_COMPLETION_RELIEF: Final[float] = 0.15
# RATIONALE: Amount by which withering level is reduced upon successful task completion, rewarding action.

# =====================================================================
# Reflection Nudge Constants (Copied from original user file)
# =====================================================================
REFLECTION_CAPACITY_NUDGE_BASE: Final[float] = 0.05
REFLECTION_SHADOW_NUDGE_BASE: Final[float] = 0.05
# RATIONALE: Base amount capacity/shadow are nudged per reflection based on sentiment analysis score.

# =====================================================================
# Other Module Constants (Examples from original user file)
# =====================================================================
# NOTE: Review if these constants are still needed or should be moved/renamed
# For example, MAGNITUDE_THRESHOLDS was defined slightly differently here than proposed for HarmonicFramework/Resonance. Using the one from the user's file for now.
MAGNITUDE_THRESHOLDS: Final[Dict[str, float]] = {
    "Seismic": 9.0, "Profound": 7.0, "Rising": 5.0, "Subtle": 3.0, "Dormant": 1.0
}
# RATIONALE: Thresholds for describing magnitude levels used in Orchestrator (from original file).

METRICS_MOMENTUM_ALPHA: Final[float] = 0.3
# RATIONALE: EWMA smoothing factor for momentum (from original file).

# --- Potentially Deprecated / Needs Review ---
# DEV_INDEX_PRIORITY_BOOST = 0.1 # Placeholder - is this used instead of TASK_EFFECT_BASE_BOOST? Need to check TaskEngine usage.
# REFLECTION_PRIORITY_BOOST = 0.05 # Placeholder - Where is this used?

# =====================================================================
# Logging Setup (Optional)
# =====================================================================
constants_logger = logging.getLogger(__name__ + ".constants")
constants_logger.info("Forest OS Constants Loaded.")
# Add validation checks here if needed
