# forest_app/config/settings.py (Added Withering Flag)

import logging
import os
from pydantic_settings import BaseSettings, SettingsConfigDict # Correct import
from typing import Optional, Dict, Any, List
from pydantic import Field, HttpUrl

logger = logging.getLogger(__name__)

class MCPServerConfig(BaseSettings):
    class Config:
        frozen = True
        extra = 'forbid'

    id: str
    type: str
    url: Optional[HttpUrl] = None
    auth_token_env_var: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    env: Optional[Dict[str, str]] = None

class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Includes configurations for specific engines AND feature flags.
    Ensures all flags defined in Feature Enum have a corresponding setting.
    """
    # --- Required environment variables ---
    GOOGLE_API_KEY: str = "test_api_key"
    DB_CONNECTION_STRING: str = "postgresql://test:test@localhost:5432/test_db"

    # --- Optional with defaults (Core LLM/App) ---
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash-latest"
    GEMINI_ADVANCED_MODEL_NAME: str = "gemini-1.5-pro-latest"
    LLM_TEMPERATURE: float = 0.7

    # --- Optional Engine Configurations ---
    # (These configure engines IF they are enabled by flags below)
    METRICS_ENGINE_ALPHA: float = 0.3
    METRICS_ENGINE_THRESHOLDS: Optional[Dict[str, float]] = None
    NARRATIVE_ENGINE_CONFIG: Optional[Dict[str, Any]] = None
    SNAPSHOT_FLOW_FREQUENCY: int = 5
    SNAPSHOT_FLOW_MAX_SNAPSHOTS: int = 100
    TASK_ENGINE_TEMPLATES: Optional[Dict[str, Any]] = None
    # PRACTICAL_CONSEQUENCE_CALIBRATION: Optional[Dict[str, float]] = None

    # --- Feature Flags ---
    # Default values represent a Bare Bones MVP target state.
    # Override via environment variables or .env file.

    # Core Functionality - Assumed ON unless specifically disabled for tests
    FEATURE_ENABLE_CORE_ONBOARDING: bool = True
    FEATURE_ENABLE_CORE_HTA: bool = True
    FEATURE_ENABLE_CORE_TASK_ENGINE: bool = True
    # FEATURE_ENABLE_SEED_MANAGER: bool = True # <-- REMOVED THIS LINE (Correct)

    # Deferred / Potentially ON for MVP
    FEATURE_ENABLE_XP_MASTERY: bool = True    # Deferred Integration (Set default based on MVP decision)
    FEATURE_ENABLE_MEMORY_SYSTEM: bool = True   # Deferred Integration (Set default based on MVP decision)

    # Modules with Flag Integration Added (Default OFF for MVP, except where noted)
    FEATURE_ENABLE_DEVELOPMENT_INDEX: bool = False
    FEATURE_ENABLE_ARCHETYPES: bool = False
    FEATURE_ENABLE_SENTIMENT_ANALYSIS: bool = True
    FEATURE_ENABLE_NARRATIVE_MODES: bool = True
    FEATURE_ENABLE_PATTERN_ID: bool = True # Default ON based on previous logs
    FEATURE_ENABLE_SHADOW_ANALYSIS: bool = False
    FEATURE_ENABLE_TRAIL_MANAGER: bool = False
    FEATURE_ENABLE_SOFT_DEADLINES: bool = True
    FEATURE_ENABLE_TRIGGER_PHRASES: bool = True # Default ON based on previous logs
    FEATURE_ENABLE_HARMONIC_RESONANCE: bool = False
    FEATURE_ENABLE_RESISTANCE_ENGINE: bool = False
    FEATURE_ENABLE_EMOTIONAL_INTEGRITY: bool = False
    FEATURE_ENABLE_RELATIONAL: bool = False
    FEATURE_ENABLE_REWARDS: bool = False            # Includes RewardIndex, OfferingRouter
    FEATURE_ENABLE_FINANCIAL_READINESS: bool = False
    FEATURE_ENABLE_DESIRE_ENGINE: bool = False
    FEATURE_ENABLE_PRACTICAL_CONSEQUENCE: bool = False
    FEATURE_ENABLE_TASK_RESOURCE_FILTER: bool = False
    FEATURE_ENABLE_POETIC_ARBITER_VOICE: bool = True
    FEATURE_ENABLE_HTA_VISUALIZATION: bool = True   # <<< ADDED - Defaulting to True (Visible)
    FEATURE_ENABLE_WITHERING: bool = False          # <<< ADDED - Defaulting to False

    # Flags Defined But Not Used in Code Checks (Skipped based on discussion)
    FEATURE_ENABLE_LOGGING_TRACKING: bool = True      # Skipped Code Check (Use Levels)
    FEATURE_ENABLE_SNAPSHOT_FLOW: bool = True         # Skipped Code Check (Process Always Runs)
    FEATURE_ENABLE_METRICS_SPECIFIC: bool = True      # Skipped Code Check (Considered Core/Not Flagged)
    FEATURE_ENABLE_HARMONIC_FRAMEWORK: bool = False   # Skipped Code Check (Resonance Handled)

    # Test configuration
    TESTING: bool = True

    mcp_servers: List[MCPServerConfig] = Field(default_factory=list, description="Configurations for connecting to MCP Servers")

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file='.env', # Load from .env file if it exists
        env_file_encoding='utf-8',
        extra='ignore', # Ignore extra environment variables
    )

# --- Create a single instance of the settings ---
settings = AppSettings()

# --- Logging & Checks (Keep your existing checks) ---
# [Logging/Checks remain unchanged]
logger.debug(">>> DEBUG SETTINGS: STARTING Pydantic settings.py <<<")
for key, value in settings.model_dump().items():
      if "KEY" in key or "STRING" in key:
          logger.debug(f">>> DEBUG SETTINGS: {key}: {'Loaded' if value else 'Missing/Empty'}")
      else:
          logger.debug(f">>> DEBUG SETTINGS: {key}: {value}")
if not settings.GOOGLE_API_KEY: logger.critical(">>> CRITICAL SETTINGS: GOOGLE_API_KEY is missing!")
if not settings.DB_CONNECTION_STRING: logger.critical(">>> CRITICAL SETTINGS: DB_CONNECTION_STRING is missing!")
logger.debug(">>> DEBUG SETTINGS: END OF Pydantic settings.py <<<")
