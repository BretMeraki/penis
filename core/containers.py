# forest_app/containers.py (REFACTORED: Incorporates Processors/Services, Logger Scope Fix)
import logging # Import logging first
import sys
from dependency_injector import containers, providers
from typing import Optional, Dict, Any, List # Ensure necessary typing

# --- Feature Flags ---
# Attempt to import the feature flag definitions and checker function
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    feature_flags_import_ok = True
    logger = logging.getLogger("containers") # Get logger after potential import
    logger.info("Successfully imported feature flags.")
except ImportError as e:
    # Configure basic logging if not already set, to show this critical error
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("containers_init_error")
    logger.error(f"CRITICAL: containers.py failed to import Feature Flags: {e}. Check path.")
    logger.warning("WARNING: ALL features will be treated as DISABLED.")
    # Define dummy Feature enum and is_enabled function if import fails
    class Feature: # Minimal dummy enum matching expected usage below
        SENTIMENT_ANALYSIS = "FEATURE_ENABLE_SENTIMENT_ANALYSIS"
        PATTERN_ID = "FEATURE_ENABLE_PATTERN_ID"
        RELATIONAL = "FEATURE_ENABLE_RELATIONAL"
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES"
        XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"
        EMOTIONAL_INTEGRITY = "FEATURE_ENABLE_EMOTIONAL_INTEGRITY"
        DESIRE_ENGINE = "FEATURE_ENABLE_DESIRE_ENGINE"
        FINANCIAL_READINESS = "FEATURE_ENABLE_FINANCIAL_READINESS"
        REWARDS = "FEATURE_ENABLE_REWARDS"
        CORE_TASK_ENGINE = "FEATURE_ENABLE_CORE_TASK_ENGINE" # Should likely default to True if flags fail
        TRIGGER_PHRASES = "FEATURE_ENABLE_TRIGGER_PHRASES"
        PRACTICAL_CONSEQUENCE = "FEATURE_ENABLE_PRACTICAL_CONSEQUENCE"
        CORE_HTA = "FEATURE_ENABLE_CORE_HTA" # Added for checks
        # Add others ONLY if directly checked via is_enabled() below

    def is_enabled(feature: Feature) -> bool: # Dummy function defaults to False
        # Use logger defined above
        logger.warning(f"Using dummy is_enabled for {getattr(feature, 'name', feature)}. Feature is DISABLED.")
        # Make core task engine default to True even in dummy mode, as it's critical
        if feature == Feature.CORE_TASK_ENGINE:
            logger.warning("Dummy is_enabled: Forcing CORE_TASK_ENGINE to True as it's essential.")
            return True
        if feature == Feature.CORE_HTA: # Also assume core HTA is essential
             logger.warning("Dummy is_enabled: Forcing CORE_HTA to True as it's essential.")
             return True
        return False
    feature_flags_import_ok = False


# --- Configuration ---
try:
    from forest_app.config.settings import settings
    settings_import_ok = True
    logger.info("Successfully imported settings.")
except ImportError as e:
    logger.warning(f"containers.py could not import settings: {e}. Using dummy settings/defaults.")
    class DummySettings: pass
    settings = DummySettings()
    # Add necessary defaults for container setup even if settings fail
    settings.GOOGLE_API_KEY = "DUMMY_KEY_REQUIRED" # Example required placeholder
    settings.DB_CONNECTION_STRING = "sqlite:///./dummy_db_required.db" # Example required placeholder
    settings.GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
    settings.GEMINI_ADVANCED_MODEL_NAME = "gemini-1.5-pro-latest"
    settings.LLM_TEMPERATURE = 0.7
    settings.METRICS_ENGINE_ALPHA = 0.3 # Provide default value
    settings.METRICS_ENGINE_THRESHOLDS = {} # Default to empty dict
    settings.NARRATIVE_ENGINE_CONFIG = {} # Default to empty dict
    settings.SNAPSHOT_FLOW_FREQUENCY = 5 # Provide default value
    settings.SNAPSHOT_FLOW_MAX_SNAPSHOTS = 100 # Provide default value
    settings.TASK_ENGINE_TEMPLATES = {} # Default to empty dict
    # Feature flags will default to False via is_enabled if settings attrs missing
    settings_import_ok = False


# --- Define DummyService BEFORE module imports ---
class DummyService: # Base dummy for fallback
    """A dummy service that does nothing, used when features are disabled or imports fail."""
    def __init__(self, *args, **kwargs):
        # Log the creation of a dummy instance for easier debugging
        logger.debug(f"Initialized DummyService for {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")
        pass
    def __call__(self, *args, **kwargs):
        logger.debug(f"Called DummyService {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")
        return None # Or perhaps return a default value if needed
    def __getattr__(self, name):
        # Return a method that logs its call and does nothing
        def _dummy_method(*args, **kwargs):
            logger.debug(f"Called dummy method '{name}' on DummyService {self.__class__.__name__} with args: {args}, kwargs: {kwargs}")
            # Check if the expected return type might need a default value
            # For example, if a method is expected to return a dict, return {}
            # if name == 'some_method_expecting_dict': return {}
            return None # Or a default value
        # Return a dummy attribute for non-callable attributes if needed
        # if not callable(getattr(super(), name, None)): return None
        return _dummy_method


# --- Integrations ---
try:
    from forest_app.integrations.llm import LLMClient
    llm_import_ok = True
except ImportError as e:
    logger.error(f"containers.py failed to import LLMClient: {e}. Using dummy.")
    class LLMClient(DummyService): pass # Inherit from DummyService
    llm_import_ok = False


# --- Modules (Engines/Managers) ---
# Import necessary real implementations (use try-except for robustness)
try:
    from forest_app.core.logging_tracking import TaskFootprintLogger
    from forest_app.modules.trigger.trigger_phrase import TriggerPhraseHandler
    from forest_app.modules.cognitive.sentiment import SecretSauceSentimentEngineHybrid
    from forest_app.modules.cognitive.pattern_id import PatternIdentificationEngine
    from forest_app.modules.cognitive.practical_consequence import PracticalConsequenceEngine
    from forest_app.modules.metrics_specific import MetricsSpecificEngine
    from forest_app.hta_tree.seed import SeedManager
    from forest_app.hta_tree.task_engine import TaskEngine
    from forest_app.hta_tree.hta_service import HTAService
    from forest_app.modules.relational.relational import RelationalManager, Profile, RelationalRepairEngine
    from forest_app.modules.cognitive.narrative_modes import NarrativeModesEngine
    from forest_app.modules.resource.offering_reward import OfferingRouter
    from forest_app.modules.resource.xp_mastery import XPMastery
    from forest_app.modules.resource.emotional_integrity import EmotionalIntegrityIndex
    from forest_app.modules.snapshot_flow.snapshot_flow import SnapshotFlowController
    from forest_app.modules.resource.desire_engine import DesireEngine
    from forest_app.modules.resource.financial_readiness import FinancialReadinessEngine
    # Core components needed
    from forest_app.core.harmonic_framework import SilentScoring, HarmonicRouting
    from forest_app.core.orchestrator import ForestOrchestrator
    # --- Processors/Services ---
    from forest_app.core.processors.reflection_processor import ReflectionProcessor
    from forest_app.core.processors.completion_processor import CompletionProcessor
    from forest_app.core.services.component_state_manager import ComponentStateManager
    from forest_app.integrations.mcp_service import MCPService
    modules_core_import_ok = True
    logger.info("Successfully imported application modules and core components.")
except ImportError as e:
    # Log the specific import error that occurred
    logger.critical(f"CRITICAL: containers.py failed to import one or more modules/core: {e}", exc_info=True)
    # Define dummy classes
    class TaskFootprintLogger(DummyService): pass
    class TriggerPhraseHandler(DummyService): pass
    class SecretSauceSentimentEngineHybrid(DummyService): pass
    class PatternIdentificationEngine(DummyService): pass
    class PracticalConsequenceEngine(DummyService): pass
    class MetricsSpecificEngine(DummyService): pass
    class SeedManager(DummyService): pass
    class RelationalManager(DummyService): pass
    class NarrativeModesEngine(DummyService): pass
    class OfferingRouter(DummyService): pass
    class XPMastery(DummyService): pass
    class EmotionalIntegrityIndex(DummyService): pass
    class TaskEngine(DummyService): pass
    class SnapshotFlowController(DummyService): pass
    class DesireEngine(DummyService): pass
    class FinancialReadinessEngine(DummyService): pass
    class SilentScoring(DummyService): pass
    class HarmonicRouting(DummyService): pass
    class ForestOrchestrator(DummyService): pass
    class ReflectionProcessor(DummyService): pass
    class CompletionProcessor(DummyService): pass
    class HTAService(DummyService): pass
    class ComponentStateManager(DummyService): pass
    modules_core_import_ok = False


class Container(containers.DeclarativeContainer):
    """
    Main Dependency Injection container for the Forest application.
    Uses feature_flags.is_enabled to provide real or dummy services directly.
    Manages providers for both individual engines and refactored processors/services.
    """
    # --- Wiring Config (UPDATED) ---
    # Define which modules need injections resolved
    wiring_config = containers.WiringConfiguration(
        modules=[
            "forest_app.core.main", "forest_app.core.dependencies",
            "forest_app.routers.core", "forest_app.routers.auth",
            "forest_app.routers.onboarding", "forest_app.routers.users",
            "forest_app.routers.hta", "forest_app.snapshot.routers",
            "forest_app.core.orchestrator",     # Original orchestrator (now refactored)
            "forest_app.core.processors",       # << ADDED Processors module
            "forest_app.core.services",         # << ADDED Services module
            # Add other modules using @inject here (if any)
        ]
    )

    # --- Configuration Provider ---
    # Loads from the 'settings' instance created from AppSettings
    config = providers.Configuration(strict=False) # strict=False ignores missing settings
    if settings_import_ok:
        try:
            # Ensure defaults are present before loading from pydantic
            config.set('METRICS_ENGINE_ALPHA', 0.3)
            config.set('METRICS_ENGINE_THRESHOLDS', {})
            config.set('NARRATIVE_ENGINE_CONFIG', {})
            config.set('SNAPSHOT_FLOW_FREQUENCY', 5)
            config.set('SNAPSHOT_FLOW_MAX_SNAPSHOTS', 100)
            config.set('TASK_ENGINE_TEMPLATES', {})
            # Load from Pydantic settings, overriding defaults if present
            config.from_pydantic(settings)
            logger.info("DI Container: Configuration loaded successfully from settings.")
        except Exception as config_load_err:
            logger.error(f"DI Container: Failed to load config from Pydantic settings: {config_load_err}", exc_info=True)
            # Apply dummy defaults if loading failed
            config.override({
                "GOOGLE_API_KEY": "DUMMY_KEY_REQUIRED",
                "DB_CONNECTION_STRING": "sqlite:///./dummy_db_required.db",
                "GEMINI_MODEL_NAME": "gemini-1.5-flash-latest",
                "GEMINI_ADVANCED_MODEL_NAME": "gemini-1.5-pro-latest",
                "LLM_TEMPERATURE": 0.7,
                "METRICS_ENGINE_ALPHA": 0.3,
                "METRICS_ENGINE_THRESHOLDS": {},
                "NARRATIVE_ENGINE_CONFIG": {},
                "SNAPSHOT_FLOW_FREQUENCY": 5,
                "SNAPSHOT_FLOW_MAX_SNAPSHOTS": 100,
                "TASK_ENGINE_TEMPLATES": {},
            })
    else:
        logger.warning("DI Container: Running with dummy configuration due to settings import failure.")
        # Apply dummy defaults directly to config provider if settings failed
        config.override({
            "GOOGLE_API_KEY": "DUMMY_KEY_REQUIRED",
            "DB_CONNECTION_STRING": "sqlite:///./dummy_db_required.db",
            "GEMINI_MODEL_NAME": "gemini-1.5-flash-latest",
            "GEMINI_ADVANCED_MODEL_NAME": "gemini-1.5-pro-latest",
            "LLM_TEMPERATURE": 0.7,
            "METRICS_ENGINE_ALPHA": 0.3,
            "METRICS_ENGINE_THRESHOLDS": {},
            "NARRATIVE_ENGINE_CONFIG": {},
            "SNAPSHOT_FLOW_FREQUENCY": 5,
            "SNAPSHOT_FLOW_MAX_SNAPSHOTS": 100,
            "TASK_ENGINE_TEMPLATES": {},
        })


    # --- Core Service Providers (LLM Client) ---
    # LLM Client (provide dummy if import failed)
    # >>> CORRECTED TO SINGLETON <<<
    llm_client = providers.Singleton(
        LLMClient # Removed arguments as LLMClient handles its own config
    ) if llm_import_ok else providers.Singleton(DummyService)

    # --- Individual Engine/Manager Providers (used by processors/services) ---
    # (Definitions remain mostly the same, using Singleton for stateful components)

    # Sentiment Engine
    sentiment_engine = providers.Singleton(
        SecretSauceSentimentEngineHybrid, llm_client=llm_client
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.SENTIMENT_ANALYSIS) else providers.Singleton(DummyService)

    # Pattern Engine
    pattern_engine = providers.Singleton(
        PatternIdentificationEngine, config=config.provided
    ) if modules_core_import_ok and is_enabled(Feature.PATTERN_ID) else providers.Singleton(DummyService)

    # Practical Consequence Engine
    practical_consequence_engine = providers.Singleton(
        PracticalConsequenceEngine
    ) if modules_core_import_ok and is_enabled(Feature.PRACTICAL_CONSEQUENCE) else providers.Singleton(DummyService)

    # Metrics Specific Engine
    metrics_engine = providers.Singleton(
        MetricsSpecificEngine,
        alpha=config.METRICS_ENGINE_ALPHA,
        thresholds=config.METRICS_ENGINE_THRESHOLDS
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # Seed Manager
    # >>> CORRECTED TO SINGLETON <<< (Assuming SeedManager holds state/cache)
    seed_manager = providers.Singleton(SeedManager) if modules_core_import_ok else providers.Singleton(DummyService)

    # Relational Manager
    relational_manager = providers.Singleton(
        RelationalManager, llm_client=llm_client
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.RELATIONAL) else providers.Singleton(DummyService)

    # Narrative Modes Engine
    narrative_engine = providers.Singleton(
        NarrativeModesEngine,
        config=config.NARRATIVE_ENGINE_CONFIG
    ) if modules_core_import_ok and is_enabled(Feature.NARRATIVE_MODES) else providers.Singleton(DummyService)

    # XP Mastery
    # >>> CORRECTED TO SINGLETON <<< (Likely holds level/progress state)
    xp_mastery = providers.Singleton(
        XPMastery
    ) if modules_core_import_ok and is_enabled(Feature.XP_MASTERY) else providers.Singleton(DummyService)

    # Emotional Integrity Engine
    emotional_integrity_engine = providers.Singleton(
        EmotionalIntegrityIndex, llm_client=llm_client
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.EMOTIONAL_INTEGRITY) else providers.Singleton(DummyService)

    # Snapshot Flow Controller
    # >>> CORRECTED TO SINGLETON <<< (Holds counter, snapshots deque)
    snapshot_flow_controller = providers.Singleton(
        SnapshotFlowController,
        frequency=config.SNAPSHOT_FLOW_FREQUENCY,
        max_snapshots=config.SNAPSHOT_FLOW_MAX_SNAPSHOTS
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # Desire Engine
    desire_engine = providers.Singleton(
        DesireEngine, llm_client=llm_client
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.DESIRE_ENGINE) else providers.Singleton(DummyService)

    # Financial Readiness Engine
    financial_engine = providers.Singleton(
        FinancialReadinessEngine, llm_client=llm_client
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.FINANCIAL_READINESS) else providers.Singleton(DummyService)

    # Offering Router
    # >>> CORRECTED TO SINGLETON <<<
    offering_router = providers.Singleton(
        OfferingRouter,
        llm_client=llm_client,
        desire_engine=desire_engine,
        financial_engine=financial_engine
    ) if modules_core_import_ok and llm_import_ok and is_enabled(Feature.REWARDS) else providers.Singleton(DummyService)

    # Task Engine
    # >>> CORRECTED TO SINGLETON <<<
    task_engine = providers.Singleton(
        TaskEngine,
        pattern_engine=pattern_engine,
        task_templates=config.TASK_ENGINE_TEMPLATES
    ) if modules_core_import_ok and is_enabled(Feature.CORE_TASK_ENGINE) else providers.Singleton(DummyService)

    # Trigger Phrase Handler
    # >>> CORRECTED TO SINGLETON <<<
    trigger_phrase_handler = providers.Singleton(
        TriggerPhraseHandler
    ) if modules_core_import_ok and is_enabled(Feature.TRIGGER_PHRASES) else providers.Singleton(DummyService)

    # Harmonic Framework Components
    # >>> CORRECTED TO SINGLETON <<<
    silent_scorer = providers.Singleton(SilentScoring) if modules_core_import_ok else providers.Singleton(DummyService)
    # >>> CORRECTED TO SINGLETON <<<
    harmonic_router = providers.Singleton(HarmonicRouting) if modules_core_import_ok else providers.Singleton(DummyService)

    # Task Footprint Logger
    # --- MODIFIED: Changed to Factory to accept db argument ---
    task_footprint_logger = providers.Factory(
        TaskFootprintLogger
    ) if modules_core_import_ok else providers.Singleton(DummyService)


    # --- ADDED: Providers for Refactored Components ---

    # Component State Manager
    # Uses providers.Dict to aggregate engine providers based on state keys
    managed_engines_dict = providers.Dict(
        # Map state key used in snapshot.component_state to the provider instance
        metrics_engine=metrics_engine,
        seed_manager=seed_manager,
        relational_manager=relational_manager,
        sentiment_engine_calibration=sentiment_engine, # Key used in original orchestrator
        practical_consequence=practical_consequence_engine,
        xp_mastery=xp_mastery,
        pattern_engine_config=pattern_engine,          # Key used in original orchestrator
        narrative_engine_config=narrative_engine,      # Key used in original orchestrator
        emotional_integrity_index=emotional_integrity_engine,
        DesireEngine=desire_engine,                    # Key used in original orchestrator
        FinancialReadinessEngine=financial_engine,      # Key used in original orchestrator
        # Add any other components that were saved/loaded by the old orchestrator
    )
    component_state_manager = providers.Singleton(
        ComponentStateManager,
        managed_engines=managed_engines_dict
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # HTA Service
    hta_service = providers.Singleton(
        HTAService,
        llm_client=llm_client,
        seed_manager=seed_manager,
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # Reflection Processor
    reflection_processor = providers.Singleton(
        ReflectionProcessor,
        llm_client=llm_client,
        task_engine=task_engine,
        sentiment_engine=sentiment_engine,
        practical_consequence_engine=practical_consequence_engine,
        narrative_engine=narrative_engine,
        silent_scorer=silent_scorer,
        harmonic_router=harmonic_router,
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # Completion Processor
    completion_processor = providers.Singleton(
        CompletionProcessor,
        hta_service=hta_service,
        task_engine=task_engine,
        xp_engine=xp_mastery,
        llm_client=llm_client, # Pass LLM if XPMastery needs it
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # --- Orchestrator Provider (UPDATED) ---
    # Now injects the refactored processors and services
    orchestrator = providers.Singleton(
        ForestOrchestrator,
        reflection_processor=reflection_processor,
        completion_processor=completion_processor,
        state_manager=component_state_manager,
        hta_service=hta_service, # Pass if orchestrator still needs direct access (maybe not)
        seed_manager=seed_manager, # Pass for delegation methods
        # Removed the direct injection of many individual engines
    ) if modules_core_import_ok else providers.Singleton(DummyService)

    # MCP Service
    mcp_service = providers.Singleton(MCPService, settings=settings)


# Create container instance at module level
container = None

def init_container():
    """Initialize the container if it hasn't been initialized yet."""
    global container
    if container is None:
        container = Container()
        # Wire the container to the modules specified in wiring_config
        try:
            container.wire(modules=Container.wiring_config.modules)
            logger.info("DI Container wiring applied successfully.")
        except Exception as wire_err:
            logger.critical(f"CRITICAL: Failed to apply DI container wiring: {wire_err}", exc_info=True)
            raise RuntimeError(f"DI Container wiring failed: {wire_err}")
    return container

# Initialize container if this module is run directly
if __name__ == "__main__":
    init_container()
