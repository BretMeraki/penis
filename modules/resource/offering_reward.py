# forest_app/modules/offering_reward.py

import json
import logging
from datetime import datetime, timezone # Use timezone aware
from typing import Any, Dict, List, Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("offering_reward_init")
    logger.warning("Feature flags module not found in offering_reward. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        REWARDS = "FEATURE_ENABLE_REWARDS" # Define the specific flag used here
        # Include flags checked by dependencies if needed for fallback logic
        DESIRE_ENGINE = "FEATURE_ENABLE_DESIRE_ENGINE"
        FINANCIAL_READINESS = "FEATURE_ENABLE_FINANCIAL_READINESS"
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

# --- Pydantic Import ---
try:
    from pydantic import BaseModel, Field, ValidationError
    pydantic_import_ok = True
except ImportError:
    logging.getLogger("offering_reward_init").critical("Pydantic not installed. OfferingRouter requires Pydantic.")
    pydantic_import_ok = False
    class BaseModel: pass
    def Field(*args, **kwargs): return None
    class ValidationError(Exception): pass

# --- Module & LLM Imports ---
# Assume these might fail if related features are off or imports broken
try:
    from forest_app.modules.resource.desire_engine import DesireEngine
    desire_engine_import_ok = True
except ImportError:
    logging.getLogger("offering_reward_init").warning("Could not import DesireEngine.")
    desire_engine_import_ok = False
    class DesireEngine: # Dummy
        def get_top_desires(self, cache, top_n): return ["Default Desire"]

try:
    from forest_app.modules.resource.financial_readiness import FinancialReadinessEngine
    financial_engine_import_ok = True
except ImportError:
    logging.getLogger("offering_reward_init").warning("Could not import FinancialReadinessEngine.")
    financial_engine_import_ok = False
    class FinancialReadinessEngine: # Dummy
        readiness = 0.5 # Default dummy value

try:
    from forest_app.integrations.llm import (
        LLMClient,
        LLMError,
        LLMValidationError,
        LLMConfigurationError,
        LLMConnectionError
    )
    llm_import_ok = True
except ImportError as e:
    logging.getLogger("offering_reward_init").critical(f"Failed to import LLM integration components: {e}.")
    llm_import_ok = False
    class LLMClient: pass
    class LLMError(Exception): pass
    class LLMValidationError(LLMError): pass
    class LLMConfigurationError(LLMError): pass
    class LLMConnectionError(LLMError): pass

logger = logging.getLogger(__name__)
# Rely on global config for level

# --- Define Response Models ---
# Only define if Pydantic import was successful
if pydantic_import_ok:
    class OfferingSuggestion(BaseModel):
        suggestion: str = Field(..., min_length=1)

    class OfferingResponseModel(BaseModel):
        suggestions: List[OfferingSuggestion] = Field(..., min_items=1)
else:
     class OfferingSuggestion: pass
     class OfferingResponseModel: pass


# Define default/error outputs
DEFAULT_OFFERING_ERROR_MSG = ["Reward suggestion generation is currently unavailable."]
DEFAULT_OFFERING_DISABLED_MSG = ["Reward suggestions are currently disabled."]
DEFAULT_RECORD_ERROR = {"error": "Cannot record acceptance; rewards disabled or error occurred."}


class OfferingRouter:
    """
    Generates personalized reward suggestions and handles totem issuance.
    Respects the REWARDS feature flag. Requires LLMClient, DesireEngine,
    and FinancialReadinessEngine.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        desire_engine: DesireEngine,
        financial_engine: FinancialReadinessEngine,
    ) -> None:
        """
        Initializes the router with required engine instances.

        Args:
            llm_client: An instance of the LLMClient.
            desire_engine: An instance of DesireEngine.
            financial_engine: An instance of FinancialReadinessEngine.

        Raises:
            TypeError: If any injected engine instance is invalid (and import succeeded).
        """
        # Check dependencies only if their respective imports were okay
        if llm_import_ok and not isinstance(llm_client, LLMClient):
            raise TypeError("OfferingRouter requires a valid LLMClient instance.")
        if desire_engine_import_ok and not isinstance(desire_engine, DesireEngine):
            raise TypeError("OfferingRouter requires a valid DesireEngine instance.")
        if financial_engine_import_ok and not isinstance(financial_engine, FinancialReadinessEngine):
            raise TypeError("OfferingRouter requires a valid FinancialReadinessEngine instance.")

        self.llm_client = llm_client
        self.desire_engine = desire_engine
        self.financial_engine = financial_engine
        logger.info("OfferingRouter initialized.")

    def _get_snapshot_data(self, snap: Any, key: str, default: Any = None) -> Any:
         """Safely get data from the snapshot object or dict."""
         if isinstance(snap, dict):
             return snap.get(key, default)
         elif hasattr(snap, key):
             return getattr(snap, key, default)
         # Add more specific checks if snapshot structure is known and complex
         logger.warning("Could not safely retrieve '%s' from snapshot object of type %s", key, type(snap))
         return default

    def preview_offering_for_task(
        self, snap: Any, task: Optional[Dict[str, Any]], reward_scale: float # task optional
    ) -> List[str]:
        """
        Quick synchronous preview of offerings. Returns empty list if feature disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.REWARDS):
            logger.debug("Skipping preview_offering_for_task: REWARDS feature disabled.")
            return [] # Return empty list, UI can handle this
        # --- End Check ---

        # Check if Desire Engine is available (could be disabled by its own flag or import failed)
        if not is_enabled(Feature.DESIRE_ENGINE) or not isinstance(self.desire_engine, DesireEngine) or not desire_engine_import_ok:
             logger.warning("Preview offering: DesireEngine unavailable. Returning generic suggestion.")
             return ["Consider a small reward for your effort."] # Simple fallback

        # Safely get wants_cache - snapshot structure might vary
        # Assuming DesireEngine holds the state internally now
        # wants_cache = self._get_snapshot_data(snap, 'wants_cache', {}) # This might be outdated if DE holds state
        # Directly access from the engine instance:
        try:
             all_wants = self.desire_engine.get_all_wants() # Assumes this method exists and returns list of strings
             if not isinstance(all_wants, list): all_wants = []
        except Exception as e:
             logger.error("Error getting wants from DesireEngine for preview: %s", e)
             all_wants = []

        # Simplified preview logic - maybe just show top 1-2 wants?
        if not all_wants:
            return ["Take a moment to relax or do something enjoyable."] # Fallback if no wants

        # Simple preview based on wants (doesn't hit LLM)
        preview_suggestions = [f"Consider a reward related to '{want}'." for want in all_wants[:2]] # Show top 2
        logger.debug("Generated preview offering: %s", preview_suggestions)
        return preview_suggestions


    async def maybe_generate_offering(
        self,
        snap: Any,
        task: Optional[Any] = None, # Task might not always be relevant
        reward_scale: float = 0.5, # Default scale
        num_suggestions: int = 3,
    ) -> List[str]:
        """
        Generate reward suggestions via LLM. Returns empty list if feature disabled or error.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.REWARDS):
            logger.debug("Skipping maybe_generate_offering: REWARDS feature disabled.")
            return [] # Return empty list
        # --- End Check ---

        # Check dependencies availability (could be disabled by own flags)
        if not is_enabled(Feature.DESIRE_ENGINE) or not isinstance(self.desire_engine, DesireEngine) or not desire_engine_import_ok:
             logger.error("Generate offering: DesireEngine unavailable.")
             return DEFAULT_OFFERING_ERROR_MSG
        if not is_enabled(Feature.FINANCIAL_READINESS) or not isinstance(self.financial_engine, FinancialReadinessEngine) or not financial_engine_import_ok:
             logger.error("Generate offering: FinancialReadinessEngine unavailable.")
             return DEFAULT_OFFERING_ERROR_MSG
        if not llm_import_ok or not isinstance(self.llm_client, LLMClient) or not hasattr(self.llm_client, 'generate'):
             logger.error("Generate offering: LLMClient unavailable.")
             return DEFAULT_OFFERING_ERROR_MSG


        # Get necessary data from engines
        try:
            all_wants = self.desire_engine.get_all_wants()
            if not isinstance(all_wants, list): all_wants = []
            top_desires = all_wants[:num_suggestions] # Use top N for prompt context
        except Exception as e:
             logger.error("Error getting wants from DesireEngine for offering: %s", e)
             top_desires = ["relaxing", "learning"] # Fallback desires

        try:
             # Financial engine holds its own state
             fin_ready = self.financial_engine.readiness
             fin_ready = _clamp01(float(fin_ready)) # Ensure clamped float
        except Exception as e:
             logger.error("Error getting financial readiness: %s. Using default.", e)
             fin_ready = FinancialReadinessEngine.DEFAULT_READINESS # Use default

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(OfferingResponseModel, BaseModel):
             try: response_model_schema = OfferingResponseModel.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for OfferingResponseModel")

        prompt = (
            f"You are a creative assistant generating personalized reward suggestions based on user context.\n"
            f"User's Top Desires: {', '.join(top_desires) if top_desires else 'None specified'}\n"
            f"Reward Scale (0=small, 1=large): {reward_scale:.2f}\n"
            f"Financial Readiness (0=low, 1=high): {fin_ready:.2f}\n"
            f"Task: Generate exactly {num_suggestions} distinct, concise, creative, and actionable reward suggestions relevant to the desires, scale, and readiness.\n"
            f"Output ONLY a valid JSON object matching this schema:\n{response_model_schema}\n"
        )

        suggestions: List[str] = []
        try:
            llm_response: Optional[OfferingResponseModel] = await self.llm_client.generate(
                 prompt_parts=[prompt],
                 response_model=OfferingResponseModel
            )
            if isinstance(llm_response, OfferingResponseModel):
                 # Extract suggestions from the validated Pydantic model
                 suggestions = [item.suggestion for item in llm_response.suggestions if isinstance(item, OfferingSuggestion)]
                 logger.info("LLM generated %d offering suggestions.", len(suggestions))
                 # Trim to exact number requested if LLM gave too many/few
                 suggestions = suggestions[:num_suggestions]
            else:
                 logger.warning("LLM offering generation did not return a valid OfferingResponseModel.")

        except (LLMError, LLMValidationError, LLMConfigurationError, LLMConnectionError, ValidationError) as llm_e:
            logger.warning("OfferingRouter LLM call or validation failed: %s.", llm_e, exc_info=False)
        except Exception as e:
            logger.exception("Unexpected error during offering generation: %s.", e)

        # Fallback only if suggestions list is empty AND LLM attempt was made
        if not suggestions:
            logger.warning("LLM offering failed or returned no suggestions. Falling back to generic.")
            # Provide simpler fallback suggestions if LLM fails
            generic_suggestions = [
                 "Take a 5-minute break and stretch.",
                 "Listen to a favorite song.",
                 "Step outside for fresh air.",
                 "Enjoy a cup of tea or coffee.",
                 "Read a chapter of a book."
            ]
            # Return a subset matching num_suggestions
            return generic_suggestions[:num_suggestions]

        return suggestions


    def record_acceptance(
        self,
        snap: Any, # Keep Any, as structure might vary or be full snapshot obj
        accepted_suggestion: str,
    ) -> Dict[str, Any]:
        """
        Record suggestion acceptance, issuing a totem. Returns error dict if feature disabled.
        Mutates the snapshot object/dict directly (CAUTION).
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.REWARDS):
            logger.debug("Skipping record_acceptance: REWARDS feature disabled.")
            return DEFAULT_RECORD_ERROR
        # --- End Check ---

        # Also check dependent flags - if desires are off, maybe don't reinforce?
        can_update_desire = is_enabled(Feature.DESIRE_ENGINE) and isinstance(self.desire_engine, DesireEngine)

        if not isinstance(accepted_suggestion, str) or not accepted_suggestion.strip():
             logger.warning("Cannot record acceptance for empty suggestion.")
             return {"error": "Invalid suggestion provided."}

        accepted_suggestion = accepted_suggestion.strip()

        # Generate Totem Info
        # Assume totems are stored directly on the snapshot object/dict for now
        current_totems = self._get_snapshot_data(snap, 'totems', [])
        if not isinstance(current_totems, list):
            logger.error("Cannot add totem: 'totems' attribute in snapshot is not a list. Found type: %s", type(current_totems))
            return {"error": "Invalid snapshot state for totems."}

        totem_id_num = len(current_totems) + 1
        new_totem = {
            "totem_id": f"totem_{totem_id_num}", # Simple sequential ID
            "name": accepted_suggestion,
            "awarded_at": datetime.now(timezone.utc).isoformat(),
            "source": "OfferingRouterAcceptance"
        }

        # --- Attempt to Mutate Snapshot State ---
        # This part is fragile and depends heavily on the snapshot structure
        # Prefer returning data and having the caller update the snapshot if possible
        mutated = False
        try:
             # Try direct attribute access first (if snap is an object like MemorySnapshot)
             if hasattr(snap, 'totems') and isinstance(snap.totems, list):
                  snap.totems.append(new_totem)
                  logger.debug("Appended new totem directly to snap.totems")
                  mutated = True
             # Try component_state access next
             elif hasattr(snap, 'component_state') and isinstance(snap.component_state, dict):
                  offering_state = snap.component_state.setdefault('OfferingRouter', {}) # Use class name?
                  totem_list = offering_state.setdefault('totems', [])
                  if isinstance(totem_list, list):
                       totem_list.append(new_totem)
                       logger.debug("Appended new totem to snap.component_state['OfferingRouter']['totems']")
                       mutated = True
                  else:
                       logger.error("Could not append totem: component_state['OfferingRouter']['totems'] is not a list.")
             # Try dict access last (if snap is just a dict)
             elif isinstance(snap, dict):
                  totem_list = snap.setdefault('totems', [])
                  if isinstance(totem_list, list):
                       totem_list.append(new_totem)
                       logger.debug("Appended new totem to snap['totems'] (dict access)")
                       mutated = True
                  else:
                       logger.error("Could not append totem: snap['totems'] is not a list.")

             if not mutated:
                  raise AttributeError("Could not find a valid 'totems' list attribute or key in snapshot.")

             # Reinforce desire (if possible and enabled)
             if can_update_desire:
                  # Assuming desire_engine directly updates an external cache or needs explicit call
                  # For simplicity, let's assume we just log it here.
                  # A better pattern might be: self.desire_engine.reinforce_want(accepted_suggestion)
                  logger.debug("Reinforcing desire (logging only): %s", accepted_suggestion)
                  # If wants_cache is directly mutable on snapshot:
                  # wants_cache = self._get_snapshot_data(snap, 'wants_cache', {})
                  # if isinstance(wants_cache, dict):
                  #    wants_cache[accepted_suggestion] = wants_cache.get(accepted_suggestion, 0.0) + 0.1

             logger.info("Recorded acceptance for suggestion '%s', issued totem %s.", accepted_suggestion, new_totem['totem_id'])
             return new_totem.copy() # Return copy of the generated totem

        except Exception as e:
             logger.exception("Failed to record acceptance/mutate snapshot: %s", e)
             return {"error": f"State update failed: {e}"}


    # --- Persistence Methods (If OfferingRouter itself needed state) ---
    # def to_dict(self) -> dict:
    #     # --- Feature Flag Check ---
    #     if not is_enabled(Feature.REWARDS): return {}
    #     # --- End Check ---
    #     # Example: return {"some_internal_router_state": self.some_state}
    #     return {} # No internal state to save currently

    # def update_from_dict(self, data: dict):
    #      # --- Feature Flag Check ---
    #      if not is_enabled(Feature.REWARDS): return
    #      # --- End Check ---
    #      # Example: self.some_state = data.get("some_internal_router_state", default_value)
    #      logger.debug("OfferingRouter state updated (if applicable).")
    #      pass # No internal state to load currently
