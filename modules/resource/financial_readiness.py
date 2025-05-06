# forest_app/modules/financial_readiness.py

import logging
import json
from datetime import datetime, timezone # Use timezone aware datetime
from typing import Dict, Any, Optional

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("fin_readiness_init")
    logger.warning("Feature flags module not found in financial_readiness. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        FINANCIAL_READINESS = "FEATURE_ENABLE_FINANCIAL_READINESS" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

# --- Pydantic Import ---
try:
    from pydantic import BaseModel, Field, ValidationError
    pydantic_import_ok = True
except ImportError:
    logging.getLogger("fin_readiness_init").critical("Pydantic not installed. FinancialReadinessEngine requires Pydantic.")
    pydantic_import_ok = False
    # Define dummy classes if Pydantic isn't installed
    class BaseModel: pass
    def Field(*args, **kwargs): return None # Dummy Field function
    class ValidationError(Exception): pass

# --- LLM Import ---
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
    logging.getLogger("fin_readiness_init").critical(f"Failed to import LLM integration components: {e}. Check llm.py.")
    llm_import_ok = False
    # Define dummy classes
    class LLMClient: pass
    class LLMError(Exception): pass
    class LLMValidationError(LLMError): pass
    class LLMConfigurationError(LLMError): pass
    class LLMConnectionError(LLMError): pass

logger = logging.getLogger(__name__)
# Rely on global config for level

# --- Helper ---
def _clamp01(x: float) -> float:
    """Clamp a float to the 0.0–1.0 range."""
    # Add type check for robustness
    if not isinstance(x, (int, float)):
        logger.warning(f"Invalid type for clamping: {type(x)}. Returning 0.5.")
        return 0.5 # Return midpoint default if type is wrong
    return max(0.0, min(1.0, float(x)))

# --- Define Response Models ---
# Only define if Pydantic import was successful
if pydantic_import_ok:
    class BaselineReadinessResponse(BaseModel):
        readiness: float = Field(..., ge=0.0, le=1.0) # Ensure value is within range

    class ReflectionDeltaResponse(BaseModel):
        delta: float # Delta could theoretically be outside -0.2 to 0.2, let apply_updates handle clamping
else:
     # Dummy versions if Pydantic failed
     class BaselineReadinessResponse: pass
     class ReflectionDeltaResponse: pass


class FinancialReadinessEngine:
    """
    Assesses and tracks the user's financial readiness level (0.0–1.0).
    Uses an LLM for assessment based on reflections or context updates.
    Respects the FINANCIAL_READINESS feature flag.
    """
    DEFAULT_READINESS = 0.5 # Default initial readiness

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the engine.

        Args:
            llm_client: An instance of the LLMClient for making calls.
        """
        if not isinstance(llm_client, LLMClient) and llm_import_ok:
            raise TypeError("FinancialReadinessEngine requires a valid LLMClient instance unless LLM imports failed.")
        self.llm_client = llm_client
        self._reset_state() # Initialize using helper
        logger.info("FinancialReadinessEngine initialized.")
        if not llm_import_ok:
            logger.error("LLM Integration components failed import. Financial readiness analysis will not function.")

    def _reset_state(self):
        """Resets readiness to default and clears timestamp."""
        self.readiness: float = self.DEFAULT_READINESS
        self.last_update: Optional[str] = None # Start with None or current time? None seems better for tracking actual updates.
        # self.last_update: str = datetime.now(timezone.utc).isoformat()
        logger.debug("Financial Readiness state reset to default.")


    def update_from_dict(self, data: Dict[str, Any]):
        """
        Rehydrate engine state. Resets state if FINANCIAL_READINESS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.FINANCIAL_READINESS):
            logger.debug("Resetting state via update_from_dict: FINANCIAL_READINESS feature disabled.")
            self._reset_state()
            return
        # --- End Check ---

        # Feature enabled, proceed with loading
        if not isinstance(data, dict):
            logger.warning("Invalid data type for FinancialReadinessEngine.update_from_dict: %s. Resetting state.", type(data))
            self._reset_state()
            return

        try:
            loaded_readiness = data.get("readiness", self.DEFAULT_READINESS)
            self.readiness = _clamp01(float(loaded_readiness)) # Clamp on load

            loaded_ts = data.get("last_update")
            if isinstance(loaded_ts, str):
                # Validate ISO format on load
                try:
                    datetime.fromisoformat(loaded_ts.replace('Z', '+00:00'))
                    self.last_update = loaded_ts
                except ValueError:
                    logger.warning("Invalid last_update format '%s' received, resetting timestamp.", loaded_ts)
                    self.last_update = None # Reset if invalid format
            else:
                 self.last_update = None # Reset if missing or wrong type

            logger.debug("Loaded FinancialReadinessEngine state: readiness=%.2f", self.readiness)
        except Exception as e:
            logger.error("Error loading FinancialReadinessEngine state: %s. Resetting state.", e, exc_info=True)
            self._reset_state()


    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize engine state. Returns empty dict if FINANCIAL_READINESS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.FINANCIAL_READINESS):
            logger.debug("Skipping FinancialReadinessEngine serialization: FINANCIAL_READINESS feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing FinancialReadinessEngine state.")
        return {
            "readiness": self.readiness,
            "last_update": self.last_update
        }

    async def assess_baseline(self, description: str) -> float:
        """
        Perform initial baseline assessment using LLM. Returns current readiness
        without LLM call or state update if FINANCIAL_READINESS feature is disabled.

        Returns:
            The readiness score (0.0-1.0) - either newly assessed or current state.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.FINANCIAL_READINESS):
            logger.debug("Skipping assess_baseline: FINANCIAL_READINESS feature disabled. Returning current readiness %.2f", self.readiness)
            return self.readiness # Return current value without processing
        # --- End Check ---

        # Feature enabled, check LLM client
        if not llm_import_ok or not isinstance(self.llm_client, LLMClient) or not hasattr(self.llm_client, 'generate'):
             logger.error("LLMClient not available for Financial Readiness baseline assessment. Returning current readiness %.2f", self.readiness)
             return self.readiness

        if not isinstance(description, str) or not description.strip():
             logger.warning("Empty description provided for baseline assessment. Returning current readiness %.2f", self.readiness)
             return self.readiness

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(BaselineReadinessResponse, BaseModel):
             try: response_model_schema = BaselineReadinessResponse.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for BaselineReadinessResponse")

        prompt = (
            "You are an objective assistant that evaluates a user's financial readiness "
            "for pursuing meaningful goals, on a scale from 0.0 (not ready) to 1.0 (fully ready). "
            "Based *only* on the following description, respond *only* with a valid JSON object matching this schema:\n"
            f'{response_model_schema}\n\n'
            f"User description:\n\"\"\"\n{description}\n\"\"\"\n"
        )

        new_readiness: Optional[float] = None
        try:
            llm_response: Optional[BaselineReadinessResponse] = await self.llm_client.generate(
                prompt_parts=[prompt],
                response_model=BaselineReadinessResponse
            )
            if isinstance(llm_response, BaselineReadinessResponse):
                 # Pydantic model already validated range [0,1]
                new_readiness = llm_response.readiness
                logger.debug("LLM baseline assessment returned readiness: %.3f", new_readiness)
            else:
                 logger.warning("LLM baseline assessment did not return a valid BaselineReadinessResponse.")

        except (LLMError, LLMValidationError, LLMConfigurationError, LLMConnectionError, ValidationError) as llm_e:
            logger.warning("Baseline financial readiness assessment failed: %s. Keeping previous readiness %.2f", llm_e, self.readiness, exc_info=False)
        except Exception as e:
            logger.exception("Unexpected error during baseline readiness assessment: %s. Keeping previous readiness %.2f", e, self.readiness)

        # Only update state if LLM call was successful and yielded a value
        if new_readiness is not None:
            self.readiness = _clamp01(new_readiness) # Clamp again just in case
            self.last_update = datetime.now(timezone.utc).isoformat()
            logger.info("Baseline readiness updated to %.3f", self.readiness)
        else:
             logger.info("Baseline readiness assessment failed or returned no value. Readiness remains %.3f", self.readiness)

        return self.readiness


    async def analyze_reflection(self, reflection: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Adjust readiness based on reflection using LLM delta. Returns current readiness
        without LLM call or state update if FINANCIAL_READINESS feature is disabled.

        Returns:
            The readiness score (0.0-1.0) - either updated or current state.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.FINANCIAL_READINESS):
            logger.debug("Skipping analyze_reflection: FINANCIAL_READINESS feature disabled. Returning current readiness %.2f", self.readiness)
            return self.readiness # Return current value without processing
        # --- End Check ---

        # Feature enabled, check LLM client
        if not llm_import_ok or not isinstance(self.llm_client, LLMClient) or not hasattr(self.llm_client, 'generate'):
             logger.error("LLMClient not available for Financial Readiness reflection analysis. Returning current readiness %.2f", self.readiness)
             return self.readiness

        if not isinstance(reflection, str) or not reflection.strip():
             logger.warning("Empty reflection provided for readiness analysis. Returning current readiness %.2f", self.readiness)
             return self.readiness

        context_str = json.dumps(context, default=str) if context else "{}" # Safe dump

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(ReflectionDeltaResponse, BaseModel):
             try: response_model_schema = ReflectionDeltaResponse.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for ReflectionDeltaResponse")

        prompt = (
            "You are an assistant analyzing how a user's new financial reflections "
            "should adjust their current financial readiness (0.0–1.0). "
            "Based *only* on the reflection and optional context, respond *only* with a valid JSON object matching this schema:\n"
            f'{response_model_schema}\n\n' # Use generated schema
            "User reflection:\n\"\"\"\n"
            f"{reflection}\n\"\"\"\n\n"
            "Optional Context:\n"
            f"{context_str}\n"
        )

        delta: Optional[float] = None
        try:
            llm_response: Optional[ReflectionDeltaResponse] = await self.llm_client.generate(
                 prompt_parts=[prompt],
                 response_model=ReflectionDeltaResponse
            )
            if isinstance(llm_response, ReflectionDeltaResponse):
                 delta = llm_response.delta # Pydantic validated type
                 logger.debug("LLM reflection analysis returned delta: %.3f", delta)
            else:
                 logger.warning("LLM reflection analysis did not return a valid ReflectionDeltaResponse.")

        except (LLMError, LLMValidationError, LLMConfigurationError, LLMConnectionError, ValidationError) as llm_e:
            logger.warning("Financial readiness reflection analysis failed: %s. No change to readiness %.2f", llm_e, self.readiness, exc_info=False)
        except Exception as e:
            logger.exception("Unexpected error during readiness reflection analysis: %s. No change to readiness %.2f", e, self.readiness)

        # Only update state if LLM call was successful and yielded a delta
        if delta is not None:
            self.readiness = _clamp01(self.readiness + delta)
            self.last_update = datetime.now(timezone.utc).isoformat()
            logger.info("Adjusted readiness via reflection to %.3f", self.readiness)
        else:
             logger.info("Readiness reflection analysis failed or returned no delta. Readiness remains %.3f", self.readiness)

        return self.readiness
