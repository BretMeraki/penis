"""
Google Gemini integration encapsulated in an LLMClient class.
Includes robust error-handling (Retry, Circuit Breaker), async support,
and Pydantic validation of JSON responses. Supports selecting
between standard and advanced models. Uses centralized Pydantic settings.
Includes specific methods for Hierarchical Task Analysis (HTA) evolution
and distilling user reflections.
"""

from __future__ import annotations

# ────────────────────────────── Std-lib ──────────────────────────────
import json
import logging
import re
# MODIFIED: Added List for type hinting
from typing import Any, Optional, Type, TypeVar, Union, Dict, List

# ───────────────────────────── Third-party ───────────────────────────
try:
    import google.generativeai as genai
    from google.generativeai.types import (
        ContentDict, GenerationConfig, GenerateContentResponse,
        HarmBlockThreshold, HarmCategory,
    )
    from google.generativeai import protos
    from google.api_core import exceptions as google_api_exceptions
    google_import_ok = True
except ImportError:
    logging.getLogger(__name__).critical(
        "Failed to import google.generativeai or related components. "
        "Install with: pip install google-generativeai"
    )
    google_import_ok = False
    # Define dummy types to avoid NameErrors if import fails
    class GenerateContentResponse: pass
    class ContentDict: pass
    class GenerationConfig: pass
    class HarmCategory:
        HARM_CATEGORY_HARASSMENT=None
        HARM_CATEGORY_HATE_SPEECH=None
        HARM_CATEGORY_SEXUALLY_EXPLICIT=None
        HARM_CATEGORY_DANGEROUS_CONTENT=None # type: ignore
    class HarmBlockThreshold:
        BLOCK_MEDIUM_AND_ABOVE=None # type: ignore
    # --- MODIFIED: Fixed dummy protos definition ---
    class protos:
        class Candidate:
            class FinishReason:
                STOP=None
                MAX_TOKENS=None
                SAFETY=None
                FINISH_REASON_UNSPECIFIED=None # type: ignore
    # --- END MODIFIED ---
    class google_api_exceptions:
        DeadlineExceeded = Exception
        ServiceUnavailable = Exception
        ResourceExhausted = Exception
        InvalidArgument = Exception
        PermissionDenied = Exception
        NotFound = Exception
        Aborted = Exception
        Unauthenticated = Exception
        InternalServerError = Exception
        GoogleAPIError = Exception

# Use PydanticBaseModel alias to avoid potential conflicts if user defines BaseModel
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, ValidationError as PydanticValidationError
from tenacity import (
    AsyncRetrying, retry_if_exception_type,
    stop_after_attempt, wait_fixed, RetryError,
)

# --- Import pybreaker ---
try:
    from pybreaker import CircuitBreaker, CircuitBreakerError
    pybreaker_import_ok = True
except ImportError:
    logging.getLogger(__name__).error(
        "pybreaker library not found. Circuit breaking disabled. Run 'pip install pybreaker'"
    )
    pybreaker_import_ok = False
    # Dummy classes/functions if pybreaker is not installed
    class CircuitBreakerError(Exception): pass
    class CircuitBreaker:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, func):
            import functools
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Basic pass-through for the dummy
                return await func(*args, **kwargs)
            return wrapper # Return the async wrapper

# ────────────────────────────── Project ──────────────────────────────
# --- Import Central Settings Object ---
try:
    # Assuming settings are in a place accessible like this
    from forest_app.config.settings import settings
    settings_import_successful = True
    _google_api_key = settings.GOOGLE_API_KEY
    _gemini_model_name = settings.GEMINI_MODEL_NAME
    _gemini_advanced_model_name = settings.GEMINI_ADVANCED_MODEL_NAME
    _llm_temperature = settings.LLM_TEMPERATURE
except ImportError as e:
    logging.getLogger(__name__).critical(f"CRITICAL: Failed to import central settings from forest_app.config.settings: {e}")
    settings_import_successful = False; _google_api_key = None
    _gemini_model_name = "gemini-1.5-flash-latest"; _gemini_advanced_model_name = "gemini-1.5-pro-latest"; _llm_temperature = 0.7
except AttributeError as e:
    logging.getLogger(__name__).critical(f"CRITICAL: Missing required attribute in settings object: {e}")
    settings_import_successful = False; _google_api_key = None
    _gemini_model_name = "gemini-1.5-flash-latest"; _gemini_advanced_model_name = "gemini-1.5-pro-latest"; _llm_temperature = 0.7
# --- END IMPORT ---

# --- HTA Model Imports ---
try:
    # Import the base HTA models needed for response structures
    from forest_app.hta_tree.hta_models import HTANodeModel, HTAResponseModel
    hta_models_import_ok = True
except ImportError:
    logging.getLogger(__name__).warning("Failed to import HTA models from forest_app.modules.hta_models. HTA-specific logic might be limited.")
    hta_models_import_ok = False
    # Define dummy classes to avoid NameErrors if import fails
    class BaseModelPlaceholder(PydanticBaseModel): pass # Inherit from Pydantic BaseModel for compatibility
    class HTANodeModel(BaseModelPlaceholder):
        # Add dummy fields if needed for type checking, adjust as per actual model
        id: str = "dummy_id"
        label: str = "dummy_label" # Assuming 'label' based on prompt, adjust if 'title'
        children: list = []
        pass
    class HTAResponseModel(BaseModelPlaceholder):
        # Add dummy fields matching the expected structure
        hta_root: Optional[HTANodeModel] = None # Use Optional if it might be missing
        pass
# ─────────────────────────────────

# ───────────────────────────── Logging ───────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────── Custom Exceptions ─────────────────────────
class LLMError(Exception): """Base exception for LLM client errors."""
class LLMValidationError(LLMError):
    """Error during response validation (JSON parsing or Pydantic schema)."""
    def __init__(self, message: str, *, validation_error: Optional[PydanticValidationError] = None, data: Any = None):
        super().__init__(message)
        self.validation_error = validation_error
        self.data = data
class LLMConnectionError(LLMError): """Error connecting to the LLM API."""
class LLMConfigurationError(LLMError): """Error in LLM client configuration."""
class LLMGenerationError(LLMError):
    """Error during the LLM generation process (e.g., empty/blocked response)."""
    def __init__(self, message: str, *, raw_response: Optional[Any] = None):
        super().__init__(message)
        self.raw_response = raw_response

# ─────────────────────── Pydantic Models ─────────────────────────────
# --- General Examples ---
class TaskDetails(PydanticBaseModel): title: str; description: Optional[str] = None
class ArbiterStandardResponse(PydanticBaseModel): task: Optional[TaskDetails] = None; narrative: Optional[str] = None
class SentimentResponseModel(PydanticBaseModel): sentiment_score: float; sentiment_label: str; key_phrases: Optional[list[str]] = None
class SnapshotCodenameResponse(PydanticBaseModel): codename: str

# --- HTA Evolution Specific Model ---
class HTAEvolveResponse(PydanticBaseModel):
    """
    Pydantic model for the response expected from an HTA evolution request.
    """
    # Expecting the LLM to return the root node directly within the 'hta_root' key
    hta_root: Optional[HTANodeModel] = Field(None, validation_alias='hta_root')

# --- Response model for Reflection Distillation ---
class DistilledReflectionResponse(PydanticBaseModel):
    """
    Expected response structure when asking the LLM to distill reflections.
    """
    distilled_text: str = Field(..., description="Concise summary of key themes/goals from reflections for HTA evolution.")

# ──────────────────── JSON Repair Function ────────────────────────
# [fix_json function remains unchanged from previous version]
def fix_json(text: str) -> str:
    """Attempts to repair potentially malformed JSON strings."""
    try:
        from json_repair import repair_json
        logger.debug("Attempting JSON repair with json_repair library.")
        # Ensure return_objects=False to get the string back
        repaired = repair_json(text, return_objects=False)
        logger.debug("json_repair finished.")
        return repaired # type: ignore
    except ImportError:
        logger.warning("json_repair library not found. Falling back to basic regex-based JSON repair.")
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'```$', '', text, flags=re.DOTALL)
        text = text.strip()
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
        open_curly = text.count('{'); close_curly = text.count('}')
        open_square = text.count('['); close_square = text.count(']')
        if open_curly > close_curly: text += '}' * (open_curly - close_curly)
        if open_square > close_square: text += ']' * (open_square - close_square)
        return text.strip()

# ──────────────────── Prompt Templates ─────────────────────────────
# [HTA_EVOLVE_PROMPT_TEMPLATE remains unchanged from previous version]
HTA_EVOLVE_PROMPT_TEMPLATE = """
You are an expert in Hierarchical Task Analysis (HTA).
Your task is to evolve the provided HTA based on a specific goal or a summary of recent user reflections.

**Current HTA Structure (JSON format):**
```json
{current_hta_json}
```

**Evolution Goal / Distilled Reflections:**
{evolution_goal}

**Instructions:**
1. Analyze the current HTA structure and the evolution goal/distilled reflections.
2. Modify the HTA by adding, removing, reordering, or refining nodes (tasks and subtasks) to align with the goal/reflections.
3. Ensure the resulting structure is a valid HTA, maintaining hierarchical consistency.
4. Maintain the same JSON format as the input, specifically the structure expected by the HTANodeModel (id, label/title, children, etc.).
5. Output *only* the complete, evolved HTA structure as a single JSON object, enclosed in ```json ... ``` markers. The root node should be under the key "hta_root".
6. Generate unique IDs for any new nodes (e.g., using a UUID format like `node_xxxxxxxx`).
7. If no changes are necessary based on the goal/reflections, return the original HTA structure under the "hta_root" key.
8. **Enrichment:** If the input HTA has nodes missing `priority` or `magnitude` fields, or if they are invalid (not numbers), add/correct them. Use a default `priority` of 0.5 and a default `magnitude` of 5.0 where needed. Ensure these fields exist in the output for all nodes.

**Evolved HTA Structure (JSON format):**
```json
{{
  "hta_root": {{
    "id": "...",
    "label": "...", // or "title" depending on your model
    "priority": 0.5, // Added/Corrected
    "magnitude": 5.0, // Added/Corrected
    "children": [ ... ] // Recursively nested structure with enriched nodes
  }}
}}
```
"""

# --- Prompt template for distilling reflections ---
DISTILL_REFLECTIONS_PROMPT_TEMPLATE = """
You are an AI assistant helping a user manage their personal growth plan using Hierarchical Task Analysis (HTA).
The user has provided several reflections during their last work cycle (completing a batch of tasks).
Your task is to distill these reflections into a concise summary (1-3 sentences) highlighting the key themes, insights, blockers, or desired changes relevant for potentially updating their HTA plan.
Focus on information that would inform *structural* changes or significant re-prioritization in their plan. Ignore minor status updates or transient feelings unless they indicate a larger shift.

**User Reflections (provided as a list):**
{reflection_list_str}

**Output:**
Provide ONLY a single valid JSON object containing the distilled summary, using the key "distilled_text".

```json
{{
  "distilled_text": "Concise summary of key points relevant for HTA evolution..."
}}
```
"""

# Generic type for validated Pydantic responses
T = TypeVar("T", bound=PydanticBaseModel)

# ====================================================================
#                     LLMClient Class Definition
# ====================================================================
class LLMClient:
    """
    An asynchronous client for interacting with Google Gemini models, featuring:
    - Centralized configuration via Pydantic settings.
    - Automatic retry mechanism for transient API errors.
    - Circuit breaking to prevent hammering a failing service.
    - Pydantic model validation for JSON responses.
    - Optional JSON repair for slightly malformed outputs.
    - Selection between standard and advanced Gemini models.
    - Specific methods for HTA evolution and reflection distillation.
    """
    # [Constants DEFAULT_SAFETY_SETTINGS, DEFAULT_RETRY_EXCEPTIONS remain unchanged]
    DEFAULT_SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    } if google_import_ok else {}

    DEFAULT_RETRY_EXCEPTIONS = (
        LLMConnectionError,
        google_api_exceptions.DeadlineExceeded,
        google_api_exceptions.ServiceUnavailable,
        google_api_exceptions.ResourceExhausted,
        google_api_exceptions.Aborted,
        google_api_exceptions.InternalServerError,
    ) if google_import_ok else (LLMConnectionError,)

    # [__init__ method remains unchanged]
    def __init__(
        self,
        fail_max: int = 5,
        reset_timeout: int = 60,
        api_timeout: int = 180
    ):
        """
        Initializes the LLMClient, configures Google GenAI, and sets up
        the circuit breaker.
        """
        logger.debug("Initializing LLMClient...")
        self.api_timeout = api_timeout

        if not google_import_ok:
            raise ImportError("google.generativeai library is required but not found.")

        # --- Configuration from Settings ---
        if not settings_import_successful:
            logger.warning("Settings import failed. Using hardcoded defaults. THIS IS NOT RECOMMENDED.")
        if not _google_api_key:
            raise LLMConfigurationError("GOOGLE_API_KEY is missing or empty in settings.")

        self.api_key = _google_api_key
        self.standard_model_name = _gemini_model_name
        self.advanced_model_name = _gemini_advanced_model_name
        self.default_temperature = _llm_temperature

        if not self.standard_model_name:
                 raise LLMConfigurationError("GEMINI_MODEL_NAME is missing or empty in settings.")

        # --- Configure Google GenAI ---
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Google GenAI configured successfully.")
            logger.info("Standard Model: %s", self.standard_model_name)
            logger.info("Advanced Model: %s", self.advanced_model_name or "Not Set")
            logger.info("Default Temperature: %s", self.default_temperature)
        except Exception as e:
            logger.exception("Failed to configure Google GenAI.")
            raise LLMConfigurationError(f"Google GenAI configuration failed: {e}") from e

        # --- Setup Circuit Breaker ---
        if pybreaker_import_ok:
            self.circuit_breaker = CircuitBreaker(fail_max=fail_max, reset_timeout=reset_timeout)
            logger.info(f"Circuit Breaker enabled (fail_max={fail_max}, reset_timeout={reset_timeout}s).")
        else:
            self.circuit_breaker = CircuitBreaker() # Dummy
            logger.warning("Circuit Breaker is disabled (pybreaker not installed).")

        logger.debug("LLMClient initialized successfully.")

    # --- Core Private Helper Methods ---
    # [_get_model_instance, _create_generation_config, _execute_gemini_request,
    #  _process_response, _parse_and_validate_json methods remain unchanged]
    def _get_model_instance(self, use_advanced_model: bool) -> genai.GenerativeModel:
        """Selects and returns the configured Gemini model instance."""
        model_name_to_use = self.standard_model_name
        if use_advanced_model:
            if self.advanced_model_name:
                model_name_to_use = self.advanced_model_name
                logger.debug("Using ADVANCED Gemini model: %s", model_name_to_use)
            else:
                logger.warning(
                    "Advanced model requested but not configured in settings. "
                    "Falling back to standard model: %s", model_name_to_use
                )
        else:
             logger.debug("Using STANDARD Gemini model: %s", model_name_to_use)

        try:
            return genai.GenerativeModel(model_name_to_use)
        except Exception as e:
            logger.exception(f"Failed to instantiate GenerativeModel '{model_name_to_use}'")
            raise LLMConfigurationError(f"Failed to create Gemini model instance '{model_name_to_use}': {e}") from e

    def _create_generation_config(
        self,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        json_mode: bool,
    ) -> GenerationConfig:
        """Creates the GenerationConfig object for the API call."""
        safe_temperature = max(0.0, min(float(temperature), 1.0))
        config = GenerationConfig(
            temperature=safe_temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
        )
        if json_mode:
            config.response_mime_type = "application/json"
            logger.debug("GenerationConfig: JSON mode enabled (response_mime_type='application/json').")
        return config

    async def _execute_gemini_request(
        self,
        model: genai.GenerativeModel,
        prompt_parts: list[Union[str, ContentDict]],
        generation_config: GenerationConfig,
        safety_settings: dict,
        retries: int,
        retry_wait: int,
    ) -> GenerateContentResponse:
        """
        Executes the asynchronous call to the Gemini API with retry logic.
        Handles specific Google API exceptions and wraps them in LLMError types.
        """
        retryer = AsyncRetrying(
            stop=stop_after_attempt(retries + 1),
            wait=wait_fixed(retry_wait),
            retry=retry_if_exception_type(self.DEFAULT_RETRY_EXCEPTIONS),
            reraise=True,
        )
        try:
            response: GenerateContentResponse = await retryer(
                model.generate_content_async,
                prompt_parts,
                generation_config=generation_config,
                safety_settings=safety_settings,
                request_options={'timeout': self.api_timeout}
            )
            return response
        except RetryError as e:
            logger.error(f"LLM request failed after {retries} retries: {e.cause}")
            final_exception = e.cause
            if isinstance(final_exception, (google_api_exceptions.DeadlineExceeded, google_api_exceptions.ServiceUnavailable, google_api_exceptions.Aborted)):
                raise LLMConnectionError(f"API call failed after retries: {final_exception}") from final_exception
            elif isinstance(final_exception, google_api_exceptions.ResourceExhausted):
                raise LLMError(f"Resource exhausted after retries (rate limit?): {final_exception}") from final_exception
            elif isinstance(final_exception, google_api_exceptions.InternalServerError):
                raise LLMError(f"Google internal server error after retries: {final_exception}") from final_exception
            else:
                raise LLMError(f"Unhandled retryable error after retries: {final_exception}") from final_exception
        except google_api_exceptions.InvalidArgument as e:
            if "API key not valid" in str(e): raise LLMConfigurationError("Invalid Google API key provided.") from e
            if "model" in str(e).lower() and "not found" in str(e).lower(): raise LLMConfigurationError(f"Invalid model name '{model.model_name}'? Error: {e}") from e
            if "application/json" in str(e).lower() and "mime type" in str(e).lower():
                logger.error(f"Model '{model.model_name}' may not support JSON mode (mime type). Error: {e}")
                raise LLMConfigurationError(f"Model '{model.model_name}' does not support JSON mode. Error: {e}") from e
            raise LLMGenerationError(f"Invalid argument passed to Google API: {e}") from e
        except google_api_exceptions.PermissionDenied as e: raise LLMConfigurationError(f"Google API permission denied: {e}") from e
        except google_api_exceptions.NotFound as e: raise LLMConfigurationError(f"Google API resource not found (check model name '{model.model_name}'): {e}") from e
        except google_api_exceptions.Unauthenticated as e: raise LLMConfigurationError(f"Google API authentication failed: {e}") from e
        except google_api_exceptions.GoogleAPIError as e:
            logger.error(f"Unhandled Google API error: {type(e).__name__} - {e}")
            raise LLMError(f"A Google API error occurred: {e}") from e
        except Exception as e:
            logger.exception("Unexpected error during Gemini API call.")
            raise LLMError(f"An unexpected error occurred during API execution: {e}") from e

    def _process_response(self, response: GenerateContentResponse) -> str:
        """
        Validates the raw GenerateContentResponse and extracts the text content.
        Raises: LLMGenerationError: If the response is blocked, empty, or malformed.
        """
        try:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name
                logger.error(f"Gemini request blocked due to prompt content. Reason: {reason}")
                raise LLMGenerationError(f"Gemini request blocked by API. Reason: {reason}", raw_response=response)

            if not response.candidates:
                reason = getattr(getattr(response, 'prompt_feedback', None), 'block_reason', "UNKNOWN").name
                logger.error(f"Gemini response has no candidates. Prompt block reason: {reason}")
                raise LLMGenerationError(f"Gemini response contained no candidates (request might be blocked: {reason})", raw_response=response)

            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED)

            if finish_reason == protos.Candidate.FinishReason.SAFETY:
                safety_ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
                logger.error(f"Gemini response stopped due to SAFETY. Ratings: {safety_ratings_str}")
                raise LLMGenerationError(f"Gemini response stopped due to safety policy.", raw_response=candidate)
            if finish_reason not in {protos.Candidate.FinishReason.STOP, protos.Candidate.FinishReason.MAX_TOKENS, protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED}:
                 finish_message = getattr(candidate, 'finish_message', '')
                 logger.warning(f"Gemini finished with unexpected reason: {finish_reason.name}. Message: {finish_message}")

            if not (hasattr(candidate, 'content') and candidate.content and
                    hasattr(candidate.content, 'parts') and candidate.content.parts):
                 if finish_reason != protos.Candidate.FinishReason.STOP and finish_reason != protos.Candidate.FinishReason.MAX_TOKENS:
                     logger.error(f"Gemini candidate has no content parts. Finish reason: {finish_reason.name}")
                     raise LLMGenerationError(f"Gemini response candidate is empty or lacks content parts (Finish reason: {finish_reason.name})", raw_response=candidate)
                 else:
                     logger.warning(f"Gemini candidate has no content parts, but finish reason was {finish_reason.name}. Returning empty string.")
                     return ""

            try:
                response_text = candidate.content.parts[0].text
                if response_text is None:
                    raise AttributeError("Text attribute is None")
                return response_text
            except (IndexError, AttributeError, TypeError) as e:
                logger.error(f"Could not extract text content from Gemini response part: {e}", exc_info=True)
                raise LLMGenerationError(f"Failed to extract text from response part: {e}", raw_response=candidate)

        except Exception as e:
             if isinstance(e, LLMGenerationError): raise
             logger.exception("Unexpected error while processing Gemini response.")
             raise LLMGenerationError(f"Unexpected error processing response: {e}") from e

    def _parse_and_validate_json(
        self,
        raw_text: str,
        response_model: Type[T],
        attempt_repair: bool = True
    ) -> T:
        """
        Parses the raw text as JSON, optionally attempts repair, and validates
        against the provided Pydantic model. Includes HTA structure checks.
        """
        cleaned_text = raw_text.strip()
        if not cleaned_text:
            logger.error("Received empty text content for JSON parsing.")
            raise LLMValidationError("Received empty response content, cannot parse JSON.", data=raw_text)

        cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned_text = re.sub(r'```$', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = cleaned_text.strip()

        if not cleaned_text:
             logger.error("Response content was empty after removing markdown fences.")
             raise LLMValidationError("Response empty after removing markdown fences.", data=raw_text)

        data: Any = None
        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError as initial_decode_error:
            logger.warning(f"Initial JSON parsing failed: {initial_decode_error}. Raw text snippet: '{cleaned_text[:100]}...'")
            if not attempt_repair:
                raise LLMValidationError(f"JSON decode error: {initial_decode_error}", data=cleaned_text) from initial_decode_error

            logger.info("Attempting JSON repair...")
            repaired_json_text = fix_json(cleaned_text)
            try:
                data = json.loads(repaired_json_text)
                logger.info("Successfully parsed repaired JSON.")
            except json.JSONDecodeError as repair_error:
                logger.error(f"JSON parsing failed even after repair attempt: {repair_error}")
                logger.debug(f"Original text (cleaned): {cleaned_text}")
                logger.debug(f"Repaired text attempt: {repaired_json_text}")
                raise LLMValidationError(
                    f"Invalid JSON structure even after repair attempt: {repair_error}",
                    data={'original': cleaned_text, 'repaired': repaired_json_text}
                ) from repair_error

        # --- Special Handling for HTA Models ---
        is_hta_target_model = False
        if hta_models_import_ok:
             if response_model is HTAEvolveResponse or response_model is HTAResponseModel or issubclass(response_model, HTAResponseModel):
                 is_hta_target_model = True

        if is_hta_target_model:
            if isinstance(data, dict) and "hta_root" not in data:
                found_dynamic_root = next((key for key in data if key.startswith("root_")), None)
                if found_dynamic_root:
                    logger.warning(f"HTA response missing 'hta_root', found dynamic key '{found_dynamic_root}'. Renaming.")
                    data["hta_root"] = data.pop(found_dynamic_root)
                else:
                    logger.warning(f"Parsing for {response_model.__name__}, but 'hta_root' key (or 'root_...' dynamic key) is missing in the JSON data.")
            elif isinstance(data, dict) and "hta_root" in data and data["hta_root"] is None:
                 logger.warning(f"HTA response has 'hta_root' key, but its value is null.")

        # --- Pydantic Validation ---
        try:
            validated_data = response_model.model_validate(data)
            logger.debug(f"Successfully validated response against {response_model.__name__}.")
            return validated_data
        except PydanticValidationError as e:
            logger.error(f"Pydantic validation failed for model {response_model.__name__}. Errors: {e.errors()}")
            try: data_preview = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception: data_preview = str(data)
            logger.debug(f"Data that failed validation:\n{data_preview}")
            raise LLMValidationError(
                f"Response schema mismatch for {response_model.__name__}.",
                validation_error=e, data=data
            ) from e
        except Exception as e:
             logger.exception(f"Unexpected error during Pydantic validation ({response_model.__name__}).")
             raise LLMValidationError(f"Unexpected validation error: {e}", data=data) from e

    # --- Main Public Method: generate ---
    # [generate method implementation remains unchanged]
    async def generate(
        self,
        prompt_parts: list[Union[str, ContentDict]],
        response_model: Type[T],
        *, # Keyword-only arguments follow
        use_advanced_model: bool = False,
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        top_k: int = 32,
        max_output_tokens: int = 8192,
        json_mode: bool = True, # Must be True if response_model is used
        retries: int = 3,
        retry_wait: int = 2,
        attempt_json_repair: bool = True
    ) -> T:
        """
        Generates content using the configured Gemini model, applying retry,
        circuit breaking, and Pydantic validation.
        """
        if not google_import_ok:
             raise ImportError("Cannot generate content, google.generativeai library not available.")
        if response_model and not json_mode:
            raise TypeError("A 'response_model' was provided, but 'json_mode' is False. Set json_mode=True for validation.")

        async def _protected_generation():
            model = self._get_model_instance(use_advanced_model)
            effective_temp = temperature if temperature is not None else self.default_temperature
            gen_config = self._create_generation_config(
                temperature=effective_temp, top_p=top_p, top_k=top_k,
                max_output_tokens=max_output_tokens, json_mode=json_mode,
            )
            safety_settings = self.DEFAULT_SAFETY_SETTINGS
            logger.info(
                f"Sending request to Gemini ({model.model_name}) -> {response_model.__name__}. "
                f"Temp={effective_temp:.1f}, MaxTokens={max_output_tokens}, Retries={retries}"
            )
            if json_mode: logger.debug("Expecting JSON response.")
            raw_response = await self._execute_gemini_request(
                model=model, prompt_parts=prompt_parts, generation_config=gen_config,
                safety_settings=safety_settings, retries=retries, retry_wait=retry_wait,
            )
            response_text = self._process_response(raw_response)
            validated_response = self._parse_and_validate_json(
                raw_text=response_text, response_model=response_model,
                attempt_repair=attempt_json_repair
            )
            logger.info(f"Successfully generated and validated {response_model.__name__} response.")
            return validated_response

        try:
            if pybreaker_import_ok and isinstance(self.circuit_breaker, CircuitBreaker) and not isinstance(self.circuit_breaker, type(CircuitBreaker())):
                 return await self.circuit_breaker.call_async(_protected_generation)
            else:
                 return await _protected_generation()
        except CircuitBreakerError as cbe:
            logger.error(f"LLM Circuit Breaker is OPEN. Request rejected: {cbe}")
            raise
        except LLMError:
            raise
        except Exception as e:
            logger.exception("An unexpected error occurred in the main generate method.")
            raise LLMError(f"Unexpected error during generation: {e}") from e

    # --- Public Method: request_hta_evolution ---
    # [request_hta_evolution method remains unchanged]
    async def request_hta_evolution(
        self,
        current_hta_json: str,
        evolution_goal: str,
        *,
        use_advanced_model: bool = False,
        temperature: Optional[float] = 0.5,
        retries: int = 3,
        retry_wait: int = 2,
        attempt_json_repair: bool = True
    ) -> HTAEvolveResponse:
        """
        Requests the LLM to evolve a given HTA structure based on a goal.
        """
        if not hta_models_import_ok:
             raise LLMConfigurationError("Cannot request HTA evolution: HTANodeModel/HTAResponseModel not imported correctly.")
        logger.info(f"Requesting HTA evolution. Goal: '{evolution_goal[:50]}...'")
        try:
            try: json.loads(current_hta_json)
            except json.JSONDecodeError as json_err:
                logger.error(f"Invalid JSON provided for current_hta_json: {json_err}")
                raise ValueError("The provided current_hta_json is not valid JSON.") from json_err
            prompt = HTA_EVOLVE_PROMPT_TEMPLATE.format(
                current_hta_json=current_hta_json, evolution_goal=evolution_goal
            )
            prompt_parts = [prompt]
        except Exception as e:
            logger.exception("Failed to format HTA evolution prompt.")
            raise LLMConfigurationError(f"Error formatting HTA evolution prompt: {e}") from e

        evolved_hta_response = await self.generate(
            prompt_parts=prompt_parts,
            response_model=HTAEvolveResponse,
            use_advanced_model=use_advanced_model,
            temperature=temperature,
            max_output_tokens=8192,
            retries=retries,
            retry_wait=retry_wait,
            attempt_json_repair=attempt_json_repair,
            json_mode=True
        )
        return evolved_hta_response

    # --- MODIFIED: Added Method for Reflection Distillation ---
    async def distill_reflections(
        self,
        reflections: List[str],
        *,
        use_advanced_model: bool = False, # Standard model likely sufficient
        temperature: Optional[float] = 0.3, # Lower temp for focused summary
        retries: int = 2, # Fewer retries might be acceptable
        retry_wait: int = 1,
        attempt_json_repair: bool = True
    ) -> Optional[DistilledReflectionResponse]:
        """
        Distills a list of user reflections into a concise summary for HTA evolution.

        Args:
            reflections: A list of reflection strings.
            use_advanced_model: If True, uses the advanced Gemini model.
            temperature: The generation temperature (creativity).
            retries: Number of retries for transient errors.
            retry_wait: Wait time between retries.
            attempt_json_repair: Whether to attempt fixing malformed JSON.

        Returns:
            A DistilledReflectionResponse object containing the summary, or None on error.
        """
        if not reflections:
            logger.info("No reflections provided to distill.")
            return None

        logger.info(f"Requesting distillation of {len(reflections)} reflections.")

        # Format reflections for the prompt (e.g., numbered list)
        # Ensure each reflection is treated as a separate item
        reflection_list_str = "\n".join(f"- {r.strip()}" for i, r in enumerate(reflections) if r and r.strip())
        if not reflection_list_str:
             logger.warning("Filtered reflections list is empty.")
             return None # Nothing to distill

        # 1. Format the prompt
        try:
            prompt = DISTILL_REFLECTIONS_PROMPT_TEMPLATE.format(
                reflection_list_str=reflection_list_str
            )
            prompt_parts = [prompt]
        except Exception as e:
            logger.exception("Failed to format reflection distillation prompt.")
            # Return None or raise specific error? Returning None for now.
            return None

        # 2. Call the generic generate method
        try:
            distilled_response = await self.generate(
                prompt_parts=prompt_parts,
                response_model=DistilledReflectionResponse, # Use the new response model
                use_advanced_model=use_advanced_model,
                temperature=temperature,
                max_output_tokens=512, # Summary should be concise
                retries=retries,
                retry_wait=retry_wait,
                attempt_json_repair=attempt_json_repair,
                json_mode=True # Required for Pydantic validation
            )
            logger.info("Successfully received distilled reflection response from LLM.")
            return distilled_response
        except LLMError as e:
            # Catch errors specifically from self.generate
            logger.error(f"LLMError during reflection distillation request: {e}")
            return None # Return None on LLM errors for this specific task
        except Exception as e:
            # Catch any other unexpected errors during the call
            logger.exception("Unexpected error during reflection distillation request.")
            return None # Return None on other errors
    # --- END MODIFIED ---

    # --- Other existing methods (get_sentiment, get_snapshot_codename, etc.) ---
    # [Implementations remain unchanged from previous version]
    async def get_sentiment(self, text: str) -> Optional[SentimentResponseModel]:
         logger.info("Requesting sentiment analysis.")
         prompt = f"""
Analyze the sentiment of the following text. Output as JSON: {{"sentiment_score": float, "sentiment_label": str, "key_phrases": ["phrase1", ...]}}
Text: {text}
JSON Output: ```json {{ ... json ... }} ```"""
         try:
             return await self.generate( [prompt], SentimentResponseModel, use_advanced_model=False, temperature=0.2)
         except LLMError as e: logger.error(f"LLMError: {e}"); return None

    async def get_snapshot_codename(self, context: str) -> Optional[SnapshotCodenameResponse]:
         logger.info("Requesting snapshot codename.")
         prompt = f"""
Generate a short (2-3 words) codename for a project state based on context. Examples: 'Quiet Dawn', 'Steady Ascent'. Output as JSON: {{"codename": "Generated Codename"}}
Context: {context}
JSON Output: ```json {{ ... json ... }} ```"""
         try:
             return await self.generate( [prompt], SnapshotCodenameResponse, use_advanced_model=False, temperature=0.8)
         except LLMError as e: logger.error(f"LLMError: {e}"); return None

    async def get_narrative(self, context: str) -> Optional[ArbiterStandardResponse]:
         logger.info("Requesting narrative.")
         prompt = f"""
You are the Forest Arbiter. Reflect on the user's situation and provide a short narrative (2-4 sentences) and optionally suggest a next task title. Output as JSON: {{"narrative": "...", "task": {{"title": "Optional Task Title"}}}}
Situation: {context}
JSON Output: ```json {{ ... json ... }} ```"""
         try:
             return await self.generate( [prompt], ArbiterStandardResponse, use_advanced_model=False, temperature=0.7, max_output_tokens=1024)
         except LLMError as e: logger.error(f"LLMError: {e}"); return None

    async def generate_hta_tree(self, context: str) -> Optional[HTAResponseModel]:
        logger.info("Requesting initial HTA generation.")
        prompt = f"""
Create an HTA tree for the goal in the context. Output as JSON with key "hta_root". Nodes need "id", "label", "children". Optional: "description", "priority", "magnitude", "depends_on", "is_milestone".
- **Enrichment:** Ensure all nodes in the output have `priority` (default 0.5) and `magnitude` (default 5.0) fields with valid numeric values.
Context: {context}
JSON Output: ```json {{"hta_root": {{ ... tree ... }} }} ```"""
        if not hta_models_import_ok: raise LLMConfigurationError("HTA models not imported.")
        try:
            return await self.generate( [prompt], HTAResponseModel, use_advanced_model=True, temperature=0.6, max_output_tokens=8192)
        except LLMError as e: logger.error(f"LLMError: {e}"); return None

# ====================================================================
#                         End LLMClient Class
# ====================================================================

# [Example Usage remains largely unchanged, but could add distillation example]
async def example_usage():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Starting LLMClient example...")
    # [Dummy HTA Data remains unchanged]
    initial_hta_data = {"hta_root": {"id": "root_0", "label": "Make Tea", "children": [{"id": "task_1", "label": "Boil Water", "children": [{"id": "sub_1.1", "label": "Fill Kettle", "children": []}, {"id": "sub_1.2", "label": "Switch On", "children": []}]}, {"id": "task_2", "label": "Prepare Cup", "children": [{"id": "sub_2.1", "label": "Add Teabag", "children": []}]}, {"id": "task_3", "label": "Combine", "children": []}]}}
    initial_hta_json = json.dumps(initial_hta_data)
    evolution_goal_example = "Refine 'Prepare Cup' and 'Combine'. Add steps for milk, sugar, pouring, stirring, removing teabag."
    # --- ADDED: Example reflections ---
    reflections_example = [
        "Felt a bit rushed making tea today, didn't enjoy it.",
        "Maybe I should focus on the ritual aspect more.",
        "Need to remember to add sugar BEFORE the milk next time.",
        "The boiling step is straightforward."
    ]

    try:
        client = LLMClient()
        # --- Example 1: Distill Reflections ---
        logger.info("\n--- Requesting Reflection Distillation ---")
        distilled_goal = evolution_goal_example # Fallback goal
        try:
            distilled = await client.distill_reflections(reflections=reflections_example)
            if distilled and distilled.distilled_text:
                logger.info(f"Distilled Reflections: {distilled.distilled_text}")
                # Use this distilled text as the goal for evolution
                distilled_goal = distilled.distilled_text
            else:
                logger.warning("Distillation failed or returned None/empty. Using fallback goal.")

        except LLMError as e: logger.error(f"Distillation failed: {e}")
        except CircuitBreakerError: logger.error("Circuit breaker open, skipping distillation.")

        # --- Example 2: Request HTA Evolution (using distilled goal) ---
        logger.info("\n--- Requesting HTA Evolution (using distilled goal) ---")
        try:
            evolved_hta = await client.request_hta_evolution(
                current_hta_json=initial_hta_json,
                evolution_goal=distilled_goal, # Use the distilled goal here
                use_advanced_model=False
            )
            logger.info("Successfully received evolved HTA.")
            if evolved_hta and evolved_hta.hta_root:
                 evolved_json = evolved_hta.hta_root.model_dump_json(indent=2) # Pydantic v2
                 logger.info(f"Evolved HTA Root:\n{evolved_json}")
            else: logger.warning("Evolved HTA response missing 'hta_root'.")
        except LLMValidationError as e: logger.error(f"HTA Evo failed validation: {e}")
        except LLMError as e: logger.error(f"HTA Evo failed: {e}")
        except CircuitBreakerError: logger.error("Circuit breaker open, skipping HTA evo.")
        except ValueError as e: logger.error(f"HTA Evo input error: {e}")

        # --- Example 3: Codename Generation ---
        # [Codename example remains unchanged]
        logger.info("\n--- Generating Codename ---")
        try:
            codename_result = await client.get_snapshot_codename(context="Project state after refining tea process.")
            if codename_result: logger.info(f"Generated Codename: {codename_result.codename}")
            else: logger.warning("Codename generation failed.")
        except LLMError as e: logger.error(f"Codename generation failed: {e}")
        except CircuitBreakerError: logger.error("Circuit breaker open.")

    except LLMConfigurationError as e: logger.critical(f"Init failed (Config): {e}")
    except ImportError as e: logger.critical(f"Init failed (Import): {e}")
    except Exception as e: logger.critical(f"Unexpected error: {e}", exc_info=True)

if __name__ == "__main__":
    import asyncio
    # [Dummy Settings Setup remains unchanged]
    if not settings_import_successful:
        logger.warning("USING DUMMY SETTINGS FOR EXAMPLE RUN - REPLACE WITH YOUR CONFIG")
        class DummySettings:
            GOOGLE_API_KEY = None
            GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
            GEMINI_ADVANCED_MODEL_NAME = "gemini-1.5-pro-latest"
            LLM_TEMPERATURE = 0.7
        import os
        settings = DummySettings()
        settings.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", None)
        _google_api_key = settings.GOOGLE_API_KEY
        _gemini_model_name = settings.GEMINI_MODEL_NAME
        _gemini_advanced_model_name = settings.GEMINI_ADVANCED_MODEL_NAME
        _llm_temperature = settings.LLM_TEMPERATURE
        settings_import_successful = True

    if not _google_api_key:
         logger.critical("\nCRITICAL: GOOGLE_API_KEY is not set.")
    elif not hta_models_import_ok:
         logger.critical("\nCRITICAL: HTA models could not be imported.")
    else:
         logger.info("API key found, HTA models imported/dummy defined. Running example...")
         asyncio.run(example_usage())
