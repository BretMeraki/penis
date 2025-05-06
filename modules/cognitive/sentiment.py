# forest_app/modules/sentiment.py
"""
Module for sentiment analysis using Google Gemini, potentially combined with other methods.
Includes fallback logic and enhanced logging for failures.
Refactored to use Pydantic models for input/output contracts and injected LLMClient.
Respects the SENTIMENT_ANALYSIS feature flag.
"""
import logging
import json
from typing import Optional, Dict, Any

# --- Import Feature Flags ---
try:
    # Assumes feature_flags.py is accessible
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if feature flags module isn't found
    logger = logging.getLogger("sentiment_init") # Ensure logger is defined early for warning
    logger.warning("Feature flags module not found in sentiment. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        SENTIMENT_ANALYSIS = "FEATURE_ENABLE_SENTIMENT_ANALYSIS" # Define the specific flag used here
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, based on desired fallback behavior


from pydantic import BaseModel, Field

# --- LLM Imports with Fallback ---
try:
    from forest_app.integrations.llm import (
        LLMClient,
        SentimentResponseModel, # Expected structure from LLM
        LLMError,
        LLMValidationError,
        LLMGenerationError
    )
    llm_import_ok = True
except ImportError as e:
    # Use the init logger here as well
    logging.getLogger("sentiment_init").critical(f"Failed to import LLM integration components: {e}. Check llm.py.")
    llm_import_ok = False
    # Define dummy classes
    class LLMClient: pass
    class SentimentResponseModel(BaseModel): # Dummy Pydantic model
         sentiment_score: float = 0.0
         sentiment_label: str = "neutral"
         key_phrases: Optional[list[str]] = None
    class LLMError(Exception): pass
    class LLMValidationError(LLMError): pass
    class LLMGenerationError(LLMError): pass
# --- End LLM Imports ---

# Define logger for the rest of the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Or adjust as needed


# --- Pydantic Contracts (Input/Output for this Engine) ---
class SentimentInput(BaseModel):
    text_to_analyze: str = Field(..., description="The text content to be analyzed for sentiment.")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for analysis (currently unused by this engine).")

class SentimentOutput(BaseModel):
    score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Sentiment score from -1 (negative) to 1 (positive). Defaults to 0.0 on error.")
    label: str = Field(default="neutral", description="Categorical label (e.g., 'positive', 'negative', 'neutral'). Defaults to 'neutral' on error.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score of the analysis, if available from LLM.")
    rationale: Optional[str] = Field(default=None, description="Explanation or rationale from the LLM, if available.")
    model_used: Optional[str] = Field(default=None, description="Identifier of the model used for analysis.")
    key_phrases: Optional[list[str]] = Field(default=None, description="Key phrases influencing sentiment, if available from LLM.")
    error_message: Optional[str] = Field(default=None, description="Description of error if analysis failed.")

# Define the default neutral output instance
NEUTRAL_SENTIMENT_OUTPUT = SentimentOutput(score=0.0, label="neutral", model_used="fallback")


class SecretSauceSentimentEngineHybrid:
    """
    Analyzes sentiment using primarily Google Gemini LLM via an injected LLMClient,
    with fallback to a neutral SentimentOutput if LLM fails or feature is disabled.
    Includes enhanced logging. Uses Pydantic models for input and output contracts.
    """

    # --- Class attribute for default prompt modifier ---
    DEFAULT_PROMPT_MODIFIER = 1.0

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the sentiment engine.

        Args:
            llm_client: An instance of the LLMClient for making API calls.
        """
        self.llm_client = llm_client
        self.prompt_modifier: float = self.DEFAULT_PROMPT_MODIFIER # Initialize state
        logger.info("SecretSauceSentimentEngineHybrid initialized.")
        if not llm_import_ok:
            logger.error("LLM Integration components failed import. Sentiment analysis may always use fallback.")


    async def analyze_emotional_field(self, input_data: SentimentInput) -> SentimentOutput:
        """
        Analyzes the sentiment of the provided text using the injected LLMClient.
        Returns neutral output if SENTIMENT_ANALYSIS feature is disabled.

        Args:
            input_data: A Pydantic model containing the text and optional context.

        Returns:
            A Pydantic SentimentOutput model containing the analysis results or neutral fallback.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SENTIMENT_ANALYSIS):
            logger.debug("Skipping sentiment analysis: SENTIMENT_ANALYSIS feature disabled. Returning neutral.")
            return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": "Sentiment analysis feature disabled"})
        # --- End Check ---

        # Check LLMClient validity (important if feature is ON)
        if not isinstance(self.llm_client, LLMClient) or not llm_import_ok or not hasattr(self.llm_client, 'generate'):
            logger.warning("LLMClient unavailable/invalid, returning neutral sentiment fallback.")
            return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": "LLMClient not available or invalid"})

        if not input_data.text_to_analyze or not isinstance(input_data.text_to_analyze, str) or not input_data.text_to_analyze.strip():
            logger.debug("Received empty or invalid text for sentiment analysis. Returning neutral.")
            return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": "Input text was empty or invalid"})

        text = input_data.text_to_analyze

        try:
            # Apply prompt modifier if needed (example - adjust prompt based on state)
            # modified_prompt_instruction = f"Analyze sentiment intensity (modifier {self.prompt_modifier}): "
            prompt = (
                # modified_prompt_instruction +
                f"Analyze the sentiment of the following text. Provide a sentiment score between -1.0 (very negative) "
                f"and 1.0 (very positive), a sentiment label ('positive', 'negative', or 'neutral'), and optionally "
                f"a list of key phrases influencing the sentiment. Respond ONLY with a valid JSON object matching the "
                f"SentimentResponseModel schema: {SentimentResponseModel.model_json_schema()}.\n"
                f"Text to analyze:\n\"\"\"\n{text}\n\"\"\""
            )

            logger.debug("Requesting LLM sentiment analysis...")
            llm_response: Optional[SentimentResponseModel] = await self.llm_client.generate(
                prompt_parts=[prompt],
                response_model=SentimentResponseModel,
                use_advanced_model=False
            )

            if isinstance(llm_response, SentimentResponseModel):
                logger.debug(f"LLM sentiment analysis successful: Score={llm_response.sentiment_score}, Label='{llm_response.sentiment_label}'")
                # Create SentimentOutput from the successful LLM response
                return SentimentOutput(
                    score=llm_response.sentiment_score,
                    label=llm_response.sentiment_label,
                    key_phrases=llm_response.key_phrases,
                    model_used="gemini-llm" # More specific model if known
                )
            else:
                logger.error(f"LLMClient.generate returned unexpected type: {type(llm_response)}. Using neutral fallback.")
                return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": f"LLM returned unexpected type: {type(llm_response)}"})

        except (LLMError, LLMValidationError, LLMGenerationError) as llm_error:
            text_snippet = (text[:250] + '...') if len(text) > 250 else text
            logger.warning(
                f"LLM sentiment analysis failed. ErrorType: {type(llm_error).__name__}, Error: {llm_error}. "
                f"Falling back to neutral sentiment. Text snippet: '{text_snippet}'",
                exc_info=False
            )
            return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": f"LLM Error: {type(llm_error).__name__} - {llm_error}"})

        except Exception as e:
            text_snippet = (text[:250] + '...') if len(text) > 250 else text
            logger.error(
                f"Unexpected error during sentiment analysis. ErrorType: {type(e).__name__}, Error: {e}. "
                f"Falling back to neutral sentiment. Text snippet: '{text_snippet}'",
                exc_info=True
            )
            return NEUTRAL_SENTIMENT_OUTPUT.model_copy(update={"error_message": f"Unexpected Error: {type(e).__name__} - {e}"})


    # --- State persistence methods ---
    def to_dict(self) -> dict:
        """
        Serializes state for persistence. Returns empty dict if
        SENTIMENT_ANALYSIS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SENTIMENT_ANALYSIS):
            logger.debug("Skipping SentimentEngine serialization: SENTIMENT_ANALYSIS feature disabled.")
            return {}
        # --- End Check ---
        # Only save state if the feature is ON
        logger.debug("Serializing SentimentEngine state.")
        return {"prompt_modifier": getattr(self, 'prompt_modifier', self.DEFAULT_PROMPT_MODIFIER)}

    def update_from_dict(self, data: dict):
        """
        Loads state from snapshot data. Resets state if SENTIMENT_ANALYSIS feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.SENTIMENT_ANALYSIS):
            logger.debug("Resetting state via update_from_dict: SENTIMENT_ANALYSIS feature disabled.")
            self.prompt_modifier = self.DEFAULT_PROMPT_MODIFIER # Reset to default
            return
        # --- End Check ---

        # Feature is enabled, proceed with loading
        if isinstance(data, dict):
            try:
                # Load and validate prompt_modifier
                loaded_modifier = data.get("prompt_modifier", self.DEFAULT_PROMPT_MODIFIER)
                self.prompt_modifier = float(loaded_modifier) # Attempt conversion
                # Add clamping or range checks if necessary for prompt_modifier
                # self.prompt_modifier = max(0.1, min(2.0, self.prompt_modifier)) # Example clamp
                logger.debug("SentimentEngine state updated from dict.")
            except (ValueError, TypeError) as e:
                 logger.warning("Invalid 'prompt_modifier' value in data: %s. Using default. Error: %s", loaded_modifier, e)
                 self.prompt_modifier = self.DEFAULT_PROMPT_MODIFIER # Fallback to default
        else:
            logger.warning(
                "Invalid data type provided to SentimentEngine.update_from_dict: Expected dict, got %s. State not updated.", type(data)
            )
            # Reset to default if data structure is wrong
            self.prompt_modifier = self.DEFAULT_PROMPT_MODIFIER
