# forest_app/modules/desire_engine.py

import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# --- Import Feature Flags ---
try:
    # Assumes feature_flags.py is accessible
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger.warning("Feature flags module not found in desire_engine. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        # Define the specific flag used in this module
        DESIRE_ENGINE = "FEATURE_ENABLE_DESIRE_ENGINE"
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True # Or False, based on desired fallback behavior

# --- Pydantic and LLM Imports ---
from pydantic import BaseModel, Field
from forest_app.integrations.llm import LLMClient, LLMError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WantsResponse(BaseModel):
    """Pydantic model for parsing LLM response."""
    wants: List[str] = Field(..., description="List of inferred wants/needs.")


class DesireEngine:
    """
    Tracks and manages the user's long‑term wants and needs ("desires").
    Respects the DESIRE_ENGINE feature flag.
    Uses an LLM to extract and update key desires from free‑form input,
    and persists a cache of accepted wants to inform reward suggestions.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initializes the DesireEngine.

        Args:
            llm_client: An instance of the LLMClient for making API calls.
        """
        self.wants_cache: List[Dict[str, Any]] = []
        self.llm_client = llm_client # Store the injected client
        logger.debug("DesireEngine initialized.") # Removed 'with LLMClient' for brevity

    def update_from_dict(self, data: Dict[str, Any]):
        """
        Rehydrate state from snapshot.component_state['desire_engine'].
        Clears state if DESIRE_ENGINE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DESIRE_ENGINE):
            logger.debug("Clearing state via update_from_dict: DESIRE_ENGINE feature disabled.")
            self.wants_cache = []
            return
        # --- End Check ---

        if not isinstance(data, dict):
             logger.warning("Invalid data format for update_from_dict: %s. State not updated.", type(data))
             self.wants_cache = [] # Reset on invalid data
             return

        cache = data.get("wants_cache")
        if isinstance(cache, list):
            # Basic validation could be added here to check structure of list items if needed
            self.wants_cache = cache
            logger.debug("DesireEngine state loaded: %d wants", len(self.wants_cache))
        else:
            logger.warning("Invalid 'wants_cache' type in data: %s. Resetting cache.", type(cache))
            self.wants_cache = [] # Reset if cache data is invalid


    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state for persistence. Returns empty dict if
        DESIRE_ENGINE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DESIRE_ENGINE):
            logger.debug("Skipping DesireEngine serialization: DESIRE_ENGINE feature disabled.")
            return {}
        # --- End Check ---

        logger.debug("Serializing DesireEngine state.")
        return {"wants_cache": list(self.wants_cache)} # Return a copy


    def add_want(self, want_text: str) -> Dict[str, Any]:
        """
        Manually record a new want/need. Returns the record added.
        Does nothing and returns empty dict if DESIRE_ENGINE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DESIRE_ENGINE):
            logger.debug("Skipping add_want: DESIRE_ENGINE feature disabled.")
            return {}
        # --- End Check ---

        if not isinstance(want_text, str) or not want_text.strip():
            logger.warning("Attempted to add empty or invalid want text.")
            return {}

        record = {
            "want": want_text.strip(),
            "timestamp": datetime.utcnow().isoformat()
        }
        self.wants_cache.append(record)
        logger.info("Added new want: %r", want_text.strip())
        return record

    def get_all_wants(self) -> List[str]:
        """
        Retrieve the list of all recorded wants (texts only).
        Returns current cache state; cache will be empty if feature disabled during load.
        """
        # No feature flag check needed here - just returns current state.
        # State will be empty if feature was disabled during update_from_dict.
        return [entry.get("want", "") for entry in self.wants_cache if isinstance(entry, dict) and "want" in entry]


    async def infer_wants(self, user_text: str, max_wants: int = 5) -> List[str]:
        """
        Uses the injected LLMClient to extract key desires. Appends *new* wants
        to the cache and returns the list of newly added wants.
        Does nothing and returns empty list if DESIRE_ENGINE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DESIRE_ENGINE):
            logger.debug("Skipping infer_wants: DESIRE_ENGINE feature disabled.")
            return []
        # --- End Check ---

        if not isinstance(user_text, str) or not user_text.strip():
             logger.debug("Skipping infer_wants: Empty user text provided.")
             return []

        prompt = (
            f"You are an assistant that extracts the user's key wants or needs "
            f"from a free-form statement. Respond ONLY with a valid JSON object matching "
            f"the schema: {{\"wants\": [list of up to {max_wants} concise string phrases]}}.\n\n"
            f"User input:\n\"\"\"\n{user_text}\n\"\"\"\n\n"
            "JSON Output:"
        )
        wants_list = []
        try:
            response: WantsResponse = await self.llm_client.generate(
                prompt_parts=[prompt],
                response_model=WantsResponse,
                use_advanced_model=False
            )
            wants_list = response.wants if response else []

        except LLMError as e:
            logger.warning("DesireEngine inference failed (LLM Error): %s", e)
            wants_list = []
        except Exception as e:
            logger.error("DesireEngine inference failed (Unexpected Error): %s", e, exc_info=True)
            wants_list = []

        new_wants_added = []
        # Use set for efficient checking of existing wants
        current_wants_set = set(self.get_all_wants())

        for want in wants_list:
            if isinstance(want, str):
                normalized = want.strip()
                if normalized and normalized not in current_wants_set:
                    # Call internal add_want - it already handles logging and timestamping
                    # Note: add_want itself checks the feature flag, but we already passed
                    # the check at the start of this method, so it's safe to call.
                    self.add_want(normalized)
                    new_wants_added.append(normalized)
                    # Add to local set immediately to prevent duplicates within the same LLM response
                    current_wants_set.add(normalized)

        if new_wants_added: # Only log if something was actually added
             logger.info("Inferred and added %d new wants from user text.", len(new_wants_added))
        else:
             logger.debug("No new wants inferred or added from user text.")

        return new_wants_added


    def clear_wants(self):
        """
        Remove all recorded wants. Does nothing if DESIRE_ENGINE feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.DESIRE_ENGINE):
            logger.debug("Skipping clear_wants: DESIRE_ENGINE feature disabled.")
            return
        # --- End Check ---

        count = len(self.wants_cache)
        if count > 0: # Only log if something was actually cleared
             self.wants_cache.clear()
             logger.info("Cleared %d wants from cache", count)
        else:
             logger.debug("clear_wants called, but cache was already empty.")
