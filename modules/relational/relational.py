# forest_app/modules/relational.py

import json
import logging
import re
from datetime import datetime, timezone # Added timezone
from typing import Optional, Dict, Any, List # Added List

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("relational_init")
    logger.warning("Feature flags module not found in relational.py. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        RELATIONAL = "FEATURE_ENABLE_RELATIONAL" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

# --- Pydantic Import ---
try:
    from pydantic import BaseModel, Field, ValidationError
    pydantic_import_ok = True
except ImportError:
    logging.getLogger("relational_init").critical("Pydantic not installed. Relational module requires Pydantic.")
    pydantic_import_ok = False
    # Define dummy classes if Pydantic isn't installed
    class BaseModel: pass
    def Field(*args, **kwargs): return None # Dummy Field function
    class ValidationError(Exception): pass

# --- LLM Integration Import ---
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
    logging.getLogger("relational_init").critical(f"Failed to import LLM integration components: {e}. Check llm.py.")
    llm_import_ok = False
    # Define dummy classes
    class LLMClient: pass
    class LLMError(Exception): pass
    class LLMValidationError(LLMError): pass
    class LLMConfigurationError(LLMError): pass
    class LLMConnectionError(LLMError): pass

logger = logging.getLogger(__name__)
# Rely on global config for level

# --- Define Response Models ---
if pydantic_import_ok:
    class RepairActionResponse(BaseModel):
        repair_action: str = Field(..., min_length=1)
        tone: str = Field(default="Gentle") # Add validation if specific tones required
        scale: str = Field(default="Medium") # Add validation if specific scales required

    class ProfileUpdateResponse(BaseModel):
        score_delta: float = Field(default=0.0)
        tag_updates: Dict[str, float] = Field(default_factory=dict)
        love_language: Optional[str] = None # Optional field

    class DeepeningSuggestionResponse(BaseModel):
        deepening_suggestion: str = Field(..., min_length=1)
        tone: str = Field(default="supportive")
else:
     # Dummy versions if Pydantic failed
     class RepairActionResponse: pass
     class ProfileUpdateResponse: pass
     class DeepeningSuggestionResponse: pass


class Profile:
    """
    Represents a profile for relational tracking. Update methods respect
    the RELATIONAL feature flag.
    """
    DEFAULT_CONNECTION_SCORE = 5.0
    DEFAULT_LOVE_LANGUAGE = "Words of Affirmation"

    def __init__(self, name: str):
        self.name: str = name
        self.emotional_tags: Dict[str, float] = {}
        self.love_language: str = self.DEFAULT_LOVE_LANGUAGE
        self.last_gifted: Optional[str] = None
        self.connection_score: float = self.DEFAULT_CONNECTION_SCORE

    def update_emotional_tags(self, new_tags: dict):
        """Updates tags only if the RELATIONAL feature is enabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping Profile.update_emotional_tags for '%s': RELATIONAL feature disabled.", self.name)
            return
        # --- End Check ---
        if not isinstance(new_tags, dict): return
        for tag, value in new_tags.items():
            try:
                current = self.emotional_tags.get(tag, 0.0)
                delta = float(value) # Assuming value itself is the delta or target? Let's assume delta.
                # Clamping ensures score stays within a reasonable range (e.g., 0-10)
                updated = max(0.0, min(10.0, current + delta))
                self.emotional_tags[tag] = round(updated, 2)
            except (ValueError, TypeError): logger.warning("Invalid value for tag '%s': %s", tag, value)
        logger.debug("Profile '%s' emotional_tags updated to %s", self.name, self.emotional_tags)

    def update_connection_score(self, delta: float):
        """Updates score only if the RELATIONAL feature is enabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping Profile.update_connection_score for '%s': RELATIONAL feature disabled.", self.name)
            return
        # --- End Check ---
        try:
            delta_float = float(delta)
            old = self.connection_score
            # Clamp ensures score stays within 0-10
            self.connection_score = max(0.0, min(10.0, self.connection_score + delta_float))
            if self.connection_score != old: # Log only if changed
                 logger.debug("Profile '%s' connection_score: %.2f → %.2f", self.name, old, self.connection_score)
        except (ValueError, TypeError): logger.warning("Invalid delta for connection_score: %s", delta)

    def update_love_language(self, new_love_language: str):
        """Updates language only if the RELATIONAL feature is enabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping Profile.update_love_language for '%s': RELATIONAL feature disabled.", self.name)
            return
        # --- End Check ---
        if isinstance(new_love_language, str) and new_love_language:
            old = self.love_language
            self.love_language = new_love_language
            if self.love_language != old: # Log only if changed
                logger.debug("Profile '%s' love_language: '%s' → '%s'", self.name, old, self.love_language)
        else: logger.warning("Invalid love_language provided: %s", new_love_language)

    def to_dict(self) -> dict:
        # No flag check needed for data serialization
        return {
            "name": self.name,
            "emotional_tags": self.emotional_tags.copy(), # Return copy
            "love_language": self.love_language,
            "last_gifted": self.last_gifted,
            "connection_score": self.connection_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
         # No flag check needed for data deserialization
        if not isinstance(data, dict): return cls("Unknown_Error") # Handle bad data
        profile = cls(data.get("name", "Unknown"))
        # Load safely with defaults and type checks
        profile.emotional_tags = data.get("emotional_tags", {})
        if not isinstance(profile.emotional_tags, dict): profile.emotional_tags = {}

        profile.love_language = data.get("love_language", cls.DEFAULT_LOVE_LANGUAGE)
        if not isinstance(profile.love_language, str): profile.love_language = cls.DEFAULT_LOVE_LANGUAGE

        profile.last_gifted = data.get("last_gifted") # Assumes string or None is ok

        conn_score = data.get("connection_score", cls.DEFAULT_CONNECTION_SCORE)
        try: profile.connection_score = max(0.0, min(10.0, float(conn_score))) # Validate and clamp
        except (ValueError, TypeError): profile.connection_score = cls.DEFAULT_CONNECTION_SCORE

        return profile


class RelationalRepairEngine:
    """Handles generation of repair actions. Respects RELATIONAL feature flag."""

    def generate_repair_action(self, profile: Profile, context: str = "") -> dict:
        """Generates a static fallback repair action. Returns {} if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping generate_repair_action (static): RELATIONAL feature disabled.")
            return {}
        # --- End Check ---
        if not isinstance(profile, Profile): return {}
        # Basic static logic based on dominant tag and score
        dominant_tag = max(profile.emotional_tags.items(), key=lambda kv: kv[1], default=("compassion", 0.0))[0]
        score = profile.connection_score
        if score < 3.0: tone, action = "Cautious", f"Write letter expressing {dominant_tag}."
        elif score < 7.0: tone, action = "Gentle", f"Send note focusing on {dominant_tag}."
        else: tone, action = "Open", f"Reach out for conversation about {dominant_tag}."
        result = {"recipient": profile.name, "tone": tone, "repair_action": action, "emotional_tag": dominant_tag, "context_hint": context, "source": "static"}
        logger.info("Generated static repair action for '%s': %s", profile.name, action)
        return result

    async def generate_dynamic_repair_action(
        self, llm_client: Optional[LLMClient], profile: Profile, snapshot: dict, context: str = "" # LLMClient is optional
    ) -> dict:
        """Generates dynamic repair action via LLM. Returns static fallback if feature disabled or LLM fails."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping generate_dynamic_repair_action: RELATIONAL feature disabled. Returning empty.")
            return {} # Return empty dict directly if feature is off
        # --- End Check ---

        if not isinstance(profile, Profile): return {} # Invalid profile
        # Check LLM client validity if feature is ON
        if not llm_import_ok or not isinstance(llm_client, LLMClient) or not hasattr(llm_client, 'generate'):
             logger.warning("LLMClient not available for dynamic repair action. Using static fallback.")
             return self.generate_repair_action(profile, context) # Static check will return {} if flag off

        # Prune snapshot for prompt brevity
        pruned = {k: snapshot.get(k) for k in ["capacity", "shadow_score", "relationship_index"] if k in snapshot} # Use relationship_index if available

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(RepairActionResponse, BaseModel):
             try: response_model_schema = RepairActionResponse.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for RepairActionResponse")

        prompt = (
            f"Relational Repair Request:\n"
            f"Profile Context: {json.dumps(profile.to_dict(), default=str)}\n" # Safe dump
            f"System Context: {json.dumps(pruned, default=str)}\n" # Safe dump
            f"User Request Hint: {context}\n"
            f"Task: Suggest a single, specific, actionable 'repair_action' suitable for the profile's state (connection_score: {profile.connection_score:.1f}, love_language: '{profile.love_language}', dominant tag: {max(profile.emotional_tags.items(), key=lambda kv: kv[1], default=('N/A', 0))[0]}). "
            f"Assign appropriate 'tone' (e.g., Cautious, Gentle, Open) and 'scale' (e.g., Small, Medium, Large).\n"
            f"Output ONLY valid JSON matching this schema:\n{response_model_schema}\n"
        )
        result = {}
        try:
            llm_response: Optional[RepairActionResponse] = await llm_client.generate(
                prompt_parts=[prompt],
                response_model=RepairActionResponse
            )
            if isinstance(llm_response, RepairActionResponse):
                # Use model_dump() for Pydantic v2+ or .dict() for v1
                if hasattr(llm_response, 'model_dump'):
                     llm_data = llm_response.model_dump()
                else:
                     llm_data = llm_response.dict()

                # Basic validation on returned values (optional, Pydantic should handle)
                tone = llm_data.get("tone", "Gentle")
                scale = llm_data.get("scale", "Medium")
                action = llm_data.get("repair_action", "No specific action suggested.")

                result = {
                    "recipient": profile.name, "tone": tone, "repair_action": action,
                    "scale": scale, "context_hint": context, "source": "dynamic"
                }
                logger.info("Dynamic repair action for '%s': %s", profile.name, action)
            else:
                logger.warning("LLMClient did not return valid RepairActionResponse. Using static fallback.")
                result = self.generate_repair_action(profile, context)

        except (LLMError, LLMValidationError, ValidationError) as llm_e:
            logger.warning("Dynamic repair action LLM/Validation error: %s. Using static fallback.", llm_e)
            result = self.generate_repair_action(profile, context)
        except Exception as e:
            logger.exception("Unexpected error during dynamic repair action: %s. Using static fallback.", e)
            result = self.generate_repair_action(profile, context)

        return result


class RelationalManager:
    """Manages relational profiles and interactions. Respects RELATIONAL feature flag."""

    def __init__(self, llm_client: LLMClient):
        """Initializes the manager with an LLMClient."""
        if not isinstance(llm_client, LLMClient) and llm_import_ok:
            raise TypeError("RelationalManager requires a valid LLMClient instance unless LLM imports failed.")
        self.llm_client = llm_client
        self._repair_engine = RelationalRepairEngine()
        self._reset_state() # Initialize state
        logger.info("RelationalManager initialized.")
        if not llm_import_ok:
            logger.error("LLM Integrations failed import. RelationalManager LLM features disabled.")


    def _reset_state(self):
        """Resets the profiles dictionary."""
        self.profiles: Dict[str, Profile] = {}
        logger.debug("RelationalManager profiles cleared.")


    def add_or_update_profile(self, profile_data: dict) -> Optional[Profile]:
        """Adds or updates profile. Returns None if feature disabled or invalid data."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping add_or_update_profile: RELATIONAL feature disabled.")
            return None
        # --- End Check ---

        if not isinstance(profile_data, dict): return None
        name = profile_data.get("name", "").strip()
        if not name: return None

        profile = None # Initialize profile to None
        if name in self.profiles:
            profile = self.profiles[name]
            # Use Profile's update methods (which also check the flag)
            profile.update_emotional_tags(profile_data.get("emotional_tags", {}))
            if "love_language" in profile_data: profile.update_love_language(profile_data["love_language"])
            if "connection_score_delta" in profile_data: profile.update_connection_score(profile_data["connection_score_delta"])
            # Update last_gifted if present
            if "last_gifted" in profile_data: profile.last_gifted = profile_data["last_gifted"]
        else:
            try:
                profile = Profile.from_dict(profile_data)
                # Ensure name consistency
                if profile.name == "Unknown" and name: profile.name = name
                elif profile.name != name:
                     logger.warning("Profile name mismatch in data ('%s') vs key ('%s'). Using key.", profile.name, name)
                     profile.name = name

                self.profiles[name] = profile
            except Exception as e:
                 logger.error("Failed to create Profile from dict: %s", e); return None

        logger.debug("Profile '%s' added/updated.", name)
        return profile


    def get_profile(self, name: str) -> Optional[Profile]:
        """Gets profile by name. Read-only, no flag check needed here."""
        return self.profiles.get(name)


    def analyze_reflection_for_interactions(self, reflection_text: str) -> dict:
        """Analyzes reflection for basic signals. Returns defaults if feature disabled."""
        default_signals = {"support": 0.0, "conflict": 0.0, "feedback": "Relational analysis disabled."}
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping analyze_reflection_for_interactions: RELATIONAL feature disabled.")
            return default_signals
        # --- End Check ---

        default_signals["feedback"] = "No significant relational signals detected." # Update default msg
        if not reflection_text or not isinstance(reflection_text, str): return default_signals

        text = reflection_text.lower()
        # Example keywords - consider making these configurable
        support_kw = ["support", "helped", "appreciated", "cared", "kind", "grateful", "listened"]
        conflict_kw = ["argued", "conflict", "hurt", "ignored", "criticized", "blamed", "upset", "angry", "frustrated"]

        support_score = sum(0.1 for w in support_kw if re.search(rf"\b{w}\b", text))
        conflict_score = sum(-0.1 for w in conflict_kw if re.search(rf"\b{w}\b", text)) # Negative score

        signals = {"support": round(support_score, 2), "conflict": round(conflict_score, 2), "feedback": ""}

        if signals["support"] > 0 and signals["conflict"] == 0: signals["feedback"] = "Positive relational signals detected."
        elif signals["conflict"] < 0 and signals["support"] == 0: signals["feedback"] = "Negative relational signals detected."
        elif signals["support"] > 0 and signals["conflict"] < 0: signals["feedback"] = "Mixed relational signals detected."
        else: signals["feedback"] = default_signals["feedback"]

        logger.debug("Relational signals analysis: %s", signals)
        return signals


    async def infer_profile_updates(self, profile_name: str, reflection_text: str) -> dict:
        """Uses LLM to infer profile updates. Returns {} if feature disabled or fails."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping infer_profile_updates: RELATIONAL feature disabled.")
            return {}
        # --- End Check ---

        profile = self.get_profile(profile_name)
        if not profile:
             logger.warning("Cannot infer updates, profile '%s' not found.", profile_name)
             return {}

        # Check LLM Client if feature is ON
        if not llm_import_ok or not isinstance(self.llm_client, LLMClient) or not hasattr(self.llm_client, 'generate'):
             logger.error("LLMClient not available for profile update inference.")
             return {}

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(ProfileUpdateResponse, BaseModel):
             try: response_model_schema = ProfileUpdateResponse.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for ProfileUpdateResponse")

        prompt = (
            f"Relational Profile Update Request:\n"
            f"Existing Profile Data: {json.dumps(profile.to_dict(), default=str)}\n"
            f"New User Reflection:\n'''\n{reflection_text}\n'''\n"
            f"Task: Analyze the reflection in context of the profile. Determine appropriate adjustments.\n"
            f"Output ONLY valid JSON matching this schema (use 0.0 delta if no change, null love_language if no change):\n{response_model_schema}\n"
        )
        updates = {}
        try:
            llm_response: Optional[ProfileUpdateResponse] = await self.llm_client.generate(
                prompt_parts=[prompt],
                response_model=ProfileUpdateResponse
            )
            if isinstance(llm_response, ProfileUpdateResponse):
                # Use model_dump() for Pydantic v2+ or .dict() for v1
                if hasattr(llm_response, 'model_dump'):
                     updates = llm_response.model_dump(exclude_none=True)
                else:
                     updates = llm_response.dict(exclude_none=True) # Fallback for Pydantic v1

                # Apply updates using Profile methods (which have flag checks)
                profile.update_connection_score(updates.get("score_delta", 0.0))
                profile.update_emotional_tags(updates.get("tag_updates", {}))
                if "love_language" in updates:
                    profile.update_love_language(updates["love_language"])
                logger.info("Profile '%s' inferred updates applied: %s", profile_name, updates)
            else:
                logger.warning("LLMClient did not return valid ProfileUpdateResponse.")

        except (LLMError, LLMValidationError, ValidationError) as llm_e:
            logger.warning("Profile inference LLM/Validation error: %s", llm_e)
        except Exception as e:
            logger.exception("Unexpected error during profile inference: %s", e)

        return updates # Return the updates dict (or empty if failed)


    async def generate_repair_for_profile(self, name: str, snapshot: dict, context: str = "") -> dict:
        """Generates repair action. Returns {} if feature disabled or fails."""
         # --- Feature Flag Check ---
        # Note: The check is implicitly handled by _repair_engine methods called below,
        # but adding one here provides an earlier exit and clearer logging.
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping generate_repair_for_profile: RELATIONAL feature disabled.")
            return {}
        # --- End Check ---

        profile = self.get_profile(name)
        if not profile: return {}
        # Pass the stored LLMClient to the repair engine method
        # The repair engine method itself checks the flag
        return await self._repair_engine.generate_dynamic_repair_action(self.llm_client, profile, snapshot, context)


    async def generate_deepening_suggestion(self, name: str, snapshot: Optional[dict] = None, context: str = "") -> dict: # snapshot optional
        """Generates deepening suggestion via LLM. Returns fallback if feature disabled or fails."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping generate_deepening_suggestion: RELATIONAL feature disabled.")
            return {"deepening_suggestion": "Relational features currently offline.", "tone": "neutral"}
        # --- End Check ---

        profile = self.get_profile(name)
        if not profile: return {}

        # Check LLM Client if feature is ON
        if not llm_import_ok or not isinstance(self.llm_client, LLMClient) or not hasattr(self.llm_client, 'generate'):
             logger.error("LLMClient not available for deepening suggestion.")
             return {"deepening_suggestion": "Suggestion engine offline.", "tone": "neutral"}

        snapshot = snapshot or {} # Ensure snapshot is a dict
        extra_ctx = {"emotional_tags": profile.emotional_tags, "connection_score": profile.connection_score}
        # Add relevant snapshot context if needed
        extra_ctx["capacity"] = snapshot.get("capacity")
        extra_ctx["shadow_score"] = snapshot.get("shadow_score")

        # Ensure response model is valid before using its schema
        response_model_schema = "{}"
        if pydantic_import_ok and issubclass(DeepeningSuggestionResponse, BaseModel):
             try: response_model_schema = DeepeningSuggestionResponse.model_json_schema(indent=0)
             except Exception: logger.error("Failed to generate Pydantic schema for DeepeningSuggestionResponse")


        prompt = (
            f"Relational Deepening Suggestion Request:\n"
            f"Profile Data: {json.dumps(profile.to_dict(), default=str)}\n"
            f"Additional Context: {json.dumps(extra_ctx, default=str)}\n"
            f"User Request Hint: {context}\n"
            f"Task: Suggest a single, actionable step or reflection prompt to deepen the connection with '{name}', considering their state.\n"
            f"Output ONLY valid JSON matching this schema:\n{response_model_schema}\n"
        )
        suggestion = {}
        fallback = {"deepening_suggestion": "Set aside dedicated time for meaningful connection.", "tone": "gentle"}

        try:
            llm_response: Optional[DeepeningSuggestionResponse] = await self.llm_client.generate(
                prompt_parts=[prompt],
                response_model=DeepeningSuggestionResponse
            )
            if isinstance(llm_response, DeepeningSuggestionResponse):
                 if hasattr(llm_response, 'model_dump'):
                      suggestion = llm_response.model_dump()
                 else:
                      suggestion = llm_response.dict()
                 logger.info("Deepening suggestion for '%s': %s", name, suggestion.get("deepening_suggestion"))
            else:
                logger.warning("LLMClient did not return valid DeepeningSuggestionResponse.")
                suggestion = fallback

        except (LLMError, LLMValidationError, ValidationError) as llm_e:
            logger.warning("Deepening suggestion LLM/Validation error: %s", llm_e)
            suggestion = fallback
        except Exception as e:
            logger.exception("Unexpected error during deepening suggestion: %s", e)
            suggestion = fallback

        return suggestion


    def to_dict(self) -> dict:
        """Serializes state. Returns {} if feature disabled."""
        # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Skipping RelationalManager serialization: RELATIONAL feature disabled.")
            return {}
        # --- End Check ---
        logger.debug("Serializing RelationalManager state.")
        return {"profiles": {n: p.to_dict() for n, p in self.profiles.items()}}


    def update_from_dict(self, data: dict):
        """Rehydrates state. Clears state if feature disabled."""
         # --- Feature Flag Check ---
        if not is_enabled(Feature.RELATIONAL):
            logger.debug("Resetting state via update_from_dict: RELATIONAL feature disabled.")
            self._reset_state()
            return
        # --- End Check ---

        # Feature enabled, proceed
        if not isinstance(data, dict):
             logger.warning("Invalid data type for RelationalManager.update_from_dict: %s", type(data))
             self._reset_state()
             return

        profiles_data = data.get("profiles", {})
        if not isinstance(profiles_data, dict):
             logger.warning("Invalid 'profiles' format in data: Expected dict, got %s.", type(profiles_data))
             self._reset_state()
             return

        loaded_profiles = {}
        loaded_count = 0
        error_count = 0
        for name, pd in profiles_data.items():
            try:
                 # Use Profile.from_dict which handles its own validation
                 prof = Profile.from_dict(pd)
                 # Ensure key matches loaded name
                 if prof.name == "Unknown_Error": # Check for error during Profile creation
                      raise ValueError(f"Profile data for key '{name}' was invalid.")
                 if prof.name != name and prof.name != "Unknown":
                      logger.warning("Profile name mismatch loading relational profiles: key '%s' vs data name '%s'. Using key.", name, prof.name)
                      prof.name = name # Force key name
                 elif prof.name == "Unknown":
                       prof.name = name # Assign key name if default was used

                 loaded_profiles[name] = prof
                 loaded_count += 1
            except Exception as e:
                 logger.error("Failed to load profile '%s': %s", name, e, exc_info=True)
                 error_count += 1

        self.profiles = loaded_profiles
        logger.info("RelationalManager state updated from dict. Loaded %d profiles, errors on %d.", loaded_count, error_count)


# Make classes available for import
__all__ = ["Profile", "RelationalRepairEngine", "RelationalManager"]
