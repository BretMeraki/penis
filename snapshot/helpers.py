# forest_app/helpers.py

import logging
import json
import os
import sys # For stderr
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Dict, Any, List

# --- SQLAlchemy Imports ---
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# --- Core Components ---
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.core.processors.reflection_processor import prune_context # Helper from reflection_processor

# --- Persistence Components ---
from forest_app.snapshot.repository import MemorySnapshotRepository
from forest_app.snapshot.models import MemorySnapshotModel

# --- LLM & Pydantic Imports ---
# Assume these imports are correct based on your provided code
from forest_app.integrations.llm import (
    LLMClient,
    SnapshotCodenameResponse,
    LLMError,
    LLMValidationError,
    LLMConfigurationError,
    LLMConnectionError
)
from pydantic import BaseModel, Field

# --- Constants ---
try:
    from forest_app.config import constants
except ImportError:
    class ConstantsPlaceholder:
        MAX_CODENAME_LENGTH = 60
        DEFAULT_RESONANCE_THEME = "neutral"
    constants = ConstantsPlaceholder()

logger = logging.getLogger(__name__)

# <<< --- ADDED HELPER FUNCTION --- >>>
# (Same helper as added to routers/core.py for consistency)
def find_node_in_dict(node_dict: Optional[Dict], node_id_to_find: str) -> Optional[Dict]:
    """Helper to recursively find a node dictionary by ID within a nested structure."""
    if not isinstance(node_dict, dict):
        return None
    if node_dict.get("id") == node_id_to_find:
        return node_dict
    children = node_dict.get("children")
    if isinstance(children, list):
        for child in children:
            found = find_node_in_dict(child, node_id_to_find)
            if found:
                return found
    return None
# <<< --- END ADDED HELPER FUNCTION --- >>>


# --- Updated Helper Function Signature ---
async def save_snapshot_with_codename(
    db: Session,
    repo: MemorySnapshotRepository,
    user_id: int,
    snapshot: MemorySnapshot, # Input is the full MemorySnapshot object
    llm_client: LLMClient,
    stored_model: Optional[MemorySnapshotModel],
    force_create_new: bool = False,
) -> Optional[MemorySnapshotModel]:
    """
    Saves or updates a snapshot model using the provided repository and session.
    Generates a codename via the injected LLMClient. Assumes the db transaction
    is managed by the caller. NOW CALLS record_feature_flags().

    Args:
        db: The SQLAlchemy Session.
        repo: The MemorySnapshotRepository.
        user_id: The ID of the user.
        snapshot: The MemorySnapshot object to save.
        llm_client: The LLMClient instance for generating the codename.
        stored_model: The existing MemorySnapshotModel if updating, else None.
        force_create_new: If True, forces creation of a new record.

    Returns:
        The newly created or updated MemorySnapshotModel object, or None on failure.
    """

    # --- CRITICAL: Record feature flags BEFORE serializing ---
    try:
        logger.debug("Calling snapshot.record_feature_flags()...")
        snapshot.record_feature_flags() # Populate the feature_flags dict
        logger.debug("Finished snapshot.record_feature_flags().")
    except Exception as ff_err:
        # Log error but proceed, snapshot might be saved without flags
        logger.error("Error calling record_feature_flags(): %s", ff_err, exc_info=True)
    # --- End Feature Flag Recording ---

    # --- Serialize the snapshot ONCE after recording flags ---
    try:
        updated_data = snapshot.to_dict()
        logger.debug("Snapshot serialized successfully after recording flags.")
    except Exception as dict_err:
        logger.error("Error calling snapshot.to_dict(): %s. Cannot save snapshot.", dict_err, exc_info=True)
        return None # Cannot proceed without serialized data
    # --- End Serialization ---


    # Logging snapshot data (keep as is)
    try:
        # Log the data we actually plan to save
        # logger.debug("SAVE_SNAPSHOT: Data prepared by snapshot.to_dict():\n%s",
        #              json.dumps(updated_data, indent=2, default=str)) # Can uncomment if needed, but verbose
        # Check presence of hta_tree in the serialized data
        core_state_check = updated_data.get('core_state', 'MISSING_CORE_STATE')
        if isinstance(core_state_check, dict):
            hta_tree_check = core_state_check.get('hta_tree', 'MISSING_HTA_TREE')
            logger.debug("SAVE_SNAPSHOT: core_state['hta_tree'] presence check BEFORE save: %s",
                         'PRESENT' if hta_tree_check != 'MISSING_HTA_TREE' else hta_tree_check)
        else:
            logger.debug("SAVE_SNAPSHOT: core_state presence check BEFORE save: %s", core_state_check)
        # Log the feature flags that were (hopefully) recorded
        logger.debug("SAVE_SNAPSHOT: Recorded feature_flags: %s", updated_data.get('feature_flags', 'MISSING_OR_ERROR'))
    except Exception as log_err:
        logger.error("SAVE_SNAPSHOT: Error logging snapshot data: %s", log_err)


    generated_codename: str = f"Snapshot_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}" # Fallback

    # --- Generate codename (Use injected LLMClient) ---
    try:
        logger.info("Attempting to generate codename for snapshot via LLMClient...")
        # Use the already serialized data for pruning context
        prompt_context = prune_context(updated_data) # Use updated_data
        theme = constants.DEFAULT_RESONANCE_THEME
        # Access component_state from the serialized dict
        if isinstance(updated_data.get('component_state'), dict):
            theme = updated_data['component_state'].get("last_resonance_theme", theme)
        prompt_context["resonance_theme"] = theme

        codename_prompt = (
            f"You are a helpful assistant specialized in creating concise, evocative codenames (2-5 words) "
            f"for user growth journey snapshots based on their current state. Use title case. "
            f"Analyze the provided context:\n{json.dumps(prompt_context, indent=2, default=str)}\n" # Added default=str
            f"Based *only* on the context, generate a suitable codename. "
            f'Return ONLY a valid JSON object in the format: {{"codename": "Generated Codename Here"}}'
        )

        llm_response: Optional[SnapshotCodenameResponse] = await llm_client.generate(
            prompt_parts=[codename_prompt],
            response_model=SnapshotCodenameResponse
        )

        if isinstance(llm_response, SnapshotCodenameResponse) and llm_response.codename:
            temp_codename = llm_response.codename.strip()[:constants.MAX_CODENAME_LENGTH]
            if temp_codename:
                generated_codename = temp_codename
                logger.info("LLM generated codename: '%s'", generated_codename)
            else:
                logger.warning("LLM returned an empty codename after stripping. Using fallback.")
        else:
            logger.warning("LLM did not return a valid SnapshotCodenameResponse object or codename was empty. Using fallback.")

    except (LLMError, LLMValidationError, LLMConfigurationError, LLMConnectionError) as llm_e:
        logger.warning("LLM call for codename failed: %s. Using fallback.", llm_e, exc_info=False)
    except Exception as e:
        logger.exception("Unexpected error generating codename: %s. Using fallback.", e)


    # --- Save or Update Snapshot Model Object ---
    new_or_updated_model: Optional[MemorySnapshotModel] = None
    action = "create" if force_create_new or not stored_model else "update"
    log_action_verb = "Prepared new" if action == "create" else "Prepared update for"

    # <<< --- ADDED LOGGING --- >>>
    try:
        # Extract the HTA node ID from the context if possible (this is brittle)
        # A better approach would be to pass the completed task_id down to this helper
        # For now, we'll try to log the whole tree's status summary if the specific ID isn't available
        hta_tree_to_log = updated_data.get('core_state', {}).get('hta_tree')
        if isinstance(hta_tree_to_log, dict) and 'root' in hta_tree_to_log:
             log_data_str = json.dumps(hta_tree_to_log, indent=2, default=str)
             if len(log_data_str) > 1000: log_data_str = log_data_str[:1000] + "... (truncated)"
             logger.debug(f"[HELPER PRE-REPO] Serialized core_state hta_tree structure being sent to repo:\n{log_data_str}")
        else:
            logger.debug("[HELPER PRE-REPO] Serialized core_state hta_tree is missing or invalid before repo call.")

    except Exception as log_ex:
        logger.error(f"[HELPER PRE-REPO] Error logging HTA state before repo call: {log_ex}")
    # <<< --- END ADDED LOGGING --- >>>


    try:
        if action == "create":
            if not isinstance(user_id, int): raise TypeError(f"User ID must be int, got: {type(user_id)}")
            # Pass the serialized data dict
            new_or_updated_model = repo.create_snapshot(user_id, updated_data, generated_codename)
        else: # action == "update"
            stored_user_id = getattr(stored_model, "user_id", None)
            if stored_user_id != user_id:
                logger.error("CRITICAL: User ID mismatch during update! Stored: %s, Requested: %d.", stored_user_id, user_id)
                raise ValueError("User ID mismatch during snapshot update.")
            # Pass the serialized data dict
            new_or_updated_model = repo.update_snapshot(stored_model, updated_data, generated_codename)

        if new_or_updated_model:
            model_id_for_log = getattr(new_or_updated_model, "id", "N/A")
            logger.info("%s snapshot model object for user ID %d (Model ID: %s, Codename: '%s'). Awaiting commit.",
                        log_action_verb, user_id, model_id_for_log, generated_codename)
        else:
            logger.error("Repository method (%s) failed to return snapshot model for user ID %d.", action, user_id)

    except (ValueError, TypeError) as val_err:
        logger.error(f"Error preparing snapshot data for User ID={user_id}, Action={action}: {val_err}", exc_info=True)
        raise # Re-raise validation errors

    except SQLAlchemyError as db_err: # Catch potential DB errors during repo interaction
         logger.error(f"Database error during snapshot {action} for User ID={user_id}: {db_err}", exc_info=True)
         raise db_err # Re-raise DB errors

    except Exception as e: # Catch any other unexpected errors
         logger.error(f"Unexpected error during snapshot {action} for User ID={user_id}: {e}", exc_info=True)
         raise e # Re-raise other errors

    return new_or_updated_model

# Other functions in helpers.py would remain unchanged...
