# forest_app/modules/trigger_phrase.py

import json
import logging
import os
import re # Import regex for parsing IDs
from datetime import datetime, timezone # Added timezone
from typing import Optional, Dict, Any # Added typing

# --- Import Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    logger = logging.getLogger("trigger_phrase_init")
    logger.warning("Feature flags module not found in trigger_phrase. Feature flag checks will be disabled.")
    class Feature: # Dummy class
        TRIGGER_PHRASES = "FEATURE_ENABLE_TRIGGER_PHRASES" # Define the specific flag
    def is_enabled(feature: Any) -> bool: # Dummy function
        logger.warning("is_enabled check defaulting to TRUE due to missing feature flags module.")
        return True

# Imports needed for type hints if using snapshot in handlers
# from forest_app.core.snapshot import MemorySnapshot # Can likely be removed if snapshot not truly needed by handlers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Default empty trigger map if feature disabled or load fails ---
DEFAULT_EMPTY_TRIGGER_MAP = {}

def load_trigger_config():
    """Loads trigger phrase mappings."""
    # --- Feature Flag Check ---
    # Decide if config loading itself should be skipped if feature is off
    # If skipped, the __init__ needs to handle the feature flag check instead
    # Let's assume we only check the flag when *handling* triggers, not loading config.
    # if not is_enabled(Feature.TRIGGER_PHRASES):
    #     logger.debug("Skipping trigger config load: TRIGGER_PHRASES feature disabled.")
    #     return DEFAULT_EMPTY_TRIGGER_MAP
    # --- End Check ---

    config_path = os.path.join("forest_app", "config", "trigger_config.json")
    # Define default triggers including new ones
    defaults = {
       "activate the forest": "activate",
       "forest, change the decor": "change_decor",
       "forest, audit the scores": "audit_scores",
       "forest, show me the running to-do list": "show_todo",
       "forest, integrate memory": "integrate_memory",
       "forest save now": "save_snapshot",
       "forest list saves": "list_snapshots",
       # Load and delete are handled by regex
    }
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                # Ensure default keys exist if file is incomplete
                for key, value in defaults.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                logger.info("Loaded trigger config from %s", config_path)
                return loaded_config
        else:
            logger.warning("trigger_config.json not found at %s. Using defaults.", config_path)
            return defaults.copy() # Return a copy
    except Exception as e:
        logger.error( # Changed to error level
            "Could not load or parse trigger configuration from %s: %s. Using default mapping.",
            config_path, e, exc_info=True # Added exc_info
        )
        return defaults.copy() # Return a copy

class TriggerPhraseHandler:
    """
    Handles simple command trigger phrases. Respects the TRIGGER_PHRASES feature flag.
    """

    def __init__(self):
        self.trigger_map = load_trigger_config()
        # Pre-compile regex patterns
        try:
            self.load_pattern = re.compile(r"forest load save (\d+)", re.IGNORECASE)
            self.delete_pattern = re.compile(r"forest delete save (\d+)", re.IGNORECASE)
        except re.error as re_err:
             logger.error("Failed to compile trigger phrase regex patterns: %s", re_err)
             # Set patterns that won't match anything if compilation fails
             self.load_pattern = re.compile(r"$^") # Matches nothing
             self.delete_pattern = re.compile(r"$^")

        # Handlers map actions to simple functions returning intent/messages
        self.handlers = {
            "activate": self._handle_activate,
            "change_decor": self._handle_change_decor,
            "audit_scores": self._handle_audit_scores,
            "show_todo": self._handle_show_todo,
            "integrate_memory": self._handle_integrate_memory,
            "save_snapshot": self._handle_save_snapshot_intent,
            "list_snapshots": self._handle_list_snapshots_intent,
        }
        logger.info("TriggerPhraseHandler initialized.")

    # --- This is the main method to guard ---
    def handle_trigger_phrase(self, user_input: str, snapshot: Optional[Any] = None) -> Dict[str, Any]:
        """
        Checks user input against trigger phrases and patterns.
        Returns immediately with {'triggered': False} if TRIGGER_PHRASES feature is disabled.
        """
        # --- Feature Flag Check ---
        if not is_enabled(Feature.TRIGGER_PHRASES):
            logger.debug("Skipping trigger phrase handling: TRIGGER_PHRASES feature disabled.")
            return {"triggered": False}
        # --- End Check ---

        # Feature enabled, proceed with matching
        if not isinstance(user_input, str):
             logger.debug("Invalid user_input type for trigger check: %s", type(user_input))
             return {"triggered": False}

        command = user_input.strip().lower()
        if not command:
             logger.debug("Empty command string received.")
             return {"triggered": False}

        action_key = None
        action_args = {}

        # 1. Check exact matches from trigger_map
        action_key = self.trigger_map.get(command)
        if action_key and action_key in self.handlers:
            logger.info("Exact trigger phrase '%s' detected (action: %s).", command, action_key)
            # Call the handler safely
            try:
                handler_result = self.handlers[action_key](snapshot)
                # Ensure result is a dict before modifying
                if not isinstance(handler_result, dict):
                     logger.error("Handler for action '%s' did not return a dict.", action_key)
                     handler_result = {} # Fallback to empty dict
            except Exception as e:
                 logger.exception("Error executing handler for action '%s': %s", action_key, e)
                 handler_result = {"error": f"Handler error: {e}"} # Add error info

            handler_result["triggered"] = True
            handler_result["action"] = action_key # Ensure action key is present
            return handler_result

        # 2. Check regex pattern for "load save {id}"
        try:
             load_match = self.load_pattern.match(command)
             if load_match:
                 action_key = "load_snapshot"
                 try:
                     action_args['snapshot_id'] = int(load_match.group(1))
                     logger.info("Load trigger phrase detected (action: %s, id: %d).", action_key, action_args['snapshot_id'])
                     return {"triggered": True, "action": action_key, "args": action_args}
                 except (ValueError, IndexError):
                     logger.warning("Could not parse snapshot ID for load command: %s", command)
                     return {"triggered": False, "error": "Invalid snapshot ID for load."}
        except re.error as re_err:
             logger.error("Regex error matching load pattern: %s", re_err)
             # Continue to next check

        # 3. Check regex pattern for "delete save {id}"
        try:
             delete_match = self.delete_pattern.match(command)
             if delete_match:
                 action_key = "delete_snapshot"
                 try:
                     action_args['snapshot_id'] = int(delete_match.group(1))
                     logger.info("Delete trigger phrase detected (action: %s, id: %d).", action_key, action_args['snapshot_id'])
                     return {"triggered": True, "action": action_key, "args": action_args}
                 except (ValueError, IndexError):
                     logger.warning("Could not parse snapshot ID for delete command: %s", command)
                     return {"triggered": False, "error": "Invalid snapshot ID for delete."}
        except re.error as re_err:
             logger.error("Regex error matching delete pattern: %s", re_err)
             # Continue to next check (although this is the last check)

        # 4. No trigger matched
        logger.debug("No specific trigger detected for input: '%s'", user_input)
        return {"triggered": False}

    # --- Simple synchronous handlers (No flag checks needed inside these) ---
    # These now primarily signal intent via the 'action' key in the return dict

    def _handle_activate(self, snapshot) -> dict:
        return {"message": "Forest activated. All systems online."}

    def _handle_change_decor(self, snapshot) -> dict:
        return {"message": "Decor changes applied."}

    def _handle_audit_scores(self, snapshot) -> dict:
        return {"message": "Scores audited."}

    def _handle_show_todo(self, snapshot) -> dict:
        # This handler logic remains, but complexity should ideally be in main.py
        if snapshot and hasattr(snapshot, "task_backlog") and snapshot.task_backlog:
            tasks = snapshot.task_backlog if isinstance(snapshot.task_backlog, list) else []
            todo_list = "\n".join([
                 f"- ID: {t.get('id', 'N/A')}, Title: {t.get('title', '')}"
                 for t in tasks[:5] if isinstance(t, dict)
            ])
            count = len(tasks)
            if count > 5: todo_list += f"\n... ({count - 5} more)"
            message = f"Current To-Do List ({count} total):\n{todo_list}"
        else:
            message = "No tasks found in the current task backlog."
        return {"message": message}

    def _handle_integrate_memory(self, snapshot) -> dict:
        return {"message": "ChatGPT memory integrated."}

    def _handle_save_snapshot_intent(self, snapshot) -> dict:
        return {"message": "Acknowledged save request."}

    def _handle_list_snapshots_intent(self, snapshot) -> dict:
        return {"message": "Acknowledged list request."}
