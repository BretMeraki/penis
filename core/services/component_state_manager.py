# forest_app/core/services/component_state_manager.py

import logging
from typing import Dict, Any, Optional, cast
from datetime import datetime, timezone

# Import core types
from forest_app.snapshot.snapshot import MemorySnapshot

logger = logging.getLogger(__name__)

class ComponentStateManager:
    """
    Manages loading and saving the state of various application components
    (engines, managers) from/to the snapshot's component_state dictionary.
    """

    def __init__(self, managed_engines: Dict[str, Any]):
        """
        Initializes the ComponentStateManager.

        Args:
            managed_engines: A dictionary where keys are the string keys used
                             in snapshot.component_state, and values are the
                             actual instances of the engines/managers to manage.
                             Example: {"seed_manager": seed_manager_instance,
                                       "metrics_engine": metrics_engine_instance}
        """
        self.managed_engines = managed_engines
        if not isinstance(managed_engines, dict):
             logger.error("ComponentStateManager received non-dict for managed_engines. State management will likely fail.")
             # Optionally raise TypeError here
             # raise TypeError("managed_engines must be a dictionary.")
        logger.info(f"ComponentStateManager initialized to manage components with keys: {list(managed_engines.keys())}")


    def load_states(self, snapshot: MemorySnapshot):
        """
        Safely loads state into each managed engine instance from the
        snapshot's component_state dictionary.

        Iterates through the engines provided during initialization.
        """
        if not hasattr(snapshot, 'component_state') or not isinstance(snapshot.component_state, dict):
            logger.warning("Snapshot component_state is missing or not a dict. Cannot load states.")
            snapshot.component_state = {} # Initialize if missing
            return

        cs = snapshot.component_state
        logger.debug("Loading component states from snapshot into managed engines...")

        for key, engine in self.managed_engines.items():
            engine_state_dict = cs.get(key) # Get the state dict for this engine's key

            if engine_state_dict is None:
                # logger.debug(f"No state found in snapshot for component key '{key}'. Skipping load for this component.")
                continue # Skip if no state exists for this key

            if not isinstance(engine_state_dict, dict):
                 logger.warning(f"Invalid state format found for component key '{key}'. Expected dict, got {type(engine_state_dict).__name__}. Skipping load.")
                 continue # Skip if state isn't a dictionary

            # Check if the engine instance is valid and has the update method
            if engine and hasattr(engine, 'update_from_dict') and callable(engine.update_from_dict):
                # Check if it's a dummy service - avoid calling update_from_dict on dummies
                if type(engine).__name__ == 'DummyService':
                    logger.debug(f"Skipping state load for component key '{key}' because engine is a DummyService.")
                    continue

                try:
                    logger.debug(f"Loading state for component key '{key}' into {type(engine).__name__}...")
                    engine.update_from_dict(engine_state_dict)
                except Exception as e:
                    logger.exception(f"Failed to load state for component key '{key}' into {type(engine).__name__}: {e}")
            elif engine:
                # Engine exists but lacks the method (or is maybe None, though __init__ checks keys)
                 logger.debug(f"State found for key '{key}', but engine {type(engine).__name__} lacks 'update_from_dict' method. Skipping load.")
            # else: # Engine instance itself might be None if dict passed to init was faulty
            #      logger.warning(f"No valid engine instance found for component key '{key}'. Skipping load.")

        logger.debug("Finished loading component states.")


    def save_states(self, snapshot: MemorySnapshot):
        """
        Safely saves state from each managed engine instance into the
        snapshot's component_state dictionary.

        Iterates through the engines provided during initialization.
        """
        if not hasattr(snapshot, 'component_state') or not isinstance(snapshot.component_state, dict):
            logger.warning("Snapshot component_state is missing or not a dict. Initializing before saving states.")
            snapshot.component_state = {}

        cs = snapshot.component_state
        logger.debug("Saving component states from managed engines into snapshot...")

        for key, engine in self.managed_engines.items():
            # Check if the engine instance is valid and has the to_dict method
            if engine and hasattr(engine, 'to_dict') and callable(engine.to_dict):
                 # Skip saving state for dummy services
                 if type(engine).__name__ == 'DummyService':
                     # Optionally remove potentially stale state for dummy services
                     if key in cs:
                         del cs[key]
                         logger.debug(f"Removed potentially stale state for dummy component key '{key}'.")
                     continue

                 try:
                     engine_data = engine.to_dict()
                     if isinstance(engine_data, dict) and engine_data:
                         cs[key] = engine_data
                         # logger.debug(f"Saved state for component key '{key}' from {type(engine).__name__}.")
                     elif key in cs:
                         # Remove key if engine returns empty state or None
                         del cs[key]
                         logger.debug(f"Removed empty/null state for component key '{key}' from {type(engine).__name__}.")

                 except Exception as e:
                     logger.exception(f"Failed to get state dictionary using to_dict() for component key '{key}' from {type(engine).__name__}: {e}")

            elif engine and type(engine).__name__ != 'DummyService':
                 # Real engine exists but lacks the method
                 logger.debug(f"Engine {type(engine).__name__} (key: '{key}') lacks 'to_dict' method. Cannot save its state.")
            # else: # Engine instance itself might be None or Dummy
                 # logger.debug(f"No valid engine instance or is DummyService for component key '{key}'. Skipping save.")
                 pass # Already handled dummy check above

        # Always update the last activity timestamp
        cs["last_activity_ts"] = datetime.now(timezone.utc).isoformat(timespec='seconds')
        snapshot.component_state = cs # Ensure the updated dict is assigned back

        logger.debug("Finished saving component states.")
