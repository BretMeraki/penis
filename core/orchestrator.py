# forest_app/core/orchestrator.py (REFACTORED)

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple # Keep necessary types
from sqlalchemy.orm import Session # Keep for type hints

# --- Core Imports ---
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.core.utils import clamp01
# Import the new Processors and Services
from forest_app.core.processors.reflection_processor import ReflectionProcessor
from forest_app.core.processors.completion_processor import CompletionProcessor
from forest_app.hta_tree.hta_service import HTAService
from forest_app.core.services.component_state_manager import ComponentStateManager
# Import necessary Modules for delegation or utility methods
from forest_app.hta_tree.seed import SeedManager, Seed
from forest_app.core.logging_tracking import TaskFootprintLogger # Keep for type hints
# --- Feature Flags (Keep if needed for logic remaining here) ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    def is_enabled(feature): return False # Minimal fallback
    class Feature:
        SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES"

# --- Constants (Keep if needed for logic remaining here) ---
from forest_app.config.constants import (
    MAGNITUDE_THRESHOLDS, # For describe_magnitude
    WITHERING_COMPLETION_RELIEF, # For _update_withering
    WITHERING_IDLE_COEFF, # For _update_withering
    WITHERING_OVERDUE_COEFF, # For _update_withering
    WITHERING_DECAY_FACTOR # For _update_withering
)
# Import helpers needed if _update_withering stays here
from forest_app.core.soft_deadline_manager import hours_until_deadline

logger = logging.getLogger(__name__)

# ═════════════════════════════ ForestOrchestrator (Refactored) ══════════════

class ForestOrchestrator:
    """
    Coordinates the main Forest application workflows by delegating to
    specialized processors and services. Manages top-level state transitions.
    """

    # ───────────────────────── 1. INITIALISATION (DI Based) ─────────────────
    def __init__(
        self,
        reflection_processor: ReflectionProcessor,
        completion_processor: CompletionProcessor,
        state_manager: ComponentStateManager,
        hta_service: HTAService, # Keep if orchestrator interacts directly, maybe not needed
        seed_manager: SeedManager, # Keep for seed operations
        # Add other direct dependencies if any logic remains here that needs them
    ):
        """Initializes the orchestrator with injected processors and services."""
        self.reflection_processor = reflection_processor
        self.completion_processor = completion_processor
        self.state_manager = state_manager
        self.hta_service = hta_service # May not be needed if processors use it directly
        self.seed_manager = seed_manager

        # Check critical dependencies
        if not isinstance(self.reflection_processor, ReflectionProcessor):
             raise TypeError("Invalid ReflectionProcessor provided.")
        if not isinstance(self.completion_processor, CompletionProcessor):
             raise TypeError("Invalid CompletionProcessor provided.")
        if not isinstance(self.state_manager, ComponentStateManager):
             raise TypeError("Invalid ComponentStateManager provided.")
        if not isinstance(self.seed_manager, SeedManager):
             raise TypeError("Invalid SeedManager provided.")
        # Add checks for other injected components like hta_service if kept

        logger.info("ForestOrchestrator (Refactored) initialized.")


    # ───────────────────────── 2. CORE WORKFLOWS ────────────────────────────

    async def process_reflection(self, user_input: str, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """
        Processes user reflection using the ReflectionProcessor after managing state.
        """
        logger.info("Orchestrator: Processing reflection...")
        # 1. Load Component States
        try:
            self.state_manager.load_states(snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error loading component states during reflection: %s", e)
             # Decide how to handle - return error or proceed cautiously?
             # For now, proceed but log critical error
             logger.critical("Proceeding with reflection despite state load error.")

        # 2. Update Withering (kept here for now)
        try:
            self._update_withering(snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error updating withering during reflection: %s", e)
             # Proceeding anyway

        # 3. Delegate to Reflection Processor
        try:
            result_payload = await self.reflection_processor.process(user_input, snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error during reflection_processor.process: %s", e)
             # Return a generic error response?
             return {"error": f"Core reflection processing failed: {type(e).__name__}", "arbiter_response": "An internal error occurred while processing your reflection."}

        # 4. Save Component States
        try:
            self.state_manager.save_states(snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error saving component states after reflection: %s", e)
             # The reflection likely succeeded, but state saving failed. Logged, but return result.

        logger.info("Orchestrator: Reflection processing complete.")
        return result_payload


    async def process_task_completion(
        self,
        task_id: str,
        success: bool,
        snapshot: MemorySnapshot,
        db: Session, # Pass DB if needed by CompletionProcessor or logging
        task_logger: TaskFootprintLogger # Pass logger
    ) -> Dict[str, Any]:
        """
        Processes task completion using the CompletionProcessor after managing state.
        """
        logger.info(f"Orchestrator: Processing task completion for {task_id}...")

        # 1. Load Component States (Important before completion logic runs)
        try:
            self.state_manager.load_states(snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error loading component states during completion: %s", e)
             logger.critical("Proceeding with completion despite state load error.")

        # 2. Delegate to Completion Processor
        try:
            result_payload = await self.completion_processor.process(
                task_id=task_id,
                success=success,
                snapshot=snapshot,
                db=db,
                task_logger=task_logger
            )
        except Exception as e:
             logger.exception("Orchestrator: Error during completion_processor.process for task %s: %s", task_id, e)
             # Return a generic error response?
             return {"error": f"Core completion processing failed for task {task_id}: {type(e).__name__}", "detail": "An internal error occurred while processing task completion."}

        # 3. Save Component States (Important AFTER completion logic runs)
        # Note: HTAService called within CompletionProcessor should handle saving the HTA tree itself.
        # This saves the state of *other* potentially modified components.
        try:
            self.state_manager.save_states(snapshot)
        except Exception as e:
             logger.exception("Orchestrator: Error saving component states after completion: %s", e)
             # Completion likely succeeded, but state saving failed. Logged, but return result.

        logger.info(f"Orchestrator: Task completion processing complete for {task_id}.")
        return result_payload


    # ───────────────────────── 3. UTILITY & DELEGATION ──────────────────────

    # Keeping _update_withering here for now, could be moved to its own class
    def _update_withering(self, snap: MemorySnapshot):
        """Adjusts withering level based on inactivity and deadlines."""
        # (Implementation is the same as the original orchestrator version)
        if not hasattr(snap, 'withering_level'): snap.withering_level = 0.0
        if not hasattr(snap, 'component_state') or not isinstance(snap.component_state, dict): snap.component_state = {}
        if not hasattr(snap, 'task_backlog') or not isinstance(snap.task_backlog, list): snap.task_backlog = []

        current_path = getattr(snap, "current_path", "structured")
        now_utc = datetime.now(timezone.utc)
        last_iso = snap.component_state.get("last_activity_ts")
        idle_hours = 0.0
        if last_iso and isinstance(last_iso, str):
            try:
                # Ensure TZ info for comparison
                last_dt_aware = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
                if last_dt_aware.tzinfo is None: last_dt_aware = last_dt_aware.replace(tzinfo=timezone.utc)
                idle_delta = now_utc - last_dt_aware
                idle_hours = max(0.0, idle_delta.total_seconds() / 3600.0)
            except ValueError: logger.warning("Could not parse last_activity_ts: %s", last_iso)
            except Exception as ts_err: logger.exception("Error processing last_activity_ts: %s", ts_err)
        elif last_iso is not None: logger.warning("last_activity_ts is not a string: %s", type(last_iso))

        idle_coeff = WITHERING_IDLE_COEFF.get(current_path, WITHERING_IDLE_COEFF["structured"])
        idle_penalty = idle_coeff * idle_hours

        overdue_hours = 0.0
        if is_enabled(Feature.SOFT_DEADLINES) and current_path != "open" and isinstance(snap.task_backlog, list):
            try:
                overdue_list = []
                for task in snap.task_backlog:
                    if isinstance(task, dict) and task.get("soft_deadline"):
                        overdue = hours_until_deadline(task) # Use imported helper
                        if isinstance(overdue, (int, float)) and overdue < 0:
                            overdue_list.append(abs(overdue))
                if overdue_list: overdue_hours = max(overdue_list)
            except Exception as e: logger.error("Error calculating overdue hours: %s", e) # Simplified error handling
        elif not is_enabled(Feature.SOFT_DEADLINES):
            logger.debug("Skipping overdue hours calculation: SOFT_DEADLINES feature disabled.")

        soft_coeff = WITHERING_OVERDUE_COEFF.get(current_path, 0.0) if is_enabled(Feature.SOFT_DEADLINES) else 0.0
        soft_penalty = soft_coeff * overdue_hours

        current_withering = getattr(snap, 'withering_level', 0.0)
        if not isinstance(current_withering, (int, float)): current_withering = 0.0
        new_level = float(current_withering) + idle_penalty + soft_penalty
        snap.withering_level = clamp01(new_level * WITHERING_DECAY_FACTOR)
        logger.debug(f"Withering updated: Level={snap.withering_level:.4f} (IdleHrs={idle_hours:.2f}, OverdueHrs={overdue_hours:.2f})")


    # Example: Keeping get_primary_active_seed here, but could be moved to SeedManager
    async def get_primary_active_seed(self) -> Optional[Seed]:
        """Retrieves the first active seed using the injected SeedManager."""
        if not self.seed_manager or not hasattr(self.seed_manager, 'get_primary_active_seed'):
             logger.error("Injected SeedManager missing or invalid for get_primary_active_seed.")
             return None
        try:
            # Assuming get_primary_active_seed is now async in SeedManager
            return await self.seed_manager.get_primary_active_seed()
        except Exception as e:
            logger.exception("Error getting primary active seed via orchestrator: %s", e)
            return None


    # Convenience APIs delegating to SeedManager
    async def plant_seed( self, intention: str, domain: str, addl_ctx: Optional[Dict[str, Any]] = None) -> Optional[Seed]:
        logger.info(f"Orchestrator: Delegating plant_seed to SeedManager...")
        if not self.seed_manager or not hasattr(self.seed_manager, 'plant_seed'):
            logger.error("Injected SeedManager missing or invalid for plant_seed.")
            return None
        try:
            # Assuming plant_seed is now async in SeedManager
            return await self.seed_manager.plant_seed(intention, domain, addl_ctx)
        except Exception as exc:
            logger.exception("Orchestrator plant_seed delegation error: %s", exc)
            return None


    async def trigger_seed_evolution( self, seed_id: str, evolution: str, new_intention: Optional[str] = None ) -> bool:
        logger.info(f"Orchestrator: Delegating trigger_seed_evolution to SeedManager...")
        if not self.seed_manager or not hasattr(self.seed_manager, 'evolve_seed'):
            logger.error("Injected SeedManager missing or invalid for evolve_seed.")
            return False
        try:
             # Assuming evolve_seed is now async in SeedManager
            return await self.seed_manager.evolve_seed(seed_id, evolution, new_intention)
        except Exception as exc:
            logger.exception("Orchestrator trigger_seed_evolution delegation error: %s", exc)
            return False


    # Static utility method can remain
    @staticmethod
    def describe_magnitude(value: float) -> str:
        # (Implementation is the same as the original orchestrator version)
        try:
            float_value = float(value)
            valid_thresholds = {k: float(v) for k, v in MAGNITUDE_THRESHOLDS.items() if isinstance(v, (int, float))}
            if not valid_thresholds: return "Unknown"
            sorted_thresholds = sorted(valid_thresholds.items(), key=lambda item: item[1], reverse=True)
            for label, thresh in sorted_thresholds:
                if float_value >= thresh: return str(label)
            return str(sorted_thresholds[-1][0]) if sorted_thresholds else "Dormant"
        except (ValueError, TypeError) as e:
            logger.error("Error converting value/threshold for magnitude: %s (Value: %s)", e, value)
            return "Unknown"
        except Exception as e:
            logger.exception("Error describing magnitude for value %s: %s", value, e)
            return "Unknown"
