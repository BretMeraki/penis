# forest_app/core/processors/completion_processor.py

import logging
import inspect
from typing import Optional, Dict, Any, List, cast
from sqlalchemy.orm import Session
from datetime import datetime

# --- Core & Module Imports ---
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.core.utils import clamp01
from forest_app.integrations.llm import LLMClient
from forest_app.core.feature_flags import is_enabled
from forest_app.core.logging_tracking import TaskFootprintLogger
from forest_app.modules.resource.xp_mastery import XPMasteryEngine

# --- Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature
except ImportError:
    # Fallback if flags cannot be imported
    class Feature:
        CORE_HTA = "FEATURE_ENABLE_CORE_HTA"
        XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"

# --- Constants ---
from forest_app.config.constants import WITHERING_COMPLETION_RELIEF

# --- New Imports ---
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.hta_tree.hta_tree import HTATree

logger = logging.getLogger(__name__)

# --- Completion Processor Class ---

class CompletionProcessor:
    """Handles the workflow for processing task completions."""

    def __init__(
        self,
        hta_service: HTAService,
        task_engine: TaskEngine,
        xp_engine: Optional[XPMasteryEngine] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """Initialize the CompletionProcessor with required and optional dependencies.
        
        Args:
            hta_service: Service for handling HTA (Hierarchical Task Analysis) operations
            task_engine: Engine for task management and generation
            xp_engine: Optional engine for XP and mastery challenge management
            llm_client: Optional language model client for XP mastery if needed
        
        Raises:
            TypeError: If hta_service or task_engine are invalid types
        """
        self.hta_service = hta_service
        self.task_engine = task_engine
        self.xp_engine = xp_engine
        self.llm_client = llm_client

        # Validate critical dependencies
        if not isinstance(self.hta_service, HTAService) or type(self.hta_service).__name__ == 'DummyService':
              logger.critical("CompletionProcessor initialized with invalid or dummy HTAService!")
              raise TypeError("Invalid HTAService provided to CompletionProcessor.")
            
        if not isinstance(self.task_engine, TaskEngine) or type(self.task_engine).__name__ == 'DummyService':
              logger.critical("CompletionProcessor initialized with invalid or dummy TaskEngine!")
              raise TypeError("Invalid TaskEngine provided to CompletionProcessor.")

        # Log warnings for optional engines if they're dummies but features are enabled
        if is_enabled(Feature.XP_MASTERY):
            if not isinstance(self.xp_engine, XPMasteryEngine) or type(self.xp_engine).__name__ == 'DummyService':
              logger.warning("CompletionProcessor: XP_MASTERY feature enabled but XPMastery engine is invalid or dummy.")
            elif not isinstance(self.llm_client, LLMClient) or type(self.llm_client).__name__ == 'DummyService':
                logger.warning("CompletionProcessor: XP_MASTERY feature enabled but LLMClient (required for XPMastery) is invalid or dummy.")

        logger.info("CompletionProcessor initialized successfully.")

    async def process_xp_update(
        self,
        task: Dict[str, Any],
        snapshot: MemorySnapshot,
        success: bool
    ) -> Dict[str, Any]:
        """
        Process XP updates for a completed task.
        
        Args:
            task: The completed task
            snapshot: Current memory snapshot
            success: Whether the task was completed successfully
            
        Returns:
            Dict containing XP update information
        """
        if not self.xp_engine or not is_enabled(Feature.XP_MASTERY):
            return {}

        try:
            # Calculate XP gain
            try:
                xp_gain = self.xp_engine.calculate_xp_gain(task) if success else 0
            except Exception as e:
                # Return 0 XP for any calculation error
                logger.warning(f"Error calculating XP gain for task {task.get('id', 'unknown')}: {e}")
                return {
                    "xp_gained": 0,
                    "error": str(e)
                }
            
            # Get previous stage
            try:
                previous_stage = self.xp_engine.get_current_stage()
            except Exception as e:
                logger.error(f"Error getting previous stage: {e}")
                return {
                    "xp_gained": xp_gain,
                    "error": f"Stage lookup failed: {e}"
                }
            
            # Update XP
            self.xp_engine.current_xp += xp_gain
            
            # Get new stage
            try:
                current_stage = self.xp_engine.get_current_stage()
            except Exception as e:
                logger.error(f"Error getting current stage: {e}")
                self.xp_engine.current_xp -= xp_gain  # Rollback XP update
                return {
                    "xp_gained": 0,
                    "error": f"Stage lookup failed: {e}"
                }
            
            # Check if stage changed
            stage_changed = previous_stage["name"] != current_stage["name"]
            
            # Generate challenge if stage changed
            challenge_content = None
            if stage_changed:
                try:
                    challenge_content = await self.xp_engine.generate_challenge_content(
                        current_stage,
                        snapshot.to_dict()
                    )
                except Exception as e:
                    logger.error(f"Error generating challenge content: {e}")
                    # Don't fail the whole operation if challenge generation fails
            
            return {
                "xp_gained": xp_gain,
                "current_xp": self.xp_engine.current_xp,
                "previous_stage": previous_stage["name"],
                "current_stage": current_stage["name"],
                "stage_changed": stage_changed,
                "challenge": challenge_content
            }
            
        except Exception as e:
            logger.error(f"Error processing XP update: {e}")
            return {
                "xp_gained": 0,
                "error": str(e)
            }

    async def process(
        self,
        task_id: str,
        success: bool,
        snapshot: MemorySnapshot,
        db: Session,
        task_logger: TaskFootprintLogger
    ) -> Dict[str, Any]:
        """Process task completion and update system state."""
        # Initialize result
        result = {
            "success": False,
            "batch_completed": False,
            "error": None
        }

        # Validate inputs
        if not isinstance(task_id, str) or not task_id.strip():
            result["error"] = "Invalid task_id: must be a non-empty string"
            return result
            
        if not isinstance(success, bool):
            result["error"] = "Invalid success parameter: must be a boolean"
            return result
            
        if not isinstance(snapshot, MemorySnapshot):
            result["error"] = "Invalid snapshot type: must be a MemorySnapshot instance"
            return result
            
        if not isinstance(db, Session):
            result["error"] = "Invalid db parameter: must be a SQLAlchemy Session"
            return result
            
        if not isinstance(task_logger, TaskFootprintLogger):
            result["error"] = "Invalid task_logger: must be a TaskFootprintLogger instance"
            return result

        try:
            # Validate task backlog
            if not isinstance(snapshot.task_backlog, list):
                result["error"] = "Invalid task backlog format"
                return result

            # Update task status in backlog
            task_found = False
            completed_task = None
            for task in snapshot.task_backlog:
                if not isinstance(task, dict) or "id" not in task:
                    logger.warning("Invalid task format in backlog")
                    continue
                    
                if task["id"] == task_id:
                    task["status"] = "complete" if success else "failed"
                    task_found = True
                    completed_task = task
                    break

            if not task_found:
                logger.warning(f"Task {task_id} not found in backlog")
                result["error"] = "Task not found in backlog"
                return result

            # Process XP update
            if completed_task:
                try:
                    xp_update = await self.process_xp_update(completed_task, snapshot, success)
                    if xp_update:
                        result["xp_update"] = xp_update
                except Exception as e:
                    logger.error(f"Error processing XP update: {e}")
                    result["xp_update"] = {
                        "xp_gained": 0,
                        "error": str(e)
                    }

            # Update withering level on successful completion
            if success:
                try:
                    old_withering = snapshot.withering_level
                    snapshot.withering_level = max(0.0, snapshot.withering_level - WITHERING_COMPLETION_RELIEF)
                    logger.info(f"Updated withering level from {old_withering} to {snapshot.withering_level}")
                except Exception as e:
                    logger.error(f"Error updating withering level: {e}")
                    # Don't fail the whole operation for withering update error

            # Remove task from frontier batch
            try:
                if not isinstance(snapshot.current_frontier_batch_ids, list):
                    snapshot.current_frontier_batch_ids = []
                if task_id in snapshot.current_frontier_batch_ids:
                    snapshot.current_frontier_batch_ids.remove(task_id)
            except Exception as e:
                logger.error(f"Error updating frontier batch: {e}")
                # Don't fail the whole operation for frontier batch error

            # Check if batch is complete
            try:
                batch_completed = len(snapshot.current_frontier_batch_ids) == 0
                result["batch_completed"] = batch_completed

                # If batch is complete, evolve the HTA tree
                if batch_completed and is_enabled(Feature.CORE_HTA):
                    try:
                        tree = await self.hta_service.load_tree(snapshot)
                        evolved_tree = await self.hta_service.evolve_tree(
                            tree, getattr(snapshot, "current_batch_reflections", [])
                        )
                        await self.hta_service.save_tree(evolved_tree, snapshot)
                        
                        # Clear batch reflections after successful evolution
                        if hasattr(snapshot, 'current_batch_reflections'):
                            snapshot.current_batch_reflections = []
                            logger.info("Cleared batch reflections after tree evolution")
                    except Exception as e:
                        logger.error(f"Error evolving HTA tree: {e}")
                        # Don't fail the whole operation for HTA evolution error
            except Exception as e:
                logger.error(f"Error checking batch completion: {e}")
                result["batch_completed"] = False

            # Remove completed task from backlog
            try:
                snapshot.task_backlog = [t for t in snapshot.task_backlog if t.get("id") != task_id]
            except Exception as e:
                logger.error(f"Error removing task from backlog: {e}")
                # Don't fail the whole operation for backlog update error

            # Log task completion
            try:
                await task_logger.log_task_completion(task_id, success, db)
            except Exception as e:
                logger.error(f"Error logging task completion: {e}")
                # Don't fail the whole operation for logging error

            result["success"] = success
            return result

        except Exception as e:
            logger.error(f"Error processing task completion: {e}")
            result["error"] = str(e)
            result["success"] = False
            return result
