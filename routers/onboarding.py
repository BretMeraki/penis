# forest_app/routers/onboarding.py (Refactored - Use Injected LLMClient)

import logging
import uuid
import json
from typing import Optional, Any, Dict, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError, BaseModel, Field
from dependency_injector.wiring import Provide, inject

# --- Dependencies & Models ---
from forest_app.snapshot.database import get_db
from forest_app.snapshot.repository import MemorySnapshotRepository
from forest_app.snapshot.models import UserModel
from forest_app.core.security import get_current_active_user
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.snapshot.helpers import save_snapshot_with_codename

# --- Updated LLM Imports ---
from forest_app.integrations.llm import (
    LLMClient,  # <-- Import Client
    LLMError,
    LLMValidationError,
    LLMConfigurationError,
    LLMConnectionError
)

from forest_app.hta_tree.hta_models import HTAResponseModel as HTAValidationModel
from forest_app.hta_tree.seed import Seed, SeedManager
from forest_app.core.orchestrator import ForestOrchestrator
from forest_app.core.containers import Container # Import Container class directly
from forest_app.hta_tree.task_engine import TaskEngine
from forest_app.hta_tree.hta_tree import HTANode
from forest_app.hta_tree.hta_service import HTAService
from forest_app.hta_tree.hta_tree import HTATree

try:
    from forest_app.config import constants
except ImportError:
     class ConstantsPlaceholder:
        MAX_CODENAME_LENGTH=60
        MIN_PASSWORD_LENGTH=8
        ONBOARDING_STATUS_NEEDS_GOAL="needs_goal"
        ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"
        ONBOARDING_STATUS_COMPLETED="completed"
        SEED_STATUS_ACTIVE="active"
        SEED_STATUS_COMPLETED="completed"
        DEFAULT_RESONANCE_THEME="neutral"
     constants = ConstantsPlaceholder()

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class SetGoalRequest(BaseModel):
    goal_description: Any = Field(...) # Keep Any for flexibility unless specific type known

class AddContextRequest(BaseModel):
    context_reflection: Any = Field(...) # Keep Any for flexibility

class OnboardingResponse(BaseModel):
    onboarding_status: str
    message: str
    refined_goal: Optional[str] = None
    first_task: Optional[dict] = None

class CompleteNodeRequest(BaseModel):
    node_id: str
    context: dict = None
# --- End Pydantic Models ---


# --- /set_goal endpoint (No LLM calls, likely no changes needed) ---
@router.post("/set_goal", response_model=OnboardingResponse, tags=["Onboarding"])
@inject
async def set_goal_endpoint(
    request: SetGoalRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    orchestrator_i: ForestOrchestrator = Depends(Provide[Container.orchestrator]),
    llm_client: LLMClient = Depends(Provide[Container.llm_client])
):
    """
    Handles the first step of onboarding: setting the user's goal.
    Saves the goal description to the snapshot and updates the onboarding status.
    """
    user_id = current_user.id
    try:
        logger.info(f"[/onboarding/set_goal] Received goal request user {user_id}.")
        repo = MemorySnapshotRepository(db)
        stored_model = repo.get_latest_snapshot(user_id)
        snapshot = MemorySnapshot()
        if stored_model and stored_model.snapshot_data:
            try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
            except Exception as load_err:
                logger.error(f"Error loading snapshot user {user_id}: {load_err}. Starting fresh.", exc_info=True)
                stored_model = None

        if snapshot.activated_state.get("activated", False):
             logger.info(f"User {user_id} /set_goal called but session previously active. Resetting goal.")

        # --- Update snapshot state ---
        if not isinstance(snapshot.component_state, dict): snapshot.component_state = {}
        snapshot.component_state["raw_goal_description"] = str(request.goal_description) # Ensure string
        snapshot.activated_state["goal_set"] = True
        snapshot.activated_state["activated"] = False

        # --- Save snapshot (requires LLMClient) ---
        if not orchestrator_i or not llm_client:
            logger.error(f"LLMClient not available for user {user_id} in set_goal.")
            raise HTTPException(status_code=500, detail="Internal configuration error: LLM service unavailable.")
        force_create = not stored_model
        saved_model = await save_snapshot_with_codename(
            db=db,
            repo=repo,
            user_id=user_id,
            snapshot=snapshot,
            llm_client=llm_client,
            stored_model=stored_model,
            force_create_new=force_create
        )
        if not saved_model: raise HTTPException(status_code=500, detail="Failed to prepare snapshot save.")

        # --- Commit and Refresh ---
        try:
            db.commit()
            db.refresh(saved_model)
            logger.info(f"Successfully committed snapshot for user {user_id} in set_goal.")
        except SQLAlchemyError as commit_err:
            db.rollback()
            logger.exception(f"Failed to commit snapshot for user {user_id} in set_goal: {commit_err}")
            raise HTTPException(status_code=500, detail="Failed to finalize goal save.")

        logger.info(f"Onboarding Step 1 complete user {user_id}.")
        return OnboardingResponse(
            onboarding_status=constants.ONBOARDING_STATUS_NEEDS_CONTEXT,
            message="Vision received. Now add context."
        )
    # (Error handling remains the same)
    except HTTPException: raise
    except (ValueError, TypeError, AttributeError) as data_err:
        logger.exception(f"Data/Type error /set_goal user {user_id}: {data_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data: {data_err}")
    except SQLAlchemyError as db_err:
        logger.exception(f"Database error /set_goal user {user_id}: {db_err}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error during goal setting.")
    except Exception as e:
        logger.exception(f"Unexpected error /set_goal user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process goal.")


# --- /add_context endpoint (Needs LLM call update) ---
@router.post("/add_context", response_model=OnboardingResponse, tags=["Onboarding"])
@inject
async def add_context_endpoint(
    request: AddContextRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    orchestrator_i: ForestOrchestrator = Depends(Provide[Container.orchestrator]),
    llm_client: LLMClient = Depends(Provide[Container.llm_client]),
    task_engine: TaskEngine = Depends(Provide[Container.task_engine])
):
    """
    Handles the second step of onboarding: adding context and generating the initial HTA.
    Uses the injected orchestrator's LLMClient for HTA generation.
    """
    user_id = current_user.id
    first_task = None
    try:
        logger.info(f"[/onboarding/add_context] Received context request user {user_id}.")

        # --- Check Orchestrator and LLMClient availability ---
        if not orchestrator_i or not llm_client:
            logger.error(f"LLMClient not available for user {user_id} in add_context.")
            raise HTTPException(status_code=500, detail="Internal configuration error: LLM service unavailable.")
        llm_client_instance = llm_client

        repo = MemorySnapshotRepository(db)
        stored_model = repo.get_latest_snapshot(user_id)
        if not stored_model or not stored_model.snapshot_data:
            raise HTTPException(status_code=404, detail="Snapshot not found. Run /set_goal first.")

        try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err:
            logger.error(f"Error loading snapshot data user {user_id}: {load_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not load session state: {load_err}")

        if not snapshot.activated_state.get("goal_set", False):
            raise HTTPException(status_code=400, detail="Goal must be set before adding context.")

        # --- Handle already active session (no LLM call needed here) ---
        if snapshot.activated_state.get("activated", False):
            # (Logic for already active session remains the same)
            logger.info(f"User {user_id} /add_context recalled, session already active.")
            first_task = None; refined_goal_desc = "N/A"
            try:
                 if orchestrator_i.seed_manager and snapshot.component_state.get("seed_manager"):
                     seeds_dict = snapshot.component_state["seed_manager"].get("seeds", {})
                     if isinstance(seeds_dict, dict) and seeds_dict:
                          first_seed = seeds_dict.get(next(iter(seeds_dict)), {})
                          refined_goal_desc = first_seed.get("description", "N/A")
                 if orchestrator_i.task_engine and snapshot.core_state.get('hta_tree'):
                     task_result = orchestrator_i.task_engine.get_next_step(snapshot.to_dict())
                     first_task = task_result.get("base_task")
            except Exception as task_e: logger.exception("Error getting existing task/goal: %s", task_e)
            return OnboardingResponse(onboarding_status=constants.ONBOARDING_STATUS_COMPLETED, message="Session already active.", refined_goal=refined_goal_desc, first_task=first_task)

        # --- Process Goal and Context ---
        raw_goal = snapshot.component_state.get("raw_goal_description")
        raw_context = request.context_reflection
        processed_goal = str(raw_goal) if raw_goal is not None else ""
        processed_context = str(raw_context) if raw_context is not None else ""
        if not processed_goal:
            logger.error(f"Internal Error: Goal description missing state user {user_id}.")
            raise HTTPException(status_code=500, detail="Internal Error: Goal description missing.")
        if not isinstance(snapshot.component_state, dict): snapshot.component_state = {}
        snapshot.component_state["raw_context_reflection"] = raw_context

        # --- Generate HTA using domain service ---
        hta_service = HTAService(llm_client=llm_client, seed_manager=orchestrator_i.seed_manager)
        hta_model_dict, seed_desc = await hta_service.generate_onboarding_hta(processed_goal, processed_context, user_id)

        if hta_model_dict is None or "hta_root" not in hta_model_dict:
             logger.critical(f"CRITICAL FAILURE: Failed to obtain HTA dict user {user_id}.")
             raise HTTPException(status_code=500, detail="Internal error processing plan structure.")

        # --- Update Snapshot State with HTA (from dict) and Activate ---
        try:
            root_node_data = hta_model_dict.get("hta_root")
            if not isinstance(root_node_data, dict): raise ValueError("Invalid HTA root node data.")
            hta_tree_dict = {"root": root_node_data}
            seed_name = str(root_node_data.get("title", processed_goal or "Primary Goal"))[:100]
            new_seed = Seed(seed_id=f"seed_{str(uuid.uuid4())[:8]}", seed_name=seed_name, seed_domain="General", description=seed_desc, status=constants.SEED_STATUS_ACTIVE, hta_tree=hta_tree_dict, created_at=datetime.now(timezone.utc))

            seed_manager = orchestrator_i.seed_manager or SeedManager() # Initialize if orchestrator didn't have one (shouldn't happen with DI)
            seed_manager.seeds.clear()
            seed_manager.add_seed(new_seed)
            snapshot.component_state["seed_manager"] = seed_manager.to_dict()
            if not isinstance(snapshot.core_state, dict): snapshot.core_state = {}
            snapshot.core_state['hta_tree'] = hta_tree_dict
            snapshot.activated_state["activated"] = True
            snapshot.component_state.pop("raw_goal_description", None)
            snapshot.component_state.pop("raw_context_reflection", None)
        except Exception as state_err:
            logger.exception(f"Internal state update error after HTA generation user {user_id}: {state_err}")
            raise HTTPException(status_code=500, detail=f"Internal state update error: {state_err}")

        # --- Save the updated snapshot ---
        force_create = not stored_model
        saved_model = await save_snapshot_with_codename(
            db=db,
            repo=repo,
            user_id=user_id,
            snapshot=snapshot,
            llm_client=llm_client_instance,
            stored_model=stored_model,
            force_create_new=force_create
        )
        if not saved_model: raise HTTPException(status_code=500, detail="Failed to prepare activated snapshot save.")

        # --- Commit and Refresh ---
        # (Commit logic remains the same)
        try:
            db.flush()
            db.commit()
            db.refresh(saved_model)
            logger.info(f"Successfully committed ACTIVATED snapshot user {user_id} in add_context.")
        except SQLAlchemyError as commit_err:
            db.rollback()
            logger.exception(f"Failed to commit ACTIVATED snapshot user {user_id}: {commit_err}")
            raise HTTPException(status_code=500, detail="Failed to finalize session activation.")

        # --- Determine First Task (Post-Activation) ---
        try:
            logger.debug(f"Reloading snapshot user {user_id} post-commit for first task.")
            repo_after_commit = MemorySnapshotRepository(db)
            stored_model_after_commit = repo_after_commit.get_latest_snapshot(user_id)
            if stored_model_after_commit and stored_model_after_commit.snapshot_data:
                snapshot_after_commit = MemorySnapshot.from_dict(stored_model_after_commit.snapshot_data)
                if snapshot_after_commit.core_state.get('hta_tree'):
                    logger.debug(f"HTA tree FOUND user {user_id} after commit/reload.")
                    snap_dict = snapshot_after_commit.to_dict()
                    print(f"DEBUG: task_engine type: {type(task_engine)}, id: {id(task_engine)}")
                    print(f"DEBUG: orchestrator_i.task_engine type: {type(getattr(orchestrator_i, 'task_engine', None))}, id: {id(getattr(orchestrator_i, 'task_engine', None))}")
                    task_result = task_engine.get_next_step(snap_dict)
                    print(f"DEBUG: task_result: {task_result}")
                    if task_result.get("base_task") is not None:
                        first_task = task_result["base_task"]
                    elif task_result.get("fallback_task") is not None:
                        first_task = task_result["fallback_task"]
                    else:
                        first_task = None
                    print(f"DEBUG: first_task: {first_task}")
                    logger.info(f"Determined first task user {user_id}: {getattr(first_task, 'id', first_task.get('id', 'N/A') if isinstance(first_task, dict) else 'N/A')}")
                else: logger.warning("Committed snapshot user %d missing hta_tree.", user_id)
            else: logger.error("Could not retrieve committed snapshot data user %d.", user_id)
        except Exception as task_e: logger.exception("Error getting first task after activation user %d: %s", user_id, task_e)

        print(f"DEBUG: Returning OnboardingResponse with first_task: {first_task}")

        logger.info(f"Onboarding Step 2 (add_context) complete user {user_id}. Session activated.")
        return OnboardingResponse(
            onboarding_status=constants.ONBOARDING_STATUS_COMPLETED,
            message="Onboarding complete! Your journey begins.",
            refined_goal=seed_desc,
            first_task=first_task or None
        )
    # (Error handling remains the same)
    except HTTPException: raise
    except (ValueError, TypeError, AttributeError, ValidationError) as data_err: # Added ValidationError
        logger.exception(f"Data/Validation error /add_context user {user_id}: {data_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data/validation: {data_err}")
    except SQLAlchemyError as db_err:
        logger.exception(f"Database error /add_context user {user_id}: {db_err}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error during context processing.")
    except Exception as e:
        logger.exception(f"Unexpected error /add_context user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")

@router.get("/current_batch", response_model=List[dict])
@inject
def get_current_batch(
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    task_engine: TaskEngine = Depends(Provide[Container.task_engine]),
):
    """Return the current batch of actionable HTA nodes (tasks) for the user."""
    # Retrieve the latest snapshot for the user
    snapshot_repo = MemorySnapshotRepository(db)
    latest_model = snapshot_repo.get_latest_snapshot(current_user.id)
    if not latest_model or not latest_model.snapshot_data:
        raise HTTPException(status_code=404, detail="No snapshot found for user.")
    snapshot = MemorySnapshot.from_dict(latest_model.snapshot_data)
    # Use injected TaskEngine to get the current batch
    batch = task_engine.get_next_step(snapshot.to_dict())
    tasks = batch.get("tasks", [])
    # --- Sync batch with snapshot ---
    batch_ids = [t["id"] for t in tasks if "id" in t]
    # Only update if batch is different
    if set(batch_ids) != set(snapshot.current_frontier_batch_ids):
        snapshot.current_frontier_batch_ids = batch_ids
        # Add new tasks to backlog if not present
        existing_ids = {t["id"] for t in snapshot.task_backlog if "id" in t}
        for t in tasks:
            if t["id"] not in existing_ids:
                snapshot.task_backlog.append(t)
        # Save updated snapshot
        snapshot_repo.save_snapshot(current_user.id, snapshot.to_dict())
        db.commit()
    return tasks

async def generate_children_for_node(node, context, llm_client):
    """Generate children for a completed node using LLM, fallback to default if LLM fails."""
    prompt = (
        f"[INST] Given the completed node below, generate 2-3 next actionable child steps as JSON list.\n"
        f"Node: {node.title}\nDescription: {node.description}\nContext: {context or {}}\n"
        f"Each child must have: id, title, description, priority (0.0-1.0), magnitude (1-10), is_milestone (bool), depends_on (list), estimated_energy, estimated_time, children (empty list), linked_tasks (empty list).\n"
        f"Respond ONLY with a JSON list of objects. No prose.\n[/INST]"
    )
    try:
        response = await llm_client.generate(
            prompt_parts=[prompt],
            response_model=List[dict],
            use_advanced_model=True
        )
        children = [HTANode.from_dict(child) for child in response]
        return children
    except Exception:
        # Fallback: generate a single default child
        import uuid
        return [HTANode(
            id=f"node_{uuid.uuid4().hex[:8]}",
            title=f"Next Step after {node.title}",
            description="Auto-generated fallback child node.",
            priority=0.5,
            magnitude=5.0,
            is_milestone=False,
            depends_on=[],
            estimated_energy="medium",
            estimated_time="medium",
            children=[],
            status="pending",
            linked_tasks=[]
        )]

@router.post("/api/complete_node", response_model=List[dict])
async def complete_node(
    request: CompleteNodeRequest,
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    orchestrator_i: ForestOrchestrator = Depends(Provide[Container.orchestrator])
):
    """Mark a node as complete, and if the batch is done, generate new children and return the new batch."""
    # Load snapshot
    snapshot_repo = MemorySnapshotRepository(db)
    latest_model = snapshot_repo.get_latest_snapshot(current_user.id)
    if not latest_model or not latest_model.snapshot_data:
        raise HTTPException(status_code=404, detail="No snapshot found for user.")
    snapshot = MemorySnapshot.from_dict(latest_model.snapshot_data)
    hta_tree_dict = snapshot.core_state.get("hta_tree")
    if not hta_tree_dict:
        raise HTTPException(status_code=404, detail="No HTA tree found.")
    from forest_app.hta_tree.hta_tree import HTATree
    hta_tree = HTATree.from_dict(hta_tree_dict)
    # Mark node as complete
    node = hta_tree.find_node_by_id(request.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found.")
    node.mark_completed()
    hta_tree.propagate_status()
    hta_tree.rebuild_node_map()
    # Save updated tree
    snapshot.core_state["hta_tree"] = hta_tree.to_dict()
    repo = MemorySnapshotRepository(db)
    repo.save_snapshot(current_user.id, snapshot.to_dict())
    db.commit()
    # Get current batch (frontier nodes)
    from forest_app.hta_tree.task_engine import TaskEngine
    task_engine = TaskEngine()
    batch = task_engine.get_next_step(snapshot.to_dict())
    incomplete_batch = [t for t in batch.get("tasks", []) if hta_tree.find_node_by_id(t["hta_node_id"]).status != "completed"]
    if incomplete_batch:
        return incomplete_batch
    # If batch is complete, generate new children for each just-completed node
    llm_client = getattr(orchestrator_i, 'llm_client', None)
    completed_nodes = [hta_tree.find_node_by_id(t["hta_node_id"]) for t in batch.get("tasks", [])]
    for node in completed_nodes:
        if node:
            children = await generate_children_for_node(node, request.context, llm_client)
            node.children.extend(children)
    hta_tree.rebuild_node_map()
    snapshot.core_state["hta_tree"] = hta_tree.to_dict()
    repo.save_snapshot(current_user.id, snapshot.to_dict())
    db.commit()
    # Return the new batch
    batch = task_engine.get_next_step(snapshot.to_dict())
    return batch.get("tasks", [])
