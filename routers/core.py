# forest_app/routers/core.py (MODIFIED: Added @inject decorators)

import logging
# MODIFIED: Added List - Ensure all needed types are here
from typing import Optional, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
# --- MODIFIED: Added Field ---
from pydantic import BaseModel, Field, ValidationError
# --- MODIFIED: Added Provide and inject for dependency injection ---
from dependency_injector.wiring import Provide, inject

# --- Dependencies & Models ---
from forest_app.snapshot.database import get_db
from forest_app.snapshot.repository import MemorySnapshotRepository, get_latest_snapshot_model
from forest_app.snapshot.models import UserModel
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.snapshot.helpers import save_snapshot_with_codename
# --- Import Classes needed for Dependency Injection Type Hints ---
from forest_app.core.orchestrator import ForestOrchestrator
from forest_app.modules.trigger.trigger_phrase import TriggerPhraseHandler
from forest_app.core.logging_tracking import TaskFootprintLogger, ReflectionLogLogger
# --- Import Dependency Injection Container ---
from forest_app.core.containers import Container
from forest_app.integrations.llm import LLMClient
from forest_app.core.security import get_current_active_user

# --- Import Constants (if needed, ensure path is correct) ---
try:
    from forest_app.config import constants
except ImportError:
    # Define placeholder if constants cannot be imported
    class ConstantsPlaceholder:
        MAX_CODENAME_LENGTH=60; MIN_PASSWORD_LENGTH=8; ONBOARDING_STATUS_NEEDS_GOAL="needs_goal";
        ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"; ONBOARDING_STATUS_COMPLETED="completed";
        SEED_STATUS_ACTIVE="active"; SEED_STATUS_COMPLETED="completed"; DEFAULT_RESONANCE_THEME="neutral"
    constants = ConstantsPlaceholder()


logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class CommandRequest(BaseModel):
    command: str

class RichCommandResponse(BaseModel):
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    offering: Optional[dict]=None
    mastery_challenge: Optional[dict]=None
    magnitude_description: str
    arbiter_response: str
    resonance_theme: str
    routing_score: float
    onboarding_status: Optional[str]=None
    action_required: Optional[str] = None
    confirmation_details: Optional[Dict[str, Any]] = None

class CompleteTaskRequest(BaseModel):
    task_id: str
    success: bool

class MessageResponse(BaseModel):
    message: str
# --- End Pydantic Models ---


@router.post("/command", response_model=RichCommandResponse, tags=["Core"])
@inject
async def command_endpoint(
    request_data: CommandRequest,
    request: Request, # Inject FastAPI Request object
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    # --- MODIFIED: Inject Dependencies using Provide ---
    trigger_h: TriggerPhraseHandler = Depends(Provide[Container.trigger_phrase_handler]),
    orchestrator_i: ForestOrchestrator = Depends(Provide[Container.orchestrator]),
    llm_client: LLMClient = Depends(Provide[Container.llm_client])
):
    user_id = current_user.id; command_text = request_data.command
    logger.info(f"Received command user {user_id}: '{command_text[:50]}...'")
    try:
        # REMINDER: Ensure get_latest_snapshot_model is sync if not using await
        repo = MemorySnapshotRepository(db); stored_model = get_latest_snapshot_model(user_id, db)
        snapshot = None
        if stored_model and stored_model.snapshot_data:
            try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
            except Exception as load_err: logger.error(f"Err load snapshot user {user_id}: {load_err}", exc_info=True); stored_model = None
        elif stored_model: logger.warning(f"Snapshot user {user_id} has no data."); stored_model = None

        # --- Use injected trigger_h ---
        trigger_result = trigger_h.handle_trigger_phrase(command_text, snapshot) # snapshot can be None
        action = trigger_result.get("action"); args = trigger_result.get("args", {})

        if trigger_result.get("triggered"):
            logger.info(f"Command trigger user {user_id}. Action: {action}")
            if action == "save_snapshot":
                if not snapshot or not stored_model: raise HTTPException(status_code=404, detail="No active session to save.")
                if not llm_client: raise HTTPException(status_code=500, detail="LLM service needed for save.")
                # REMINDER: Ensure save_snapshot_with_codename is async if using await
                saved_model = await save_snapshot_with_codename(db=db, repo=repo, user_id=user_id, snapshot=snapshot, llm_client=llm_client, stored_model=stored_model)
                if not saved_model: raise HTTPException(status_code=500, detail="Save failed")
                try: db.commit(); db.refresh(saved_model)
                except SQLAlchemyError as commit_err: db.rollback(); logger.exception(f"Failed commit: {commit_err}"); raise HTTPException(status_code=500, detail="Failed finalize save.")
                codename = saved_model.codename or f"ID {saved_model.id}";
                return RichCommandResponse(
                        tasks=[], arbiter_response=f"Snapshot saved ('{codename}')",
                        magnitude_description="N/A", resonance_theme="N/A", routing_score=0.0
                )
            else:
                return RichCommandResponse(
                        tasks=[], arbiter_response=trigger_result.get("message", "Acknowledged trigger."),
                        magnitude_description="N/A", resonance_theme="N/A", routing_score=0.0
                )

        # If not triggered, proceed to reflection processing
        if not snapshot or not stored_model:
            onboarding_status = constants.ONBOARDING_STATUS_NEEDS_GOAL
            # REMINDER: Ensure get_latest_snapshot_model is sync if not using await
            temp_stored_model = get_latest_snapshot_model(user_id, db)
            if temp_stored_model and temp_stored_model.snapshot_data:
                try:
                    temp_snap_data = temp_stored_model.snapshot_data
                    if isinstance(temp_snap_data, dict) and temp_snap_data.get("activated_state", {}).get("goal_set"):
                        onboarding_status = constants.ONBOARDING_STATUS_NEEDS_CONTEXT
                except Exception as snap_peek_err: logger.error("Error peeking snapshot: %s", snap_peek_err)
            detail = "Onboarding: Please provide context." if onboarding_status == constants.ONBOARDING_STATUS_NEEDS_CONTEXT else "Onboarding: Please set a goal."
            raise HTTPException(status_code=403, detail=detail)

        if not snapshot.activated_state.get("activated", False):
            raise HTTPException(status_code=403, detail="Onboarding incomplete.")

        logger.info(f"Processing command user {user_id} as reflection.")

        # REMINDER: Ensure process_reflection is async if using await
        result_dict = await orchestrator_i.process_reflection(
            user_input=command_text,
            snapshot=snapshot # Pass the MemorySnapshot object
        )

        if not llm_client: raise HTTPException(status_code=500, detail="LLM service needed for save.")
         # REMINDER: Ensure save_snapshot_with_codename is async if using await
        saved_model = await save_snapshot_with_codename( db=db, repo=repo, user_id=user_id, snapshot=snapshot, llm_client=llm_client, stored_model=stored_model)
        if not saved_model: raise HTTPException(status_code=500, detail="Failed save state after reflection.")

        try: db.commit(); db.refresh(saved_model)
        except SQLAlchemyError as commit_err: db.rollback(); logger.exception(f"Failed commit: {commit_err}"); raise HTTPException(status_code=500, detail="Failed finalize reflection save.")

        # Process result_dict and return RichCommandResponse
        response_payload = {
                "tasks": result_dict.get("tasks", []),
                "arbiter_response": result_dict.get("arbiter_response", ""),
                "offering": result_dict.get("offering"),
                "mastery_challenge": result_dict.get("mastery_challenge"),
                "magnitude_description": result_dict.get("magnitude_description", "N/A"),
                "resonance_theme": result_dict.get("resonance_theme", constants.DEFAULT_RESONANCE_THEME),
                "routing_score": result_dict.get("routing_score", 0.0),
                "action_required": result_dict.get("action_required"),
                "confirmation_details": result_dict.get("confirmation_details"),
            }
        try:
            return RichCommandResponse.model_validate(response_payload)
        except ValidationError as val_err:
            logger.error("Validation error RichCommandResponse: %s Payload: %s", val_err, response_payload)
            raise HTTPException(status_code=500, detail=f"Internal Error: Could not format valid response.")

    except HTTPException: raise
    except (SQLAlchemyError, ValueError, TypeError) as db_val_err:
        # Ensure rollback happens even if commit wasn't reached or failed before rollback was called
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in outer exception handler")
        logger.exception(f"DB/Data error /command user {user_id}: {db_val_err}")
        detail = "Database error." if isinstance(db_val_err, SQLAlchemyError) else f"Invalid data request: {db_val_err}"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(db_val_err, SQLAlchemyError) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
         # Ensure rollback happens even if commit wasn't reached or failed before rollback was called
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in outer exception handler")
        logger.exception(f"Unexpected internal error /command user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal server error: {type(e).__name__}")

# --- ADDED: Task Completion Endpoint ---
@router.post("/complete_task", response_model=Dict[str, Any], tags=["Core"])
@inject
async def complete_task_endpoint(
    request_data: CompleteTaskRequest, # Use the renamed request model
    request: Request, # Inject FastAPI Request object if needed for context
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user), # Use the specific active user function
    orchestrator: ForestOrchestrator = Depends(Provide[Container.orchestrator]),
    task_logger: TaskFootprintLogger = Depends(lambda db=Depends(get_db): Container.task_footprint_logger(db=db)),
    llm_client: LLMClient = Depends(Provide[Container.llm_client]),
):
    """
    Endpoint to mark a task as completed (or failed) and trigger
    subsequent processing like HTA updates, logging, etc.
    """
    user_id = current_user.id
    task_id = request_data.task_id
    success = request_data.success
    logger.info(f"Received /complete_task request user {user_id}, Task: {task_id}, Success: {success}")

    try:
        # 1. Load the latest snapshot
        repo = MemorySnapshotRepository(db)
        # REMINDER: Ensure get_latest_snapshot_model is sync if not using await
        stored_model = get_latest_snapshot_model(user_id, db)
        if not stored_model:
            logger.error(f"Snapshot not found for user {user_id} during task completion.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active session found.")
        if not stored_model.snapshot_data:
            logger.error(f"Snapshot data empty for user {user_id} (ID: {stored_model.id}).")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active session data is empty.")

        try:
            snap = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err:
            logger.error(f"Error loading snapshot user {user_id} for task completion: {load_err}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed load session state: {load_err}")

        # 2. Check Onboarding Status
        if not snap.activated_state.get("activated"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot complete task: Onboarding incomplete.")

        # 3. Call Orchestrator Logic
        completion_result = await orchestrator.process_task_completion(
            task_id=task_id,
            success=success,
            snapshot=snap,
            db=db,
            task_logger=task_logger
        )

        # If the orchestrator returned an error, raise HTTPException
        if completion_result.get("error"):
            raise HTTPException(status_code=404, detail=completion_result["error"])

        # 4. Save Updated Snapshot
        # Pass the potentially modified 'snap' object to the save helper
        # REMINDER: Ensure save_snapshot_with_codename is async if using await
        saved_model = await save_snapshot_with_codename(
            db=db,
            repo=repo,
            user_id=user_id,
            snapshot=snap, # Pass the updated snapshot object
            llm_client=llm_client,
            stored_model=stored_model # Pass the original DB model for update context
        )
        if not saved_model:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed save state after task completion.")

        # 5. Commit Transaction
        try:
            db.commit()
            db.refresh(saved_model) # Refresh to get updated data if needed
            logger.info(f"Successfully processed completion for task {task_id} and saved snapshot.")
        except SQLAlchemyError as commit_err:
            db.rollback()
            logger.exception(f"Failed commit after task completion: {commit_err}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed finalize task completion state save.")

        # 6. Return Result
        return {"detail": f"Task '{task_id}' processed.", "result": completion_result}

    except HTTPException:
        # Rollback might have happened in specific failure points (like save fail),
        # but ensure it happens for other HTTPExceptions raised before commit.
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in HTTPException handler")
        raise # Re-raise HTTPExceptions directly
    except ValueError as val_err: # Handle specific errors like task not found from orchestrator
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in ValueError handler")
        status_code = 404 if "not found" in str(val_err).lower() else 400
        logger.warning(f"Value error completing task {task_id} user {user_id}: {val_err}")
        raise HTTPException(status_code=status_code, detail=f"Error: {val_err}")
    except (SQLAlchemyError, TypeError) as db_type_err:
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in DB/Type error handler")
        logger.exception(f"DB/Type error /complete_task user {user_id} task {task_id}: {db_type_err}")
        detail = "Database error during task completion." if isinstance(db_type_err, SQLAlchemyError) else f"Data processing error: {db_type_err}"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(db_type_err, SQLAlchemyError) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        try: db.rollback()
        except Exception: logger.error("Exception during rollback in generic Exception handler")
        logger.exception(f"Unexpected internal error /complete_task user {user_id} task {task_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Unexpected internal server error: {type(e).__name__}")
# --- END ADDED ---
