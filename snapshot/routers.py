# forest_app/routers/snapshots.py (MODIFIED: Corrected SnapshotInfo model)

import logging
from typing import Optional, List, Dict, Any # Added Dict, Any
from datetime import datetime # Added datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError, BaseModel # Added BaseModel
from dependency_injector.wiring import Provide, inject

# --- Dependencies & Models ---
from forest_app.snapshot.database import get_db
from forest_app.snapshot.repository import MemorySnapshotRepository
from forest_app.snapshot.models import UserModel # Import if type hints need it
from forest_app.snapshot.snapshot import MemorySnapshot
from forest_app.snapshot.helpers import save_snapshot_with_codename # Import helper
from forest_app.core.security import get_current_active_user
# from forest_app.core.pydantic_models import SnapshotInfo, LoadSessionRequest, MessageResponse # Import if centralized
from forest_app.integrations.llm import LLMClient
from forest_app.core.containers import Container

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models (Copied from main.py or moved to core/pydantic_models.py) ---
# Define models here if not centralized
class SnapshotInfo(BaseModel):
    id: int
    codename: Optional[str] = None
    created_at: datetime # <<< CORRECTED FIELD NAME HERE
    class Config:
        from_attributes = True # For Pydantic v2 (was orm_mode=True in v1)

class LoadSessionRequest(BaseModel):
    snapshot_id: int

class MessageResponse(BaseModel):
    message: str
# --- End Pydantic Models ---


# Route path corrected based on previous analysis
@router.get("/list", response_model=List[SnapshotInfo], tags=["Snapshots"])
async def list_user_snapshots(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """Lists all saved snapshots for the current user."""
    user_id = current_user.id
    logger.info(f"Request list snapshots user {user_id}")
    try:
        repo = MemorySnapshotRepository(db); models = repo.list_snapshots(user_id);
        if not models: return []
        # Use model_validate for Pydantic v2+
        # from_attributes=True in Config enables conversion from ORM model
        return [SnapshotInfo.model_validate(m) for m in models] # Use model_validate directly
    except SQLAlchemyError as db_err:
        logger.error("DB error listing snapshots user %d: %s", user_id, db_err, exc_info=True)
        raise HTTPException(status_code=503, detail="DB error listing snapshots.")
    except ValidationError as val_err:
         # Log the detailed validation error
         logger.error("Validation error formatting snapshot list user %d: %s", user_id, val_err, exc_info=True)
         raise HTTPException(status_code=500, detail="Internal error formatting snapshot list.")
    except Exception as e:
        logger.error("Error listing snapshots user %d: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error listing snapshots.")

@router.post("/session/load", response_model=MessageResponse, tags=["Snapshots"])
@inject
async def load_session_from_snapshot(
    request: LoadSessionRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    llm_client: LLMClient = Depends(Provide[Container.llm_client])
):
    """Loads a previous snapshot as the new active session."""
    user_id = current_user.id; snapshot_id = request.snapshot_id
    logger.info(f"Request load session user {user_id} from snapshot {snapshot_id}")
    try:
        repo = MemorySnapshotRepository(db)
        model_to_load = repo.get_snapshot_by_id(snapshot_id, user_id)
        if not model_to_load: raise HTTPException(status_code=404, detail="Snapshot not found.")
        if not model_to_load.snapshot_data: raise HTTPException(status_code=404, detail="Snapshot empty.")

        try: loaded_snapshot = MemorySnapshot.from_dict(model_to_load.snapshot_data)
        except Exception as load_err: raise HTTPException(status_code=500, detail=f"Failed parse snapshot: {load_err}")

        if not isinstance(loaded_snapshot.activated_state, dict): loaded_snapshot.activated_state = {}
        loaded_snapshot.activated_state.update({"activated": True, "goal_set": True})

        new_model = await save_snapshot_with_codename(
             db=db,
             repo=repo,
             user_id=user_id,
             snapshot=loaded_snapshot,
             llm_client=llm_client,
             force_create_new=True
        )
        if not new_model: raise HTTPException(status_code=500, detail="Failed save loaded session.")

        codename = new_model.codename or f"ID {new_model.id}";
        logger.info(f"Loaded snap {snapshot_id} user {user_id}. New ID: {new_model.id}")
        return MessageResponse(message=f"Session loaded from '{codename}'.")

    except HTTPException: raise
    except (SQLAlchemyError, ValueError, TypeError) as db_val_err:
        logger.exception(f"DB/Data error load session user {user_id} snap {snapshot_id}: {db_val_err}")
        detail = "DB error." if isinstance(db_val_err, SQLAlchemyError) else f"Invalid data: {db_val_err}"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(db_val_err, SQLAlchemyError) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.exception(f"Error load session user {user_id} snap {snapshot_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error.")

@router.delete("/snapshots/{snapshot_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Snapshots"])
async def delete_user_snapshot(
    snapshot_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    """Deletes a specific snapshot."""
    user_id = current_user.id
    logger.info(f"Request delete snap {snapshot_id} user {user_id}")
    try:
        repo = MemorySnapshotRepository(db)
        # Assuming delete handles commit/rollback
        deleted = repo.delete_snapshot_by_id(snapshot_id, user_id)
        if not deleted: raise HTTPException(status_code=404, detail="Snapshot not found")
        logger.info(f"Deleted snap {snapshot_id} user {user_id}")
        return None # Return None for 204 response
    except HTTPException: raise
    except SQLAlchemyError as db_err:
        logger.exception(f"DB error delete snap {snapshot_id} user {user_id}: {db_err}")
        raise HTTPException(status_code=503, detail="DB error.")
    except Exception as e:
        logger.exception(f"Error delete snap {snapshot_id} user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error.")
