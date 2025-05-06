# forest_app/routers/hta.py

import logging
from typing import Optional, Any, Dict
# <<< --- ADDED IMPORT --- >>>
import json
# <<< --- END ADDED IMPORT --- >>>

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
# --- Pydantic Imports ---
from pydantic import BaseModel # Import base pydantic needs

# --- Dependencies & Models ---
from forest_app.snapshot.database import get_db
from forest_app.snapshot.repository import MemorySnapshotRepository
from forest_app.snapshot.models import UserModel
from forest_app.core.security import get_current_active_user
from forest_app.snapshot.snapshot import MemorySnapshot
# --- REMOVED INCORRECT IMPORT ---
# from forest_app.core.pydantic_models import HTAStateResponse
try:
    from forest_app.config import constants
except ImportError:
    class ConstantsPlaceholder: ONBOARDING_STATUS_NEEDS_GOAL="needs_goal"; ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"; ONBOARDING_STATUS_COMPLETED="completed"
    constants = ConstantsPlaceholder()


logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class HTAStateResponse(BaseModel):
    hta_tree: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
# --- End Pydantic Models ---


@router.get("/state", response_model=HTAStateResponse, tags=["HTA"]) # Prefix is in main.py
async def get_hta_state(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    user_id = current_user.id
    logger.info(f"Request HTA state user {user_id}")
    try:
        repo = MemorySnapshotRepository(db); stored_model = repo.get_latest_snapshot(user_id);
        if not stored_model: return HTAStateResponse(hta_tree=None, message="No active session found.")
        if not stored_model.snapshot_data: return HTAStateResponse(hta_tree=None, message="Session data missing.")

        try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err: raise HTTPException(status_code=500, detail=f"Failed load session: {load_err}")

        if not snapshot.activated_state.get("activated", False):
            status_msg = constants.ONBOARDING_STATUS_NEEDS_CONTEXT if snapshot.activated_state.get("goal_set") else constants.ONBOARDING_STATUS_NEEDS_GOAL
            message = "Onboarding incomplete. Provide context." if status_msg == constants.ONBOARDING_STATUS_NEEDS_CONTEXT else "Onboarding incomplete. Set goal."
            return HTAStateResponse(hta_tree=None, message=message)

        hta_tree_data = snapshot.core_state.get("hta_tree")

        # <<< --- ADDED LOGGING --- >>>
        try:
            if hta_tree_data:
                log_data_str = json.dumps(hta_tree_data, indent=2, default=str)
                if len(log_data_str) > 1000: log_data_str = log_data_str[:1000] + "... (truncated)"
                logger.debug(f"[ROUTER HTA LOAD] HTA data loaded from core_state to be returned:\n{log_data_str}")
            else:
                logger.debug("[ROUTER HTA LOAD] HTA data loaded from core_state is None or empty.")
        except Exception as log_ex:
            logger.error(f"[ROUTER HTA LOAD] Error logging loaded HTA state: {log_ex}")
        # <<< --- END ADDED LOGGING --- >>>

        if not hta_tree_data or not isinstance(hta_tree_data, dict) or not hta_tree_data.get("root"):
            # Log this specific condition too
            logger.warning(f"[ROUTER HTA LOAD] HTA data is invalid/missing root just before returning 404-like response. Type: {type(hta_tree_data)}")
            return HTAStateResponse(hta_tree=None, message="HTA data not found or invalid.")

        return HTAStateResponse(hta_tree=hta_tree_data, message="HTA structure retrieved.")

    except HTTPException: raise
    except SQLAlchemyError as db_err:
        logger.error("DB error getting HTA state user %d: %s", user_id, db_err, exc_info=True)
        raise HTTPException(status_code=503, detail="DB error.")
    except Exception as e:
        logger.error("Error getting HTA state user %d: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error.")
