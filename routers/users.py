# forest_app/routers/users.py (Corrected Import and Type Hint)

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError, BaseModel, EmailStr

# --- Dependencies & Models ---
from forest_app.snapshot.database import get_db
from forest_app.snapshot.repository import MemorySnapshotRepository
# --- Import the ACTUAL UserModel ---
from forest_app.snapshot.models import UserModel
# --- Import only get_current_active_user from security ---
from forest_app.core.security import get_current_active_user
from forest_app.snapshot.snapshot import MemorySnapshot
try:
    from forest_app.config import constants
except ImportError:
    class ConstantsPlaceholder: ONBOARDING_STATUS_NEEDS_GOAL="needs_goal"; ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"; ONBOARDING_STATUS_COMPLETED="completed"
    constants = ConstantsPlaceholder()

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserRead(UserBase):
    id: int
    is_active: bool
    onboarding_status: Optional[str] = None
    class Config: from_attributes = True
# --- End Pydantic Models ---


@router.get("/me", response_model=UserRead, tags=["Users"])
async def read_users_me(
    # --- Use the correct UserModel for type hint ---
    current_user: UserModel = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Gets the current logged-in user's details, including onboarding status."""
    user_id = current_user.id
    logger.info("Fetching details for current user ID: %d (%s)", user_id, current_user.email)
    onboarding_status = constants.ONBOARDING_STATUS_NEEDS_GOAL
    try:
        repo = MemorySnapshotRepository(db)
        stored_model = repo.get_latest_snapshot(user_id)
        if stored_model and stored_model.snapshot_data:
            try:
                snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
                if snapshot.activated_state.get("activated", False): onboarding_status = constants.ONBOARDING_STATUS_COMPLETED
                elif snapshot.activated_state.get("goal_set", False): onboarding_status = constants.ONBOARDING_STATUS_NEEDS_CONTEXT
            except (ValidationError, TypeError, KeyError, Exception) as snap_load_err:
                logger.error("Error loading/parsing snapshot for user %d status: %s.", user_id, snap_load_err, exc_info=True)
        elif stored_model: logger.warning("Latest snapshot (ID: %d) for user %d has no data.", stored_model.id, user_id)
        else: logger.info("No snapshot found for user %d.", user_id)
    except SQLAlchemyError as db_err: logger.error("DB error fetching snapshot user %d status: %s", user_id, db_err, exc_info=True)
    except Exception as e: logger.error("Unexpected error fetching snapshot user %s status: %s", user_id, e, exc_info=True)

    try:
        user_data = UserRead.model_validate(current_user, from_attributes=True).model_dump()
        user_data["onboarding_status"] = onboarding_status
        return UserRead.model_validate(user_data)
    except ValidationError as val_err:
        logger.error("Validation failed creating final UserRead response user %d: %s", user_id, val_err)
        # Fallback to returning basic user info without onboarding status if final validation fails
        return UserRead.model_validate(current_user, from_attributes=True)
