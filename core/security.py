# forest_app/core/security.py (Refactored - Updated Type Hint)

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Callable, TypeVar, Union, cast, TYPE_CHECKING

# --- Security Libraries ---
from passlib.context import CryptContext
from jose import JWTError, jwt

# --- FastAPI/Pydantic ---
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ValidationError

# --- Database & Models ---
from sqlalchemy.orm import Session
from forest_app.snapshot.database import get_db
from forest_app.snapshot.models import UserModel

# Use TYPE_CHECKING to prevent circular imports
if TYPE_CHECKING:
    from forest_app.persistence.models import UserModel

# Configure logger with more detailed format
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Type Variables for Generic User Model ---
T = TypeVar('T')
UserType = TypeVar('UserType', bound='UserModel')

# --- Globals for dependencies (initialized via function) ---
UserModel: Optional[type] = None
get_user_by_email: Optional[Callable[[Session, str], Optional[UserType]]] = None

def initialize_security_dependencies(
    _get_user_by_email: Callable[[Session, str], Optional[UserType]],
    _user_model: type[UserType]
) -> None:
    """Initialize the security module's dependencies with proper type checking."""
    global get_user_by_email, UserModel

    if not callable(_get_user_by_email):
        raise TypeError("get_user_by_email must be a callable")
    if not isinstance(_user_model, type):
        raise TypeError("user_model must be a class type")

    # Verify UserModel has required attributes
    required_attrs = ['email', 'is_active', 'hashed_password']
    missing_attrs = [attr for attr in required_attrs if not hasattr(_user_model, attr)]
    if missing_attrs:
        raise ValueError(f"UserModel missing required attributes: {missing_attrs}")

    get_user_by_email = _get_user_by_email
    UserModel = _user_model
    logger.info("Security dependencies initialized successfully with type checking.")

# --- Configuration (SECRET_KEY, ALGORITHM, etc. - unchanged) ---
SECRET_KEY = os.getenv("SECRET_KEY", "dummy_insecure_secret_key_replace_in_env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
if SECRET_KEY == "dummy_insecure_secret_key_replace_in_env":
     logger.critical("FATAL SECURITY WARNING: SECRET_KEY environment variable not set.")

# --- Password Hashing Setup (unchanged) ---
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated=["bcrypt"])

# --- OAuth2 Scheme Setup (unchanged) ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token") # Matches endpoint in main.py

# --- Pydantic model for token data (unchanged) ---
class TokenData(BaseModel):
    email: Optional[str] = None

# --- Helper Functions (verify_password, get_password_hash, create_access_token - unchanged) ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password: return False
    try: return pwd_context.verify(plain_password, hashed_password)
    except Exception as e: logger.error(f"Error verifying password: {e}", exc_info=True); return False

def get_password_hash(password: str) -> str:
    if not password: raise ValueError("Password cannot be empty")
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta: expire = datetime.now(timezone.utc) + expires_delta
    else: expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    if "sub" not in to_encode and "email" in to_encode: to_encode["sub"] = to_encode.pop("email")
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Dependency to Decode Token (decode_access_token - unchanged) ---
async def decode_access_token(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: Optional[str] = payload.get("sub")
        if email is None: logger.warning("Token payload missing 'sub'."); raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError as e: logger.warning(f"JWTError decoding token: {e}"); raise credentials_exception
    except ValidationError as e: logger.warning(f"Token Data validation error: {e}"); raise credentials_exception
    except Exception as e: logger.exception(f"Unexpected error decoding token: {e}"); raise credentials_exception
    return token_data

# --- Dependency to Get Current User (Refactored previously - unchanged here) ---
async def get_current_user(
    token_data: TokenData = Depends(decode_access_token),
    db: Session = Depends(get_db) # Uses managed session
) -> Any:
    """
    Dependency to fetch the user from DB based on token data.
    Uses a managed DB session from get_db dependency.
    """
    # Check if dependencies were initialized correctly
    if not callable(get_user_by_email) or UserModel is None:
        logger.critical("Security dependencies (get_user_by_email/UserModel) not initialized!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error [Security Dep Init]"
        )
    if token_data.email is None:
        logger.error("Token data email is None in get_current_user.")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token data", headers={"WWW-Authenticate": "Bearer"})

    try:
        logger.debug("Fetching user in get_current_user for email: %s", token_data.email)
        # Use the injected, managed db session directly
        user = get_user_by_email(db=db, email=token_data.email)

        if user is None:
            logger.warning("User '%s' from token not found in database.", token_data.email)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User associated with token not found", headers={"WWW-Authenticate": "Bearer"})

        logger.debug("User '%s' found in database.", token_data.email)
        return user

    except HTTPException:
        # Let HTTPException pass through, no rollback needed here
        raise
    except Exception as e:
        # Log other unexpected errors during DB lookup
        logger.exception(f"Error fetching user '{token_data.email}' in get_current_user: {e}")
        # No rollback needed here, get_db handles it if error propagates
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error retrieving user information.")

# --- Dependency to Get Current Active User (Unchanged) ---
async def get_current_active_user(
    current_user: Any = Depends(get_current_user) # Depends on refactored get_current_user
) -> Any:
    """Dependency to get the current *active* user."""
    if not getattr(current_user, "is_active", False):
        logger.warning("Authentication attempt by inactive user: %s", getattr(current_user, 'email', 'N/A'))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user
