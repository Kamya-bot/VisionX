"""
VisionX — Authentication API Routes
Real auth: bcrypt password hashing + JWT tokens
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import logging

from jose import JWTError, jwt
from passlib.context import CryptContext

from database import get_db
from crud import create_user, get_user_by_email, get_user_by_id
from config import settings
from schemas.auth_models import (
    RegisterRequest, LoginRequest, TokenResponse, UserProfileResponse
)
import models

router = APIRouter()
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer scheme
bearer_scheme = HTTPBearer(auto_error=False)

# Cluster labels (mirrors config.py)
CLUSTER_LABELS = {
    0: "Casual User",
    1: "Analytical Researcher",
    2: "High Intent Buyer",
    3: "Power Decision Maker"
}


# ─── Password helpers ───────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ─── JWT helpers ────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None


# ─── Auth dependency ─────────────────────────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Extract and validate JWT; return the User row."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db)
) -> Optional[models.User]:
    """Same as get_current_user but returns None instead of raising if not authed."""
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    return get_user_by_id(db, payload["sub"])


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user.

    - Hashes password with bcrypt
    - Stores user in DB
    - Returns a JWT access token immediately (auto-login on register)
    """
    # Check duplicate
    existing = get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists"
        )

    # Derive username from email
    username_base = body.email.split("@")[0]
    username = username_base

    # Ensure username unique
    counter = 1
    while db.query(models.User).filter(models.User.username == username).first():
        username = f"{username_base}{counter}"
        counter += 1

    hashed = hash_password(body.password)
    user = create_user(
        db,
        email=body.email,
        username=username,
        hashed_password=hashed,
        full_name=body.full_name or username_base.capitalize()
    )

    token = create_access_token(user.id, user.email)
    cluster_label = CLUSTER_LABELS.get(user.cluster_id) if user.cluster_id is not None else None

    logger.info(f"New user registered: {user.email} (id={user.id})")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        cluster_id=user.cluster_id,
        cluster_label=cluster_label
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user and return JWT token.
    """
    user = get_user_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )

    token = create_access_token(user.id, user.email)
    cluster_label = CLUSTER_LABELS.get(user.cluster_id) if user.cluster_id is not None else None

    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        cluster_id=user.cluster_id,
        cluster_label=cluster_label
    )


@router.get("/auth/me", response_model=UserProfileResponse)
async def get_me(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current authenticated user's profile.
    Requires Bearer token in Authorization header.
    """
    # Count their predictions
    total_predictions = db.query(models.PredictionLog)\
        .filter(models.PredictionLog.user_id == current_user.id)\
        .count()

    cluster_label = CLUSTER_LABELS.get(current_user.cluster_id) if current_user.cluster_id is not None else None

    return UserProfileResponse(
        user_id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        cluster_id=current_user.cluster_id,
        cluster_label=cluster_label,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
        total_predictions=total_predictions
    )
