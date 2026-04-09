"""
VisionX — Authentication API Routes

Implements:
  - Email/password register + login
  - JWT access token (15 min) + refresh token (7 days) with rotation-on-use
  - Refresh token theft detection via family invalidation
  - Google OAuth2 PKCE flow
  - GitHub OAuth2 flow
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from config import settings
from crud import create_user, get_user_by_email, get_user_by_id
from database import get_db
from schemas.auth_models import (
    AccessTokenResponse,
    LoginRequest,
    OAuthURLResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserProfileResponse,
)
import models

router = APIRouter()
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

CLUSTER_LABELS = {
    0: "Independent Thinker & Risk-Averse",
    1: "Growth-Oriented & Value-Conscious",
    2: "Budget Pragmatist & Stability-Seeker",
    3: "Socially-Validated & Speed-Driven",
}

# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Access token ──────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


# ── Refresh token ─────────────────────────────────────────────────────────────

def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def create_refresh_token(
    user_id: str,
    db: Session,
    family_id: Optional[str] = None,
) -> str:
    """
    Creates a signed refresh JWT, stores its hash in the DB.
    family_id groups tokens for rotation — if a used token is replayed,
    we revoke the whole family (theft detection).
    """
    jti = str(uuid.uuid4())
    raw_token = secrets.token_urlsafe(48)   # high-entropy random part
    family = family_id or str(uuid.uuid4())
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    # Embed jti + family in the JWT so the client holds them
    payload = {
        "sub": user_id,
        "jti": jti,
        "family": family,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
        "rand": raw_token,   # makes each token unique even with same jti
    }
    signed = jwt.encode(payload, settings.REFRESH_SECRET_KEY, algorithm=settings.ALGORITHM)

    rt = models.RefreshToken(
        user_id=user_id,
        jti=jti,
        token_hash=_hash_token(signed),
        family_id=family,
        expires_at=expire,
    )
    db.add(rt)
    db.commit()
    return signed


def rotate_refresh_token(old_token: str, db: Session) -> tuple[str, str]:
    """
    Validates old refresh token, issues new access + refresh token pair.
    Returns (new_access_token, new_refresh_token).
    Raises HTTPException on any failure.
    """
    try:
        payload = jwt.decode(
            old_token, settings.REFRESH_SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    jti = payload.get("jti")
    family = payload.get("family")
    user_id = payload.get("sub")

    if not jti or not family or not user_id:
        raise HTTPException(status_code=401, detail="Malformed refresh token")

    rt_row = db.query(models.RefreshToken).filter(models.RefreshToken.jti == jti).first()

    if not rt_row:
        raise HTTPException(status_code=401, detail="Refresh token not found")

    if rt_row.revoked:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    if rt_row.used:
        # THEFT DETECTED — revoke entire family
        db.query(models.RefreshToken)\
            .filter(models.RefreshToken.family_id == family)\
            .update({"revoked": True})
        db.commit()
        logger.warning(f"Refresh token replay detected for user {user_id}, family {family} revoked")
        raise HTTPException(
            status_code=401,
            detail="Refresh token already used. Please log in again."
        )

    if rt_row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # Mark old token as used
    rt_row.used = True
    rt_row.used_at = datetime.utcnow()
    db.commit()

    user = get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    new_access = create_access_token(user.id, user.email)
    new_refresh = create_refresh_token(user.id, db, family_id=family)
    return new_access, new_refresh


# ── Auth dependency ───────────────────────────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.User:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = get_user_by_id(db, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    if not credentials:
        return None
    payload = decode_access_token(credentials.credentials)
    if not payload:
        return None
    return get_user_by_id(db, payload["sub"])


# ── Helper: build TokenResponse ───────────────────────────────────────────────

def _token_response(user: models.User, db: Session, family_id: Optional[str] = None) -> TokenResponse:
    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id, db, family_id=family_id)
    label = CLUSTER_LABELS.get(user.cluster_id) if user.cluster_id is not None else None
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_id=user.id,
        email=user.email,
        full_name=user.full_name,
        cluster_id=user.cluster_id,
        cluster_label=label,
    )


# ── Routes: email/password ────────────────────────────────────────────────────

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    username_base = body.email.split("@")[0]
    username = username_base
    counter = 1
    while db.query(models.User).filter(models.User.username == username).first():
        username = f"{username_base}{counter}"
        counter += 1

    user = create_user(
        db,
        email=body.email,
        username=username,
        hashed_password=hash_password(body.password),
        full_name=body.full_name or username_base.capitalize(),
    )
    logger.info(f"New user registered: {user.email} (id={user.id})")
    return _token_response(user, db)


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, body.email)
    if not user or not user.hashed_password or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    logger.info(f"User logged in: {user.email}")
    return _token_response(user, db)


@router.post("/auth/refresh", response_model=AccessTokenResponse)
async def refresh_token(body: RefreshRequest, db: Session = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token + rotated refresh token.
    Old refresh token is immediately invalidated.
    """
    new_access, new_refresh = rotate_refresh_token(body.refresh_token, db)
    return AccessTokenResponse(
        access_token=new_access,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/auth/logout")
async def logout(
    body: RefreshRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Revoke the provided refresh token (client should discard access token)."""
    try:
        payload = jwt.decode(
            body.refresh_token, settings.REFRESH_SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        jti = payload.get("jti")
        if jti:
            db.query(models.RefreshToken)\
                .filter(models.RefreshToken.jti == jti)\
                .update({"revoked": True})
            db.commit()
    except JWTError:
        pass   # invalid token — still return 200, client is logging out anyway
    return {"message": "Logged out successfully"}


@router.get("/auth/me", response_model=UserProfileResponse)
async def get_me(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total_predictions = db.query(models.PredictionLog)\
        .filter(models.PredictionLog.user_id == current_user.id)\
        .count()
    label = CLUSTER_LABELS.get(current_user.cluster_id) if current_user.cluster_id is not None else None
    return UserProfileResponse(
        user_id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        cluster_id=current_user.cluster_id,
        cluster_label=label,
        created_at=current_user.created_at.isoformat() if current_user.created_at else "",
        total_predictions=total_predictions,
    )


# ── Routes: OAuth — Google ────────────────────────────────────────────────────

@router.get("/auth/oauth/google/url", response_model=OAuthURLResponse)
async def google_oauth_url():
    """Frontend calls this to get the Google OAuth URL + state param."""
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    state = secrets.token_urlsafe(32)
    params = (
        f"client_id={settings.GOOGLE_CLIENT_ID}"
        f"&redirect_uri={settings.GOOGLE_REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&state={state}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    return OAuthURLResponse(
        url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}",
        state=state,
    )


@router.get("/auth/oauth/google/callback")
async def google_callback(code: str, state: str, db: Session = Depends(get_db)):
    """
    Google redirects here after user approves.
    Exchanges code for tokens, upserts user, returns VisionX tokens.
    """
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")
    async with httpx.AsyncClient() as client:
        # Exchange code for tokens
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Google token exchange failed")
        token_data = token_resp.json()

        # Get user info
        userinfo_resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not fetch Google user info")
        info = userinfo_resp.json()

    user = _upsert_oauth_user(
        db,
        provider="google",
        sub=info["sub"],
        email=info["email"],
        full_name=info.get("name"),
        avatar_url=info.get("picture"),
    )
    return _token_response(user, db)


# ── Routes: OAuth — GitHub ────────────────────────────────────────────────────

@router.get("/auth/oauth/github/url", response_model=OAuthURLResponse)
async def github_oauth_url():
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    state = secrets.token_urlsafe(32)
    params = (
        f"client_id={settings.GITHUB_CLIENT_ID}"
        f"&redirect_uri={settings.GITHUB_REDIRECT_URI}"
        f"&scope=user:email"
        f"&state={state}"
    )
    return OAuthURLResponse(
        url=f"https://github.com/login/oauth/authorize?{params}",
        state=state,
    )


@router.get("/auth/oauth/github/callback")
async def github_callback(code: str, state: str, db: Session = Depends(get_db)):
    if not settings.GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": settings.GITHUB_REDIRECT_URI,
            },
        )
        if token_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="GitHub token exchange failed")
        gh_token = token_resp.json().get("access_token")

        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/json"},
        )
        if user_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not fetch GitHub user info")
        gh_user = user_resp.json()

        # GitHub may not expose email directly — fetch from emails endpoint
        email = gh_user.get("email")
        if not email:
            emails_resp = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/json"},
            )
            if emails_resp.status_code == 200:
                emails = emails_resp.json()
                primary = next((e["email"] for e in emails if e.get("primary") and e.get("verified")), None)
                email = primary or (emails[0]["email"] if emails else None)

    if not email:
        raise HTTPException(status_code=400, detail="Could not retrieve email from GitHub")

    user = _upsert_oauth_user(
        db,
        provider="github",
        sub=str(gh_user["id"]),
        email=email,
        full_name=gh_user.get("name") or gh_user.get("login"),
        avatar_url=gh_user.get("avatar_url"),
    )
    return _token_response(user, db)


# ── OAuth user upsert ─────────────────────────────────────────────────────────

def _upsert_oauth_user(
    db: Session,
    provider: str,
    sub: str,
    email: str,
    full_name: Optional[str],
    avatar_url: Optional[str],
) -> models.User:
    """
    Find existing user by oauth_sub+provider or email.
    Create if not found. Never overwrites a password-auth user's password.
    """
    # 1. Try to find by provider sub
    user = db.query(models.User).filter(
        models.User.oauth_provider == provider,
        models.User.oauth_sub == sub,
    ).first()

    if not user:
        # 2. Try by email (user might have registered with email first)
        user = get_user_by_email(db, email)

    if user:
        # Update OAuth fields if missing
        if not user.oauth_sub:
            user.oauth_sub = sub
            user.oauth_provider = provider
        if avatar_url and not user.avatar_url:
            user.avatar_url = avatar_url
        db.commit()
        db.refresh(user)
        return user

    # 3. Create new OAuth user
    username_base = email.split("@")[0]
    username = username_base
    counter = 1
    while db.query(models.User).filter(models.User.username == username).first():
        username = f"{username_base}{counter}"
        counter += 1

    user = models.User(
        email=email,
        username=username,
        hashed_password=None,   # no password for OAuth users
        full_name=full_name or username_base.capitalize(),
        avatar_url=avatar_url,
        oauth_provider=provider,
        oauth_sub=sub,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"New OAuth user ({provider}): {user.email}")
    return user