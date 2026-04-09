"""
VisionX — Authentication Schemas
Pydantic models for auth request/response validation
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
import re


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int          # seconds until access token expires
    user_id: str
    email: str
    full_name: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None


class AccessTokenResponse(BaseModel):
    """Returned by /auth/refresh — only new access token, no new refresh token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    created_at: str
    total_predictions: int


class OAuthCallbackRequest(BaseModel):
    """Used when frontend sends back the OAuth code."""
    code: str
    state: str
    provider: str   # "google" | "github"


class OAuthURLResponse(BaseModel):
    url: str
    state: str