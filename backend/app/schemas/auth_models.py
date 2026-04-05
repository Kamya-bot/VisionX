"""
Auth Schemas — request/response models for authentication
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    full_name: Optional[str] = Field(None, description="Optional display name")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword",
                "full_name": "Jane Smith"
            }
        }


class LoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securepassword"
            }
        }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    full_name: Optional[str]
    cluster_id: Optional[int]
    cluster_label: Optional[str]


class UserProfileResponse(BaseModel):
    user_id: str
    email: str
    full_name: Optional[str]
    cluster_id: Optional[int]
    cluster_label: Optional[str]
    created_at: str
    total_predictions: int
