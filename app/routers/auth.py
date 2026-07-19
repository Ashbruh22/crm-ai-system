from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta

from app.config import settings

router = APIRouter()

VALID_USERS = {
    "test_rep":   {"password": "test_pass", "role": "sales_rep"},
    "test_manager": {"password": "test_pass", "role": "manager"},
    "test_admin": {"password": "test_pass", "role": "admin"},
}

class TokenRequest(BaseModel):
    username: str
    password: str
    role: str | None = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str

@router.post("/token", response_model=TokenResponse)
def login(req: TokenRequest):
    user = VALID_USERS.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    payload = {
        "sub": req.username,
        "role": user["role"],
        "exp": datetime.utcnow() + timedelta(hours=8),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    return TokenResponse(access_token=token, role=user["role"])
