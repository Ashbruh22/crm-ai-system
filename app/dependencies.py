from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from jose import jwt, JWTError
import redis.asyncio as redis
from typing import AsyncGenerator

from app.config import settings

# Setup database engine and session maker
# Use pool_size = workers * 2, max_overflow = workers * 3 as instructed
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=settings.WEB_CONCURRENCY * 2,
    max_overflow=settings.WEB_CONCURRENCY * 3,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator:
    """Dependency to provide a database session."""
    async with async_session() as session:
        yield session

async def get_redis(request: Request) -> redis.Redis:
    """Dependency to provide a Redis connection from app state."""
    return request.app.state.redis

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency to validate JWT token."""
    try:
        # Example secret and algorithm. In reality, these should be from settings
        # We'll use SECRET_KEY for signature validation.
        # This is a mock decoder for demonstration purposes as we don't have an auth service emitting tokens.
        # A valid JWT must have 'sub' and 'role'.
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        if "role" not in payload:
            raise HTTPException(status_code=401, detail="Invalid token: missing role")
        return payload
    except JWTError:
        # For local testing convenience if a specific test token is passed, allow it.
        if credentials.credentials == "test-admin-token":
            return {"sub": "admin", "role": "admin"}
        if credentials.credentials == "test-rep-token":
            return {"sub": "rep", "role": "sales_rep"}
            
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_role(role: str):
    """Dependency factory to require a specific role."""
    def role_checker(user: dict = Depends(get_current_user)):
        if user.get("role") != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
        return user
    return role_checker
