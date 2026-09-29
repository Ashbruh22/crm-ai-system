"""Admin authentication (spec section 10).

A single bearer token, compared in constant time. Not a user system: there is
one operator and the only privileged action is resetting a demo full of
generated data.

The service starts with ``ADMIN_TOKEN`` unset, and in that state admin routes
refuse everything rather than falling open. A default token would be worse than
no token, because it would look protected.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

#: auto_error=False so a missing header produces our 401 with a useful message
#: rather than FastAPI's bare "Not authenticated".
_scheme = HTTPBearer(auto_error=False)

#: Below this, a token is guessable enough not to count as protection.
MIN_TOKEN_LENGTH = 16


def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Depends(_scheme),
) -> str:
    """Reject anything without a valid admin bearer token."""
    configured = (settings.ADMIN_TOKEN or "").strip()

    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Admin actions are disabled because ADMIN_TOKEN is not set. "
                "Set it in the environment to enable them."
            ),
        )

    if len(configured) < MIN_TOKEN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"ADMIN_TOKEN is shorter than {MIN_TOKEN_LENGTH} characters; "
                "admin actions stay disabled rather than pretend to be protected."
            ),
        )

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time: a plain == leaks the token a character at a time.
    if not hmac.compare_digest(credentials.credentials, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "admin"
