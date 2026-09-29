"""Signed webhook verification (spec section 10).

HMAC-SHA256 over ``{timestamp}.{raw body}``, compared in constant time, with a
five-minute freshness window.

What changed from the previous implementation
---------------------------------------------
``app/utils/hmac_validator.py`` signed the body alone and never checked
freshness. It used ``hmac.compare_digest`` correctly, so the signature could not
be forged — but a valid request captured once could be replayed forever, and
``validate_hubspot_signature`` even accepted a ``timestamp`` argument, folded it
into the digest, and then never looked at it. Binding the timestamp into the
signed string *and* rejecting stale timestamps is what actually stops replay:
either half alone does not.

The raw body must be signed exactly as received. Re-serialising parsed JSON
reorders keys and changes whitespace, which breaks verification for reasons that
are painful to debug.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass


class SignatureError(Exception):
    """Verification failed. The message is safe to return to the caller."""


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    age_seconds: float


def sign(payload: bytes, secret: str, timestamp: int | str) -> str:
    """Produce the hex signature for a payload. Also used by tests and clients."""
    signed_payload = f"{timestamp}.".encode() + payload
    return hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()


def _strip_prefix(signature: str) -> str:
    """Accept both ``sha256=abc...`` and a bare hex digest."""
    signature = signature.strip()
    if "=" in signature:
        scheme, _, value = signature.partition("=")
        if scheme.lower() in ("sha256", "v1"):
            return value.strip()
    return signature


def verify(
    payload: bytes,
    signature_header: str | None,
    timestamp_header: str | None,
    secret: str,
    tolerance_seconds: int = 300,
    now: float | None = None,
) -> VerificationResult:
    """Verify a signed webhook, or raise SignatureError.

    Args:
        payload: the raw request body, exactly as received.
        signature_header: ``X-CRM-Signature``.
        timestamp_header: ``X-CRM-Timestamp``, unix seconds.
        tolerance_seconds: how old a request may be. Also bounds clock skew in
            the future direction, so a forged far-future timestamp cannot buy
            an indefinitely valid signature.
    """
    if not secret:
        raise SignatureError("webhook secret is not configured")
    if not signature_header:
        raise SignatureError("missing signature header")
    if not timestamp_header:
        raise SignatureError("missing timestamp header")

    try:
        timestamp = int(str(timestamp_header).strip())
    except (TypeError, ValueError):
        raise SignatureError("timestamp header is not an integer")

    now = time.time() if now is None else now
    age = now - timestamp
    if age > tolerance_seconds:
        raise SignatureError(
            f"request is {age:.0f}s old; tolerance is {tolerance_seconds}s"
        )
    if age < -tolerance_seconds:
        raise SignatureError("timestamp is too far in the future")

    expected = sign(payload, secret, timestamp)
    provided = _strip_prefix(signature_header)

    # Constant-time: a plain == leaks the position of the first wrong byte
    # through timing, which is enough to forge a signature given enough tries.
    if not hmac.compare_digest(expected, provided):
        raise SignatureError("signature mismatch")

    return VerificationResult(ok=True, age_seconds=age)
