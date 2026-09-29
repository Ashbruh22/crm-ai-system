"""Webhook signature verification (spec section 10)."""

from __future__ import annotations

import time

import pytest

from app.security.webhook import SignatureError, sign, verify

SECRET = "test-webhook-secret"
BODY = b'{"deal_id":"LIVE-0001","type":"email_replied"}'


def headers_for(body: bytes = BODY, secret: str = SECRET, ts: int | None = None):
    ts = int(time.time()) if ts is None else ts
    return sign(body, secret, ts), str(ts)


def test_valid_signature_passes():
    signature, ts = headers_for()
    result = verify(BODY, signature, ts, SECRET)
    assert result.ok is True
    assert abs(result.age_seconds) < 5


def test_sha256_prefix_is_accepted():
    signature, ts = headers_for()
    assert verify(BODY, f"sha256={signature}", ts, SECRET).ok


def test_wrong_secret_is_rejected():
    signature, ts = headers_for(secret="a-different-secret")
    with pytest.raises(SignatureError, match="signature mismatch"):
        verify(BODY, signature, ts, SECRET)


def test_tampered_body_is_rejected():
    signature, ts = headers_for()
    with pytest.raises(SignatureError, match="signature mismatch"):
        verify(BODY + b" ", signature, ts, SECRET)


def test_signature_is_bound_to_the_timestamp():
    """A replay with a fresh timestamp must not verify.

    This is the property the old implementation lacked: it signed the body
    alone, so a captured signature stayed valid for any timestamp forever.
    """
    signature, original_ts = headers_for(ts=int(time.time()) - 200)
    replay_ts = str(int(time.time()))

    with pytest.raises(SignatureError, match="signature mismatch"):
        verify(BODY, signature, replay_ts, SECRET)

    # The same signature still verifies against its own timestamp...
    assert verify(BODY, signature, original_ts, SECRET).ok


def test_stale_request_is_rejected_even_with_a_valid_signature():
    old = int(time.time()) - 600
    signature, ts = headers_for(ts=old)
    with pytest.raises(SignatureError, match="old"):
        verify(BODY, signature, ts, SECRET, tolerance_seconds=300)


def test_request_inside_the_window_is_accepted():
    recent = int(time.time()) - 120
    signature, ts = headers_for(ts=recent)
    assert verify(BODY, signature, ts, SECRET, tolerance_seconds=300).ok


def test_far_future_timestamp_is_rejected():
    """Otherwise a forged future timestamp buys an indefinitely valid signature."""
    future = int(time.time()) + 4000
    signature, ts = headers_for(ts=future)
    with pytest.raises(SignatureError, match="future"):
        verify(BODY, signature, ts, SECRET, tolerance_seconds=300)


@pytest.mark.parametrize(
    "signature,timestamp,expected",
    [
        (None, "123", "missing signature"),
        ("abc", None, "missing timestamp"),
        ("abc", "not-a-number", "not an integer"),
    ],
)
def test_missing_or_malformed_headers(signature, timestamp, expected):
    with pytest.raises(SignatureError, match=expected):
        verify(BODY, signature, timestamp, SECRET)


def test_unconfigured_secret_fails_closed():
    signature, ts = headers_for()
    with pytest.raises(SignatureError, match="not configured"):
        verify(BODY, signature, ts, "")
