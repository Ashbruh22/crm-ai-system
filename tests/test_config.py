"""Settings validation (spec section 2: "fail fast on bad env")."""

from __future__ import annotations

import pytest

BASE = {
    "SECRET_KEY": "x" * 16,
    "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
}


def build(**overrides):
    from app.config import Settings

    return Settings(**{**BASE, **overrides})


def test_a_pasted_shell_command_is_rejected_by_name():
    """The mistake that took a real deploy down.

    Upstash presents the credential as a runnable command. Pasted whole into a
    deployment's environment, redis-py raised a bare scheme error forty frames
    deep inside a connection pool, and the service crashed on boot with no clue
    as to which variable was wrong.
    """
    with pytest.raises(ValueError, match="shell command"):
        build(REDIS_URL="redis-cli --tls -u rediss://d:p@h.upstash.io:6379")


def test_upstash_without_tls_is_rejected():
    with pytest.raises(ValueError, match="two s's"):
        build(REDIS_URL="redis://default:tok@fun-cobra.upstash.io:6379")


def test_a_bare_host_and_port_is_rejected():
    with pytest.raises(ValueError, match="must start with"):
        build(REDIS_URL="localhost:6379")


@pytest.mark.parametrize(
    "url",
    [
        "rediss://default:tok@fun-cobra.upstash.io:6379",
        "redis://localhost:6379/0",
        "unix:///var/run/redis.sock",
    ],
)
def test_valid_urls_pass(url):
    assert build(REDIS_URL=url).REDIS_URL == url


def test_surrounding_whitespace_is_tolerated():
    """Copying from a terminal or a web console often picks up a newline."""
    assert build(
        REDIS_URL="  rediss://default:tok@h.upstash.io:6379\n"
    ).REDIS_URL == "rediss://default:tok@h.upstash.io:6379"


def test_a_sync_postgres_url_is_normalised_to_async():
    """SQLAlchemy's async engine needs an async driver; the sync form fails at
    first query with a confusing error rather than at startup."""
    settings = build(
        REDIS_URL="redis://localhost:6379/0",
        DATABASE_URL="postgresql://u:p@host:5432/db",
    )
    assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")


def test_wildcard_cors_origin_is_refused():
    settings = build(REDIS_URL="redis://localhost:6379/0", ALLOWED_ORIGINS="*")
    with pytest.raises(ValueError, match="explicitly"):
        _ = settings.allowed_origins
