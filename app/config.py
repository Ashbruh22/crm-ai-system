"""Settings, validated at import time (spec section 2: "fail fast on bad env").

A bad or missing env var should stop the process at startup with a readable
message, not surface later as a 500 from a handler.
"""

from __future__ import annotations

import os
from functools import cached_property

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    # --- required ------------------------------------------------------------
    SECRET_KEY: str = Field(min_length=8)
    DATABASE_URL: str
    REDIS_URL: str

    # --- service -------------------------------------------------------------
    WEB_CONCURRENCY: int = Field(default=4, ge=1, le=32)
    DOCS_ENABLED: bool = True
    BATCH_CONCURRENCY: int = Field(default=10, ge=1, le=100)
    MODEL_VERSION: str | None = None
    ARTIFACT_DIR: str = os.path.join(REPO_ROOT, "artifacts")
    SYNTHETIC_DATA_DIR: str = os.path.join(REPO_ROOT, "data", "synthetic")

    # --- CORS ----------------------------------------------------------------
    #: Comma-separated dashboard origins. The old code sent
    #: allow_origins=["*"] together with allow_credentials=True, which browsers
    #: reject outright and which the spec forbids (section 10).
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- ingestion -----------------------------------------------------------
    CRM_WEBHOOK_SECRET: str = "your-hmac-secret-here"
    CRM_SIMULATE_ENABLED: bool = True
    #: Replay window for signed webhooks, in seconds (spec section 10).
    WEBHOOK_TIMESTAMP_TOLERANCE: int = Field(default=300, ge=30, le=3600)

    # --- admin ---------------------------------------------------------------
    ADMIN_TOKEN: str | None = None

    # --- startup behaviour ---------------------------------------------------
    #: Seed the live demo set on boot when the deals table is empty.
    SEED_ON_STARTUP: bool = True
    #: Refuse to start when artifacts are missing or the schema has drifted.
    #: Only turn this off for tests that never touch the models.
    REQUIRE_ARTIFACTS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @field_validator("DATABASE_URL")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        """Catch the sync-driver mistake early.

        SQLAlchemy's async engine needs an async driver. A plain
        ``postgresql://`` URL fails at first query with a confusing error, so
        normalise it here instead.
        """
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        if v.startswith("sqlite://") and "+aiosqlite" not in v:
            return v.replace("sqlite://", "sqlite+aiosqlite://", 1)
        return v

    @cached_property
    def allowed_origins(self) -> list[str]:
        """ALLOWED_ORIGINS parsed into a list, wildcard rejected."""
        origins = [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]
        if "*" in origins:
            raise ValueError(
                "ALLOWED_ORIGINS must name the dashboard origins explicitly; "
                "'*' cannot be combined with credentialed requests"
            )
        return origins

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


settings = Settings()
