"""Dialect-portable column types.

The legacy models import ``sqlalchemy.dialects.postgresql.UUID`` directly, which
pins every primary key to PostgreSQL and makes both a SQLite test run and any
future MySQL move a rewrite. ``GUID`` below stores a native ``UUID`` on
PostgreSQL and a ``CHAR(36)`` everywhere else, so the same models run against
Postgres in Docker, SQLite in tests, and MySQL if that ever comes up.
"""

from __future__ import annotations

import uuid

from sqlalchemy import CHAR, TypeDecorator
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


class GUID(TypeDecorator):
    """UUID primary key that works on any dialect.

    Values are handed back as ``uuid.UUID`` regardless of storage, so callers
    never have to care which database they are on.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            value = uuid.UUID(str(value))
        if dialect.name == "postgresql":
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))


def new_uuid() -> uuid.UUID:
    """Default factory for GUID primary keys."""
    return uuid.uuid4()
