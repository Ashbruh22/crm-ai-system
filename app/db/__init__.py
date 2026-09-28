"""Demo database layer (spec section 6).

`app/db/models.py` holds the schema the hosted demo runs on: deals, activities,
scores, actions, processed_events. It shares ``Base`` with the legacy models in
`app/models/db.py`, so both live in one Alembic metadata and one migration
history.

Types are dialect-portable on purpose (see `types.py`): PostgreSQL in Docker and
on Render, SQLite for fast local test runs.
"""
