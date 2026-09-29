#!/bin/sh
# Migrate, then serve (spec section 12).
#
# Migrations run here rather than in the app's lifespan so that N workers do not
# race to apply the same revision. Seeding stays in the lifespan, where it is
# guarded by a row count and is safe to attempt more than once.
set -e

echo "==> alembic upgrade head"
alembic upgrade head

echo "==> starting: $*"
exec "$@"
