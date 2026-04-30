# Database migrations

Schema is owned by Alembic (introduced in Tier 2.3). The legacy
`backend/scripts/init_db.py` was removed — do not bring it back.

## New environment from scratch

```bash
# 1. Create the prod and test databases (one-time, postgres superuser).
psql -U postgres -c 'CREATE DATABASE connect4_arena OWNER "user";'
psql -U postgres -c 'CREATE DATABASE connect4_test  OWNER "user";'

# 2. Apply the full migration history.
cd backend
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5433/connect4_arena \
    alembic upgrade head
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5433/connect4_test \
    alembic upgrade head
```

Inside Docker, the `db` host resolves to the `postgres` container — same
commands, swap the URL host:

```bash
docker exec connect4-backend-1 sh -c "cd /app/backend && \
    DATABASE_URL=postgresql+asyncpg://user:password@db:5432/connect4_arena \
    alembic upgrade head"
```

## Existing database that pre-dates Alembic

If the schema is already correct (created by an old `init_db.py` run), stamp
it at the baseline so Alembic skips the initial migration:

```bash
DATABASE_URL=… alembic stamp 0001_initial_schema
DATABASE_URL=… alembic upgrade head   # applies any later migrations
```

## Adding a schema change

1. Edit ORM models in `backend/app/models/`.
2. Generate the migration:

   ```bash
   cd backend
   DATABASE_URL=… alembic revision --autogenerate -m "short description"
   ```

3. Open the generated file in `backend/alembic/versions/` and **inspect it**
   — autogenerate is a starting point, not gospel. Pay attention to:
   - server defaults (Alembic doesn't always emit them)
   - JSONB column types
   - data migrations (autogenerate never does these)
4. Apply it:

   ```bash
   DATABASE_URL=… alembic upgrade head
   ```

5. Commit the migration file alongside the model change.

## Useful one-liners

| Command | Meaning |
|---|---|
| `alembic current` | Which revision the DB is on. |
| `alembic history` | All revisions. |
| `alembic check` | Does the DB match the ORM? (Empty diff = yes.) |
| `alembic downgrade -1` | Roll back one revision. Use sparingly. |
