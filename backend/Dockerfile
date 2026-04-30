# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL driver (psycopg2/asyncpg)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (Needed for the initial build context)
COPY . backend/

# Set Python path so imports work correctly
ENV PYTHONPATH=/app

# Start Command with Hot Reload enabled.
# Schema is owned by Alembic since Tier 2.3 — `alembic upgrade head` is a
# no-op when the DB is already at head and applies any pending migrations
# otherwise. The legacy backend.scripts.init_db was removed in Tier 4.A.
CMD ["sh", "-c", "cd backend && alembic upgrade head && cd .. && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"]