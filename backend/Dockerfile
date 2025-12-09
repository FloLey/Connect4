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

# Start Command with Hot Reload enabled
# 1. Initialize DB tables
# 2. Start Uvicorn with --reload
CMD ["sh", "-c", "python -m backend.scripts.init_db && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"]