import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi import Request
from dotenv import load_dotenv

load_dotenv()

# Explicitly load both from env
DATABASE_URL = os.getenv("DATABASE_URL")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

# Strict check: prevent startup if URLs are missing
if not DATABASE_URL or not TEST_DATABASE_URL:
    raise ValueError("Missing DATABASE_URL or TEST_DATABASE_URL in environment.")

# Factory for engines
def create_engine(url):
    return create_async_engine(
        url, 
        pool_size=50,         # Match max expected tournament concurrency
        max_overflow=30,      # Allow temporary spikes
        pool_timeout=30,      # Wait 30s for a connection before failing
        pool_recycle=1800, 
        echo=False
    )

engines = {
    "prod": create_engine(DATABASE_URL),
    "test": create_engine(TEST_DATABASE_URL)
}

session_makers = {
    "prod": sessionmaker(bind=engines["prod"], class_=AsyncSession, expire_on_commit=False),
    "test": sessionmaker(bind=engines["test"], class_=AsyncSession, expire_on_commit=False)
}

Base = declarative_base()

# Helper to get specific session factory (for background scripts)
def get_session_maker(env: str):
    return session_makers.get(env, session_makers["prod"])

# Helper for backward compatibility (used by scripts)
def get_database_url():
    """Helper to retrieve prod DB URL in scripts context"""
    return DATABASE_URL

# Dependency Injection for API
async def get_db(request: Request):
    # Logic: Read Header -> Select DB
    env = request.headers.get("x-db-env", "prod")
    if env not in ["prod", "test"]: 
        env = "prod"
    
    async with session_makers[env]() as session:
        yield session

# Backward compatibility: keep AsyncSessionLocal for existing imports
AsyncSessionLocal = session_makers["prod"]