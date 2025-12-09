from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_database_url():
    """Helper to retrieve DB URL in scripts context"""
    return DATABASE_URL

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    # CHANGE: Increase pool size to handle concurrent games + admin usage
    pool_size=20,     
    max_overflow=20
)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# Dependency for API routes
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session