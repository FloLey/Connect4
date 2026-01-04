import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from backend.app.core.database import Base
# IMPORT ALL MODELS
from backend.app.models.game_model import Game
from backend.app.models.elo_model import EloRating, EloHistory
from backend.app.models.tournament_model import Tournament

async def create_database_if_not_exists(db_url: str, db_name: str):
    """Create database if it doesn't exist"""
    # Connect to default postgres database
    # Replace database name with postgres
    if "/" in db_url:
        parts = db_url.rsplit("/", 1)
        default_url = parts[0] + "/postgres"
    else:
        default_url = db_url + "/postgres"
    
    engine = create_async_engine(default_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            # Check if database exists
            result = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": db_name}
            )
            exists = result.scalar() is not None
            if not exists:
                print(f"Creating database '{db_name}'...")
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                print(f"Database '{db_name}' created.")
            else:
                print(f"Database '{db_name}' already exists.")
    finally:
        await engine.dispose()

async def init_models_for_db(db_url: str):
    """Create tables in specified database"""
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        # For Dev: Drop to ensure schema update (WARNING: DELETES DATA)
        # If you want to keep data, you must use Alembic or SQL ALTER commands manually.
        # For this feature, I recommend a clean slate if possible.
        # await conn.run_sync(Base.metadata.drop_all) 
        
        # Safe create (only creates if missing)
        await conn.run_sync(Base.metadata.create_all)
        print(f"Database tables updated for {db_url}.")

async def main():
    # Get prod URL from env
    prod_db_url = os.getenv("DATABASE_URL")
    if not prod_db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        return
    
    # Derive test URL
    test_db_url = prod_db_url.replace("/connect4_arena", "/connect4_test")
    
    # Extract database names
    prod_db_name = prod_db_url.split("/")[-1]
    test_db_name = test_db_url.split("/")[-1]
    
    print(f"Prod DB: {prod_db_name}, Test DB: {test_db_name}")
    
    # Create test database if needed
    await create_database_if_not_exists(prod_db_url, test_db_name)
    
    # Initialize tables for prod
    print("Initializing prod database tables...")
    await init_models_for_db(prod_db_url)
    
    # Initialize tables for test
    print("Initializing test database tables...")
    await init_models_for_db(test_db_url)
    
    print("All databases initialized.")

if __name__ == "__main__":
    asyncio.run(main())