import asyncio
import sys
import os
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.app.core.database import get_database_url

async def migrate_schema():
    print("🔄 Connecting to database...")
    database_url = get_database_url()
    # Create a temporary engine just for this script
    engine = create_async_engine(database_url)

    async with engine.begin() as conn:
        print("🛠️  Adding new columns to 'elo_ratings' table...")
        
        # 1. Add total_moves
        await conn.execute(text("""
            ALTER TABLE elo_ratings 
            ADD COLUMN IF NOT EXISTS total_moves BIGINT DEFAULT 0;
        """))
        
        # 2. Add total_duration_seconds
        await conn.execute(text("""
            ALTER TABLE elo_ratings 
            ADD COLUMN IF NOT EXISTS total_duration_seconds FLOAT DEFAULT 0.0;
        """))

        # 3. Add total_input_tokens (Just in case)
        await conn.execute(text("""
            ALTER TABLE elo_ratings 
            ADD COLUMN IF NOT EXISTS total_input_tokens BIGINT DEFAULT 0;
        """))

        # 4. Add total_output_tokens (Just in case)
        await conn.execute(text("""
            ALTER TABLE elo_ratings 
            ADD COLUMN IF NOT EXISTS total_output_tokens BIGINT DEFAULT 0;
        """))

    print("✅ Schema update complete!")
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(migrate_schema())