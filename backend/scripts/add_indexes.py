#!/usr/bin/env python3
"""
Script to add database indexes for performance optimization.
Specifically adds indexes for retry_after column to fix tournament completion issues.
"""

import asyncio
import os
import sys

# Add project root to path so we can import from backend.app
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sqlalchemy import text
from backend.app.core.database import get_session_maker

async def add_indexes():
    """Add necessary indexes for tournament performance"""
    SessionLocal = get_session_maker("prod")
    async with SessionLocal() as db:
        print("Adding database indexes for tournament performance optimization...")
        
        # 1. Check existing indexes
        check_query = text("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename = 'games' AND indexname LIKE '%retry_after%'
        """)
        result = await db.execute(check_query)
        existing_indexes = result.scalars().all()
        
        if existing_indexes:
            print(f"Found existing retry_after indexes: {existing_indexes}")
        else:
            print("No retry_after indexes found.")
        
        # 2. Add retry_after index
        print("\nAdding idx_games_retry_after index...")
        try:
            add_index_query = text("""
                CREATE INDEX IF NOT EXISTS idx_games_retry_after 
                ON games(retry_after)
            """)
            await db.execute(add_index_query)
            await db.commit()
            print("✅ Added idx_games_retry_after index")
        except Exception as e:
            print(f"❌ Failed to add idx_games_retry_after: {e}")
            await db.rollback()
        
        # 3. Add composite index for tournament queries
        print("\nAdding idx_games_tournament_status_retry index...")
        try:
            composite_index_query = text("""
                CREATE INDEX IF NOT EXISTS idx_games_tournament_status_retry 
                ON games(tournament_id, status, retry_after) 
                WHERE status = 'PAUSED'
            """)
            await db.execute(composite_index_query)
            await db.commit()
            print("✅ Added idx_games_tournament_status_retry index")
        except Exception as e:
            print(f"❌ Failed to add composite index: {e}")
            await db.rollback()
        
        # 4. Verify indexes
        print("\nVerifying indexes...")
        verify_query = text("""
            SELECT 
                indexname, 
                indexdef 
            FROM pg_indexes 
            WHERE tablename = 'games' 
            ORDER BY indexname
        """)
        result = await db.execute(verify_query)
        indexes = result.fetchall()
        
        print("\nCurrent indexes on 'games' table:")
        for idx_name, idx_def in indexes:
            print(f"  {idx_name}: {idx_def[:80]}...")
        
        # Check for the specific indexes we need
        required_indexes = ['idx_games_retry_after', 'idx_games_tournament_status_retry']
        existing_index_names = [idx[0] for idx in indexes]
        
        print("\nRequired index status:")
        for req_idx in required_indexes:
            if req_idx in existing_index_names:
                print(f"  ✅ {req_idx}: EXISTS")
            else:
                print(f"  ❌ {req_idx}: MISSING")

if __name__ == "__main__":
    asyncio.run(add_indexes())