#!/usr/bin/env python3
"""
Global Janitor Script for Connect Four Games

This script finds and marks old IN_PROGRESS games as ABANDONED.
Games older than 1 hour that are still IN_PROGRESS are considered stale
and will be marked as ABANDONED to prevent database bloat.

Usage:
    python backend/scripts/cleanup_games.py
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from backend.app.models.game_model import Game
from backend.app.core.database import get_database_url

async def cleanup_abandoned_games():
    """
    Find and mark games as ABANDONED if they are:
    - Status is IN_PROGRESS
    - Created more than 1 hour ago
    """
    # Initialize database connection
    engine = create_async_engine(get_database_url())
    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session_maker() as db:
        try:
            # Calculate cutoff time (1 hour ago)
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            # Query for old IN_PROGRESS games
            result = await db.execute(
                select(Game).where(
                    Game.status == 'IN_PROGRESS',
                    Game.created_at < cutoff_time
                )
            )
            stale_games = result.scalars().all()
            
            if not stale_games:
                print("No stale games found. Database is clean.")
                return
            
            # Update each stale game to ABANDONED
            abandoned_count = 0
            for game in stale_games:
                print(f"Marking game {game.id} as ABANDONED (created: {game.created_at})")
                game.status = "ABANDONED"
                abandoned_count += 1
            
            # Commit all changes
            await db.commit()
            print(f"✅ Successfully marked {abandoned_count} games as ABANDONED")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            await db.rollback()
        finally:
            await engine.dispose()

if __name__ == "__main__":
    print("🧹 Starting Connect Four game cleanup...")
    asyncio.run(cleanup_abandoned_games())
    print("🧹 Cleanup complete!")