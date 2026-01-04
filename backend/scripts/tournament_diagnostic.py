#!/usr/bin/env python3
"""
Tournament Diagnostic Script
Identifies unfinished games in "Completed" tournaments.
"""

import asyncio
import os
import sys

# Add project root to path so we can import from backend.app
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from sqlalchemy.future import select
from sqlalchemy import func
from backend.app.core.database import get_session_maker
from backend.app.models.game_model import Game
from backend.app.models.tournament_model import Tournament
from backend.app.models.enums import GameStatus, TournamentStatus

async def diagnose():
    SessionLocal = get_session_maker("prod")
    async with SessionLocal() as db:
        # Get latest tournament
        res = await db.execute(select(Tournament).order_by(Tournament.created_at.desc()).limit(1))
        t = res.scalar_one_or_none()
        
        if not t:
            print("No tournaments found in database.")
            return
        
        print(f"--- Diagnostic: Tournament #{t.id} ({t.status}) ---")
        print(f"Created: {t.created_at}")
        print(f"Total matches: {t.total_matches}")
        print(f"Config: {t.config}")
        
        # Count by status
        print("\nGame Status Breakdown:")
        for s in [GameStatus.COMPLETED, GameStatus.DRAW, GameStatus.PENDING, GameStatus.PAUSED, GameStatus.IN_PROGRESS]:
            count = await db.execute(select(func.count(Game.id)).where(Game.tournament_id == t.id, Game.status == s))
            print(f"  {s}: {count.scalar()}")

        # Calculate completion percentage
        completed_query = await db.execute(
            select(func.count(Game.id)).where(
                Game.tournament_id == t.id,
                Game.status.in_([GameStatus.COMPLETED, GameStatus.DRAW])
            )
        )
        completed_count = completed_query.scalar() or 0
        
        if t.total_matches > 0:
            completion_pct = (completed_count / t.total_matches) * 100
            print(f"\nCompletion: {completed_count}/{t.total_matches} ({completion_pct:.1f}%)")
        else:
            print(f"\nCompletion: {completed_count}/0 (N/A%)")
        
        # List Ghost Games (unfinished games)
        ghosts = await db.execute(
            select(Game).where(
                Game.tournament_id == t.id,
                Game.status.in_([GameStatus.PENDING, GameStatus.PAUSED, GameStatus.IN_PROGRESS])
            ).order_by(Game.id.asc())
        )
        ghost_list = ghosts.scalars().all()
        
        print(f"\nGhost Games Found: {len(ghost_list)}")
        if ghost_list:
            print("First 10 ghost games:")
            for g in ghost_list[:10]:
                retry_info = f" (retry_after: {g.retry_after})" if g.retry_after else ""
                print(f"  Game {g.id}: {g.player_1_type} vs {g.player_2_type} (Status: {g.status}{retry_info})")
        
        # Check if tournament is prematurely marked as completed
        if t.status == TournamentStatus.COMPLETED and len(ghost_list) > 0:
            print(f"\n⚠️  WARNING: Tournament #{t.id} is marked as COMPLETED but has {len(ghost_list)} unfinished games!")
            print("  This indicates the 'Ghost Match' issue.")
        
        # Model performance summary
        print("\nModel Performance Summary:")
        
        # Get all models in tournament
        models = set()
        if t.config and 'model_ids' in t.config:
            models = set(t.config['model_ids'])
        elif t.config and 'benchmark_models' in t.config:
            models = set(t.config['benchmark_models'])
            if 'target_model' in t.config:
                models.add(t.config['target_model'])
        
        if models:
            print("  Models in tournament:", ", ".join(sorted(models)))
            
            # Count games per model (as player 1)
            for model in sorted(models):
                p1_count = await db.execute(
                    select(func.count(Game.id)).where(
                        Game.tournament_id == t.id,
                        Game.player_1_type == model
                    )
                )
                p2_count = await db.execute(
                    select(func.count(Game.id)).where(
                        Game.tournament_id == t.id,
                        Game.player_2_type == model
                    )
                )
                total_games = (p1_count.scalar() or 0) + (p2_count.scalar() or 0)
                print(f"  {model}: {total_games} total games")

if __name__ == "__main__":
    asyncio.run(diagnose())