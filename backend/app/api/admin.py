from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.services.stats_aggregator import stats_aggregator


def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Gate admin routes behind CONNECT4_ADMIN_TOKEN.

    When the token is unset (default), the gate is a no-op so single-user dev
    workflows keep working. In any deployment that exposes /admin/* publicly,
    set CONNECT4_ADMIN_TOKEN.
    """
    expected = settings.admin_token
    if expected and x_admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Token header",
        )


router = APIRouter(dependencies=[Depends(require_admin)])

@router.delete("/reset", status_code=status.HTTP_200_OK)
async def reset_database(confirmation: str, db: AsyncSession = Depends(get_db)):
    """
    Resets the database.
    Query Param 'confirmation' must equal 'I-UNDERSTAND-THIS-DELETES-EVERYTHING'.
    """
    if confirmation != "I-UNDERSTAND-THIS-DELETES-EVERYTHING":
        raise HTTPException(
            status_code=400, 
            detail="Invalid confirmation string. Operation aborted."
        )

    try:
        # Truncate tables in specific order to handle foreign keys if they existed
        # Using RESTART IDENTITY to reset ID counters to 1
        await db.execute(text("TRUNCATE TABLE elo_history RESTART IDENTITY CASCADE;"))
        await db.execute(text("TRUNCATE TABLE elo_ratings RESTART IDENTITY CASCADE;"))
        await db.execute(text("TRUNCATE TABLE games RESTART IDENTITY CASCADE;"))
        
        await db.commit()
        return {"message": "Database successfully wiped."}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rebuild-stats")
async def rebuild_stats(db: AsyncSession = Depends(get_db)):
    """Rebuild materialized stats tables from EloRating + Game.

    Used after the alembic migration introduces the new tables, or for manual
    recovery if a listener missed an event.
    """
    summary = await stats_aggregator.rebuild_all(db)
    return {"message": "Stats rebuilt", **summary}


@router.get("/status")
async def get_admin_status(db: AsyncSession = Depends(get_db)):
    """
    Get database statistics for admin dashboard.
    """
    try:
        # Count records in each table
        games_result = await db.execute(text("SELECT COUNT(*) FROM games;"))
        games_count = games_result.scalar()
        
        ratings_result = await db.execute(text("SELECT COUNT(*) FROM elo_ratings;"))
        ratings_count = ratings_result.scalar()
        
        history_result = await db.execute(text("SELECT COUNT(*) FROM elo_history;"))
        history_count = history_result.scalar()
        
        return {
            "games": games_count,
            "elo_ratings": ratings_count, 
            "elo_history": history_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))