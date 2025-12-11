from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import List, Optional

from backend.app.core.database import get_db
from backend.app.services.tournament_service import tournament_service
from backend.app.models.tournament_model import Tournament

router = APIRouter()

class TournamentCreate(BaseModel):
    models: List[str]
    rounds: int = 1
    concurrency: int = 2

@router.post("/create")
async def create_tournament(payload: TournamentCreate, db: AsyncSession = Depends(get_db)):
    t = await tournament_service.create_tournament(
        db, payload.models, payload.rounds, payload.concurrency
    )
    return {"id": t.id, "total_matches": t.total_matches, "status": t.status}

@router.post("/{id}/start")
async def start_tournament(id: int, db: AsyncSession = Depends(get_db)):
    success = await tournament_service.start_tournament(db, id)
    if not success:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {"message": "Tournament started"}

@router.post("/{id}/stop")
async def stop_tournament(id: int, db: AsyncSession = Depends(get_db)):
    success = await tournament_service.stop_tournament(db, id)
    return {"message": "Tournament stopped"}

@router.get("/current")
async def get_current_status(db: AsyncSession = Depends(get_db)):
    # Get the most recent active or setup tournament
    result = await db.execute(
        select(Tournament).order_by(Tournament.created_at.desc()).limit(1)
    )
    t = result.scalar_one_or_none()
    if not t:
        return None
    
    # Calculate progress
    # We could optimize this with a count query
    from backend.app.models.game_model import Game
    from sqlalchemy import func
    
    completed = await db.execute(
        select(func.count(Game.id)).where(
            Game.tournament_id == t.id, 
            Game.status.in_(["COMPLETED", "DRAW"])
        )
    )
    completed_count = completed.scalar() or 0
    
    return {
        "id": t.id,
        "status": t.status,
        "config": t.config,
        "total": t.total_matches,
        "completed": completed_count
    }

class TournamentConfigUpdate(BaseModel):
    concurrency: int

@router.post("/{id}/pause")
async def pause_tournament(id: int, db: AsyncSession = Depends(get_db)):
    """Pause a tournament - stops all running games but preserves state."""
    success = await tournament_service.pause_tournament(db, id)
    if not success:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {"message": "Tournament paused"}

@router.post("/{id}/resume")
async def resume_tournament(id: int, db: AsyncSession = Depends(get_db)):
    """Resume a paused tournament."""
    result = await db.execute(select(Tournament).where(Tournament.id == id))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Tournament not found")
    
    if t.status != "PAUSED":
        raise HTTPException(status_code=400, detail="Tournament is not paused")
    
    t.status = "IN_PROGRESS"
    await db.commit()
    
    # Trigger tick to resume games
    await tournament_service.tick(db)
    
    return {"message": "Tournament resumed"}

@router.patch("/{id}/config")
async def update_tournament_config(
    id: int, 
    payload: TournamentConfigUpdate, 
    db: AsyncSession = Depends(get_db)
):
    """Update tournament configuration (e.g., concurrency limit)."""
    success = await tournament_service.update_concurrency(db, id, payload.concurrency)
    if not success:
        raise HTTPException(status_code=404, detail="Tournament not found")
    return {"message": "Tournament configuration updated"}