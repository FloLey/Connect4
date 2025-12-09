import asyncio
from backend.app.core.database import engine, Base
# IMPORT ALL MODELS
from backend.app.models.game_model import Game
from backend.app.models.elo_model import EloRating, EloHistory 
from backend.app.models.tournament_model import Tournament # <--- NEW

async def init_models():
    async with engine.begin() as conn:
        # For Dev: Drop to ensure schema update (WARNING: DELETES DATA)
        # If you want to keep data, you must use Alembic or SQL ALTER commands manually.
        # For this feature, I recommend a clean slate if possible.
        # await conn.run_sync(Base.metadata.drop_all) 
        
        # Safe create (only creates if missing)
        await conn.run_sync(Base.metadata.create_all)
        print("Database tables updated.")

if __name__ == "__main__":
    asyncio.run(init_models())