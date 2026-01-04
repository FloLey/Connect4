from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Any
from datetime import datetime
from backend.app.models.enums import PlayerType

class MoveRecord(BaseModel):
    # Allow extra fields in the JSON to prevent crashes if schema evolves
    model_config = ConfigDict(extra='ignore') 

    player: int
    column: int
    reasoning: Optional[str] = None
    
    # Allow None to be coerced to 0, or handle optional integers gracefully
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0
    
    # --- ADDED THIS FIELD ---
    duration: Optional[float] = 0.0

class GameCreate(BaseModel):
    player_1: str = PlayerType.HUMAN
    player_2: str = PlayerType.AI

class GameResponse(BaseModel):
    id: int
    status: str
    winner: Optional[int] = None # Ensure explicit default
    
    # Use MoveRecord with relaxed validation to prevent 422s on malformed history data
    history: List[MoveRecord] 
    
    created_at: datetime
    player_1_type: str
    player_2_type: str
    player_1_token: Optional[str] = None
    player_2_token: Optional[str] = None

    class Config:
        from_attributes = True