from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.app.core.database import Base
from backend.app.models.enums import GameStatus, PlayerType

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Tournament Links
    tournament_id = Column(Integer, ForeignKey("tournaments.id"), nullable=True)
    round_number = Column(Integer, nullable=True) # For round-robin tracking
    
    tournament = relationship("Tournament", back_populates="games")

    # Metadata
    player_1_type = Column(String, default=PlayerType.HUMAN) 
    player_2_type = Column(String, default=PlayerType.AI)
    
    # Game State
    winner = Column(Integer, nullable=True)
    status = Column(String, default=GameStatus.IN_PROGRESS)
    
    history = Column(JSONB, default=list)
    stats = Column(JSONB, nullable=True) 
    # Store: {"total_cost": 0.05, "total_tokens": 1500, "duration": 45.2}
    
    # Session tokens for human player security
    player_1_token = Column(String, nullable=True)
    player_2_token = Column(String, nullable=True)
    
    # Rate limit cooldown
    retry_after = Column(DateTime(timezone=True), nullable=True, index=True)