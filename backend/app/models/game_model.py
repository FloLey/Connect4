from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.app.core.database import Base

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Tournament Links
    tournament_id = Column(Integer, ForeignKey("tournaments.id"), nullable=True)
    round_number = Column(Integer, nullable=True) # For round-robin tracking
    
    tournament = relationship("Tournament", back_populates="games")

    # Metadata
    player_1_type = Column(String, default="human") 
    player_2_type = Column(String, default="ai")
    
    # Game State
    winner = Column(Integer, nullable=True)
    status = Column(String, default="IN_PROGRESS") # IN_PROGRESS, COMPLETED, DRAW, PENDING
    
    history = Column(JSONB, default=list)