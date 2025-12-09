from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.app.core.database import Base

class Tournament(Base):
    __tablename__ = "tournaments"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="SETUP") # SETUP, IN_PROGRESS, COMPLETED, STOPPED
    
    # Config: { "concurrency": 5, "rounds": 2, "models": [...] }
    config = Column(JSONB, default=dict)
    
    total_matches = Column(Integer, default=0)
    
    # Relationship
    games = relationship("Game", back_populates="tournament", cascade="all, delete-orphan")