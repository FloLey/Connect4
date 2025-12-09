from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger # Added BigInteger
from sqlalchemy.sql import func
from backend.app.core.database import Base

class EloRating(Base):
    __tablename__ = "elo_ratings"

    model_name = Column(String, primary_key=True, index=True)
    rating = Column(Float, default=1200.0)
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    
    # --- Token Usage ---
    total_input_tokens = Column(BigInteger, default=0)
    total_output_tokens = Column(BigInteger, default=0)
    
    # --- NEW: Time & Move Stats for Averages ---
    total_moves = Column(BigInteger, default=0)
    total_duration_seconds = Column(Float, default=0.0)
    # -------------------------------------------
    
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class EloHistory(Base):
    """
    Time-series data for the graph. 
    A row is inserted every time a model finishes a ranked match.
    """
    __tablename__ = "elo_history"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True) # Foreign key logic handled manually or via string match
    rating = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    match_id = Column(Integer, nullable=True) # Reference to the Game ID