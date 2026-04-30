"""Materialized stats tables (Tier 2.4).

Snapshots derived from games + ELO records, kept in sync via the
``stats_aggregator`` listener that subscribes to ``game_events.notify_complete``.
The leaderboard and matrix endpoints read from these tables directly so the
hot path is a simple SELECT.
"""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.sql import func

from backend.app.core.database import Base


class LeaderboardSnapshot(Base):
    """One row per model — mirrors EloRating plus precomputed costs."""

    __tablename__ = "leaderboard_snapshots"

    model_name = Column(String, primary_key=True, index=True)
    rating = Column(Float, default=0.0)

    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)

    total_input_tokens = Column(BigInteger, default=0)
    total_output_tokens = Column(BigInteger, default=0)
    total_duration_seconds = Column(Float, default=0.0)
    total_moves = Column(BigInteger, default=0)

    # Precomputed via registry pricing × tokens / 1M; recomputed by the
    # aggregator each time the row is refreshed.
    cost_input_total = Column(Float, default=0.0)
    cost_output_total = Column(Float, default=0.0)

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class MatrixCell(Base):
    """One row per directed pair (player_a vs player_b).

    Symmetric pairs are written twice — once as (a,b) and once as (b,a) —
    so the matrix endpoint can read from a single column ordering and never
    has to flip semantics at read time.
    """

    __tablename__ = "matrix_cells"

    player_a = Column(String, primary_key=True, index=True)
    player_b = Column(String, primary_key=True, index=True)

    wins_a = Column(Integer, default=0)
    wins_b = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    total = Column(Integer, default=0)

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
