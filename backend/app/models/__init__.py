"""Convenience re-exports for ORM models.

Importing this package guarantees every model class is registered on
``Base.metadata`` — handy for ``Base.metadata.create_all`` and for Alembic's
autogenerate to pick up every table in one import.
"""

from backend.app.models.app_settings import AppSettings
from backend.app.models.elo_model import EloHistory, EloRating
from backend.app.models.game_model import Game
from backend.app.models.stats import LeaderboardSnapshot, MatrixCell
from backend.app.models.tournament_model import Tournament

__all__ = [
    "AppSettings",
    "EloHistory",
    "EloRating",
    "Game",
    "LeaderboardSnapshot",
    "MatrixCell",
    "Tournament",
]
