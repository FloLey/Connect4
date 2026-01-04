from enum import StrEnum

class GameStatus(StrEnum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    DRAW = "DRAW"
    PAUSED = "PAUSED"

class PlayerType(StrEnum):
    HUMAN = "human"
    AI = "ai"

class TournamentStatus(StrEnum):
    SETUP = "SETUP"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"