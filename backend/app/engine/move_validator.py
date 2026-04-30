"""Pure-function move validation. No DB, no logging — just checks.

Owners can call these from inside or outside a transaction; the result is a
plain dataclass. This was previously inlined in ``game_service.process_human_move``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from backend.app.engine.game import ConnectFour
    from backend.app.models.game_model import Game


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: Optional[str] = None

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(ok=True, error=None)

    @classmethod
    def failure(cls, error: str) -> "ValidationResult":
        return cls(ok=False, error=error)


def validate_column(col: int, board: "ConnectFour") -> ValidationResult:
    """Check that ``col`` is a legal drop on the current board.

    Mirrors ``ConnectFour.is_valid_move`` but returns a structured reason
    instead of a bool.
    """
    cols = len(board.board[0])
    if not isinstance(col, int):
        return ValidationResult.failure(f"Column must be an integer, got {type(col).__name__}")
    if col < 0 or col >= cols:
        return ValidationResult.failure(f"Column {col} out of range [0, {cols - 1}]")
    if board.board[0][col] != 0:
        return ValidationResult.failure(f"Column {col} is full")
    return ValidationResult.success()


def validate_player_token(game: "Game", provided: Optional[str]) -> ValidationResult:
    """Check the provided session token against the player whose turn it is.

    Player turn is derived from ``len(game.history) % 2`` to match the
    existing convention (P1 on even move counts).
    """
    history = game.history if game.history is not None else []
    is_p1_turn = (len(history) % 2) == 0
    required_token = game.player_1_token if is_p1_turn else game.player_2_token

    # No token configured for this side → no auth required (e.g. AI vs AI).
    if not required_token:
        return ValidationResult.success()

    if required_token != provided:
        return ValidationResult.failure("Unauthorized: Invalid player token")
    return ValidationResult.success()
