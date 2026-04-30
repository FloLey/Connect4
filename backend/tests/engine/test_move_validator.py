"""Pure-function tests for the move validator (no DB)."""

import pytest

from backend.app.engine.game import COLS, ConnectFour
from backend.app.engine.move_validator import (
    ValidationResult,
    validate_column,
    validate_player_token,
)
from backend.app.models.game_model import Game


class TestValidationResult:
    def test_success_helper(self):
        r = ValidationResult.success()
        assert r.ok is True
        assert r.error is None

    def test_failure_helper(self):
        r = ValidationResult.failure("nope")
        assert r.ok is False
        assert r.error == "nope"


class TestValidateColumn:
    def test_legal_column_passes(self):
        engine = ConnectFour()
        assert validate_column(3, engine).ok is True

    def test_negative_column_rejected(self):
        engine = ConnectFour()
        result = validate_column(-1, engine)
        assert result.ok is False
        assert "out of range" in result.error

    def test_too_high_column_rejected(self):
        engine = ConnectFour()
        result = validate_column(COLS, engine)
        assert result.ok is False
        assert "out of range" in result.error

    def test_full_column_rejected(self):
        engine = ConnectFour()
        for _ in range(6):
            engine.drop_piece(0)
        result = validate_column(0, engine)
        assert result.ok is False
        assert "full" in result.error.lower()

    def test_non_int_rejected(self):
        engine = ConnectFour()
        result = validate_column("3", engine)  # type: ignore[arg-type]
        assert result.ok is False
        assert "integer" in result.error.lower()


class TestValidatePlayerToken:
    def test_no_required_token_passes(self):
        # AI vs AI game — no tokens minted.
        game = Game(
            player_1_type="model-a",
            player_2_type="model-b",
            history=[],
            player_1_token=None,
            player_2_token=None,
        )
        assert validate_player_token(game, provided=None).ok is True

    def test_valid_token_for_p1_turn(self):
        game = Game(
            player_1_type="human",
            player_2_type="model-a",
            history=[],
            player_1_token="alpha",
            player_2_token=None,
        )
        assert validate_player_token(game, provided="alpha").ok is True

    def test_wrong_token_for_p1_turn(self):
        game = Game(
            player_1_type="human",
            player_2_type="model-a",
            history=[],
            player_1_token="alpha",
        )
        result = validate_player_token(game, provided="beta")
        assert result.ok is False
        assert "Unauthorized" in result.error

    def test_token_check_uses_p2_token_on_p2_turn(self):
        # 1 move played → P2's turn.
        game = Game(
            player_1_type="human",
            player_2_type="human",
            history=[{"player": 1, "column": 3}],
            player_1_token="alpha",
            player_2_token="beta",
        )
        assert validate_player_token(game, provided="beta").ok is True
        assert validate_player_token(game, provided="alpha").ok is False
