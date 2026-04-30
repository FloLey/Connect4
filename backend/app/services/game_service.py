"""Game Service — coordinator for game state mutations.

Tier 2 split: validation lives in ``engine.move_validator``, AI orchestration
in ``services.ai_turn_executor``, and event fan-out in
``services.game_event_publisher``. This module owns the DB locking dance
(READ → think → WRITE-with-lock) and persistence, nothing more.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.attributes import flag_modified

from backend.app.core.logging import get_logger
from backend.app.engine.game import ConnectFour
from backend.app.engine.move_validator import (
    validate_column,
    validate_player_token,
)
from backend.app.engine.rate_limit import RateLimitedError
from backend.app.models.enums import GameStatus, PlayerType
from backend.app.models.game_model import Game
from backend.app.services.ai_turn_executor import ai_turn_executor
from backend.app.services.game_event_publisher import game_event_publisher

logger = get_logger(__name__)


class GameState:
    """Snapshot of a game's runtime state for API responses."""

    def __init__(self, game_db: Game, engine: ConnectFour):
        self.game_id = game_db.id
        self.board = engine.board
        self.current_turn = engine.current_turn
        self.winner = engine.winner
        self.status = game_db.status
        self.is_draw = engine.is_draw()
        self.last_move = game_db.history[-1] if game_db.history else None
        self.player_1_type = game_db.player_1_type
        self.player_2_type = game_db.player_2_type


class GameService:
    """Coordinator for game operations. DB writes happen here; AI / validation don't."""

    async def create_game(
        self, db: AsyncSession, player_1_type: str, player_2_type: str
    ) -> Game:
        new_game = Game(
            player_1_type=player_1_type,
            player_2_type=player_2_type,
            history=[],
        )
        db.add(new_game)
        await db.commit()
        await db.refresh(new_game)
        return new_game

    async def get_game_state(
        self, db: AsyncSession, game_id: int
    ) -> Tuple[Game, ConnectFour]:
        """READ-ONLY: load + reconstruct engine from history."""
        game_db = (
            await db.execute(select(Game).where(Game.id == game_id))
        ).scalar_one_or_none()
        if not game_db:
            raise ValueError(f"Game {game_id} not found")
        return game_db, self._engine_from_history(game_db.history)

    async def _get_game_for_update(
        self, db: AsyncSession, game_id: int
    ) -> Tuple[Game, ConnectFour]:
        """SELECT … FOR UPDATE: blocks concurrent writers on the same row."""
        game_db = (
            await db.execute(
                select(Game).where(Game.id == game_id).with_for_update()
            )
        ).scalar_one_or_none()
        if not game_db:
            raise ValueError(f"Game {game_id} not found")
        return game_db, self._engine_from_history(game_db.history)

    @staticmethod
    def _engine_from_history(history) -> ConnectFour:
        engine = ConnectFour()
        for move in history or []:
            engine.drop_piece(move["column"])
        return engine

    # -- Move processing -----------------------------------------------------

    async def process_human_move(
        self,
        db: AsyncSession,
        game_id: int,
        column: int,
        provided_token: Optional[str] = None,
    ) -> GameState:
        start_time = time.time()
        game_db, engine = await self._get_game_for_update(db, game_id)

        if game_db.status != GameStatus.IN_PROGRESS:
            return GameState(game_db, engine)

        token_check = validate_player_token(game_db, provided_token)
        if not token_check.ok:
            raise ValueError(token_check.error)

        column_check = validate_column(column, engine)
        if not column_check.ok:
            raise ValueError(f"Invalid move: {column_check.error}")

        previous_player = engine.current_turn  # captured before drop_piece toggles it
        if not engine.drop_piece(column):
            raise ValueError(f"Invalid move: column {column}")

        move_record = {
            "player": previous_player,
            "column": column,
            "reasoning": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "duration": round(time.time() - start_time, 3),
            "is_fallback": False,
            "cost_usd": 0.0,
        }
        await self._save_move_to_db(db, game_db, move_record, engine)
        return GameState(game_db, engine)

    async def step_ai_turn(
        self, db: AsyncSession, game_id: int
    ) -> Optional[GameState]:
        # 1. READ snapshot — no lock, so the LLM call doesn't block writers.
        try:
            game_db_snapshot, engine_snapshot = await self.get_game_state(db, game_id)
        except ValueError:
            return None

        if (
            engine_snapshot.winner
            or engine_snapshot.is_draw()
            or game_db_snapshot.status != GameStatus.IN_PROGRESS
        ):
            return GameState(game_db_snapshot, engine_snapshot)

        current_player = engine_snapshot.current_turn
        ai_model = (
            game_db_snapshot.player_1_type if current_player == 1
            else game_db_snapshot.player_2_type
        )
        if ai_model == PlayerType.HUMAN:
            return GameState(game_db_snapshot, engine_snapshot)

        snapshot_move_count = len(game_db_snapshot.history or [])

        # 2. THINK — AI executor handles LLM call + rate-limit detection.
        try:
            turn = await ai_turn_executor.run(
                engine_snapshot=engine_snapshot,
                player_id=current_player,
                model_name=ai_model,
            )
        except RateLimitedError as rl:
            await self._snooze_game(db, game_id, ai_model, rl.snooze_seconds)
            return None
        except Exception as e:
            logger.error("ai_turn_failed", game_id=game_id, model_name=ai_model, error=str(e))
            return GameState(game_db_snapshot, engine_snapshot)

        if turn is None:
            return GameState(game_db_snapshot, engine_snapshot)

        # 3. WRITE under lock — re-check state hasn't drifted, then persist.
        game_db, engine = await self._get_game_for_update(db, game_id)

        if len(game_db.history or []) != snapshot_move_count:
            logger.warning(
                "stale_ai_move_discarded",
                game_id=game_id,
                snapshot=snapshot_move_count,
                current=len(game_db.history or []),
            )
            return GameState(game_db, engine)
        if game_db.status != GameStatus.IN_PROGRESS:
            logger.info("game_finished_during_think", game_id=game_id)
            return GameState(game_db, engine)
        if not engine.drop_piece(turn.decision.column):
            logger.warning("ai_invalid_move", game_id=game_id, column=turn.decision.column)
            return GameState(game_db, engine)

        await self._save_move_to_db(db, game_db, turn.move_record, engine)
        return GameState(game_db, engine)

    # -- Persistence helpers -------------------------------------------------

    async def _snooze_game(
        self, db: AsyncSession, game_id: int, model_name: str, snooze_seconds: int
    ) -> None:
        logger.warning(
            "rate_limit_snooze",
            game_id=game_id,
            model_name=model_name,
            snooze_seconds=snooze_seconds,
        )
        game_db, _ = await self._get_game_for_update(db, game_id)
        game_db.status = GameStatus.PAUSED
        game_db.retry_after = datetime.now(timezone.utc) + timedelta(seconds=snooze_seconds)
        await db.commit()
        logger.info("game_snoozed", game_id=game_id, retry_after=game_db.retry_after.isoformat())

    @staticmethod
    def _calculate_game_stats(game_db: Game) -> Dict[str, Any]:
        if not game_db.history:
            return {}
        totals = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_duration": 0.0,
            "total_cost_usd": 0.0,
        }
        for move in game_db.history:
            totals["total_input_tokens"] += move.get("input_tokens", 0) or 0
            totals["total_output_tokens"] += move.get("output_tokens", 0) or 0
            totals["total_duration"] += move.get("duration", 0.0) or 0.0
            totals["total_cost_usd"] += move.get("cost_usd", 0.0) or 0.0
        totals["total_tokens"] = totals["total_input_tokens"] + totals["total_output_tokens"]
        totals["total_duration"] = round(totals["total_duration"], 3)
        totals["total_cost_usd"] = round(totals["total_cost_usd"], 6)
        return totals

    async def _save_move_to_db(
        self,
        db: AsyncSession,
        game_db: Game,
        move_record: Dict[str, Any],
        engine: ConnectFour,
    ) -> None:
        try:
            new_history = list(game_db.history or [])
            new_history.append(move_record)
            game_db.history = new_history
            flag_modified(game_db, "history")

            if engine.winner:
                game_db.winner = engine.winner
                game_db.status = GameStatus.COMPLETED
            elif engine.is_draw():
                game_db.status = GameStatus.DRAW

            if game_db.status in (GameStatus.COMPLETED, GameStatus.DRAW):
                winner_id = engine.winner or 0
                game_db.stats = self._calculate_game_stats(game_db)
                await game_event_publisher.publish_complete(db, game_db, winner_id)

            await db.commit()
            await db.refresh(game_db)
            game_event_publisher.signal_tournament_tick()
        except Exception as e:
            logger.error("db_save_error", error=str(e))
            await db.rollback()
            raise ValueError(f"Failed to save move: {e}")


# Singleton instance.
game_service = GameService()
