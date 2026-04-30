"""Tests for the cleanup helper extracted from main.run_cleanup_periodically.

Seeds a mix of game rows and asserts only old non-tournament IN_PROGRESS games
get marked ABANDONED.
"""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.future import select

from backend.app.main import _cleanup_once
from backend.app.models.enums import GameStatus
from backend.app.models.game_model import Game
from backend.app.models.tournament_model import Tournament


def _now():
    return datetime.now(timezone.utc)


def _ago(hours):
    return _now() - timedelta(hours=hours)


async def test_returns_zero_when_no_eligible_games(test_db):
    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 0


async def test_marks_old_non_tournament_in_progress_as_abandoned(test_db):
    # Seed two old standalone games — both should get cleaned.
    test_db.add_all([
        Game(player_1_type="a", player_2_type="b", status=GameStatus.IN_PROGRESS, history=[]),
        Game(player_1_type="c", player_2_type="d", status=GameStatus.IN_PROGRESS, history=[]),
    ])
    await test_db.commit()
    # Backdate created_at to 8h ago so cutoff (6h) catches them.
    games = (await test_db.execute(select(Game))).scalars().all()
    for g in games:
        g.created_at = _ago(8)
    await test_db.commit()

    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 2

    statuses = {
        g.status
        for g in (await test_db.execute(select(Game))).scalars().all()
    }
    assert statuses == {"ABANDONED"}


async def test_recent_games_are_left_alone(test_db):
    test_db.add(
        Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.IN_PROGRESS,
            history=[],
        )
    )
    await test_db.commit()
    # Recent — created_at ~ now, cutoff is 6h ago.
    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 0


async def test_tournament_games_never_cleaned(test_db):
    tournament = Tournament(status="IN_PROGRESS", config={}, total_matches=2)
    test_db.add(tournament)
    await test_db.commit()
    await test_db.refresh(tournament)

    test_db.add(
        Game(
            tournament_id=tournament.id,
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.IN_PROGRESS,
            history=[],
        )
    )
    await test_db.commit()
    # Backdate so it would be eligible if not for tournament_id.
    g = (await test_db.execute(select(Game))).scalar_one()
    g.created_at = _ago(48)
    await test_db.commit()

    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 0
    g = (await test_db.execute(select(Game))).scalar_one()
    assert g.status == GameStatus.IN_PROGRESS


async def test_paused_games_never_cleaned(test_db):
    test_db.add(
        Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.PAUSED,
            history=[],
            retry_after=_now() + timedelta(minutes=5),
        )
    )
    await test_db.commit()
    g = (await test_db.execute(select(Game))).scalar_one()
    g.created_at = _ago(48)
    await test_db.commit()

    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 0


async def test_completed_games_never_cleaned(test_db):
    test_db.add(
        Game(
            player_1_type="a",
            player_2_type="b",
            status=GameStatus.COMPLETED,
            winner=1,
            history=[],
        )
    )
    await test_db.commit()
    g = (await test_db.execute(select(Game))).scalar_one()
    g.created_at = _ago(48)
    await test_db.commit()

    cleaned = await _cleanup_once(test_db, cutoff=_ago(6))
    assert cleaned == 0
