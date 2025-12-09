import math
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from backend.app.models.elo_model import EloRating, EloHistory
from backend.app.models.game_model import Game

K_FACTOR = 32

async def get_or_create_rating(db: AsyncSession, model_name: str) -> EloRating:
    result = await db.execute(select(EloRating).where(EloRating.model_name == model_name))
    rating_obj = result.scalar_one_or_none()
    if not rating_obj:
        rating_obj = EloRating(model_name=model_name, rating=1200.0)
        db.add(rating_obj)
        # We don't commit here, we let the caller commit transactionally
    return rating_obj

def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

async def update_elo(db: AsyncSession, model_a_name: str, model_b_name: str, winner_id: int, game_id: int):
    """
    Updates ratings for both models after a match.
    winner_id: 1 (Model A), 2 (Model B), or 0 (Draw)
    """
    # 0. IDEMPOTENCY CHECK
    # Check if we have already processed this game to prevent double counting
    existing_history = await db.execute(
        select(EloHistory).where(EloHistory.match_id == game_id)
    )
    if existing_history.first():
        print(f"Skipping ELO update for Game {game_id}: Already processed.")
        return

    # 1. Fetch Current Ratings
    model_a = await get_or_create_rating(db, model_a_name)
    model_b = await get_or_create_rating(db, model_b_name)

    # 2. Calculate Expected Scores
    expected_a = calculate_expected_score(model_a.rating, model_b.rating)
    expected_b = calculate_expected_score(model_b.rating, model_a.rating)

    # 3. Determine Actual Scores (safely handle nullable counters)
    if winner_id == 1:
        score_a, score_b = 1.0, 0.0
        model_a.wins = (model_a.wins or 0) + 1
        model_b.losses = (model_b.losses or 0) + 1
    elif winner_id == 2:
        score_a, score_b = 0.0, 1.0
        model_a.losses = (model_a.losses or 0) + 1
        model_b.wins = (model_b.wins or 0) + 1
    else:
        score_a, score_b = 0.5, 0.5
        model_a.draws = (model_a.draws or 0) + 1
        model_b.draws = (model_b.draws or 0) + 1

    # 4. Update Ratings
    new_rating_a = model_a.rating + K_FACTOR * (score_a - expected_a)
    new_rating_b = model_b.rating + K_FACTOR * (score_b - expected_b)

    model_a.rating = new_rating_a
    model_b.rating = new_rating_b
    model_a.matches_played = (model_a.matches_played or 0) + 1
    model_b.matches_played = (model_b.matches_played or 0) + 1

    # 5. AGGREGATE STATS FROM GAME HISTORY
    # Fetch the game to look at history
    result = await db.execute(select(Game).where(Game.id == game_id))
    game = result.scalar_one()
    
    # Calculate sums for Player 1 (Model A)
    # Filter moves by player 1
    moves_a = [m for m in game.history if m['player'] == 1]
    tokens_a_in = sum(m.get('input_tokens', 0) for m in moves_a)
    tokens_a_out = sum(m.get('output_tokens', 0) for m in moves_a)
    duration_a = sum(m.get('duration', 0.0) for m in moves_a)
    count_moves_a = len(moves_a)

    # Calculate sums for Player 2 (Model B)
    moves_b = [m for m in game.history if m['player'] == 2]
    tokens_b_in = sum(m.get('input_tokens', 0) for m in moves_b)
    tokens_b_out = sum(m.get('output_tokens', 0) for m in moves_b)
    duration_b = sum(m.get('duration', 0.0) for m in moves_b)
    count_moves_b = len(moves_b)

    # Apply updates to Model A
    model_a.total_input_tokens = (model_a.total_input_tokens or 0) + tokens_a_in
    model_a.total_output_tokens = (model_a.total_output_tokens or 0) + tokens_a_out
    model_a.total_moves = (model_a.total_moves or 0) + count_moves_a
    model_a.total_duration_seconds = (model_a.total_duration_seconds or 0.0) + duration_a
    
    # Apply updates to Model B
    model_b.total_input_tokens = (model_b.total_input_tokens or 0) + tokens_b_in
    model_b.total_output_tokens = (model_b.total_output_tokens or 0) + tokens_b_out
    model_b.total_moves = (model_b.total_moves or 0) + count_moves_b
    model_b.total_duration_seconds = (model_b.total_duration_seconds or 0.0) + duration_b

    # 6. Log History (For Graphs)
    hist_a = EloHistory(model_name=model_a_name, rating=new_rating_a, match_id=game_id)
    hist_b = EloHistory(model_name=model_b_name, rating=new_rating_b, match_id=game_id)
    
    db.add(hist_a)
    db.add(hist_b)
    
    # Caller must await db.commit()