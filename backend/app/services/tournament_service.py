import random
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, or_
from sqlalchemy.orm.attributes import flag_modified
from typing import List

from backend.app.models.tournament_model import Tournament
from backend.app.models.game_model import Game
from backend.app.models.enums import TournamentStatus, GameStatus, PlayerType
from backend.app.services.game_runner import game_runner

class TournamentService:
    async def create_tournament(
        self, 
        db: AsyncSession, 
        models: List[str], 
        rounds: int, 
        concurrency: int
    ) -> Tournament:
        """
        Generates the schedule and creates the Tournament record + PENDING games.
        Does NOT start them yet.
        """
        # 1. Create Tournament Record
        config = {
            "model_ids": models,
            "rounds": rounds,
            "concurrency": concurrency
        }
        
        # Calculate total math for validation
        n = len(models)
        matches_per_round = n * (n - 1) 
        total_games = matches_per_round * rounds

        tournament = Tournament(
            status=TournamentStatus.SETUP,
            config=config,
            total_matches=total_games
        )
        db.add(tournament)
        await db.commit()
        await db.refresh(tournament)

        # 2. Generate Schedule (Round Robin)
        games_to_create = []
        
        for r in range(1, rounds + 1):
            # Create all pairs for this round
            round_pairs = []
            for i in range(len(models)):
                for j in range(len(models)):
                    if i == j: continue # Can't play self
                    
                    # Model A vs Model B
                    p1 = models[i]
                    p2 = models[j]
                    
                    round_pairs.append({
                        "p1": p1,
                        "p2": p2,
                        "round": r
                    })
            
            # Shuffle pairs within the round to avoid "Model A plays 10 times in a row"
            random.shuffle(round_pairs)
            
            for pair in round_pairs:
                games_to_create.append(Game(
                    tournament_id=tournament.id,
                    round_number=pair["round"],
                    player_1_type=pair["p1"],
                    player_2_type=pair["p2"],
                    status=GameStatus.PENDING, # Waiting for runner
                    history=[]
                ))

        # Bulk insert is faster
        db.add_all(games_to_create)
        await db.commit()
        
        return tournament

    async def create_evaluation_tournament(
        self, 
        db: AsyncSession, 
        target: str, 
        benchmarks: List[str], 
        rounds: int, 
        concurrency: int
    ) -> Tournament:
        # 1. Initialize Tournament Record
        config = {
            "type": "EVALUATION",
            "target_model": target,
            "benchmark_models": benchmarks,
            "rounds": rounds,
            "concurrency": concurrency
        }
        
        # Calculation: 2 games (both sides) per opponent per round
        total_games = len(benchmarks) * 2 * rounds

        tournament = Tournament(
            status=TournamentStatus.SETUP,
            config=config,
            total_matches=total_games
        )
        db.add(tournament)
        await db.commit()
        await db.refresh(tournament)

        # 2. Generate Schedule
        games_to_create = []
        for r in range(1, rounds + 1):
            round_pairs = []
            for opponent in benchmarks:
                if opponent == target:
                    continue
                
                # Game A: Target goes first
                round_pairs.append({"p1": target, "p2": opponent})
                # Game B: Opponent goes first
                round_pairs.append({"p1": opponent, "p2": target})
                
            # Shuffle within the round to prevent the same model 
            # from being hammered by the target model sequentially
            random.shuffle(round_pairs)
            
            for pair in round_pairs:
                games_to_create.append(Game(
                    tournament_id=tournament.id,
                    round_number=r,
                    player_1_type=pair["p1"],
                    player_2_type=pair["p2"],
                    status=GameStatus.PENDING,
                    history=[]
                ))

        db.add_all(games_to_create)
        await db.commit()
        return tournament

    async def start_tournament(self, db: AsyncSession, tournament_id: int):
        result = await db.execute(select(Tournament).where(Tournament.id == tournament_id))
        t = result.scalar_one_or_none()
        if t:
            t.status = TournamentStatus.IN_PROGRESS
            await db.commit()
            return True
        return False

    async def stop_tournament(self, db: AsyncSession, tournament_id: int):
        result = await db.execute(select(Tournament).where(Tournament.id == tournament_id))
        t = result.scalar_one_or_none()
        if t:
            t.status = TournamentStatus.STOPPED
            await db.commit()
            # Also cancel all pending games? Optional.
            return True
        return False

    async def pause_tournament(self, db: AsyncSession, tournament_id: int, env: str = "prod"):
        """Pause a tournament - stops all running games but preserves state."""
        result = await db.execute(select(Tournament).where(Tournament.id == tournament_id))
        t = result.scalar_one_or_none()
        if not t:
            return False
        
        # Update tournament status
        t.status = TournamentStatus.PAUSED
        await db.commit()
        
        # Find all IN_PROGRESS games for this tournament
        games_query = await db.execute(
            select(Game).where(
                Game.tournament_id == tournament_id,
                Game.status == GameStatus.IN_PROGRESS
            )
        )
        games = games_query.scalars().all()
        
        # Stop all running games in GameRunner
        for game in games:
            await game_runner.stop_game(game.id, env)
        
        print(f"⏸️ Tournament {tournament_id} Paused ({len(games)} games stopped)")
        return True

    async def update_concurrency(self, db: AsyncSession, tournament_id: int, new_concurrency: int):
        """Update concurrency limit for a tournament."""
        result = await db.execute(select(Tournament).where(Tournament.id == tournament_id))
        t = result.scalar_one_or_none()
        if not t:
            return False
        
        # Update concurrency in config
        t.config = {**t.config, "concurrency": new_concurrency}
        flag_modified(t, "config")  # Ensure SQLAlchemy sees the change in the JSONB dict
        
        await db.commit()
        print(f"⚙️ Tournament {tournament_id} concurrency updated to {new_concurrency}")
        return True

    async def tick(self, db: AsyncSession, env: str = "prod"):
        """
        The Heartbeat. Checks active tournament and feeds games to the runner.
        Called periodically by main.py
        """
        # 1. Find active tournament
        result = await db.execute(
            select(Tournament).where(Tournament.status == TournamentStatus.IN_PROGRESS)
        )
        active_tournament = result.scalars().first()
        
        if not active_tournament:
            return

        # 2. GLOBAL UNFINISHED CHECK
        # A tournament is ONLY finished if ZERO games are PENDING, IN_PROGRESS, or PAUSED
        unfinished_query = await db.execute(
            select(func.count(Game.id)).where(
                Game.tournament_id == active_tournament.id,
                Game.status.in_([GameStatus.PENDING, GameStatus.IN_PROGRESS, GameStatus.PAUSED])
            )
        )
        total_unfinished = unfinished_query.scalar() or 0

        if total_unfinished == 0:
            active_tournament.status = TournamentStatus.COMPLETED
            await db.commit()
            print(f"🏆 Tournament {active_tournament.id} TRULY finished.")
            return

        concurrency_limit = active_tournament.config.get("concurrency", 1)
        
        # 3. CONCURRENCY CHECK: Get all IN_PROGRESS games for this tournament
        active_games_query = await db.execute(
            select(Game).where(
                and_(
                    Game.tournament_id == active_tournament.id,
                    Game.status == GameStatus.IN_PROGRESS
                )
            )
        )
        active_games_db = active_games_query.scalars().all()
        
        # 4. Count games actually running in memory (GameRunner)
        current_running = 0
        orphaned_games = []
        for game in active_games_db:
            if game_runner.is_game_running(game.id, env):
                current_running += 1
            else:
                orphaned_games.append(game)
        
        slots_available = concurrency_limit - current_running
        
        if slots_available > 0:
            games_to_start = []
            
            # 5. First, fill slots with orphaned games (resume paused games)
            for game in orphaned_games:
                if slots_available <= 0:
                    break
                games_to_start.append(game)
                slots_available -= 1
            
            # 6. If slots still available, fetch PENDING or expired PAUSED games
            if slots_available > 0:
                now = datetime.now(timezone.utc)
                pending_query = await db.execute(
                    select(Game)
                    .where(
                        and_(
                            Game.tournament_id == active_tournament.id,
                            or_(
                                Game.status == GameStatus.PENDING,
                                and_(
                                    Game.status == GameStatus.PAUSED,
                                    Game.retry_after <= now  # 10 minutes have passed!
                                )
                            )
                        )
                    )
                    .with_for_update(skip_locked=True)
                    .order_by(Game.round_number.asc(), Game.id.asc())
                    .limit(slots_available)
                )
                pending_games = pending_query.scalars().all()
                games_to_start.extend(pending_games)

            # 7. Launch them
            for game in games_to_start:
                if game.status == GameStatus.PENDING:
                    game.status = GameStatus.IN_PROGRESS
                    await db.commit() # Commit status change first so other workers don't grab it
                elif game.status == GameStatus.PAUSED:
                    print(f"🚀 Resuming Game {game.id} after cooldown.")
                    game.status = GameStatus.IN_PROGRESS
                    game.retry_after = None  # Clear the snooze timer
                    await db.commit()
                
                print(f"🚀 Tournament Launching: Game {game.id} (Round {game.round_number})")
                await game_runner.start_game_if_ai_vs_ai(game.id, env)

tournament_service = TournamentService()