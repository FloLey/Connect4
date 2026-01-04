# training/core/solver.py
from .constants import MAX_SCORE, MIN_SCORE, INVALID_MOVE_SCORE, COLS, MAX_MOVES, COLUMN_ORDER, HEIGHT
from .bitboard import Bitboard
from .transposition import TranspositionTable

class Solver:
    def __init__(self):
        self.tt = TranspositionTable()
        self.nodes = 0

    def solve(self, board: Bitboard) -> dict:
        """
        Root Entry Point.
        Evaluates EVERY column to provide a full RL reward signal.
        """
        self.nodes = 0
        self.tt.reset()
        
        move_scores = {}
        best_score = MIN_SCORE - 1
        best_move = -1

        # 1. Iterate ALL columns (0..6) - No sorting here, we need index mapping
        for col in range(COLS):
            if board.can_play(col):
                next_board = board.play(col)
                
                # Check if this move wins immediately (optimization)
                # Note: next_board is swapped. We check if the piece we just placed (opponent of next_board) won.
                # Ideally, simple Negamax recursion handles it, but let's call it.
                
                # We search with FULL WINDOW (MIN_SCORE, MAX_SCORE)
                # We do NOT pass 'best_score' as alpha, because we want the
                # exact value of *this* specific column, even if it's sub-optimal.
                score = -self.negamax(next_board, MIN_SCORE, MAX_SCORE)
                
                move_scores[col] = score

                if score > best_score:
                    best_score = score
                    best_move = col
            else:
                # 2. Penalty for Invalid Moves
                move_scores[col] = INVALID_MOVE_SCORE

        outcome, distance = self._analyze_score(best_score)

        return {
            "best_move": best_move,
            "best_score": best_score,
            "outcome": outcome,
            "distance_to_outcome": distance,
            "scores": move_scores, # Map {0: -40, 1: 38, ...}
            "nodes_explored": self.nodes
        }

    def _analyze_score(self, score):
        if score > 0: return "WIN", (MAX_SCORE - score)
        if score < 0: return "LOSS", (MAX_SCORE + score)
        return "DRAW", 0

    def negamax(self, board: Bitboard, alpha: int, beta: int) -> int:
        self.nodes += 1
        
        # 1. Check if current player has won
        # In negamax, we check if the player who just moved (opponent in this node) won
        # The pieces of the player who just moved are: board.position ^ board.mask
        opponent_pieces = board.position ^ board.mask
        if self._check_win_mask(opponent_pieces):
            # If opponent won, current player loses
            # Score is negative of (MAX_SCORE - moves_count)
            return -(MAX_SCORE - board.moves_count)

        # 2. Check Draw
        if board.moves_count == MAX_MOVES:
            return 0

        # 3. Transposition Table Cache
        key = board.get_key()
        if (tt_val := self.tt.get(key)) is not None:
            # For exact solving, we can treat TT values as exact if we aren't using iterative deepening bounds
            # To be safe: if tt_val > MAX_SCORE - 100 or tt_val < MIN_SCORE + 100: return tt_val
            return tt_val

        # 4. Max theoretical score possible from this depth
        # If we win next turn, score is (MAX_SCORE - (moves+1))
        # If alpha is already better than that, prune.
        max_possible = MAX_SCORE - board.moves_count
        if alpha >= max_possible:
            return max_possible

        # 5. Recursive Search
        for col in COLUMN_ORDER: # 3, 2, 4, 1...
            if board.can_play(col):
                next_board = board.play(col)
                score = -self.negamax(next_board, -beta, -alpha)

                if score >= beta:
                    return score # Beta Cutoff
                if score > alpha:
                    alpha = score

        self.tt.put(key, alpha)
        return alpha

    def _check_win_mask(self, p: int) -> bool:
        # Horizontal
        m = p & (p >> HEIGHT); 
        if m & (m >> (2 * HEIGHT)): return True
        # Diagonal \
        m = p & (p >> (HEIGHT - 1)); 
        if m & (m >> (2 * (HEIGHT - 1))): return True
        # Diagonal /
        m = p & (p >> (HEIGHT + 1)); 
        if m & (m >> (2 * (HEIGHT + 1))): return True
        # Vertical
        m = p & (p >> 1); 
        if m & (m >> 2): return True
        return False