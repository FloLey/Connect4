import unittest
from training.core.bitboard import Bitboard
from training.core.solver import Solver
from training.core.constants import INVALID_MOVE_SCORE, MAX_SCORE, MIN_SCORE

class TestOracle(unittest.TestCase):
    def setUp(self):
        self.solver = Solver()

    def test_vertical_win_endgame(self):
        """
        Scenario: P1 to move. P1 has 3 vertical pieces in Col 0.
        Verifies simple vertical detection and positive scoring.
        """
        matrix = [[0]*7 for _ in range(6)]
        
        # Fill junk columns 2-6 to reduce search depth
        for c in range(2, 7):
            for r in range(6):
                # Pattern 1, 2, 1, 2... shifted to avoid accidental connections
                val = 1 if r % 2 == 0 else 2
                if c in [4, 5]: val = 3 - val
                matrix[r][c] = val
        
        # Setup specific scenario in Col 0 & 1
        for r in range(6): matrix[r][0] = 0; matrix[r][1] = 0
            
        # P1 has 3 in Col 0
        matrix[5][0] = 1; matrix[4][0] = 1; matrix[3][0] = 1
        # P2 has 3 in Col 1 (keep move count even for P1 turn)
        matrix[5][1] = 2; matrix[4][1] = 2; matrix[3][1] = 2
        
        board = Bitboard.from_matrix(matrix)
        result = self.solver.solve(board)
        
        self.assertEqual(result['best_move'], 0)
        self.assertEqual(result['outcome'], "WIN")
        self.assertGreater(result['best_score'], 0)

    def test_horizontal_win(self):
        """
        Scenario: P1 to move. P1 has pieces at Bottom Row of Cols 0, 1, 2.
        Winning move is Col 3.
        """
        matrix = [[0]*7 for _ in range(6)]
        
        # P1: (5,0), (5,1), (5,2)
        matrix[5][0] = 1
        matrix[5][1] = 1
        matrix[5][2] = 1
        
        # P2: Junk to balance move count (3 moves)
        matrix[4][0] = 2
        matrix[4][1] = 2
        matrix[4][2] = 2
        
        # Total 6 moves. P1 Turn.
        # We fill upper rows with junk to limit depth and prevent other wins
        for r in range(0, 4): # Rows 0-3
            for c in range(7):
                matrix[r][c] = 1 if (r+c)%2==0 else 2
                
        # Clear the target winning spot (5,3) and above
        for r in range(6): matrix[r][3] = 0
            
        board = Bitboard.from_matrix(matrix)
        result = self.solver.solve(board)
        
        # Expect Col 3 to be the win
        self.assertEqual(result['best_move'], 3)
        self.assertGreater(result['scores'][3], 0)

    def test_diagonal_win(self):
        """
        Scenario: P1 to move. P1 can complete a '/' diagonal.
        Target slot is (2,3).
        Required pieces for the diagonal: (5,0), (4,1), (3,2).
        
        We must carefully construct the board so that:
        1. There are NO pre-existing wins (vertical, horizontal, or diagonal).
        2. The piece count is balanced (Total pieces is even -> P1's turn).
        """
        matrix = [[0]*7 for _ in range(6)]
        
        # --- 1. Set up the specific diagonal scenario in Cols 0-3 ---
        # Col 0: P1 at bottom
        matrix[5][0] = 1 
        
        # Col 1: P2 at bottom, P1 on top
        matrix[5][1] = 2; matrix[4][1] = 1
        
        # Col 2: P2, P2, P1
        matrix[5][2] = 2; matrix[4][2] = 2; matrix[3][2] = 1
        
        # Col 3: P2, P2, P2 (Empty at 2,3 - The Winning Spot)
        matrix[5][3] = 2; matrix[4][3] = 2; matrix[3][3] = 2
        
        # Count so far:
        # P1: 3 pieces ((5,0), (4,1), (3,2))
        # P2: 6 pieces ((5,1), (5,2), (4,2), (5,3), (4,3), (3,3))
        # Diff: P2 +3. We need 3 more P1 pieces to equalize (Total Even -> P1 Turn).
        
        # --- 2. Balance using Cols 4-6 ---
        # We need to add 3 P1 pieces without creating lines.
        # We will simply place them in Col 6 vertically, but ensuring no 4-in-a-row.
        # P1, P2, P1, P2, P1
        matrix[5][6] = 1
        matrix[4][6] = 2
        matrix[3][6] = 1
        matrix[2][6] = 2
        matrix[1][6] = 1
        
        # Now P1 has added 3 pieces. P2 has added 2.
        # Total P1: 3 + 3 = 6
        # Total P2: 6 + 2 = 8
        # Still need 2 more P1 pieces.
        
        # Let's use Col 5.
        matrix[5][5] = 1
        matrix[4][5] = 1
        # Danger: (5,5)=1, (5,0)=1. Horizontal is far.
        # Check verticals: Col 5 has 2 P1s. Safe.
        
        # Total P1: 6 + 2 = 8.
        # Total P2: 8.
        # Grand Total: 16 (Even). P1's Turn. Correct.
        
        # Verify no accidental wins in the balancing:
        # Col 6: 1, 2, 1, 2, 1. Safe.
        # Col 5: 1, 1. Safe.
        # Diagonals between 5 and 6?
        # (5,5)=1, (4,6)=2. Safe.
        # (4,5)=1, (3,6)=1. 2-in-row. Safe.
        
        board = Bitboard.from_matrix(matrix)
        result = self.solver.solve(board)
        
        # P1 needs to play Col 3 to land at (2,3) and complete the / diagonal.
        self.assertEqual(result['best_move'], 3)
        self.assertEqual(result['outcome'], "WIN")
        self.assertGreater(result['scores'][3], 0)

    def test_forced_block(self):
        """
        Scenario: P1 to move. P2 has 3 horizontal pieces in a row.
        P1 CANNOT win immediately.
        P1 MUST block P2, otherwise P2 wins next turn (Score -42).
        Therefore, Blocking Score > -42.
        """
        matrix = [[0]*7 for _ in range(6)]
        
        # Fill columns 4,5,6 to limit depth with safe pattern
        for c in range(4, 7):
            for r in range(6):
                 matrix[r][c] = 1 if (r+c)%2==0 else 2

        # P2 Threat: (5,0), (5,1), (5,2)
        matrix[5][0] = 2
        matrix[5][1] = 2
        matrix[5][2] = 2
        
        # P1 Distractions (Cols 0,1,2, Row 4)
        matrix[4][0] = 1
        matrix[4][1] = 1
        matrix[4][2] = 1
        
        # P1 to move.
        # If P1 plays Col 3, piece lands at (5,3) -> BLOCKS P2.
        # If P1 plays anywhere else, P2 plays Col 3 next and wins.
        
        board = Bitboard.from_matrix(matrix)
        result = self.solver.solve(board)
        
        # Verify Col 3 is the best move
        self.assertEqual(result['best_move'], 3)
        
        # Verify Score Logic:
        # Playing 3 prevents immediate loss (-42). 
        block_score = result['scores'][3]
        
        # Assert blocking is better than the worst possible outcome
        self.assertGreater(block_score, MIN_SCORE + 1)
        
        # Assert not blocking is a disaster (if valid)
        if result['scores'][0] != INVALID_MOVE_SCORE:
            self.assertLess(result['scores'][0], 0)

if __name__ == '__main__':
    unittest.main()