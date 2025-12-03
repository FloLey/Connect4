import logging
from typing import List, Optional, Tuple, Dict, Any

# Logger setup
logger = logging.getLogger(__name__)

ROWS = 6
COLS = 7

class ConnectFour:
    def __init__(self):
        """
        Board uses (row, col) indexing.
        Row 0 is the TOP of the board.
        Row 5 is the BOTTOM of the board.
        Values: 0=Empty, 1=Player1, 2=Player2
        """
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.current_turn = 1
        self.winner: Optional[int] = None
        self.history: List[Dict[str, Any]] = []

    def get_valid_moves(self) -> List[int]:
        """Returns a list of column indices (0-6) that are not full."""
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def is_valid_move(self, col: int) -> bool:
        if col < 0 or col >= COLS:
            return False
        return self.board[0][col] == 0

    def drop_piece(self, col: int) -> bool:
        """
        Drops a piece into the specified column. 
        Returns True if successful, False if invalid or game over.
        """
        if self.winner is not None or not self.is_valid_move(col):
            return False

        # Gravity: Find the lowest empty row
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_turn
                
                # Record move logic (metadata will be added by API layer later)
                self.history.append({
                    "player": self.current_turn,
                    "column": col
                })

                if self.check_win(r, col):
                    self.winner = self.current_turn
                else:
                    self.switch_turn()
                return True
        return False

    def switch_turn(self):
        self.current_turn = 1 if self.current_turn == 2 else 2

    def check_win(self, r: int, c: int) -> bool:
        """Checks for 4-in-a-row originating from the placed piece."""
        player = self.board[r][c]
        # Directions: Horizontal, Vertical, Diagonal /, Diagonal \
        directions = [(0, 1), (1, 0), (1, -1), (1, 1)]

        for dr, dc in directions:
            count = 1
            # Check positive direction
            for i in range(1, 4):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            # Check negative direction
            for i in range(1, 4):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < ROWS and 0 <= nc < COLS and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
            
            if count >= 4:
                return True
        return False

    def is_draw(self) -> bool:
        """Returns True if board is full and no winner."""
        return self.winner is None and all(self.board[0][c] != 0 for c in range(COLS))

    # --- Formatting for LLM Consumption ---

    def get_visual_board(self) -> str:
        """Generates an ASCII grid representation."""
        symbols = {0: ".", 1: "X", 2: "O"}
        header = " " + " ".join([str(i) for i in range(COLS)])
        rows_str = []
        for r in range(ROWS):
            row_cells = [symbols[self.board[r][c]] for c in range(COLS)]
            rows_str.append("|" + "|".join(row_cells) + "|")
        return header + "\n" + "\n".join(rows_str)

    def get_textual_description(self) -> str:
        """
        Describes the board column by column, listing pieces from Bottom to Top.
        Example: 'Col 0: P1, P2, Empty'
        """
        lines = []
        for c in range(COLS):
            pieces = []
            # Scan from Bottom (Row 5) to Top (Row 0)
            for r in range(ROWS - 1, -1, -1):
                val = self.board[r][c]
                if val == 0:
                    break # Stop at first empty space
                pieces.append("P1" if val == 1 else "P2")
            
            desc = ", ".join(pieces) if pieces else "Empty"
            lines.append(f"Column {c}: {desc}")
        return "\n".join(lines)