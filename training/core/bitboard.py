# training/core/bitboard.py
from typing import List
from .constants import ROWS, COLS, HEIGHT

class Bitboard:
    def __init__(self, position: int = 0, mask: int = 0, moves_count: int = 0):
        self.position = position
        self.mask = mask
        self.moves_count = moves_count

    @classmethod
    def from_matrix(cls, matrix: List[List[int]]):
        """
        Converts app 2D matrix (Row 0=Top) to Bitboard (Row 0=Bottom).
        Automatically detects whose turn it is based on piece count.
        """
        mask = 0
        p1_pieces = 0
        moves_count = 0
        
        # 1. Build Mask and P1 Map
        for c in range(COLS):
            for r in range(ROWS - 1, -1, -1):
                val = matrix[r][c]
                if val != 0:
                    moves_count += 1
                    logical_row = (ROWS - 1) - r
                    bit_idx = c * HEIGHT + logical_row
                    
                    mask |= (1 << bit_idx)
                    if val == 1:
                        p1_pieces |= (1 << bit_idx)

        # 2. Determine Current Player (P1 if even moves, P2 if odd)
        # Bitboard 'position' is always CURRENT player.
        # If it is P1's turn, position = p1_pieces.
        # If it is P2's turn, position = p2_pieces (which is mask ^ p1_pieces).
        if moves_count % 2 == 0:
            position = p1_pieces
        else:
            position = mask ^ p1_pieces

        return cls(position, mask, moves_count)

    def can_play(self, col: int) -> bool:
        """Checks if the top cell of the column is empty."""
        # Check bit at Row 5 of specific column
        top_mask = 1 << (col * HEIGHT + (ROWS - 1))
        return (self.mask & top_mask) == 0

    def play(self, col: int) -> 'Bitboard':
        """
        Returns a NEW Bitboard with the move applied and turn swapped.
        """
        # XOR swap logic: New Position = Current Position ^ Mask
        # This effectively flips the board to the perspective of the other player.
        
        # 1. Find the first empty bit in column
        move_bit = (self.mask + (1 << (col * HEIGHT))) & ((1 << (col * HEIGHT + ROWS)) - 1)
        
        new_mask = self.mask | move_bit
        new_position = self.position ^ self.mask # Swap roles
        
        return Bitboard(new_position, new_mask, self.moves_count + 1)

    def check_win(self) -> bool:
        """
        Checks if 'position' (current player) has 4 connected.
        """
        p = self.position
        # Horizontal (Shift 7)
        m = p & (p >> HEIGHT); 
        if m & (m >> (2 * HEIGHT)): return True
        # Diagonal \ (Shift 6)
        m = p & (p >> (HEIGHT - 1)); 
        if m & (m >> (2 * (HEIGHT - 1))): return True
        # Diagonal / (Shift 8)
        m = p & (p >> (HEIGHT + 1)); 
        if m & (m >> (2 * (HEIGHT + 1))): return True
        # Vertical (Shift 1)
        m = p & (p >> 1); 
        if m & (m >> 2): return True

        return False

    def get_key(self):
        """Unique ID for caching: Position + Mask"""
        return self.position + self.mask