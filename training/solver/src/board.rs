pub const COLS: u32 = 7;
pub const ROWS: u32 = 6;
pub const STRIDE: u32 = 7; // bits per column (6 rows + 1 sentinel)

/// Returns the bit index for a given column and row.
pub fn bit_index(col: u32, row: u32) -> u32 {
    col * STRIDE + row
}

#[derive(Clone)]
pub struct Board {
    pub position: u64, // bitmask for current player's pieces
    pub mask: u64,     // bitmask for ALL pieces (both players)
    pub moves: u32,    // number of moves played so far
    move_history: Vec<u32>,
}

impl Board {
    pub fn new() -> Self {
        Board {
            position: 0,
            mask: 0,
            moves: 0,
            move_history: Vec::with_capacity(42),
        }
    }

    /// Returns the height (0..=6) of a given column.
    /// Height 6 means full (no legal move).
    pub fn height(&self, col: u32) -> u32 {
        let bottom = col * STRIDE;
        let col_mask = self.mask >> bottom;
        // Count trailing ones = number of pieces stacked in this column
        (!col_mask).trailing_zeros().min(STRIDE)
    }

    /// Play a piece in column `col` for the current player.
    /// Panics if the column is full.
    pub fn play(&mut self, col: u32) {
        let h = self.height(col);
        assert!(h < ROWS, "Column {} is full", col);

        let bit = 1u64 << bit_index(col, h);

        // After placing, switch perspective: XOR position with the new mask
        // so `position` always represents the current (next-to-move) player.
        self.position ^= self.mask;
        self.mask |= bit;
        self.moves += 1;
        self.move_history.push(col);
    }

    /// Undo the last move in column `col`.
    pub fn undo(&mut self, col: u32) {
        let h = self.height(col);
        assert!(h > 0, "Column {} is empty, cannot undo", col);

        let bit = 1u64 << bit_index(col, h - 1);
        self.mask ^= bit;
        self.position ^= self.mask;
        self.moves -= 1;
        self.move_history.pop();
    }

    /// Returns true if the given bitboard contains a four-in-a-row.
    pub fn is_win(&self, board: u64) -> bool {
        let mut m: u64;
        // Horizontal (stride = 7)
        m = board & (board >> 7);
        if m & (m >> 14) != 0 {
            return true;
        }
        // Vertical (stride = 1)
        m = board & (board >> 1);
        if m & (m >> 2) != 0 {
            return true;
        }
        // Diagonal / (stride = 8 = 7+1)
        m = board & (board >> 8);
        if m & (m >> 16) != 0 {
            return true;
        }
        // Diagonal \ (stride = 6 = 7-1)
        m = board & (board >> 6);
        if m & (m >> 12) != 0 {
            return true;
        }
        false
    }

    /// Returns the bitboard of the opponent (the player who just moved).
    pub fn opponent_board(&self) -> u64 {
        self.mask ^ self.position
    }

    /// Returns a bitmask of all cells where the current player can win immediately.
    pub fn winning_moves(&self) -> u64 {
        self.compute_winning_positions(self.position) & self.legal_moves_mask()
    }

    /// Returns a bitmask of all cells that are legal next moves.
    pub fn legal_moves_mask(&self) -> u64 {
        let mut result = 0u64;
        for col in 0..COLS {
            let h = self.height(col);
            if h < ROWS {
                result |= 1u64 << bit_index(col, h);
            }
        }
        result
    }

    /// Returns the unique key for this position (used in transposition table).
    /// key = position + mask (provably unique for all reachable positions)
    pub fn key(&self) -> u64 {
        self.position.wrapping_add(self.mask)
    }

    /// Returns the move sequence as a string of column digits (0-indexed).
    pub fn move_sequence(&self) -> String {
        self.move_history.iter().map(|c| char::from(b'0' + *c as u8)).collect()
    }

    /// Returns true if the board is completely full (draw).
    pub fn is_full(&self) -> bool {
        self.moves >= ROWS * COLS
    }

    /// Returns true if the game is over (someone won or board is full).
    /// "Someone won" means the player who just moved won.
    pub fn is_game_over(&self) -> bool {
        self.is_full() || self.is_win(self.opponent_board())
    }

    /// Compute all positions where `player` would complete a four-in-a-row
    /// if they placed a piece there.
    fn compute_winning_positions(&self, player: u64) -> u64 {
        let mut result: u64;

        // Vertical (stride = 1): 3 in a column, winning cell is 3 above the bottom
        let m_v = player & (player >> 1) & (player >> 2);
        result = m_v << 3;

        // Horizontal (stride = 7)
        result |= Self::direction_wins(player, 7);
        // Diagonal / (stride = 8)
        result |= Self::direction_wins(player, 8);
        // Diagonal \ (stride = 6)
        result |= Self::direction_wins(player, 6);

        // Only return cells that are within the playable area (not sentinel rows)
        let mut board_mask = 0u64;
        for col in 0..COLS {
            for row in 0..ROWS {
                board_mask |= 1u64 << bit_index(col, row);
            }
        }
        result & board_mask & !self.mask
    }

    /// For a given direction stride, compute all cells where placing a piece
    /// would complete a 4-in-a-row for the given player in that direction.
    fn direction_wins(player: u64, stride: u32) -> u64 {
        let s = stride;
        let mut result = 0u64;

        // Pattern: _ X X X
        let m = player & (player >> s) & (player >> (2 * s));
        result |= m << s;

        // Pattern: X _ X X
        let m = (player << s) & (player >> s) & (player >> (2 * s));
        result |= m;

        // Pattern: X X _ X
        let m = (player << (2 * s)) & (player << s) & (player >> s);
        result |= m;

        // Pattern: X X X _
        let m = player & (player << s) & (player << (2 * s));
        result |= m >> s;

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board() {
        let b = Board::new();
        assert_eq!(b.position, 0);
        assert_eq!(b.mask, 0);
        assert_eq!(b.moves, 0);
    }

    #[test]
    fn test_play_and_height() {
        let mut b = Board::new();
        assert_eq!(b.height(3), 0);
        b.play(3);
        assert_eq!(b.height(3), 1);
        b.play(3);
        assert_eq!(b.height(3), 2);
        assert_eq!(b.moves, 2);
    }

    #[test]
    fn test_undo() {
        let mut b = Board::new();
        b.play(3);
        b.play(2);
        let key_before = b.key();
        b.play(4);
        b.undo(4);
        assert_eq!(b.key(), key_before);
        assert_eq!(b.moves, 2);
    }

    #[test]
    fn test_vertical_win() {
        let mut b = Board::new();
        // Player 1: col 0, Player 2: col 1, repeat
        b.play(0); // P1
        b.play(1); // P2
        b.play(0); // P1
        b.play(1); // P2
        b.play(0); // P1
        b.play(1); // P2
        b.play(0); // P1 — 4 in col 0
        assert!(b.is_win(b.opponent_board()));
    }

    #[test]
    fn test_horizontal_win() {
        let mut b = Board::new();
        // P1: 0,1,2,3 with P2 stacking on col 6
        b.play(0); // P1
        b.play(6); // P2
        b.play(1); // P1
        b.play(6); // P2
        b.play(2); // P1
        b.play(6); // P2
        b.play(3); // P1 — horizontal 4
        assert!(b.is_win(b.opponent_board()));
    }

    #[test]
    fn test_move_sequence() {
        let mut b = Board::new();
        b.play(3);
        b.play(3);
        b.play(4);
        assert_eq!(b.move_sequence(), "334");
    }

    #[test]
    fn test_is_full() {
        let b = Board::new();
        assert!(!b.is_full());
    }

    #[test]
    fn test_key_different_for_different_ownership() {
        // Different move orders lead to different piece ownership → different keys
        let mut b1 = Board::new();
        b1.play(0);
        b1.play(1);

        let mut b2 = Board::new();
        b2.play(1);
        b2.play(0);

        assert_ne!(b1.key(), b2.key());
    }

    #[test]
    fn test_key_same_for_same_position() {
        // Same position reached via same moves should always have same key
        let mut b1 = Board::new();
        b1.play(3);
        b1.play(4);

        let mut b2 = Board::new();
        b2.play(3);
        b2.play(4);

        assert_eq!(b1.key(), b2.key());
    }
}
