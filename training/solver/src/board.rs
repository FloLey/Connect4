pub const COLS: u32 = 7;
pub const ROWS: u32 = 6;
pub const STRIDE: u32 = 7; // bits per column (6 rows + 1 sentinel)

/// Bottom bit of each column (row 0).
const BOTTOM_MASK: u64 = {
    let mut m = 0u64;
    let mut c = 0u32;
    while c < COLS {
        m |= 1u64 << (c * STRIDE);
        c += 1;
    }
    m
};

/// All playable bits (rows 0 through ROWS-1 of each column).
const BOARD_MASK: u64 = {
    let col_mask = (1u64 << ROWS) - 1; // 0b111111
    let mut m = 0u64;
    let mut c = 0u32;
    while c < COLS {
        m |= col_mask << (c * STRIDE);
        c += 1;
    }
    m
};

/// Returns the bit index for a given column and row.
pub fn bit_index(col: u32, row: u32) -> u32 {
    col * STRIDE + row
}

#[derive(Clone)]
pub struct Board {
    pub position: u64, // bitmask for current player's pieces
    pub mask: u64,     // bitmask for ALL pieces (both players)
    pub moves: u32,    // number of moves played so far
    move_history: [u32; 42],
    history_len: usize,
}

impl Board {
    pub fn new() -> Self {
        Board {
            position: 0,
            mask: 0,
            moves: 0,
            move_history: [0u32; 42],
            history_len: 0,
        }
    }

    /// Returns the height (0..=6) of a given column.
    /// Height 6 means full (no legal move).
    #[inline(always)]
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
        self.move_history[self.history_len] = col;
        self.history_len += 1;
    }

    /// Undo the last move. Uses move_history to determine which column.
    pub fn undo(&mut self) {
        assert!(self.history_len > 0, "Cannot undo from empty move history");
        self.history_len -= 1;
        let col = self.move_history[self.history_len];
        let h = self.height(col);
        assert!(h > 0, "Column {} is empty, cannot undo", col);

        let bit = 1u64 << bit_index(col, h - 1);
        self.mask ^= bit;
        self.position ^= self.mask;
        self.moves -= 1;
    }

    /// Returns true if the given bitboard contains a four-in-a-row.
    #[inline(always)]
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
    #[inline(always)]
    pub fn opponent_board(&self) -> u64 {
        self.mask ^ self.position
    }

    /// Returns a bitmask of all cells where the current player can win immediately.
    #[inline(always)]
    pub fn winning_moves(&self) -> u64 {
        Self::compute_winning_positions(self.position) & self.legal_moves_mask()
    }

    /// Returns a bitmask of all cells where the opponent could win if it were their turn.
    #[cfg(test)]
    pub fn opponent_winning_moves(&self) -> u64 {
        Self::compute_winning_positions(self.mask ^ self.position) & self.legal_moves_mask()
    }

    /// Returns a bitmask of all cells that are legal next moves.
    /// Uses the bottom-mask trick: adding BOTTOM_MASK to mask carries through
    /// consecutive 1s to land on the first empty row in each column.
    #[inline(always)]
    pub fn legal_moves_mask(&self) -> u64 {
        (self.mask + BOTTOM_MASK) & BOARD_MASK
    }

    /// Returns the unique key for this position (used in transposition table).
    /// key = position + mask (provably unique for all reachable positions)
    #[inline(always)]
    pub fn key(&self) -> u64 {
        self.position.wrapping_add(self.mask)
    }

    /// Returns a compact key encoding the first `n` moves, used to group positions
    /// with the same opening prefix into the same thread chunk for TT locality.
    pub fn prefix_key(&self, n: usize) -> u32 {
        let depth = self.history_len.min(n);
        let mut key = 0u32;
        for i in 0..depth {
            key = key * COLS + self.move_history[i];
        }
        key
    }

    /// Returns the move sequence as a string of column digits (0-indexed).
    pub fn move_sequence(&self) -> String {
        self.move_history[..self.history_len].iter().map(|c| char::from(b'0' + *c as u8)).collect()
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

    /// Returns a bitmask of legal moves that don't give the opponent an immediate win.
    /// Excludes moves that play directly below an opponent winning cell.
    /// If forced to block (opponent has a threat on a legal cell), returns only the blocking move(s).
    #[allow(dead_code)]
    #[inline(always)]
    pub fn non_losing_moves(&self) -> u64 {
        let possible = self.legal_moves_mask();
        // CRITICAL: mask to empty board cells only — prevents phantom bits
        // in sentinel rows from poisoning the >> 1 shift
        let opponent_wins =
            Self::compute_winning_positions(self.mask ^ self.position) & (BOARD_MASK & !self.mask);
        let forced = opponent_wins & possible;

        if forced != 0 {
            if forced & (forced - 1) != 0 {
                // Multiple forced moves = double threat = we lose
                return 0;
            }
            // Single forced move: must play there, but also filter below-threat cells
            return forced & !(opponent_wins >> 1);
        }

        // No forced blocking: exclude any move where the cell directly above
        // is an opponent winning position (playing there hands them the win).
        possible & !(opponent_wins >> 1)
    }

    /// Scores a move by counting how many winning positions it creates for the current player.
    /// Higher score = more threats created = better move to explore first.
    pub fn move_score(&self, col: u32) -> u32 {
        let h = self.height(col);
        let bit = 1u64 << bit_index(col, h);
        Self::compute_winning_positions(self.position | bit).count_ones()
    }

    /// Pre-compute (legal_moves, winning_positions_current, winning_positions_opponent) in one pass.
    /// Called once per negamax node to avoid redundant compute_winning_positions calls.
    #[inline(always)]
    pub fn precompute_threats(&self) -> (u64, u64, u64) {
        let legal = (self.mask + BOTTOM_MASK) & BOARD_MASK;
        let winning_cur = Self::compute_winning_positions(self.position);
        let winning_opp = Self::compute_winning_positions(self.mask ^ self.position);
        (legal, winning_cur, winning_opp)
    }

    /// winning_moves using pre-computed values from precompute_threats().
    #[inline(always)]
    pub fn winning_moves_with(&self, winning_cur: u64, legal: u64) -> u64 {
        winning_cur & legal
    }

    /// non_losing_moves using pre-computed opponent winning positions from precompute_threats().
    #[inline(always)]
    pub fn non_losing_moves_with(&self, legal: u64, winning_opp: u64) -> u64 {
        let opponent_wins = winning_opp & (BOARD_MASK & !self.mask);
        let forced = opponent_wins & legal;
        if forced != 0 {
            if forced & (forced - 1) != 0 { return 0; }
            return forced & !(opponent_wins >> 1);
        }
        legal & !(opponent_wins >> 1)
    }

    /// Compute all cells where placing a piece would complete a four-in-a-row
    /// for the given player. Uses Pascal Pons' method: for each direction,
    /// enumerate all 4 gap patterns (XXX_, _XXX, XX_X, X_XX).
    #[inline(always)]
    fn compute_winning_positions(pos: u64) -> u64 {
        let mut r = 0u64;

        // Vertical (stride 1) — only one pattern due to gravity: pieces below, win on top
        r |= (pos << 1) & (pos << 2) & (pos << 3);

        // Horizontal (stride 7) — all 4 patterns
        let p = (pos << 7) & (pos << 14);
        r |= p & (pos << 21); // XXX_
        r |= p & (pos >> 7); // XX_X
        let p = (pos >> 7) & (pos >> 14);
        r |= p & (pos << 7); // X_XX
        r |= p & (pos >> 21); // _XXX

        // Diagonal / (stride 8) — all 4 patterns
        let p = (pos << 8) & (pos << 16);
        r |= p & (pos << 24);
        r |= p & (pos >> 8);
        let p = (pos >> 8) & (pos >> 16);
        r |= p & (pos << 8);
        r |= p & (pos >> 24);

        // Diagonal \ (stride 6) — all 4 patterns
        let p = (pos << 6) & (pos << 12);
        r |= p & (pos << 18);
        r |= p & (pos >> 6);
        let p = (pos >> 6) & (pos >> 12);
        r |= p & (pos << 6);
        r |= p & (pos >> 18);

        r
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
        b.undo();
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

        // Fill the board with a pattern that avoids 4-in-a-row
        // (alternating pairs per column, reverse order for upper half)
        let mut full = Board::new();
        let moves = [
            0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5,
            1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2,
            5, 4, 5, 4, 5, 4,
            6, 6, 6, 6, 6, 6,
        ];
        for &col in &moves {
            if full.is_game_over() { break; }
            full.play(col);
        }
        assert!(full.is_full() || full.is_game_over());
    }

    #[test]
    fn test_winning_moves_vertical() {
        let mut b = Board::new();
        // P1: col 0, P2: col 1, repeat 3 times → P1 has 3 in col 0
        b.play(0); b.play(1); b.play(0); b.play(1); b.play(0); b.play(1);
        // P1 to move with 3 in col 0 (rows 0,1,2), winning cell is row 3
        let wm = b.winning_moves();
        assert_ne!(wm, 0, "Should detect vertical win opportunity");
        // The winning cell should be bit_index(0, 3) = 3
        assert_ne!(wm & (1u64 << bit_index(0, 3)), 0);
    }

    #[test]
    fn test_winning_moves_horizontal() {
        let mut b = Board::new();
        // P1 places at cols 0,1,2 (bottom row), P2 at col 6
        b.play(0); b.play(6); b.play(1); b.play(6); b.play(2); b.play(6);
        // P1 to move with 3 horizontal at row 0, can win with col 3
        let wm = b.winning_moves();
        assert_ne!(wm, 0, "Should detect horizontal win opportunity");
        assert_ne!(wm & (1u64 << bit_index(3, 0)), 0);
    }

    #[test]
    fn test_opponent_winning_moves() {
        let mut b = Board::new();
        // P1: col 0, P2: col 1, repeat → P2 has 3 in col 1
        b.play(0); b.play(1); b.play(0); b.play(1); b.play(0); b.play(1);
        // After P2's last move (move 6), P1 is to move.
        b.play(6); // P1 wastes a move on col 6
        // Now P2 to move, but let's check from P1's perspective of opponent threats
        // Actually after 7 moves, P2 is current player. P1 (opponent) has 3 in col 0.
        let opp_wm = b.opponent_winning_moves();
        assert_ne!(opp_wm, 0, "Should detect opponent's win threat");
    }

    #[test]
    fn test_non_losing_moves_excludes_below_opponent_win() {
        // Build a position where P2 has 3 horizontal on row 0 (cols 3,4,5)
        // and threatens col 2 at row 0 and col 6 at row 0.
        // Playing col 2 or col 6 at row 0 should be allowed (it blocks/doesn't
        // create a threat above). But if P2 threatened at row 1, playing at
        // row 0 in that column would give P2 row 1 — that should be excluded.
        let mut b = Board::new();
        b.play(0); b.play(3); b.play(1); b.play(4); b.play(6); b.play(5);
        // P1 to move. P2 has 3 horizontal (cols 3,4,5 row 0).
        // P2's winning cells: col 2 row 0 (extends left) and col 6 row 0 (extends right).
        // But col 6 already has P1's piece at row 0, so only col 2 row 0 is a threat.
        let nlm = b.non_losing_moves();
        let legal = b.legal_moves_mask();
        // Non-losing moves must be subset of legal
        assert_eq!(nlm & !legal, 0, "non_losing_moves should be subset of legal_moves");
        // Since P2 threatens col 2 row 0 (a legal cell), this is a forced blocking move
        let forced_bit = 1u64 << bit_index(2, 0);
        assert_ne!(nlm & forced_bit, 0, "Must include the forced blocking move at col 2");
    }

    #[test]
    fn test_non_losing_moves_forced_blocking() {
        let mut b = Board::new();
        // Set up: P2 has 3 in col 1, P1 must block
        b.play(0); b.play(1); b.play(0); b.play(1); b.play(0); b.play(1);
        b.play(6); // P1 wastes move
        // P2 to move, P1 has 3 in col 0 threatening row 3
        // P2 must block col 0 — this should be the only non-losing move that blocks
        let nlm = b.non_losing_moves();
        // The forced blocking move (col 0, row 3) should be included
        assert_ne!(nlm & (1u64 << bit_index(0, 3)), 0, "Blocking move should be included");
    }

    #[test]
    fn test_move_score() {
        let mut b = Board::new();
        // P1 places at cols 0, 1, 2. P2 at col 6.
        b.play(0); b.play(6); b.play(1); b.play(6); b.play(2); b.play(6);
        // P1 to move. Playing col 3 would complete horizontal 4.
        // move_score(3) should be high (creates winning position).
        let score_3 = b.move_score(3);
        let score_5 = b.move_score(5);
        assert!(score_3 > score_5, "Col 3 (winning) should score higher than col 5");
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
