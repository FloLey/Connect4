use crate::board::{Board, COLS, ROWS};

const MOVE_ORDER: [u32; 7] = [3, 2, 4, 1, 5, 0, 6];
const MAX_SCORE: i32 = ((COLS * ROWS - 1) / 2) as i32; // 20

/// Fixed-size transposition table with simple replacement.
pub struct TranspositionTable {
    keys: Vec<u64>,
    values: Vec<i8>,
    size: usize,
}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        TranspositionTable {
            keys: vec![0; size],
            values: vec![0; size],
            size,
        }
    }

    pub fn get(&self, key: u64) -> Option<i8> {
        let idx = (key as usize) % self.size;
        if self.keys[idx] == key && self.values[idx] != 0 {
            Some(self.values[idx])
        } else {
            None
        }
    }

    pub fn put(&mut self, key: u64, value: i8) {
        let idx = (key as usize) % self.size;
        self.keys[idx] = key;
        self.values[idx] = value;
    }

    pub fn reset(&mut self) {
        self.keys.fill(0);
        self.values.fill(0);
    }
}

pub struct Solver {
    pub table: TranspositionTable,
    pub node_count: u64,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            table: TranspositionTable::new(8_388_593),
            node_count: 0,
        }
    }

    /// Returns the exact score for the given position.
    pub fn solve(&mut self, board: &mut Board) -> i32 {
        self.node_count = 0;

        // Check if the previous player already won
        if board.is_win(board.opponent_board()) {
            // The player who just moved won; current player loses.
            let moves_at_win = board.moves;
            return -(((COLS * ROWS) as i32 + 1 - moves_at_win as i32) / 2);
        }

        if board.is_full() {
            return 0;
        }

        let alpha = -MAX_SCORE;
        let beta = MAX_SCORE;
        self.negamax(board, alpha, beta)
    }

    fn negamax(&mut self, board: &mut Board, mut alpha: i32, mut beta: i32) -> i32 {
        self.node_count += 1;

        // Draw check
        if board.is_full() {
            return 0;
        }

        // Check if current player can win immediately
        let winning = board.winning_moves();
        if winning != 0 {
            return ((COLS * ROWS) as i32 + 1 - board.moves as i32) / 2;
        }

        // Upper bound: best possible score (win on next move is impossible, so at least 2 more moves)
        let max_possible = ((COLS * ROWS) as i32 - 1 - board.moves as i32) / 2;
        if max_possible <= 0 {
            return 0; // Can't win anymore, guaranteed draw at best
        }
        if beta > max_possible {
            beta = max_possible;
            if alpha >= beta {
                return beta;
            }
        }

        // Check if opponent can win next move — if so, we must block
        let opponent_win = board.opponent_winning_moves();
        if opponent_win != 0 {
            // Count opponent winning moves — if more than 1, we lose
            if (opponent_win & (opponent_win - 1)) != 0 {
                // Multiple threats, we can only block one
                return -(((COLS * ROWS) as i32 - board.moves as i32) / 2);
            }
            // Exactly one threat — we must play there (forced move)
            for col in 0..COLS {
                let h = board.height(col);
                if h < ROWS {
                    let bit = 1u64 << (col * 7 + h);
                    if opponent_win & bit != 0 {
                        board.play(col);
                        let score = -self.negamax(board, -beta, -alpha);
                        board.undo(col);
                        return score;
                    }
                }
            }
        }

        // Transposition table lookup
        let key = board.key();
        if let Some(stored) = self.table.get(key) {
            let val = stored as i32 - 18; // offset decoding
            if val >= beta {
                return val;
            }
            if val > alpha {
                alpha = val;
            }
        }

        let mut best = i32::MIN;

        for &col in &MOVE_ORDER {
            if board.height(col) >= ROWS {
                continue;
            }

            board.play(col);

            // Check if this move lets opponent win immediately — skip if so
            // (already handled by forced move logic above, but this is for non-forced cases)

            let score = -self.negamax(board, -beta, -alpha);
            board.undo(col);

            if score > best {
                best = score;
            }

            if score >= beta {
                self.table.put(key, (score + 18) as i8);
                return score;
            }

            if score > alpha {
                alpha = score;
            }
        }

        if best == i32::MIN {
            // No legal moves (shouldn't happen if is_full check is correct)
            return 0;
        }

        self.table.put(key, (alpha + 18) as i8);
        alpha
    }

    /// Returns a Vec of (col, score) pairs for all legal moves, sorted best to worst.
    pub fn rank_moves(&mut self, board: &mut Board) -> Vec<(u32, i32)> {
        // If game is already over, return empty
        if board.is_win(board.opponent_board()) || board.is_full() {
            return vec![];
        }

        let mut results = vec![];
        for &col in &MOVE_ORDER {
            if board.height(col) >= ROWS {
                continue;
            }
            board.play(col);
            let score = -self.solve(board);
            board.undo(col);
            results.push((col, score));
        }
        results.sort_by(|a, b| b.1.cmp(&a.1));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_from_moves(moves: &[u32]) -> Board {
        let mut b = Board::new();
        for &col in moves {
            b.play(col);
        }
        b
    }

    #[test]
    fn test_solve_immediate_win() {
        let mut solver = Solver::new();
        // P1: 0, P2: 1, P1: 0, P2: 1, P1: 0, P2: 1 → P1 has 3 in col 0, can win
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1]);
        let score = solver.solve(&mut board);
        assert!(score > 0, "P1 should be winning, got {}", score);
    }

    #[test]
    fn test_solve_opponent_wins_next() {
        let mut solver = Solver::new();
        // After these moves, P2 has 3 in col 1 and can win
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 6]);
        let score = solver.solve(&mut board);
        assert!(score > 0, "P2 (current) should be winning, got {}", score);
    }

    #[test]
    fn test_solve_near_full_board() {
        let mut solver = Solver::new();
        // Fill most of the board (36 moves, alternating col pairs)
        let moves = [
            0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5,
            1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2,
            5, 4, 5, 4, 5, 4,
        ];
        let mut board = Board::new();
        for &col in &moves {
            if board.is_game_over() { break; }
            board.play(col);
        }
        if !board.is_game_over() {
            let score = solver.solve(&mut board);
            assert!(score >= -MAX_SCORE && score <= MAX_SCORE);
        }
    }

    #[test]
    fn test_rank_moves_near_full() {
        let mut solver = Solver::new();
        // Use a near-full board where only col 6 has space and solutions are trivial
        let moves = [
            0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5,
            1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2,
            5, 4, 5, 4, 5, 4,
        ];
        let mut board = Board::new();
        for &col in &moves {
            if board.is_game_over() { break; }
            board.play(col);
        }
        if board.is_game_over() { return; }
        let ranked = solver.rank_moves(&mut board);
        assert!(!ranked.is_empty());
        // Verify sorted descending
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 >= ranked[i].1);
        }
    }

    #[test]
    fn test_transposition_table() {
        let mut tt = TranspositionTable::new(1024);
        assert!(tt.get(42).is_none());
        tt.put(42, 25); // score = 25 - 18 = 7
        assert_eq!(tt.get(42), Some(25));
        tt.reset();
        assert!(tt.get(42).is_none());
    }
}
