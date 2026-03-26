use crate::board::{Board, COLS, ROWS};

const MOVE_ORDER: [u32; 7] = [3, 2, 4, 1, 5, 0, 6];
const MAX_SCORE: i32 = ((COLS * ROWS - 1) / 2) as i32; // 20

/// Bound type stored alongside transposition table entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Bound {
    Exact,
    Lower, // true value >= stored value (beta cutoff)
    Upper, // true value <= stored value (no move improved alpha)
}

/// Fixed-size transposition table with bound type tracking.
pub struct TranspositionTable {
    keys: Vec<u64>,
    values: Vec<i8>,
    bounds: Vec<u8>, // 0 = empty, 1 = Exact, 2 = Lower, 3 = Upper
    size: usize,
}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        TranspositionTable {
            keys: vec![0; size],
            values: vec![0; size],
            bounds: vec![0; size],
            size,
        }
    }

    pub fn get(&self, key: u64) -> Option<(i8, Bound)> {
        let idx = (key as usize) % self.size;
        if self.keys[idx] == key && self.bounds[idx] != 0 {
            let bound = match self.bounds[idx] {
                1 => Bound::Exact,
                2 => Bound::Lower,
                3 => Bound::Upper,
                _ => return None,
            };
            Some((self.values[idx], bound))
        } else {
            None
        }
    }

    pub fn put(&mut self, key: u64, value: i8, bound: Bound) {
        let idx = (key as usize) % self.size;
        self.keys[idx] = key;
        self.values[idx] = value;
        self.bounds[idx] = match bound {
            Bound::Exact => 1,
            Bound::Lower => 2,
            Bound::Upper => 3,
        };
    }

    pub fn reset(&mut self) {
        self.keys.fill(0);
        self.values.fill(0);
        self.bounds.fill(0);
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
        // Check if the previous player already won.
        // They chose their winning move when board.moves was (board.moves - 1),
        // so their win score was (43 - (board.moves - 1)) / 2. We negate it.
        if board.is_win(board.opponent_board()) {
            return -(((COLS * ROWS) as i32 + 2 - board.moves as i32) / 2);
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
            return 0;
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

        // Transposition table lookup with proper bound handling
        let key = board.key();
        let original_alpha = alpha;
        if let Some((stored_raw, bound)) = self.table.get(key) {
            let val = stored_raw as i32 - 18;
            match bound {
                Bound::Exact => return val,
                Bound::Lower => {
                    if val >= beta {
                        return val;
                    }
                    if val > alpha {
                        alpha = val;
                    }
                }
                Bound::Upper => {
                    if val <= alpha {
                        return val;
                    }
                    if val < beta {
                        beta = val;
                    }
                }
            }
            if alpha >= beta {
                return val;
            }
        }

        let mut best = i32::MIN;

        for &col in &MOVE_ORDER {
            if board.height(col) >= ROWS {
                continue;
            }

            board.play(col);
            let score = -self.negamax(board, -beta, -alpha);
            board.undo(col);

            if score > best {
                best = score;
            }

            if score >= beta {
                self.table.put(key, (score + 18) as i8, Bound::Lower);
                return score;
            }

            if score > alpha {
                alpha = score;
            }
        }

        if best == i32::MIN {
            return 0;
        }

        // Determine bound type for storage
        let bound = if alpha > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.table.put(key, (alpha + 18) as i8, bound);
        alpha
    }

    /// Returns a Vec of (col, score) pairs for all legal moves, sorted best to worst.
    /// Also accumulates total node_count across all column solves.
    pub fn rank_moves(&mut self, board: &mut Board) -> Vec<(u32, i32)> {
        // If game is already over, return empty
        if board.is_win(board.opponent_board()) || board.is_full() {
            return vec![];
        }

        self.node_count = 0;
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
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1]);
        let score = solver.solve(&mut board);
        assert!(score > 0, "P1 should be winning, got {}", score);
        assert_eq!(score, 18, "Should be max win score (immediate win)");
    }

    #[test]
    fn test_solve_opponent_wins_next() {
        let mut solver = Solver::new();
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 6]);
        let score = solver.solve(&mut board);
        assert!(score > 0, "P2 (current) should be winning, got {}", score);
        assert_eq!(score, 18, "Should be max win score (can win immediately)");
    }

    #[test]
    fn test_solve_terminal_position() {
        let mut solver = Solver::new();
        // P1 wins with 4 in col 0
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 0]);
        // Game is over, P2 to move but P1 already won
        let score = solver.solve(&mut board);
        assert!(score < 0, "Current player (P2) should be losing, got {}", score);
    }

    #[test]
    fn test_solve_near_full_board() {
        let mut solver = Solver::new();
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
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 >= ranked[i].1);
        }
    }

    #[test]
    fn test_rank_moves_node_count_accumulates() {
        let mut solver = Solver::new();
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
        solver.rank_moves(&mut board);
        // Node count should be > 1 (accumulated across all column solves)
        assert!(solver.node_count > 1, "Node count should accumulate, got {}", solver.node_count);
    }

    #[test]
    fn test_self_consistency() {
        let mut solver = Solver::new();
        // Solve parent position
        let mut parent = board_from_moves(&[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]);
        let ranked = solver.rank_moves(&mut parent);
        if ranked.is_empty() { return; }
        let (best_col, parent_score) = ranked[0];

        // Solve child position (after playing best move)
        parent.play(best_col);
        let child_score = solver.solve(&mut parent);

        // Parent's best score should equal negation of child's score
        assert_eq!(parent_score, -child_score,
            "Self-consistency failed: parent best={} but -child={}",
            parent_score, -child_score);
    }

    #[test]
    fn test_transposition_table_bounds() {
        let mut tt = TranspositionTable::new(1024);
        assert!(tt.get(42).is_none());

        tt.put(42, 25, Bound::Exact);
        let (val, bound) = tt.get(42).unwrap();
        assert_eq!(val, 25);
        assert_eq!(bound, Bound::Exact);

        tt.put(42, 20, Bound::Lower);
        let (val, bound) = tt.get(42).unwrap();
        assert_eq!(val, 20);
        assert_eq!(bound, Bound::Lower);

        tt.put(42, 15, Bound::Upper);
        let (val, bound) = tt.get(42).unwrap();
        assert_eq!(val, 15);
        assert_eq!(bound, Bound::Upper);

        tt.reset();
        assert!(tt.get(42).is_none());
    }
}
