use crate::board::{bit_index, Board, COLS, ROWS};

const MOVE_ORDER: [u32; 7] = [3, 4, 2, 5, 1, 6, 0];

// Score bounds for 7x6 board
const MIN_SCORE: i32 = -(((COLS * ROWS) as i32) / 2) + 3; // -18
const MAX_SCORE: i32 = ((COLS * ROWS) as i32 + 1) / 2 - 3; // 18

/// Compact transposition table using Pons' encoding:
/// - Upper bound stored as: alpha - MIN_SCORE + 1 (range 1..37)
/// - Lower bound stored as: score + MAX_SCORE - 2*MIN_SCORE + 2 (range 38..74)
/// - 0 = empty
/// Uses u32 partial keys (truncated from u64) to save memory.
pub struct TranspositionTable {
    keys: Vec<u32>,
    values: Vec<u8>,
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

    fn index(&self, key: u64) -> usize {
        (key as usize) % self.size
    }

    pub fn get(&self, key: u64) -> u8 {
        let idx = self.index(key);
        if self.keys[idx] == key as u32 {
            self.values[idx]
        } else {
            0
        }
    }

    pub fn put(&mut self, key: u64, value: u8) {
        let idx = self.index(key);
        self.keys[idx] = key as u32;
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
            // ~8.4M entries * 5 bytes = ~42MB, fits near L3 cache
            table: TranspositionTable::new(8_388_593), // ~8.4M entries, ~42MB (L3 cache friendly)
            node_count: 0,
        }
    }

    /// Returns the exact score using iterative deepening with null-window search.
    pub fn solve(&mut self, board: &mut Board) -> i32 {
        if board.is_win(board.opponent_board()) {
            return -(((COLS * ROWS) as i32 + 2 - board.moves as i32) / 2);
        }
        if board.is_full() {
            return 0;
        }
        if board.winning_moves() != 0 {
            return ((COLS * ROWS) as i32 + 1 - board.moves as i32) / 2;
        }

        let mut min = -(((COLS * ROWS) as i32 - 2 - board.moves as i32) / 2);
        let mut max = ((COLS * ROWS) as i32 - 1 - board.moves as i32) / 2;

        // Narrow bounds from TT
        let val = self.table.get(board.key());
        if val != 0 {
            if val > (MAX_SCORE - MIN_SCORE + 1) as u8 {
                // Lower bound
                let lb = val as i32 + 2 * MIN_SCORE - MAX_SCORE - 2;
                if lb > min { min = lb; }
            } else {
                // Upper bound
                let ub = val as i32 + MIN_SCORE - 1;
                if ub < max { max = ub; }
            }
        }

        while min < max {
            let mut med = min + (max - min) / 2;
            if med <= 0 && min / 2 < med {
                med = min / 2;
            } else if med >= 0 && max / 2 > med {
                med = max / 2;
            }

            let r = self.negamax(board, med, med + 1);
            if r <= med {
                max = r;
            } else {
                min = r;
            }
        }
        min
    }

    fn negamax(&mut self, board: &mut Board, mut alpha: i32, mut beta: i32) -> i32 {
        self.node_count += 1;

        if board.is_full() {
            return 0;
        }

        if board.winning_moves() != 0 {
            return ((COLS * ROWS) as i32 + 1 - board.moves as i32) / 2;
        }

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

        let non_losing = board.non_losing_moves();
        if non_losing == 0 {
            return -(((COLS * ROWS) as i32 - board.moves as i32) / 2);
        }

        let min_possible = -(((COLS * ROWS) as i32 - 2 - board.moves as i32) / 2);
        if alpha < min_possible {
            alpha = min_possible;
            if alpha >= beta {
                return alpha;
            }
        }

        // TT lookup (Pons encoding)
        let key = board.key();
        let val = self.table.get(key);
        if val != 0 {
            if val > (MAX_SCORE - MIN_SCORE + 1) as u8 {
                // Lower bound
                let lb = val as i32 + 2 * MIN_SCORE - MAX_SCORE - 2;
                if lb >= beta { return lb; }
                if lb > alpha { alpha = lb; }
            } else {
                // Upper bound
                let ub = val as i32 + MIN_SCORE - 1;
                if ub <= alpha { return ub; }
                if ub < beta { beta = ub; }
            }
        }

        // Build move list sorted by move_score (Pons-style)
        let mut move_entries: [(u32, u32); 7] = [(0, 0); 7];
        let mut n_moves = 0;
        for &col in &MOVE_ORDER {
            let h = board.height(col);
            if h >= ROWS {
                continue;
            }
            let bit = 1u64 << bit_index(col, h);
            if non_losing & bit == 0 {
                continue;
            }
            let score = board.move_score(col);
            move_entries[n_moves] = (col, score);
            n_moves += 1;
        }

        // Insertion sort descending (like Pons' MoveSorter)
        for i in 1..n_moves {
            let mut j = i;
            while j > 0 && move_entries[j].1 > move_entries[j - 1].1 {
                move_entries.swap(j, j - 1);
                j -= 1;
            }
        }

        for i in 0..n_moves {
            let col = move_entries[i].0;
            board.play(col);
            let score = -self.negamax(board, -beta, -alpha);
            board.undo();

            if score >= beta {
                // Store lower bound
                self.table.put(
                    key,
                    (score + MAX_SCORE - 2 * MIN_SCORE + 2) as u8,
                );
                return score;
            }
            if score > alpha {
                alpha = score;
            }
        }

        // Store upper bound
        self.table.put(key, (alpha - MIN_SCORE + 1) as u8);
        alpha
    }

    /// Returns a Vec of (col, score) pairs for all legal moves, sorted best to worst.
    pub fn rank_moves(&mut self, board: &mut Board) -> Vec<(u32, i32)> {
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
            board.undo();
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
        assert_eq!(score, 18);
    }

    #[test]
    fn test_solve_opponent_wins_next() {
        let mut solver = Solver::new();
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 6]);
        let score = solver.solve(&mut board);
        assert!(score > 0);
        assert_eq!(score, 18);
    }

    #[test]
    fn test_solve_terminal_position() {
        let mut solver = Solver::new();
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 0]);
        let score = solver.solve(&mut board);
        assert_eq!(score, -18);
    }

    #[test]
    fn test_solve_near_full_board() {
        let mut solver = Solver::new();
        let moves = [
            0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5, 1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2, 5, 4, 5, 4, 5, 4,
        ];
        let mut board = Board::new();
        for &col in &moves {
            if board.is_game_over() { break; }
            board.play(col);
        }
        if !board.is_game_over() {
            let score = solver.solve(&mut board);
            assert!(score >= MIN_SCORE && score <= MAX_SCORE);
        }
    }

    #[test]
    fn test_rank_moves_sorted() {
        let mut solver = Solver::new();
        let moves = [
            0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5, 1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2, 5, 4, 5, 4, 5, 4,
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
    fn test_node_count_accumulates() {
        let mut solver = Solver::new();
        let moves = [
            0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5, 1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2, 5, 4, 5, 4, 5, 4,
        ];
        let mut board = Board::new();
        for &col in &moves {
            if board.is_game_over() { break; }
            board.play(col);
        }
        if board.is_game_over() { return; }
        solver.rank_moves(&mut board);
        assert!(solver.node_count > 1);
    }

    #[test]
    fn test_self_consistency() {
        let mut solver = Solver::new();
        let mut parent = board_from_moves(&[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]);
        let ranked = solver.rank_moves(&mut parent);
        if ranked.is_empty() { return; }
        let (best_col, parent_score) = ranked[0];

        parent.play(best_col);
        let child_score = solver.solve(&mut parent);
        assert_eq!(parent_score, -child_score,
            "parent best={} but -child={}", parent_score, -child_score);
    }

    #[test]
    fn test_tt_operations() {
        let mut tt = TranspositionTable::new(1024);
        assert_eq!(tt.get(42), 0);
        tt.put(42, 25);
        assert_eq!(tt.get(42), 25);
        tt.reset();
        assert_eq!(tt.get(42), 0);
    }
}
