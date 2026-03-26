use crate::board::{bit_index, Board, COLS, ROWS};

const MOVE_ORDER: [u32; 7] = [3, 2, 4, 1, 5, 0, 6];

/// Bound type stored alongside transposition table entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Bound {
    Exact,
    Lower, // true value >= stored value (beta cutoff)
    Upper, // true value <= stored value (no move improved alpha)
}

/// Packed TT entry: value (6 bits) + bound (2 bits) + best_move (3 bits) = 11 bits → u16
/// Value is stored with +21 offset so range [-21,21] maps to [0,42] (fits in 6 bits).
/// Bound: 0=empty, 1=Exact, 2=Lower, 3=Upper.
/// Best move: column 0-6, or 7 = unknown.
const VAL_OFFSET: i32 = 21;

fn pack_entry(value: i32, bound: Bound, best_move: u32) -> u16 {
    let v = (value + VAL_OFFSET) as u16 & 0x3F; // 6 bits
    let b = match bound {
        Bound::Exact => 1u16,
        Bound::Lower => 2u16,
        Bound::Upper => 3u16,
    };
    let m = (best_move.min(7) as u16) & 0x07; // 3 bits
    v | (b << 6) | (m << 8)
}

fn unpack_entry(packed: u16) -> Option<(i32, Bound, u32)> {
    let b = (packed >> 6) & 0x03;
    if b == 0 {
        return None; // empty
    }
    let v = (packed & 0x3F) as i32 - VAL_OFFSET;
    let bound = match b {
        1 => Bound::Exact,
        2 => Bound::Lower,
        3 => Bound::Upper,
        _ => return None,
    };
    let m = ((packed >> 8) & 0x07) as u32;
    Some((v, bound, m))
}

/// Fixed-size transposition table with packed entries (key + value/bound/best_move).
pub struct TranspositionTable {
    keys: Vec<u64>,
    entries: Vec<u16>, // packed: value + bound + best_move
    size: usize,
}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        TranspositionTable {
            keys: vec![0; size],
            entries: vec![0; size],
            size,
        }
    }

    pub fn get(&self, key: u64) -> Option<(i32, Bound, u32)> {
        let idx = (key as usize) % self.size;
        if self.keys[idx] == key {
            unpack_entry(self.entries[idx])
        } else {
            None
        }
    }

    pub fn put(&mut self, key: u64, value: i32, bound: Bound, best_move: u32) {
        let idx = (key as usize) % self.size;
        self.keys[idx] = key;
        self.entries[idx] = pack_entry(value, bound, best_move);
    }

    pub fn reset(&mut self) {
        self.keys.fill(0);
        self.entries.fill(0);
    }
}

pub struct Solver {
    pub table: TranspositionTable,
    pub node_count: u64,
}

impl Solver {
    pub fn new() -> Self {
        Solver {
            table: TranspositionTable::new(24_000_001),
            node_count: 0,
        }
    }

    /// Returns the exact score using iterative deepening with null-window search.
    pub fn solve(&mut self, board: &mut Board) -> i32 {
        // Terminal: previous player already won
        if board.is_win(board.opponent_board()) {
            return -(((COLS * ROWS) as i32 + 2 - board.moves as i32) / 2);
        }

        if board.is_full() {
            return 0;
        }

        // Immediate win check
        if board.winning_moves() != 0 {
            return ((COLS * ROWS) as i32 + 1 - board.moves as i32) / 2;
        }

        // Initial bounds
        let mut min = -(((COLS * ROWS) as i32 - 2 - board.moves as i32) / 2);
        let mut max = ((COLS * ROWS) as i32 - 1 - board.moves as i32) / 2;

        // Narrow bounds from TT
        if let Some((val, bound, _)) = self.table.get(board.key()) {
            match bound {
                Bound::Exact => return val,
                Bound::Lower => {
                    if val > min {
                        min = val;
                    }
                }
                Bound::Upper => {
                    if val < max {
                        max = val;
                    }
                }
            }
        }

        // Iterative deepening with null-window search
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

        // Non-losing moves filter
        let non_losing = board.non_losing_moves();
        if non_losing == 0 {
            return -(((COLS * ROWS) as i32 - board.moves as i32) / 2);
        }

        // Lower bound pruning
        let min_possible = -(((COLS * ROWS) as i32 - 2 - board.moves as i32) / 2);
        if alpha < min_possible {
            alpha = min_possible;
            if alpha >= beta {
                return alpha;
            }
        }

        // TT lookup — also retrieve best move for ordering
        let key = board.key();
        let original_alpha = alpha;
        let mut tt_best_move: Option<u32> = None;

        if let Some((val, bound, best_move)) = self.table.get(key) {
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
            if best_move < COLS {
                tt_best_move = Some(best_move);
            }
        }

        // Anticipation: filter moves that allow opponent double threat
        let mut dominated = non_losing;
        for &col in &MOVE_ORDER {
            let h = board.height(col);
            if h >= ROWS {
                continue;
            }
            let bit = 1u64 << bit_index(col, h);
            if non_losing & bit == 0 {
                continue;
            }

            board.play(col);
            if board.can_create_double_threat() {
                dominated &= !bit;
            }
            board.undo(col);
        }

        if dominated == 0 {
            dominated = non_losing;
        }

        // Build move list with scores, TT best move gets highest priority
        let mut move_entries: [(u32, u32); 7] = [(0, 0); 7];
        let mut n_moves = 0;
        for &col in &MOVE_ORDER {
            let h = board.height(col);
            if h >= ROWS {
                continue;
            }
            let bit = 1u64 << bit_index(col, h);
            if dominated & bit == 0 {
                continue;
            }
            let mut score = board.move_score(col);
            // Boost TT best move to ensure it's explored first
            if tt_best_move == Some(col) {
                score += 1000;
            }
            move_entries[n_moves] = (col, score);
            n_moves += 1;
        }

        // Sort by score descending (insertion sort, small array)
        for i in 1..n_moves {
            let mut j = i;
            while j > 0 && move_entries[j].1 > move_entries[j - 1].1 {
                move_entries.swap(j, j - 1);
                j -= 1;
            }
        }

        let mut best = i32::MIN;
        let mut best_col = 7u32; // invalid sentinel

        for i in 0..n_moves {
            let col = move_entries[i].0;
            board.play(col);
            let score = -self.negamax(board, -beta, -alpha);
            board.undo(col);

            if score > best {
                best = score;
                best_col = col;
            }

            if score >= beta {
                self.table.put(key, score, Bound::Lower, col);
                return score;
            }

            if score > alpha {
                alpha = score;
            }
        }

        if best == i32::MIN {
            return 0;
        }

        let bound = if alpha > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.table.put(key, alpha, bound, best_col);
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
        let mut board = board_from_moves(&[0, 1, 0, 1, 0, 1, 0]);
        let score = solver.solve(&mut board);
        assert!(score < 0, "Current player (P2) should be losing, got {}", score);
        assert_eq!(score, -18, "Terminal loss should be -18");
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
            assert!(score >= -21 && score <= 21);
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
        assert!(solver.node_count > 1, "Node count should accumulate, got {}", solver.node_count);
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
            "Self-consistency failed: parent best={} but -child={}",
            parent_score, -child_score);
    }

    #[test]
    fn test_transposition_table() {
        let mut tt = TranspositionTable::new(1024);
        assert!(tt.get(42).is_none());

        tt.put(42, 7, Bound::Exact, 3);
        let (val, bound, best_move) = tt.get(42).unwrap();
        assert_eq!(val, 7);
        assert_eq!(bound, Bound::Exact);
        assert_eq!(best_move, 3);

        tt.put(42, -5, Bound::Lower, 1);
        let (val, bound, best_move) = tt.get(42).unwrap();
        assert_eq!(val, -5);
        assert_eq!(bound, Bound::Lower);
        assert_eq!(best_move, 1);

        tt.reset();
        assert!(tt.get(42).is_none());
    }
}
