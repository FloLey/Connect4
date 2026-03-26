use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

use crate::board::{Board, COLS, ROWS};
use crate::score::{self, INVALID_SCORE};
use crate::solver::Solver;

thread_local! {
    static THREAD_SOLVER: RefCell<Solver> = RefCell::new(Solver::new());
}

pub struct DataPoint {
    pub move_sequence: String,
    pub scores: [f32; 7],
    pub best_col: u32,
}

impl DataPoint {
    pub fn from_board(board: &mut Board, solver: &mut Solver) -> Option<Self> {
        let ranked = solver.rank_moves(board);
        if ranked.is_empty() {
            return None; // Game already over
        }

        let mut scores = [INVALID_SCORE; 7];
        for col in 0..COLS {
            scores[col as usize] = score::score_for_col(col, &ranked);
        }

        // Best col: highest score, tie-broken by lowest column index
        let best_col = (0..COLS)
            .filter(|&c| scores[c as usize] > INVALID_SCORE)
            .max_by(|&a, &b| {
                scores[a as usize]
                    .partial_cmp(&scores[b as usize])
                    .unwrap()
                    .then(b.cmp(&a)) // tie-break: prefer lower column index
            })
            .unwrap_or(0);

        Some(DataPoint {
            move_sequence: board.move_sequence(),
            scores,
            best_col,
        })
    }

    fn from_board_thread_local(board: &Board) -> Option<Self> {
        THREAD_SOLVER.with(|s| {
            let mut solver = s.borrow_mut();
            let mut b = board.clone();
            Self::from_board(&mut b, &mut solver)
        })
    }
}

/// Write a batch of DataPoints to a CSV file.
pub fn write_csv(path: &Path, data: &[DataPoint]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(
        w,
        "move_sequence,col0,col1,col2,col3,col4,col5,col6,best_col"
    )?;
    for dp in data {
        writeln!(
            w,
            "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
            dp.move_sequence,
            dp.scores[0],
            dp.scores[1],
            dp.scores[2],
            dp.scores[3],
            dp.scores[4],
            dp.scores[5],
            dp.scores[6],
            dp.best_col,
        )?;
    }
    w.flush()?;
    Ok(())
}

/// Generate all positions up to `max_depth` moves by DFS.
/// Only solves positions at depth >= `min_solve_depth` (use 0 to solve all).
/// Returns (data_points, seen_keys) so the seen set can be reused.
pub fn generate_systematic(
    max_depth: u32,
    min_solve_depth: u32,
) -> (Vec<DataPoint>, HashSet<u64>) {
    let mut seen = HashSet::new();
    let mut board = Board::new();
    let mut boards = Vec::new();

    eprintln!(
        "Generating systematic positions up to depth {} (solving from depth {})...",
        max_depth, min_solve_depth
    );

    // Phase 1: Collect all positions via DFS (sequential, cheap)
    systematic_collect(&mut board, max_depth, min_solve_depth, &mut seen, &mut boards);

    eprintln!(
        "Collected {} positions to solve ({} total enumerated)",
        boards.len(),
        seen.len()
    );

    // Phase 2: Solve in parallel using rayon
    let results: Vec<DataPoint> = boards
        .par_iter()
        .filter_map(|b| DataPoint::from_board_thread_local(b))
        .collect();

    eprintln!(
        "Systematic generation complete: {} positions solved",
        results.len()
    );
    (results, seen)
}

fn systematic_collect(
    board: &mut Board,
    max_depth: u32,
    min_solve_depth: u32,
    seen: &mut HashSet<u64>,
    boards: &mut Vec<Board>,
) {
    let key = board.key();
    if !seen.insert(key) {
        return; // Already visited this position
    }

    // Only collect for solving if at sufficient depth
    if board.moves >= min_solve_depth {
        boards.push(board.clone());

        if boards.len() % 10000 == 0 {
            eprintln!("  ... {} positions collected", boards.len());
        }
    }

    // Recurse if not at max depth
    if board.moves >= max_depth {
        return;
    }

    for col in 0..COLS {
        if board.height(col) >= ROWS {
            continue;
        }

        board.play(col);

        // Only recurse if game isn't over
        if !board.is_game_over() {
            systematic_collect(board, max_depth, min_solve_depth, seen, boards);
        }

        board.undo(col);
    }
}

/// Generate `count` random positions by playing random moves to a random depth.
/// Uses shared `seen` HashSet for deduplication across calls.
/// Solves positions in parallel using rayon.
pub fn generate_random(
    count: usize,
    rng: &mut StdRng,
    seen: &mut HashSet<u64>,
) -> Vec<DataPoint> {
    eprintln!("Generating {} random positions...", count);

    // Phase 1: Generate board positions (sequential, cheap)
    let mut boards: Vec<Board> = Vec::with_capacity(count);
    let mut attempts = 0u64;

    while boards.len() < count {
        attempts += 1;

        // Random depth between 8 and 36
        let target_depth: u32 = rng.gen_range(8..=36);
        let mut board = Board::new();

        let mut valid = true;
        for _ in 0..target_depth {
            let legal: Vec<u32> = (0..COLS).filter(|&c| board.height(c) < ROWS).collect();
            if legal.is_empty() {
                valid = false;
                break;
            }

            let col = legal[rng.gen_range(0..legal.len())];
            board.play(col);

            if board.is_game_over() {
                valid = false;
                break;
            }
        }

        if !valid {
            continue;
        }

        let key = board.key();
        if !seen.insert(key) {
            continue; // Duplicate position
        }

        boards.push(board);

        if boards.len() % 10000 == 0 {
            eprintln!(
                "  ... {}/{} positions generated ({} attempts)",
                boards.len(),
                count,
                attempts
            );
        }
    }

    eprintln!(
        "Generated {} positions in {} attempts, solving in parallel...",
        boards.len(),
        attempts
    );

    // Phase 2: Solve in parallel using rayon
    let results: Vec<DataPoint> = boards
        .par_iter()
        .filter_map(|b| DataPoint::from_board_thread_local(b))
        .collect();

    eprintln!(
        "Random generation complete: {} positions solved",
        results.len()
    );
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a near-full board (36 moves) with no wins — alternating columns.
    fn make_near_full_board() -> Board {
        let mut b = Board::new();
        let moves = [
            0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5,
            1, 0, 1, 0, 1, 0,
            3, 2, 3, 2, 3, 2,
            5, 4, 5, 4, 5, 4,
        ];
        for &col in &moves {
            if b.height(col) < ROWS && !b.is_game_over() {
                b.play(col);
            }
        }
        b
    }

    #[test]
    fn test_datapoint_from_near_full() {
        let mut solver = Solver::new();
        let mut board = make_near_full_board();
        assert!(!board.is_game_over(), "Board should not be game over (moves={})", board.moves);
        let dp = DataPoint::from_board(&mut board, &mut solver).unwrap();
        assert_eq!(dp.scores.len(), 7);
        assert!(dp.scores.iter().any(|&s| s > INVALID_SCORE));
    }

    #[test]
    fn test_csv_output() {
        let dp = DataPoint {
            move_sequence: "test".to_string(),
            scores: [0.5, 0.3, -0.2, 0.9, -1.0, 0.1, -0.5],
            best_col: 3,
        };
        let path = Path::new("/tmp/test_solver_output.csv");
        write_csv(path, &[dp]).unwrap();
        let content = fs::read_to_string(path).unwrap();
        assert!(content.starts_with("move_sequence,col0,"));
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 2);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_csv_format() {
        let dp = DataPoint {
            move_sequence: "334".to_string(),
            scores: [-0.245, 0.243, 0.905, 0.982, 0.905, 0.243, -0.245],
            best_col: 3,
        };
        let path = Path::new("/tmp/test_csv_format.csv");
        write_csv(path, &[dp]).unwrap();

        let content = fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines[0], "move_sequence,col0,col1,col2,col3,col4,col5,col6,best_col");
        assert!(lines[1].starts_with("334,"));
        assert!(lines[1].ends_with(",3"));
        let _ = fs::remove_file(path);
    }
}
