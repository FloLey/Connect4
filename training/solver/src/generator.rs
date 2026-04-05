use std::collections::HashSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

use crate::board::{Board, COLS, ROWS};
use crate::book::OpeningBook;
use crate::score::{self, INVALID_SCORE};
use crate::solver::Solver;

thread_local! {
    // Lazy-initialized per-thread solver. Reset to None when the book/mode changes.
    static THREAD_SOLVER: RefCell<Option<Solver>> = const { RefCell::new(None) };
    // Book to use when initializing the thread-local solver.
    static THREAD_BOOK: RefCell<Option<Arc<OpeningBook>>> = const { RefCell::new(None) };
    // When true, use Solver::for_book() (larger TT) instead of Solver::new().
    static THREAD_BOOK_MODE: RefCell<bool> = const { RefCell::new(false) };
}

/// Seed a book into all rayon worker threads and reset their solvers.
pub fn set_thread_book(book: Option<Arc<OpeningBook>>) {
    rayon::broadcast(|_| {
        THREAD_BOOK.with(|b| *b.borrow_mut() = book.clone());
        THREAD_SOLVER.with(|s| *s.borrow_mut() = None);
    });
    THREAD_BOOK.with(|b| *b.borrow_mut() = book);
    THREAD_SOLVER.with(|s| *s.borrow_mut() = None);
}

/// Enable/disable book-build mode (uses Solver::for_book() with larger TT).
pub fn set_book_mode(enabled: bool) {
    rayon::broadcast(|_| {
        THREAD_BOOK_MODE.with(|m| *m.borrow_mut() = enabled);
        THREAD_SOLVER.with(|s| *s.borrow_mut() = None);
    });
    THREAD_BOOK_MODE.with(|m| *m.borrow_mut() = enabled);
    THREAD_SOLVER.with(|s| *s.borrow_mut() = None);
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
        with_thread_solver(|solver| {
            let mut b = board.clone();
            Self::from_board(&mut b, solver)
        })
    }
}

/// Run a closure with the thread-local solver, lazily initializing it from
/// THREAD_BOOK if not yet created.
fn with_thread_solver<F, R>(f: F) -> R
where
    F: FnOnce(&mut Solver) -> R,
{
    THREAD_SOLVER.with(|s| {
        let mut opt = s.borrow_mut();
        if opt.is_none() {
            let book = THREAD_BOOK.with(|b| b.borrow().clone());
            let book_mode = THREAD_BOOK_MODE.with(|m| *m.borrow());
            let mut solver = if book_mode { Solver::for_book() } else { Solver::new() };
            if let Some(arc) = book { solver.book = Some(arc); }
            *opt = Some(solver);
        }
        f(opt.as_mut().unwrap())
    })
}

fn solve_boards_with_progress(boards: &[Board]) -> Vec<DataPoint> {
    let done = AtomicUsize::new(0);
    let total = boards.len();
    let start = Instant::now();

    // One chunk per thread: each thread processes a contiguous slice of the
    // (DFS-ordered or prefix-sorted) boards array, so its TT accumulates
    // sub-position results that are reused by subsequent positions in the chunk.
    let n_threads = rayon::current_num_threads();
    let chunk_size = (total + n_threads - 1) / n_threads;

    let results: Vec<DataPoint> = boards
        .par_chunks(chunk_size.max(1))
        .flat_map_iter(|chunk| {
            chunk.iter().filter_map(|b| {
                let result = DataPoint::from_board_thread_local(b);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 1000 == 0 || n == total {
                    let elapsed = start.elapsed().as_secs_f64();
                    let rate = n as f64 / elapsed;
                    let eta = (total - n) as f64 / rate;
                    eprint!(
                        "\rSolving... {} / {} ({:.1}%) [depth ~{}] [{:.0} pos/sec, ETA: {}]  ",
                        n, total, 100.0 * n as f64 / total as f64, b.moves, rate, fmt_eta(eta)
                    );
                }
                result
            })
        })
        .collect();
    eprintln!();
    results
}

pub fn fmt_eta(secs: f64) -> String {
    let s = secs as u64;
    if s >= 60 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{}s", s)
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

    // Sort by opening prefix first, then deepest within each prefix group.
    // Positions sharing the same first 2 moves go to the same thread chunk,
    // maximising TT hit rate (sub-positions explored for one position are reused
    // by the next position in the same opening line).
    boards.sort_unstable_by_key(|b| (b.prefix_key(2), std::cmp::Reverse(b.moves)));

    // Phase 2: Solve in parallel — each thread has its own 160MB TT
    let results = solve_boards_with_progress(&boards);

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

        board.undo();
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

    while boards.len() < count {
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

        if boards.len() % 10000 == 0 || boards.len() == count {
            eprint!(
                "\rGenerating... {} / {} ({:.1}%)   ",
                boards.len(), count, 100.0 * boards.len() as f64 / count as f64
            );
        }
    }
    eprintln!();

    // Sort by opening prefix so each thread chunk stays in the same subtree
    boards.sort_unstable_by_key(|b| (b.prefix_key(2), std::cmp::Reverse(b.moves)));

    // Phase 2: Solve in parallel — each thread has its own 150MB TT
    solve_boards_with_progress(&boards)
}

/// Collect all unique positions at exactly `depth` moves played.
pub fn collect_boards_at_depth(depth: u32) -> Vec<Board> {
    let mut seen = HashSet::new();
    let mut board = Board::new();
    let mut boards = Vec::new();
    collect_exact_depth(&mut board, depth, &mut seen, &mut boards);
    boards
}

fn collect_exact_depth(
    board: &mut Board,
    target: u32,
    seen: &mut HashSet<u64>,
    boards: &mut Vec<Board>,
) {
    if board.moves > target {
        return;
    }
    let key = board.key();
    if !seen.insert(key) {
        return;
    }
    if board.moves == target {
        boards.push(board.clone());
        if boards.len() % 100_000 == 0 {
            eprintln!("  ... {} positions collected", boards.len());
        }
        return;
    }
    for col in 0..COLS {
        if board.height(col) >= ROWS {
            continue;
        }
        board.play(col);
        if !board.is_game_over() {
            collect_exact_depth(board, target, seen, boards);
        }
        board.undo();
    }
}

/// Build a book layer by streaming DFS: no upfront board collection, solve on-the-fly.
/// Split points at depth min(4, target_depth) are collected (tiny), then each thread
/// DFS-es its sub-trees and solves positions at target_depth sequentially.
/// The thread-local TT is preserved across sub-trees within the same chunk for
/// maximum cache locality. Uses Solver::for_book() (larger TT) automatically.
pub fn build_book_layer(target_depth: u32) -> Vec<(u64, i32)> {
    set_book_mode(true);

    let split_depth = 4.min(target_depth);
    eprintln!("Collecting split points at depth {}...", split_depth);
    let mut split_boards = collect_boards_at_depth(split_depth);
    eprintln!("Split into {} sub-trees.", split_boards.len());
    split_boards.sort_unstable_by_key(|b| b.prefix_key(2));

    use std::sync::atomic::{AtomicBool, AtomicU64};
    let solved       = Arc::new(AtomicUsize::new(0));
    let subtrees_done = Arc::new(AtomicUsize::new(0));
    // Benchmarking counters
    let total_solver_ns  = Arc::new(AtomicU64::new(0)); // ns spent inside solver.solve()
    let total_solver_nodes = Arc::new(AtomicU64::new(0)); // nodes explored by solver
    let total_dfs_ns     = Arc::new(AtomicU64::new(0)); // ns spent in DFS traversal
    let dfs_nodes = AtomicUsize::new(0);
    let total_subtrees = split_boards.len();
    let start = Instant::now();
    let n_threads = rayon::current_num_threads();
    let chunk_size = ((split_boards.len() + n_threads - 1) / n_threads).max(1);
    let stop_progress = Arc::new(AtomicBool::new(false));

    // Background thread: prints every 5s with timing breakdown
    {
        let solved          = Arc::clone(&solved);
        let subtrees_done   = Arc::clone(&subtrees_done);
        let solver_ns       = Arc::clone(&total_solver_ns);
        let solver_nodes    = Arc::clone(&total_solver_nodes);
        let dfs_ns          = Arc::clone(&total_dfs_ns);
        let stop            = Arc::clone(&stop_progress);
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_secs(5));
                if stop.load(Ordering::Relaxed) { break; }
                let n      = solved.load(Ordering::Relaxed).max(1);
                let st     = subtrees_done.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64();
                let rate   = n as f64 / elapsed.max(0.001);
                let eta    = if st > 0 {
                    fmt_eta((total_subtrees - st) as f64 * elapsed / st as f64)
                } else { "estimating...".to_string() };

                let s_ns      = solver_ns.load(Ordering::Relaxed) as f64;
                let d_ns      = dfs_ns.load(Ordering::Relaxed) as f64;
                // wall_ns = elapsed × n_threads (total thread-seconds)
                let wall_ns   = elapsed * 24.0 * 1e9;
                let avg_nodes = solver_nodes.load(Ordering::Relaxed) as f64 / n as f64;
                let avg_ms    = s_ns / n as f64 / 1_000_000.0;

                eprintln!(
                    "  {:.0}s | {n} solved [{rate:.0}/s] | {st}/{total_subtrees} trees | ETA: {eta}\n  \
                     solver: {:.1}% wall | {avg_nodes:.0} nodes/pos | {avg_ms:.1}ms/pos | \
                     HashSet: {:.1}% wall",
                    elapsed,
                    s_ns / wall_ns * 100.0,
                    d_ns / wall_ns * 100.0,
                );
            }
        });
    }

    eprintln!("Dispatching {} chunks across {} threads...",
        (split_boards.len() + chunk_size - 1) / chunk_size, n_threads);

    let results: Vec<(u64, i32)> = split_boards
        .par_chunks(chunk_size)
        .flat_map_iter(|chunk| {
            with_thread_solver(|solver| {
                let mut all_results = Vec::new();
                let mut seen = HashSet::new();
                for split in chunk {
                    seen.clear();
                    let mut board = split.clone();
                    dfs_solve_at_depth(
                        &mut board, target_depth, solver,
                        &mut seen, &mut all_results,
                        &*solved, &dfs_nodes, &start,
                        &*total_solver_ns, &*total_solver_nodes, &*total_dfs_ns,
                    );
                    subtrees_done.fetch_add(1, Ordering::Relaxed);
                }
                all_results
            })
        })
        .collect();

    stop_progress.store(true, Ordering::Relaxed);
    set_book_mode(false);
    eprintln!("\rSolving complete: {} positions in {:.1?}                    ",
        solved.load(Ordering::Relaxed), start.elapsed());
    results
}

fn dfs_solve_at_depth(
    board: &mut Board,
    target: u32,
    solver: &mut Solver,
    seen: &mut HashSet<u64>,
    results: &mut Vec<(u64, i32)>,
    solved: &AtomicUsize,
    dfs_nodes: &AtomicUsize,
    start: &Instant,
    solver_ns: &AtomicU64,
    solver_nodes: &AtomicU64,
    dfs_overhead_ns: &AtomicU64,
) {
    if board.moves > target { return; }
    let key = board.key();

    let t_dfs = Instant::now();
    let is_new = seen.insert(key);
    dfs_overhead_ns.fetch_add(t_dfs.elapsed().as_nanos() as u64, Ordering::Relaxed);

    if !is_new { return; }
    dfs_nodes.fetch_add(1, Ordering::Relaxed);

    if board.moves == target {
        solver.node_count = 0;
        let t_solve = Instant::now();
        let score = solver.solve(board);
        let ns = t_solve.elapsed().as_nanos() as u64;
        solver_ns.fetch_add(ns, Ordering::Relaxed);
        solver_nodes.fetch_add(solver.node_count, Ordering::Relaxed);
        results.push((key, score));
        solved.fetch_add(1, Ordering::Relaxed);
        return;
    }

    for col in 0..COLS {
        if board.height(col) >= ROWS { continue; }
        board.play(col);
        if !board.is_game_over() {
            dfs_solve_at_depth(board, target, solver, seen, results,
                solved, dfs_nodes, start, solver_ns, solver_nodes, dfs_overhead_ns);
        }
        board.undo();
    }
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
