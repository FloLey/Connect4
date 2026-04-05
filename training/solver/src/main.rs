mod board;
mod book;
mod generator;
mod score;
mod solver;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rand::rngs::StdRng;
use rand::SeedableRng;

use board::{Board, COLS, ROWS};
use book::OpeningBook;
use generator::{build_book_layer, collect_boards_at_depth, generate_random, generate_systematic, set_thread_book, write_csv};
use score::INVALID_SCORE;
use solver::Solver;

#[derive(Parser)]
#[command(name = "connect4-solver")]
#[command(about = "Connect4 solver and training data generator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve a single position and print ranked moves
    Solve {
        /// Move sequence as column digits (0-indexed), e.g. "3344252"
        move_sequence: String,
        /// Opening book file to use for exact lookups
        #[arg(long)]
        book: Option<String>,
    },
    /// Generate training data
    Generate {
        /// Opening book file to speed up solving
        #[arg(long)]
        book: Option<String>,
        #[command(subcommand)]
        mode: GenerateMode,
    },
    /// Build an opening book by chaining from start_depth down to save_depth.
    /// Depth start_depth is solved with full alpha-beta, each lower depth is
    /// solved in ~1 ply using the layer above as an oracle.
    BuildBook {
        /// Deepest layer to compute first (full alpha-beta, hardest step)
        #[arg(long)]
        start_depth: u32,
        /// Build down to this depth; all layers save_depth..=start_depth are saved
        #[arg(long, default_value = "0")]
        save_depth: u32,
        /// Output file path
        #[arg(long)]
        output: String,
    },
    /// Run a quick benchmark
    Bench,
    /// Benchmark against Pons' test suite (pass a test file path)
    BenchPons {
        /// Path to Pons test file (format: "<move_sequence> <score>" per line)
        #[arg(long)]
        data: String,
    },
    /// Generate opening book using Pons' C++ solver as oracle
    GenerateBookPons {
        /// Path to compiled Pons solver binary (c4solver)
        #[arg(long)]
        solver: String,
        /// Path to Pons' .book file (7x6.book)
        #[arg(long, name = "pons-book")]
        pons_book: String,
        /// Maximum depth to include in the book
        #[arg(long, default_value = "12")]
        max_depth: u32,
        /// Output file path
        #[arg(long)]
        output: String,
    },
}

#[derive(Subcommand)]
enum GenerateMode {
    /// Generate all positions up to a given depth
    Systematic {
        #[arg(long, default_value = "10")]
        max_depth: u32,
        /// Minimum depth to start solving (skip expensive early positions)
        #[arg(long, default_value = "0")]
        min_depth: u32,
        #[arg(long, default_value = "output")]
        output: String,
    },
    /// Generate random mid-game positions
    Random {
        #[arg(long, default_value = "100000")]
        count: usize,
        #[arg(long, default_value = "output")]
        output: String,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Run both systematic and random generation (recommended)
    Full {
        #[arg(long, default_value = "output")]
        output: String,
        #[arg(long)]
        seed: Option<u64>,
        /// Minimum depth to start solving in systematic phase (default: 0 = solve all)
        #[arg(long, default_value = "0")]
        min_depth: u32,
        /// Maximum depth for systematic generation
        #[arg(long, default_value = "10")]
        max_depth: u32,
        /// Number of random positions to generate
        #[arg(long, default_value = "50000")]
        random_count: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Solve { move_sequence, book } => cmd_solve(&move_sequence, book.as_deref()),
        Commands::Generate { book, mode } => cmd_generate(mode, book.as_deref()),
        Commands::BuildBook { start_depth, save_depth, output } => cmd_build_book(start_depth, save_depth, &output),
        Commands::Bench => cmd_bench(),
        Commands::BenchPons { data } => cmd_bench_pons(&data),
        Commands::GenerateBookPons { solver, pons_book, max_depth, output } =>
            cmd_generate_book_pons(&solver, &pons_book, max_depth, &output),
    }
}

fn cmd_solve(move_sequence: &str, book_path: Option<&str>) {
    let mut board = Board::new();
    for ch in move_sequence.chars() {
        let col = ch.to_digit(10).expect("Invalid character in move sequence");
        assert!(col < COLS, "Column {} out of range", col);
        board.play(col);
    }

    println!(
        "Position: {} ({} moves played)",
        if move_sequence.is_empty() {
            "(empty)"
        } else {
            move_sequence
        },
        board.moves
    );

    let start = Instant::now();
    let mut solver = match book_path {
        Some(path) => {
            let b = OpeningBook::load(Path::new(path))
                .unwrap_or_else(|e| panic!("Failed to load book: {}", e));
            eprintln!("Loaded book: {} entries", b.len());
            Solver::with_book(Arc::new(b))
        }
        None => Solver::new(),
    };

    // Check if the game is already over
    if board.is_win(board.opponent_board()) {
        let elapsed = start.elapsed();
        let score = solver.solve(&mut board);
        println!("Game is already over. The player who just moved WON, meaning the current player LOST.");
        println!("Score: {} (current player's loss)", score);
        println!("Time: {:.2?}", elapsed);
        return;
    }
    if board.is_full() {
        println!("Game is already over. DRAW (board full).");
        return;
    }

    println!("Solving...\n");

    let ranked = solver.rank_moves(&mut board);
    let elapsed = start.elapsed();

    println!("Move ranking:");
    let mut best_printed = false;
    for &(col, raw_score) in &ranked {
        let norm = score::normalize(raw_score);
        let label = if !best_printed {
            best_printed = true;
            "  [BEST]"
        } else {
            ""
        };
        let outcome_hint = if raw_score > 0 {
            let moves_to_win = (43 - board.moves as i32) / 2 - raw_score + 1;
            format!("  win in ~{} moves", moves_to_win.max(1))
        } else if raw_score < 0 {
            String::from("  loss")
        } else {
            String::from("  draw")
        };
        println!(
            "  col {}  score {:+.3}  (raw {:+}){}{}",
            col, norm, raw_score, label, outcome_hint
        );
    }

    // Show illegal columns
    for col in 0..COLS {
        if ranked.iter().all(|(c, _)| *c != col) {
            println!("  col {}  ILLEGAL (full)     → {:.3}", col, INVALID_SCORE);
        }
    }

    let best_score = ranked.first().map(|(_, s)| *s).unwrap_or(0);
    let result = if best_score > 0 {
        "current player WINS with best play"
    } else if best_score < 0 {
        "current player LOSES with best play"
    } else {
        "DRAW with best play"
    };

    println!("\nResult: {}.", result);
    println!("Nodes explored: {}", format_number(solver.node_count));
    println!("Time: {:.2?}", elapsed);
}

fn cmd_generate(mode: GenerateMode, book_path: Option<&str>) {
    if let Some(path) = book_path {
        let book = OpeningBook::load(Path::new(path))
            .unwrap_or_else(|e| panic!("Failed to load book: {}", e));
        eprintln!("Loaded book: {} entries", book.len());
        set_thread_book(Some(Arc::new(book)));
    }

    match mode {
        GenerateMode::Systematic { max_depth, min_depth, output } => {
            let start = Instant::now();
            let (data, _seen) = generate_systematic(max_depth, min_depth);
            let path = Path::new(&output).join(format!("systematic_d{}.csv", max_depth));
            write_csv(&path, &data).expect("Failed to write CSV");
            println!(
                "Wrote {} positions to {} ({:.1?})",
                data.len(),
                path.display(),
                start.elapsed()
            );
        }
        GenerateMode::Random {
            count,
            output,
            seed,
        } => {
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let start = Instant::now();
            let batch_size = 100_000;
            let mut total = 0;
            let mut batch_idx = 0;
            let mut seen = std::collections::HashSet::new();

            while total < count {
                let this_batch = (count - total).min(batch_size);
                let data = generate_random(this_batch, &mut rng, &mut seen);
                let path =
                    Path::new(&output).join(format!("random_batch_{:03}.csv", batch_idx));
                write_csv(&path, &data).expect("Failed to write CSV");
                println!(
                    "Wrote {} positions to {}",
                    data.len(),
                    path.display()
                );
                total += data.len();
                batch_idx += 1;
            }

            println!(
                "Total: {} positions ({:.1?})",
                total,
                start.elapsed()
            );
        }
        GenerateMode::Full { output, seed, min_depth, max_depth, random_count } => {
            let start = Instant::now();

            // Systematic phase
            eprintln!("=== Phase 1: Systematic generation (depth <= {}, solving >= {}) ===", max_depth, min_depth);
            let (systematic, mut seen) = generate_systematic(max_depth, min_depth);
            let path = Path::new(&output).join(format!("systematic_d{}.csv", max_depth));
            write_csv(&path, &systematic).expect("Failed to write CSV");
            println!(
                "Wrote {} systematic positions to {}",
                systematic.len(),
                path.display()
            );

            // Random phase
            eprintln!("\n=== Phase 2: Random generation ({} positions) ===", random_count);
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let total_random = random_count;
            let batch_size = 100_000;
            let mut total = 0;
            let mut batch_idx = 0;

            while total < total_random {
                let this_batch = (total_random - total).min(batch_size);
                let data = generate_random(this_batch, &mut rng, &mut seen);
                let path =
                    Path::new(&output).join(format!("random_batch_{:03}.csv", batch_idx));
                write_csv(&path, &data).expect("Failed to write CSV");
                println!(
                    "Wrote {} positions to {}",
                    data.len(),
                    path.display()
                );
                total += data.len();
                batch_idx += 1;
            }

            println!(
                "\n=== Complete ===\nSystematic: {} positions\nRandom: {} positions\nTotal: {} positions\nTime: {:.1?}",
                systematic.len(),
                total,
                systematic.len() + total,
                start.elapsed()
            );
        }
    }
}

fn cmd_build_book(start_depth: u32, save_depth: u32, output_path: &str) {
    assert!(save_depth <= start_depth, "--save-depth must be <= --start-depth");
    let total_start = Instant::now();

    let mut merged_book = OpeningBook::new();
    let mut oracle: Option<Arc<OpeningBook>> = None;

    // Build from start_depth down to 0.
    // Depths > save_depth are oracle-only (in memory, not written to disk).
    // Depths <= save_depth are saved in the output book.
    for depth in (0..=start_depth).rev() {
        let step_start = Instant::now();
        let label = if depth > save_depth { " (oracle only, not saved)" } else { "" };
        eprintln!("\n=== Depth {}{} ===", depth, label);

        // Seed oracle and stream DFS-and-solve (no upfront board collection)
        set_thread_book(oracle.clone());
        let solved = build_book_layer(depth);
        set_thread_book(None);

        // Only persist to output book if within save range
        if depth <= save_depth {
            for &(key, score) in &solved {
                merged_book.insert(key, score);
            }
        }

        // Always update oracle so the next (shallower) layer can use it
        let mut new_oracle = OpeningBook::new();
        for &(key, score) in &solved {
            new_oracle.insert(key, score);
        }
        if let Some(ref prev) = oracle {
            new_oracle.merge_from(prev);
        }
        new_oracle.finalize();
        oracle = Some(Arc::new(new_oracle));

        eprintln!("Depth {} done in {:.1?}", depth, step_start.elapsed());
    }

    // Finalize and save the merged book (contains all depths save_depth..=start_depth)
    merged_book.finalize();
    merged_book.save(Path::new(output_path))
        .unwrap_or_else(|e| panic!("Failed to save book: {}", e));

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "\nBook saved: {} entries, {:.1} MB → {} ({:.1?})",
        merged_book.len(),
        file_size as f64 / 1_048_576.0,
        output_path,
        total_start.elapsed()
    );
}

fn cmd_bench() {
    use rand::Rng;

    println!("Running benchmark: solving 1000 random mid-game positions (depth 14-28)...");
    let mut solver = Solver::new();
    let mut rng = StdRng::seed_from_u64(42);
    let mut total_nodes = 0u64;
    let mut total_tt_hits = 0u64;
    let mut total_tt_useful = 0u64;
    let mut total_moves_sum = 0u64;
    let mut solved = 0u32;

    let start = Instant::now();

    while solved < 1000 {
        let depth: u32 = rng.gen_range(14..=28);
        let mut board = Board::new();
        let mut valid = true;

        for _ in 0..depth {
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

        solver.reset_diagnostics();
        solver.solve(&mut board);
        total_nodes += solver.node_count;
        total_tt_hits += solver.tt_hits;
        total_tt_useful += solver.tt_useful;
        total_moves_sum += solver.total_moves;
        solved += 1;

        if solved % 100 == 0 || solved == 1000 {
            eprint!("\rSolving... {} / 1000 positions  ", solved);
        }
    }
    eprintln!();

    let elapsed = start.elapsed();
    let positions_per_sec = 1000.0 / elapsed.as_secs_f64();

    println!("Solved: 1000 positions");
    println!("Time: {:.2?}", elapsed);
    println!("Positions/sec: {:.0}", positions_per_sec);
    println!("Total nodes: {}", format_number(total_nodes));
    println!("Avg nodes/position: {}", format_number(total_nodes / 1000));
    println!("TT hit rate: {:.1}%  ({} hits / {} nodes)",
        100.0 * total_tt_hits as f64 / total_nodes as f64,
        format_number(total_tt_hits),
        format_number(total_nodes));
    println!("TT useful rate: {:.1}%  ({} narrowed bounds / {} hits)",
        if total_tt_hits > 0 { 100.0 * total_tt_useful as f64 / total_tt_hits as f64 } else { 0.0 },
        format_number(total_tt_useful),
        format_number(total_tt_hits));
    println!("Avg branching factor: {:.2}  ({} moves / {} nodes)",
        total_moves_sum as f64 / total_nodes as f64,
        format_number(total_moves_sum),
        format_number(total_nodes));
}

fn cmd_bench_pons(data_path: &str) {
    use std::io::{BufRead, BufReader};
    use std::fs::File;

    let file = File::open(data_path).unwrap_or_else(|e| panic!("Cannot open {}: {}", data_path, e));
    let reader = BufReader::new(file);

    let mut solver = Solver::new();
    let mut total_nodes = 0u64;
    let mut wrong = 0u32;
    let mut solved = 0u32;
    let start = Instant::now();

    for line in reader.lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut parts = line.split_whitespace();
        let moves = parts.next().unwrap_or("");
        let expected: i32 = parts.next().unwrap_or("0").parse().unwrap_or(0);

        let mut board = Board::new();
        let mut valid = true;
        for ch in moves.chars() {
            let col = match ch.to_digit(10) {
                Some(c) if c >= 1 && c <= 7 => c - 1, // Pons uses 1-indexed columns
                _ => { valid = false; break; }
            };
            if board.height(col) >= ROWS || board.is_game_over() { valid = false; break; }
            board.play(col);
        }
        if !valid { continue; }

        solver.reset_diagnostics();
        let score = solver.solve(&mut board);
        total_nodes += solver.node_count;
        solved += 1;
        if score != expected { wrong += 1; }

        if solved % 100 == 0 {
            eprint!("\rTested {} positions, {} wrong  ", solved, wrong);
        }
    }
    eprintln!();

    let elapsed = start.elapsed();
    println!("Tested: {} positions", solved);
    println!("Wrong scores: {} ({:.1}%)", wrong, 100.0 * wrong as f64 / solved.max(1) as f64);
    println!("Time: {:.2?}", elapsed);
    println!("Positions/sec: {:.0}", solved as f64 / elapsed.as_secs_f64());
    println!("Avg nodes/position: {}", format_number(total_nodes / solved.max(1) as u64));
    println!("TT hit rate: {:.1}%", 100.0 * solver.tt_hits as f64 / total_nodes.max(1) as f64);
    println!("Avg branching: {:.2}", solver.total_moves as f64 / total_nodes.max(1) as f64);
}

fn cmd_generate_book_pons(pons_solver: &str, pons_book_path: &str, max_depth: u32, output_path: &str) {
    use std::process::Command;
    use std::io::{Write as _, BufWriter};

    let start = Instant::now();

    // Collect all positions at depth 0..=max_depth
    eprintln!("Collecting positions at depth 0-{}...", max_depth);
    let mut all_boards: Vec<Board> = Vec::new();
    for depth in 0..=max_depth {
        let boards = collect_boards_at_depth(depth);
        eprintln!("  depth {}: {} positions", depth, boards.len());
        all_boards.extend(boards);
    }
    eprintln!("Total: {} positions to solve.", all_boards.len());

    // Parallel: spawn N Pons solver processes, each handling a chunk
    let n_workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let chunk_size = (all_boards.len() + n_workers - 1) / n_workers;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let total = all_boards.len();

    eprintln!("Solving with {} parallel Pons workers...", n_workers);

    let mut book = OpeningBook::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = all_boards.chunks(chunk_size).enumerate().map(|(i, chunk)| {
            let done = &done;
            s.spawn(move || {
                // Write positions to temp file
                let temp_path = format!("/tmp/c4_chunk_{}.txt", i);
                {
                    let mut f = BufWriter::new(std::fs::File::create(&temp_path).unwrap());
                    for board in chunk {
                        let seq: String = board.move_sequence()
                            .chars().map(|c| char::from(c as u8 + 1)).collect();
                        writeln!(f, "{}", seq).unwrap();
                    }
                }

                // Run Pons solver on the temp file
                let output = Command::new(pons_solver)
                    .arg("-b").arg(pons_book_path)
                    .stdin(std::fs::File::open(&temp_path).unwrap())
                    .output()
                    .unwrap_or_else(|e| panic!("Worker {}: Pons solver failed: {}", i, e));

                std::fs::remove_file(&temp_path).ok();

                // Parse results
                let stdout = String::from_utf8(output.stdout).unwrap();
                let results: Vec<(u64, i32)> = chunk.iter()
                    .zip(stdout.lines())
                    .map(|(board, line)| {
                        let score: i32 = line.split_whitespace().last()
                            .unwrap_or("0").parse().unwrap_or(0);
                        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if n % 500_000 == 0 {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = n as f64 / elapsed;
                            eprint!("\r  {}/{} ({:.1}%) [{:.0}/s, ETA: {}]  ",
                                n, total, 100.0 * n as f64 / total as f64,
                                rate, generator::fmt_eta((total - n) as f64 / rate));
                        }
                        (board.key(), score)
                    }).collect();
                results
            })
        }).collect();

        for handle in handles {
            for (key, score) in handle.join().unwrap() {
                book.insert(key, score);
            }
        }
    });
    eprintln!("\rImported {} positions.                              ", done.load(std::sync::atomic::Ordering::Relaxed));

    book.finalize();
    book.save(Path::new(output_path))
        .unwrap_or_else(|e| panic!("Failed to save book: {}", e));

    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "Book saved: {} entries, {:.1} MB → {} ({:.1?})",
        book.len(), file_size as f64 / 1_048_576.0, output_path, start.elapsed()
    );
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}
