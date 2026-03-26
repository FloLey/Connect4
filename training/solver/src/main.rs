mod board;
mod generator;
mod score;
mod solver;

use std::path::Path;
use std::time::Instant;

use clap::{Parser, Subcommand};
use rand::rngs::StdRng;
use rand::SeedableRng;

use board::{Board, COLS};
use generator::{generate_random, generate_systematic, write_csv};
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
    },
    /// Generate training data
    Generate {
        #[command(subcommand)]
        mode: GenerateMode,
    },
    /// Run a quick benchmark
    Bench,
}

#[derive(Subcommand)]
enum GenerateMode {
    /// Generate all positions up to a given depth
    Systematic {
        #[arg(long, default_value = "10")]
        max_depth: u32,
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
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Solve { move_sequence } => cmd_solve(&move_sequence),
        Commands::Generate { mode } => cmd_generate(mode),
        Commands::Bench => cmd_bench(),
    }
}

fn cmd_solve(move_sequence: &str) {
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
    let mut solver = Solver::new();

    // Check if the game is already over
    if board.is_win(board.opponent_board()) {
        let elapsed = start.elapsed();
        let score = solver.solve(&mut board);
        println!("Game is already over. The player who just moved WON.");
        println!("Score: {} (from current player's perspective)", score);
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

fn cmd_generate(mode: GenerateMode) {
    let mut solver = Solver::new();

    match mode {
        GenerateMode::Systematic { max_depth, output } => {
            let start = Instant::now();
            let data = generate_systematic(max_depth, &mut solver);
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

            while total < count {
                let this_batch = (count - total).min(batch_size);
                let data = generate_random(this_batch, &mut solver, &mut rng);
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
        GenerateMode::Full { output, seed } => {
            let start = Instant::now();

            // Systematic: depth <= 10
            eprintln!("=== Phase 1: Systematic generation (depth <= 10) ===");
            let systematic = generate_systematic(10, &mut solver);
            let path = Path::new(&output).join("systematic_d10.csv");
            write_csv(&path, &systematic).expect("Failed to write CSV");
            println!(
                "Wrote {} systematic positions to {}",
                systematic.len(),
                path.display()
            );

            // Random: 1M positions
            eprintln!("\n=== Phase 2: Random generation (1M positions) ===");
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let total_random = 1_000_000;
            let batch_size = 100_000;
            let mut total = 0;
            let mut batch_idx = 0;

            while total < total_random {
                let this_batch = (total_random - total).min(batch_size);
                let data = generate_random(this_batch, &mut solver, &mut rng);
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

fn cmd_bench() {
    use rand::Rng;

    println!("Running benchmark: solving 1000 random mid-game positions...");
    let mut solver = Solver::new();
    let mut rng = StdRng::seed_from_u64(12345);
    let mut total_nodes = 0u64;
    let mut solved = 0u32;

    let start = Instant::now();

    while solved < 1000 {
        let depth: u32 = rng.gen_range(8..=28);
        let mut board = Board::new();
        let mut valid = true;

        for _ in 0..depth {
            let legal: Vec<u32> = (0..COLS).filter(|&c| board.height(c) < 6).collect();
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

        solver.node_count = 0;
        solver.solve(&mut board);
        total_nodes += solver.node_count;
        solved += 1;
    }

    let elapsed = start.elapsed();
    let positions_per_sec = 1000.0 / elapsed.as_secs_f64();

    println!("Solved: 1000 positions");
    println!("Time: {:.2?}", elapsed);
    println!("Positions/sec: {:.0}", positions_per_sec);
    println!("Total nodes: {}", format_number(total_nodes));
    println!(
        "Avg nodes/position: {}",
        format_number(total_nodes / 1000)
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
