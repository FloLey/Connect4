"""
Generate teacher-model reasoning traces for Connect 4 SFT.

For each position in connect4_data.csv, call Gemini with the board +
oracle scores and ask it to produce 2-4 sentences of thinking that lead
to the best column. Output a JSONL file that prepare_sft_dataset can load
in place of the random FILLER_THOUGHTS, so SFT trains on board-specific
reasoning instead of generic filler.

Usage:
  python generate_thoughts.py \
      --csv connect4_data.csv \
      --out connect4_thoughts.jsonl \
      --n 5000 \
      --model gemini-2.5-pro \
      --concurrency 10

Cost estimate (Gemini 2.5 Pro at $1.25/M in, $10/M out):
  5000 positions ~= 2M input + 0.5M output = ~$2.50 + ~$5.00 = ~$7.50.
  Use gemini-2.5-flash for ~10-15x cheaper.

Reads GOOGLE_API_KEY from (in order):
  ./.env  ->  training/.env  ->  backend/.env  ->  ../.env  ->  os.environ

Output format (one JSON object per line):
  {"move_sequence": "0303", "best_col": 3, "thought": "Column 3 is ..."}

Resume-safe: rerunning reads existing output and skips already-processed
move_sequences, appending only the missing ones.
"""

import argparse
import concurrent.futures
import csv
import json
import os
import random
import sys
import time
from pathlib import Path


# =============================================================================
# ENV + CLIENT
# =============================================================================

def _load_env():
    """Walk candidate .env paths and populate os.environ with keys not yet set."""
    candidates = [Path(".env"), Path("training/.env"), Path("backend/.env"), Path("../.env")]
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)
        break


# =============================================================================
# BOARD RECONSTRUCTION (copied from connect4_train.py to keep this script
# importable on its own without pulling the full training deps)
# =============================================================================

ROWS = 6
COLS = 7


class ConnectFour:
    def __init__(self):
        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
        self.current_turn = 1

    def drop_piece(self, col):
        if col < 0 or col >= COLS or self.board[0][col] != 0:
            return False
        for r in range(ROWS - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = self.current_turn
                self.current_turn = 2 if self.current_turn == 1 else 1
                return True
        return False

    def get_valid_moves(self):
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def get_visual_board(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        header = " " + " ".join([str(i) for i in range(COLS)])
        rows_str = []
        for r in range(ROWS):
            row_cells = [symbols[self.board[r][c]] for c in range(COLS)]
            rows_str.append("|" + "|".join(row_cells) + "|")
        return header + "\n" + "\n".join(rows_str)


def reconstruct_board(move_sequence):
    game = ConnectFour()
    for ch in move_sequence:
        game.drop_piece(int(ch))
    return game


def current_player_of(move_sequence):
    num_moves = len(move_sequence)
    player_id = 1 if num_moves % 2 == 0 else 2
    symbol = "X" if player_id == 1 else "O"
    return player_id, symbol


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_data(csv_path):
    data = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "move_sequence": row["move_sequence"].strip(),
                "scores": [float(row[f"col{i}"]) for i in range(7)],
                "best_col": int(row["best_col"]),
            })
    return data


def load_existing_thoughts(out_path):
    """Return set of move_sequence strings already in the output file."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("move_sequence") and obj.get("thought"):
                    done.add(obj["move_sequence"])
            except json.JSONDecodeError:
                continue
    return done


# =============================================================================
# PROMPT
# =============================================================================

TEACHER_SYSTEM = """You are a Connect Four expert explaining move choices. Given the current board, the valid columns, the player to move, and the best column (based on perfect-play analysis), write 2-4 sentences of concise first-person thinking that leads to choosing that column.

Rules:
- Under 80 words. 2-4 sentences.
- Focus on concrete tactics: immediate threats, forced wins, blocks, central control, four-in-a-row lines, unused squares.
- Reference columns by number (0-6) when useful.
- Do NOT say "the best column is X", "therefore I choose X", or "the oracle says". Just reason naturally.
- First person, present tense, as if you are about to make the move.
- Output only the thinking. No preamble, no bullet points, no code blocks, no quotes."""


def build_teacher_user_prompt(entry):
    seq = entry["move_sequence"]
    game = reconstruct_board(seq)
    player_id, symbol = current_player_of(seq)
    valid = game.get_valid_moves()
    scores = entry["scores"]
    best = entry["best_col"]
    # Show oracle scores so the teacher understands *why* the best column is
    # best — improves reasoning quality — but we tell it in the system prompt
    # not to reveal the scores in its output.
    scores_str = ", ".join([f"col {i}: {scores[i]:+.2f}" for i in range(7)])
    return (
        f"Board (you are player {player_id} = {symbol}):\n{game.get_visual_board()}\n\n"
        f"Valid columns: {valid}\n"
        f"Per-column perfect-play scores (do NOT mention these verbatim): {scores_str}\n"
        f"Best column: {best}\n\n"
        f"Write the thinking that leads to column {best}."
    )


# =============================================================================
# GEMINI CALL
# =============================================================================

def _diagnose_empty(resp):
    """Return a short human-readable reason when resp.text is empty."""
    try:
        pf = getattr(resp, "prompt_feedback", None)
        if pf and getattr(pf, "block_reason", None):
            return f"prompt blocked: {pf.block_reason}"
        cands = getattr(resp, "candidates", None) or []
        if not cands:
            return "no candidates"
        c = cands[0]
        fr = getattr(c, "finish_reason", None)
        sr = getattr(c, "safety_ratings", None)
        bits = [f"finish_reason={fr}"]
        if sr:
            blocked = [str(s) for s in sr if getattr(s, "blocked", False)]
            if blocked:
                bits.append(f"safety_blocked={blocked}")
        # usage hints: did the model burn budget on thinking?
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            for attr in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count"):
                val = getattr(um, attr, None)
                if val is not None:
                    bits.append(f"{attr}={val}")
        return "; ".join(bits)
    except Exception as e:
        return f"<diagnose failed: {e}>"


def call_gemini(client, model_name, user_prompt, system_prompt, temperature=0.7, max_output_tokens=2048, thinking_budget=512, max_retries=4, verbose=False):
    """Call Gemini with retries + exponential backoff. Returns the text or raises."""
    import time
    from google.genai import types

    last_err = None
    for attempt in range(max_retries):
        try:
            cfg_kwargs = dict(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            # Gemini 2.5* models do internal "thinking" that eats into
            # max_output_tokens. Cap the thinking budget so the final
            # response has room. ThinkingConfig is ignored on models that
            # don't support it, but our Gemini 2.5 client supports it.
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            except AttributeError:
                pass
            resp = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            text = (resp.text or "").strip()
            # Strip any stray markdown fences the model might emit
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()
            if not text:
                diag = _diagnose_empty(resp)
                raise ValueError(f"empty response ({diag})")
            return text
        except Exception as e:
            last_err = e
            if verbose:
                print(f"    attempt {attempt+1}/{max_retries} failed: {e}", file=sys.stderr)
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Generate teacher-model reasoning traces for Connect 4 SFT.")
    ap.add_argument("--csv", default="connect4_data.csv", help="Input CSV with move_sequence + col0..col6 + best_col.")
    ap.add_argument("--out", default="connect4_thoughts.jsonl", help="Output JSONL. Appended to on resume.")
    ap.add_argument("--n", type=int, default=5000, help="Number of positions to cover (after sort_by_difficulty + shuffle).")
    ap.add_argument("--model", default="gemini-2.5-pro", help="Gemini model id. gemini-2.5-pro ($1.25/M in, $10/M out) or gemini-2.5-flash (cheaper) are the sensible options.")
    ap.add_argument("--concurrency", type=int, default=10, help="Parallel Gemini calls. Respect your rate limit.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens (Gemini 2.5* thinking counts toward this — keep generous).")
    ap.add_argument("--thinking-budget", type=int, default=512, help="Cap on Gemini 2.5* internal thinking tokens per call. Set 0 to disable thinking on supported models.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed — match connect4_train.py split_data.")
    ap.add_argument("--dry-run", action="store_true", help="Print the first teacher prompt and exit without API calls.")
    ap.add_argument("--verbose", action="store_true", help="Print each attempt + a preview of every successful trace as it lands.")
    args = ap.parse_args()

    _load_env()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: GOOGLE_API_KEY not found in env or .env files.", file=sys.stderr)
        print("Create a .env with GOOGLE_API_KEY=your-key in one of: ./, training/, backend/, ../", file=sys.stderr)
        sys.exit(1)

    # Load + sample positions. Must match connect4_train.split_data EXACTLY:
    #   shuffle(seed=42) -> eval=last_10k -> train=sort_by_difficulty(rest, +std desc)
    # Otherwise the move_sequences we label don't line up with the positions
    # SFT actually trains on, and prepare_sft_dataset ends up using almost
    # none of the traces.
    raw = load_csv_data(args.csv)
    print(f"Loaded {len(raw)} positions from {args.csv}")
    random.seed(args.seed)
    random.shuffle(raw)
    train_unsorted = raw[:-10_000]

    def _diff_score(e):
        """Std of the 7 column scores — matches connect4_train.difficulty_score."""
        s = e["scores"]
        m = sum(s) / len(s)
        return (sum((x - m) ** 2 for x in s) / len(s)) ** 0.5

    train = sorted(train_unsorted, key=_diff_score, reverse=True)   # easy (high std) first
    subset = train[: args.n]
    print(f"Targeting {len(subset)} training positions for teacher traces (sorted: easy → hard)")

    # Resume: drop already-done move_sequences
    done = load_existing_thoughts(args.out)
    if done:
        print(f"Resuming — {len(done)} positions already have traces in {args.out}")
    todo = [e for e in subset if e["move_sequence"] not in done]
    print(f"Positions remaining to generate: {len(todo)}")
    if not todo:
        print("Nothing to do. Exiting.")
        return

    # Dry run — show what one prompt looks like, skip API
    if args.dry_run:
        first = todo[0]
        print("\n--- SYSTEM PROMPT ---")
        print(TEACHER_SYSTEM)
        print("\n--- USER PROMPT (first position) ---")
        print(build_teacher_user_prompt(first))
        return

    # Import + client lazily so --dry-run works without the SDK installed
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Simple rate-limited thread pool. Writes results to disk immediately as
    # each one lands (flush + fsync) so progress survives Ctrl+C and anyone
    # tailing the file sees lines appear in real time.
    n_done = 0
    n_err = 0
    t_start = time.time()
    # Line-buffered (buffering=1) + explicit flush + fsync. Belt and suspenders.
    with open(args.out, "a", encoding="utf-8", buffering=1) as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(
                    call_gemini,
                    client,
                    args.model,
                    build_teacher_user_prompt(entry),
                    TEACHER_SYSTEM,
                    args.temperature,
                    args.max_tokens,
                    args.thinking_budget,
                    4,               # max_retries
                    args.verbose,
                ): entry
                for entry in todo
            }
            for fut in concurrent.futures.as_completed(futures):
                entry = futures[fut]
                seq = entry["move_sequence"]
                try:
                    thought = fut.result()
                except Exception as e:
                    n_err += 1
                    print(f"  ! [{n_done+n_err}/{len(todo)}] seq={seq!r}  {e}", file=sys.stderr, flush=True)
                    continue
                record = {
                    "move_sequence": seq,
                    "best_col": entry["best_col"],
                    "thought": thought,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                try:
                    os.fsync(f_out.fileno())
                except OSError:
                    pass  # fsync can fail on some filesystems; not fatal
                n_done += 1
                if args.verbose or n_done <= 3:
                    preview = thought.replace("\n", " ")
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    print(f"  ✓ [{n_done+n_err}/{len(todo)}] seq={seq!r} best={entry['best_col']}  {preview}", flush=True)
                elif n_done % 10 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - n_done) / rate if rate > 0 else 0.0
                    print(f"  [{n_done}/{len(todo)}] rate={rate:.2f}/s  eta={eta/60:.1f} min  errors={n_err}", flush=True)

    print(f"\nDone. Wrote {n_done} traces to {args.out} (errors: {n_err}).", flush=True)


if __name__ == "__main__":
    main()
