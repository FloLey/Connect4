# training/core/constants.py

# --- Board Dimensions ---
ROWS = 6
COLS = 7
# Height includes a sentinel row to prevent bit-shift overflows
HEIGHT = ROWS + 1 
MAX_MOVES = ROWS * COLS

# --- Scoring System ---
# Logic: Score = (MAX_SCORE - depth)
# Win in 1 move  = +42
# Loss in 1 move = -42
# Draw           = 0
MAX_SCORE = 43 
MIN_SCORE = -43

# Penalties
# Strictly worse than an immediate loss (-42)
# Used for RL feedback when AI attempts to play in a full column
INVALID_MOVE_SCORE = -100

# --- Optimization ---
# Search center columns first to maximize Alpha-Beta pruning efficiency
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]