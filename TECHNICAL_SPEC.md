# 🎯 Technical Specification: Connect Four LLM Arena

## 1. Project Overview
**Goal:** Build a full-stack web platform to benchmark Large Language Models (LLMs) on the game of Connect Four.
**Key Functionality:**
1.  **Live Arena:** Human vs. AI (Real-time).
2.  **Replay System:** "Time-travel" through past games to view moves and **LLM Reasoning**.
3.  **Automated Tournament:** Background matches between different LLMs (AI vs. AI) to calculate ELO ratings.

---

## 2. Technology Stack
*   **Backend:** Python 3.11+ (FastAPI).
*   **Real-Time:** WebSockets (Native FastAPI).
*   **Database:** PostgreSQL (SQLAlchemy + Alembic).
*   **AI Orchestration:** LangChain (w/ Pydantic for strict JSON outputs).
*   **Frontend:** React (Vite) + Tailwind CSS.
*   **Containerization:** Docker & Docker Compose.

---

## 3. Project Structure
We follow a **Monorepo** structure for ease of development.

```text
connect4-llm-arena/
├── docker-compose.yml           # Postgres & App orchestration
├── .env.example                 # API Keys (OpenAI/Anthropic) & DB URL
├── README.md
│
├── backend/                     # Python / FastAPI
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # Entry point (FastAPI app init)
│   │   │
│   │   ├── api/                 # Route Handlers
│   │   │   ├── routes.py        # REST endpoints (Game creation, Listing)
│   │   │   └── websocket.py     # WS Manager (Live Gameplay loop)
│   │   │
│   │   ├── core/                # Config
│   │   │   ├── config.py        # Environment variables
│   │   │   └── database.py      # SQLAlchemy Session
│   │   │
│   │   ├── engine/              # Pure Game Logic (Independent of API)
│   │   │   ├── game.py          # ConnectFour rules & board rendering
│   │   │   └── ai.py            # LangChain Agent & Prompts
│   │   │
│   │   ├── models/              # Database Models
│   │   │   └── game_model.py    # SQL Table Definitions
│   │   │
│   │   └── schemas/             # Pydantic Schemas
│   │       └── game_schema.py   # API Request/Response shapes
│   │
│   ├── migrations/              # Alembic (DB Migrations)
│   ├── requirements.txt
│   │
│   └── scripts/
│       └── tournament.py        # Background worker for AI vs AI
│
└── frontend/                    # React (Vite)
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── api/                 # Axios wrappers
        ├── components/          # Reusable UI (Board, Slider, Chat)
        ├── hooks/               # useGameSocket.js
        └── pages/               # Arena, Gallery, Replay
```

---

## 4. Database Schema
We use a **Hybrid Relational/Document** approach. High-level metadata is relational; specific move history is stored as an appended **JSONB** array.

**Table:** `games`

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `UUID` (PK) | Unique Game ID. |
| `created_at` | `DateTime` | Timestamp. |
| `player_1_model` | `String` | e.g., "Human", "gpt-4o". |
| `player_2_model` | `String` | e.g., "claude-3.5-sonnet". |
| `status` | `String` | `IN_PROGRESS`, `COMPLETED`, `DRAW`. |
| `winner` | `Integer` | `1`, `2`, or `NULL`. |
| `history` | `JSONB` | **The Source of Truth** for replays. |

### `history` JSON Structure
This array grows with every move.
```json
[
  {
    "move_num": 1,
    "player": 1,
    "column": 3,
    "reasoning": null,  // Human moves have null reasoning
    "timestamp": "2023-10-27T10:00:00Z"
  },
  {
    "move_num": 2,
    "player": 2,
    "column": 4,
    "reasoning": "Blocking vertical threat on col 3...", // AI move
    "timestamp": "2023-10-27T10:00:05Z"
  }
]
```

---

## 5. API & Communication Protocols

### 5.1 REST Endpoints
*   `POST /games`: Create a new game entry in DB. Returns `game_id`.
*   `GET /games`: List completed games (for Leaderboard/Gallery).
*   `GET /games/{id}`: Fetch static game data and full history (for Replay).

### 5.2 WebSocket Protocol (Live Play)
**URL:** `ws://api_host/games/{id}/ws`

**Workflow:**
1.  **Connect:** Client connects. Server sends current board state.
2.  **Human Move (Client -> Server):**
    ```json
    { "action": "MOVE", "column": 3 }
    ```
3.  **Update (Server -> Client):**
    ```json
    {
      "type": "UPDATE",
      "board_visual": "....... \n ...X...", 
      "current_turn": 2,
      "status": "IN_PROGRESS"
    }
    ```
4.  **AI Thinking (Server -> Client):**
    ```json
    { "type": "THINKING_START" }
    ```
    *(Server invokes LangChain. Wait 2-5s)*
    ```json
    { "type": "THINKING_END" }
    ```
5.  **AI Move Broadcast (Server -> Client):** Server sends "UPDATE" again with new piece.

---

## 6. Implementation Guidelines

### AI Context Management
*   **Input:** When prompting the LLM, send **only the current board state** (Visual ASCII + Textual Column Description).
*   **History:** Do **not** send the list of previous moves in the prompt context to save tokens, unless specific testing proves it necessary.

### Latency & UX
*   **Loading State:** LLM generation is slow (2-5s). The Frontend must handle `THINKING_START` events to disable the board and show a spinner.
*   **Timeouts:** If the LLM times out (>15s), the backend should retry once, then fallback to a random valid move to keep the game alive.

### Replay Logic
*   **Frontend-Driven:** The backend sends the full list of moves (e.g., `[3, 3, 4, 5]`). The Frontend Replay Component starts with an empty board and re-runs the logic locally to visualize the state at any given turn.

---

## 7. Development Phases

### Phase 1: Core Engine (Logic & CLI)
*   **Objective:** Validate game rules and LLM Prompts.
*   **Tasks:**
    *   Implement `ConnectFour` class (Grid, Valid Moves, Win Check).
    *   Implement `ConnectFourAI` class (LangChain, Pydantic JSON parser).
    *   Run `console_test.py` to verify Human vs GPT-4o.

### Phase 2: Backend & Database
*   **Objective:** Functional API and State Persistence.
*   **Tasks:**
    *   Setup PostgreSQL with Docker.
    *   Implement `games` table via SQLAlchemy.
    *   Build `POST /games` endpoint.
    *   Build WebSocket manager to handle the game loop (Receive Move -> Save -> Trigger AI -> Save -> Reply).

### Phase 3: Frontend (React)
*   **Objective:** Interactive UI.
*   **Tasks:**
    *   Build the **Board Component** (CSS Grid, click handlers).
    *   Connect `useGameSocket` hook to the Backend.
    *   Build **Replay View** (Slider to scrub through history).
    *   Display "AI Reasoning" in a sidebar text box.

### Phase 4: Tournament System
*   **Objective:** Automated Benchmarking.
*   **Tasks:**
    *   Create `scripts/tournament.py`.
    *   Logic: Select 2 Models -> Play Match -> Update DB.
    *   Run this script in a background loop or cron job.
    *   Calculate/Display Win Rates.