# 🎯 Technical Specification: Connect Four LLM Arena (v2.1)

## 1. Project Overview
**Goal:** Build a full-stack competitive platform to benchmark Large Language Models (LLMs) on the game of Connect Four.
**Key Features:**
- **Live Arena:** Human vs. AI (Real-time) or Spectate AI vs. AI.
- **Unified Replay System:** Watch past games with a timeline slider and view LLM Reasoning for every move.
- **Automated Tournament:** A background system runs continuous matches between different models to calculate rankings.
- **Analytics Dashboard:** Live Leaderboards including ELO scores, Win Rates, and Token Usage/Cost tracking, plus ELO history graphs.

---

## 2. Technology Stack
- **Backend:** Python 3.11+ (FastAPI).
- **Real-Time:** WebSockets (Native FastAPI).
- **Database:** PostgreSQL (using SQLAlchemy + Alembic).
- **AI Orchestration:** LangChain (using with_structured_output for JSON).
- **Frontend:** React (Vite) + Tailwind CSS + Recharts (for analytics).
- **Infrastructure:** Docker & Docker Compose.

---

## 3. Project Structure
Monorepo Approach:

```text
connect4-llm-arena/
├── docker-compose.yml           # Database & App orchestration
├── .env                         # API Keys (OpenAI/Anthropic/Google) & DB URL
├── README.md
│
├── backend/                     # Python / FastAPI
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # App Entry Point
│   │   │
│   │   ├── api/                 # Route Handlers
│   │   │   ├── routes.py        # Game Management (CRUD)
│   │   │   ├── stats.py         # Leaderboard & Analytics Endpoints
│   │   │   └── websocket_manager.py # WebSocket Logic (Singleton)
│   │   │
│   │   ├── core/                # Configuration
│   │   │   └── database.py      # Async DB Session
│   │   │
│   │   ├── engine/              # Core Logic
│   │   │   ├── game.py          # Connect Four Rules Engine
│   │   │   ├── ai.py            # LangChain Agent Factory & Prompts
│   │   │   └── elo.py           # Ranking Math & Updates
│   │   │
│   │   ├── models/              # SQLAlchemy Models
│   │   │   ├── game_model.py    # Game Tables
│   │   │   └── elo_model.py     # Ranking Tables
│   │   │
│   │   └── schemas/             # Pydantic Schemas
│   │       └── game_schema.py   # API Types
│   │
│   ├── scripts/
│   │   ├── init_db.py           # DB Initialization Script
│   │   └── tournament.py        # Background Worker (AI vs AI)
│   └── requirements.txt
│
└── frontend/                    # React (Vite)
    ├── src/
        ├── components/          # Reusable UI (Board, Chat, Graph)
        ├── hooks/               # useGameSocket.js
        └── pages/               # Arena, Tournament, Leaderboard
```

---

## 4. Database Schema

### 4.1 Table: `games`
Stores match metadata and the full move log.

| Column | Type | Notes |
| :--- | :--- | :--- |
| `id` | Integer (PK) | Unique ID. |
| `created_at` | DateTime | |
| `player_1_type` | String | "human" or model name (e.g., "gpt-4o"). |
| `player_2_type` | String | "human" or model name. |
| `winner` | Integer | 1, 2, or NULL. |
| `status` | String | IN_PROGRESS, COMPLETED, DRAW. |
| `history` | JSONB | Critical: Stores the full game log. |

**Structure of history JSONB array:**
```json
[
  {
    "move_num": 1,
    "player": 1,
    "column": 3,
    "reasoning": "Starting center...",
    "input_tokens": 450,
    "output_tokens": 40
  }
]
```

### 4.2 Table: `elo_ratings`
Stores current standings.

| Column | Type | Notes |
| :--- | :--- | :--- |
| `model_name` | String (PK) | e.g., "claude-3-opus". |
| `rating` | Float | Standard ELO (starts at 1200.0). |
| `matches_played` | Integer | |
| `wins`/`losses`/`draws` | Integer | |
| `total_input_tokens` | BigInteger | Lifetime input token count. |
| `total_output_tokens` | BigInteger | Lifetime output token count. |

### 4.3 Table: `elo_history`
Stores time-series data for graphing.

| Column | Type | Notes |
| :--- | :--- | :--- |
| `id` | Integer (PK) | |
| `model_name` | String | |
| `rating` | Float | Rating snapshot after a match. |
| `timestamp` | DateTime | |
| `match_id` | Integer | Link to the game that caused the update. |

---

## 5. API & Communication Protocols

### 5.1 REST Endpoints
- `POST /games`: Start a new game.
- `GET /games/{id}`: Fetch game state and full history.
- `GET /stats/leaderboard`: Returns models sorted by Rank (including Token Usage columns).
- `GET /stats/history`: Returns historical ELO data points for graphing.
- `GET /stats/active-games`: Returns list of currently running AI vs AI games.

### 5.2 WebSocket Protocol (Live Play)
**URL:** `ws://api_host/games/{id}/ws`

**Architecture:** Singleton Connection Manager.
- The Backend maintains one Game Engine instance per Game ID in memory.
- Multiple clients (players + spectators) connecting to the same ID share this state.

**Events:**
- **Client -> Server:** `{ "action": "MOVE", "column": 3 }`
- **Server -> Client:**
  - `UPDATE`: Contains Board (ASCII), Turn, Status, and Last Move (w/ Reasoning).
  - `THINKING_START`: Signal to show spinner/loading state.
  - `THINKING_END`: Signal to unlock UI.

---

## 6. Implementation Logic

### 6.1 AI Agent Logic
- **Prompting:** The AI receives the current board state only (Visual Grid + Text Description) to save context window.
- **Output:** Strict JSON via Pydantic (reasoning, column).
- **Token Tracking:** We must use `include_raw=True` in LangChain to capture `usage_metadata` (Input/Output tokens) and save this to the DB history.

### 6.2 ELO Calculation
- **Trigger:** Runs only when a game finishes AND both players are AI models.
- **Algorithm:** Standard ELO (K=32).
- **Updates:**
  - Update `elo_ratings` (Score + Win/Loss counters + Token Sums).
  - Insert row into `elo_history`.

### 6.3 Tournament Worker (`scripts/tournament.py`)
- **Type:** Background Python Process (AsyncIO).
- **Loop:**
  1. Select 2 random models from config.
  2. Play Game (Headless - no WebSockets, direct DB calls).
  3. Save result.
  4. Update ELO.
  5. Repeat.

---

## 7. Frontend Views (React)

**Arena (Unified View):**
- **Board:** Interactive 7x6 Grid.
- **Reasoning Panel:** Chat-like history of "Model Thoughts".
- **Controls:** If Live -> "Drop Piece" buttons. If Replay -> Timeline Slider.

**Tournament Dashboard:**
- **Live Grid:** Thumbnails of active background games.
- **Leaderboard:** Table showing Rank, Model, ELO, Win Rate, Total Tokens.
- **Analytics:** Line chart of ELO history over time.

**New Game:**
- Dropdowns to select Player 1 and Player 2 (Human or Specific Model).

---

## 8. Development Roadmap

**Phase 1: Core Engine (Done)**
- Game Logic, AI Integration, CLI Test.

**Phase 2: Backend & DB (Done/Ready)**
- FastAPI setup, Postgres Schema (Games + ELO), WebSocket Manager.

**Phase 3: Frontend (Next)**
- React App, Game Board, Replay Logic, Dashboard UI.

**Phase 4: Automation**
- Tournament Script implementation and deployment.