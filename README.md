# Connect Four LLM Arena 🎯

A production-grade competitive platform to benchmark Large Language Models (LLMs) against humans and each other.

The platform orchestrates tournaments, calculates ELO ratings, tracks token usage/costs, and provides deep analytics on model reasoning capabilities using the game of Connect Four.

![Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20React%20%7C%20PostgreSQL-blue)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED)

## 🌟 Key Features

### 🏟️ The Arena
*   **Human vs. AI:** Real-time gameplay via WebSockets.
*   **AI vs. AI:** Spectate live matches between different models.
*   **Unified Replay System:** "Time-travel" through past games with a scrubber timeline.
*   **Reasoning Visibility:** View the raw "Chain of Thought" and token cost for every AI move.

### 🏆 Tournament System
*   **Automated Benchmarking:** Run background round-robin tournaments.
*   **Concurrency Control:** Adjust the number of parallel workers live without restarting the server.
*   **Resiliency:** System automatically resumes interrupted tournaments/games on startup.
*   **ELO Engine:** Real-time rating updates ($K=32$) with idempotency checks.

### 📊 Analytics & Economics
*   **Live Leaderboard:** Track Win Rates, ELO, and Games Played.
*   **Economic Analysis:** Scatter plots for Cost vs. Performance and Speed vs. Performance.
*   **Win Rate Matrix:** Heatmap detection of model-specific weaknesses.
*   **Multi-Provider Support:** First-class support for OpenAI, Anthropic, Google Gemini, and DeepSeek.

### 🛠️ Architecture
*   **Dual-Environment:** Instant toggle between `Production` (Ranked) and `Test` (Sandbox) databases via the UI.
*   **Row-Level Locking:** Prevents race conditions during simultaneous moves.
*   **Hybrid Architecture:** WebSockets for live interactions; Async Background Runners for tournament execution.

---

## 🏗 Tech Stack

### Backend
*   **Framework:** Python 3.11, FastAPI
*   **Database:** PostgreSQL (AsyncSQLAlchemy + Alembic)
*   **AI Orchestration:** LangChain (Structured Output / Function Calling)
*   **Tasks:** AsyncIO Native Background Tasks

### Frontend
*   **Framework:** React 18 (Vite)
*   **Styling:** Tailwind CSS (Dark Mode supported)
*   **Visualization:** Recharts (Analytics), Framer Motion (Animations)
*   **State:** Context API

---

## 🚀 Getting Started (Docker)

The preferred way to run the application is via Docker Compose.

### 1. Prerequisites
*   Docker & Docker Compose
*   API Keys for desired providers (OpenAI, Anthropic, Google, etc.)

### 2. Configuration
Copy the example environment file:
```bash
cp backend/.env.example backend/.env
```
Edit `backend/.env` and add your API keys:
```ini
DATABASE_URL=postgresql+asyncpg://user:password@db:5432/connect4_arena
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant...
# Add other keys as needed
```

### 3. Run the Stack
```bash
docker compose up --build
```
*   **Frontend:** http://localhost:5173
*   **Backend API:** http://localhost:8000
*   **Docs:** http://localhost:8000/docs

**Note:** The `init_db.py` script runs automatically on container start to create tables in both Production and Test databases.

---

## 🛠️ Management & Scripts

All scripts should be run from the project root inside the backend container.

### Verify Model Configuration
Before starting a tournament, verify your API keys and model availability:
```bash
docker compose exec backend python backend/scripts/verify_ai_moves.py
```
Runs a parallelized check against all configured LLMs to ensure they can produce valid JSON game moves.

### Database Cleanup
To archive abandoned games (older than 1 hour):
```bash
docker compose exec backend python backend/scripts/cleanup_games.py
```

### Database Snapshot
Create a SQL dump of the current database state:
```bash
./backend/scripts/snapshot_db.sh
```

---

## 🧠 Adding New Models

Model configurations are centrally managed in the backend via YAML configuration.

1.  Open `backend/config/models.yaml`.
2.  Add a new entry to the `models` section.
3.  Restart the backend container.

See `MODELS.md` for detailed configuration options and currently supported models.

---

## 🧪 Development Workflow

### Hot Reloading
Both Frontend and Backend containers are configured for Hot Reloading.

*   **Backend:** Changes to `*.py` files trigger a Uvicorn reload.
*   **Frontend:** Changes to `*.jsx` trigger a Vite HMR update.

### Test Environment
Use the toggle in the Top Navigation Bar to switch between **Production** and **Test Sandbox**.

*   **Production:** Affects the main leaderboard.
*   **Test:** Isolated database for testing new models or features without affecting stats.

**Note:** API costs are real in both environments.

---

## 📂 Project Structure

```
Connect4/
├── backend/
│   ├── app/
│   │   ├── api/            # Endpoints (Games, Stats, Admin)
│   │   ├── engine/         # Core Logic (Connect4, AI, ELO)
│   │   ├── services/       # Business Logic (Tournament, Runner)
│   │   └── models/         # DB Models
│   ├── scripts/            # Utility Scripts
│   └── main.py             # Entry Point & Lifespan Manager
├── frontend/
│   ├── src/
│   │   ├── components/     # Reusable UI
│   │   ├── context/        # Theme & DB Context
│   │   └── pages/          # Views (Arena, Dashboard, Stats)
└── docker-compose.yml
```

---

## 📈 API Endpoints

### Core Game API
*   `POST /games` - Create a new game
*   `GET /games/{id}` - Get game details and history
*   `GET /games/history` - Paginated game history
*   `WebSocket /games/{id}/ws` - Real-time gameplay

### Statistics & Analytics
*   `GET /stats/leaderboard` - Model performance rankings
*   `GET /stats/matrix` - Win rate matrix (model vs model)
*   `GET /stats/history` - ELO progression over time
*   `GET /stats/active-games` - Live tournament matches

### Tournament Management
*   `POST /tournament/create` - Create a new tournament
*   `GET /tournament/current` - Get active tournament status
*   `POST /tournament/{id}/start` - Start tournament execution
*   `POST /tournament/{id}/pause` - Pause tournament (adjust concurrency)
*   `POST /tournament/{id}/resume` - Resume paused tournament
*   `POST /tournament/{id}/stop` - Gracefully stop tournament

### Administration
*   `GET /admin/status` - System health and queue status
*   `DELETE /admin/reset` - **DANGER** - Reset entire database (requires confirmation)

---

## 🔒 Race Condition Prevention

The system implements robust race condition prevention:

1.  **Row-Level Locking:** Database `FOR UPDATE` locks prevent simultaneous moves
2.  **Move Count Validation:** AI moves validate `len(history)` before committing
3.  **Idempotent ELO Updates:** ELO calculations include game ID to prevent double-counting
4.  **Background Runner Isolation:** Tournament games run in isolated processes

---

## 📝 License

MIT

---

## 🙏 Acknowledgments

*   Built with FastAPI, React, and LangChain
*   Inspired by AI benchmarking challenges
*   Connect Four game logic implementation