# Connect Four LLM Arena 🎯

A full-stack web platform to benchmark Large Language Models (LLMs) on the game of Connect Four.

## Features

- **Live Arena**: Human vs AI real-time gameplay
- **Replay System**: "Time-travel" through past games with LLM reasoning
- **Automated Tournament**: Background AI vs AI matches for ELO ratings

## Tech Stack

- **Backend**: Python 3.11+ (FastAPI)
- **Real-Time**: WebSockets (Native FastAPI)
- **Database**: PostgreSQL (SQLAlchemy + Alembic)
- **AI Orchestration**: LangChain with Pydantic JSON outputs
- **Frontend**: React (Vite) + Tailwind CSS
- **Containerization**: Docker & Docker Compose

## Project Structure

```
connect4-llm-arena/
├── backend/                    # Python / FastAPI
│   ├── app/
│   │   ├── api/               # Route Handlers
│   │   │   └── websocket_manager.py
│   │   ├── core/              # Config
│   │   │   ├── config.py
│   │   │   └── database.py
│   │   ├── engine/            # Pure Game Logic
│   │   │   ├── game.py        # ConnectFour rules
│   │   │   └── ai.py          # LangChain Agent
│   │   ├── models/            # Database Models
│   │   │   └── game_model.py
│   │   ├── schemas/           # Pydantic Schemas
│   │   │   └── game_schema.py
│   │   └── main.py            # FastAPI app
│   ├── migrations/            # Alembic (DB Migrations)
│   ├── requirements.txt
│   └── scripts/
│       └── init_db.py         # Database initialization
├── frontend/                  # React (Vite)
│   └── src/
│       ├── api/              # Axios wrappers
│       ├── components/       # Reusable UI
│       ├── hooks/           # useGameSocket.js
│       └── pages/           # Arena, Gallery, Replay
├── docker-compose.yml        # Postgres orchestration
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Connect4
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

3. **Set up PostgreSQL**
   ```bash
   # Install PostgreSQL (Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install -y postgresql postgresql-contrib
   
   # Start PostgreSQL service
   sudo service postgresql start
   
   # Create database and user
   sudo -u postgres psql -c "CREATE DATABASE connect4_arena;"
   sudo -u postgres psql -c "CREATE USER connect4_user WITH PASSWORD 'connect4';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE connect4_arena TO connect4_user;"
   sudo -u postgres psql -c "ALTER USER connect4_user WITH SUPERUSER;"
   ```

4. **Configure environment variables**
   ```bash
   # Copy .env.example to .env and update values
   cp backend/.env.example backend/.env
   # Edit backend/.env with your OpenAI API key
   ```

5. **Initialize database**
   ```bash
   python -m backend.scripts.init_db
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **API will be available at**: http://localhost:8000
   - API documentation: http://localhost:8000/docs
   - WebSocket endpoint: ws://localhost:8000/games/{id}/ws

### API Endpoints

- `POST /games` - Create a new game
- `GET /games/{id}` - Get game details and history
- `GET /games` - List completed games (for leaderboard)
- `WebSocket /games/{id}/ws` - Real-time gameplay

### WebSocket Protocol

**Connection**: `ws://localhost:8000/games/{game_id}/ws`

**Client → Server** (Human move):
```json
{ "action": "MOVE", "column": 3 }
```

**Server → Client** messages:
- `{"type": "UPDATE", "board_visual": "...", "current_turn": 1, ...}` - Game state update
- `{"type": "THINKING_START"}` - AI is thinking
- `{"type": "THINKING_END"}` - AI finished thinking

## Development Status

### ✅ Phase 1: Core Engine (Completed)
- ConnectFour game logic
- LangChain AI agent with structured outputs
- CLI testing script (`console_test.py`)

### ✅ Phase 2: Backend & Database (Completed)
- PostgreSQL database setup
- FastAPI REST endpoints
- WebSocket real-time game loop
- Game state persistence

### 🔄 Phase 3: Frontend (Next)
- React board component
- WebSocket hook integration
- Replay view with slider
- AI reasoning display

### 📋 Phase 4: Tournament System
- Automated AI vs AI matches
- ELO rating calculation
- Background tournament runner

## Testing

Run the console test:
```bash
python console_test.py
```

Test the API:
```bash
# Create a game
curl -X POST "http://localhost:8000/games" \
  -H "Content-Type: application/json" \
  -d '{"player_1": "human", "player_2": "ai"}'

# Get game details
curl "http://localhost:8000/games/1"
```

## License

MIT

## Acknowledgments

- Built with FastAPI, React, and LangChain
- Inspired by AI benchmarking challenges
- Connect Four game logic implementation