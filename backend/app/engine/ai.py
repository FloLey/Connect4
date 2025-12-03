import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load env variables
load_dotenv()

# --- 1. Define Structured Output ---
class MoveDecision(BaseModel):
    reasoning: str = Field(
        description="A concise explanation of the strategy (e.g., 'Blocking opponent vertical win' or 'Building diagonal threat')."
    )
    column: int = Field(
        description="The column index (0-6) where the piece should be dropped."
    )

# --- 2. Setup LLM & Prompt ---
# We use a low temperature for gameplay stability, but non-zero for variety.
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
structured_llm = llm.with_structured_output(MoveDecision)

SYSTEM_PROMPT = """
You are an expert Connect Four player engine.
The board is 6 Rows x 7 Columns (Cols 0-6).
Gravity applies: Pieces fall to the lowest available row in a column.
Goal: Connect 4 pieces (Horizontal, Vertical, Diagonal).

You are Player {player_id} (Symbol: {symbol}).
Your Opponent is Player {opponent_id} (Symbol: {opp_symbol}).
"""

USER_TEMPLATE = """
Current Board State (Visual):
{visual_board}

Current Board State (Textual - Bottom to Top):
{textual_board}

Analyze the board. Provide your reasoning and select a column.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_TEMPLATE)
])

class ConnectFourAI:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        self.symbol = "X" if player_id == 1 else "O"
        self.opp_symbol = "O" if player_id == 1 else "X"
        self.chain = prompt | structured_llm

    def get_move(self, game_engine) -> MoveDecision:
        """
        Accepts the GameEngine instance, extracts state, prompts LLM, 
        and returns the MoveDecision object.
        """
        visual = game_engine.get_visual_board()
        textual = game_engine.get_textual_description()
        valid_moves = game_engine.get_valid_moves()

        try:
            decision = self.chain.invoke({
                "player_id": self.player_id,
                "opponent_id": self.opponent_id,
                "symbol": self.symbol,
                "opp_symbol": self.opp_symbol,
                "visual_board": visual,
                "textual_board": textual,
                "valid_moves": valid_moves
            })
            
            # Basic validation fallback
            if decision.column not in valid_moves:
                print(f"Warning: AI suggested invalid col {decision.column}. Picking random valid move.")
                import random
                decision.column = random.choice(valid_moves)
                decision.reasoning = "Fallback: Original move was invalid."
            
            return decision

        except Exception as e:
            print(f"LLM Error: {e}")
            # Crash safe fallback
            import random
            return MoveDecision(
                reasoning="Error in generation, playing random.",
                column=random.choice(valid_moves)
            )