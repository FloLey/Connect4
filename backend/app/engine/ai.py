import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from backend.app.core.model_registry import registry

load_dotenv()

# --- 1. Define Structured Output ---
class MoveDecision(BaseModel):
    reasoning: str = Field(description="your reasoning that lead to the choice. step by step.")
    column: int = Field(description="Column index (0-6).")
    is_fallback: bool = Field(default=False, description="Whether this move was a system fallback due to AI failure.")

def get_llm(model_key: str, temperature: float = 0.2):
    """
    Factory function to return the correct LangChain Chat Model.
    Uses the new provider strategy pattern.
    """
    from backend.app.engine.ai_factory import get_llm as factory_get_llm
    return factory_get_llm(model_key, temperature)

# --- 3. Prompt Template ---
SYSTEM_PROMPT = """
You are an expert Connect Four player engine.
You are Player {player_id} (Symbol: {symbol}).
Opponent is Player {opponent_id} (Symbol: {opp_symbol}).
Board: 6 Rows x 7 Columns.
Goal: Connect 4 pieces in a row (Horizontal, Vertical, Diagonal).
Gravity: Pieces fall to the lowest empty slot.
"""

USER_TEMPLATE = """
Board (Visual):
{visual_board}

Board (Textual):
{textual_board}

Valid Columns: {valid_moves}

Analyze the board state carefully. Output valid JSON.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_TEMPLATE)
])

class ConnectFourAI:
    def __init__(self, player_id: int, model_name: str = "gpt-4o"):
        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        self.symbol = "X" if player_id == 1 else "O"
        self.opp_symbol = "O" if player_id == 1 else "X"
        self.model_name = model_name
        
        try:
            llm = get_llm(model_name)
            
            # Simple divergence: DeepSeek Chat requires explicit function_calling
            # All other models (OpenAI, Anthropic, Google) use auto-mode.
            if "deepseek" in model_name:
                self.structured_llm = llm.with_structured_output(
                    MoveDecision, 
                    method="function_calling", 
                    include_raw=True
                )
            else:
                self.structured_llm = llm.with_structured_output(MoveDecision, include_raw=True)
            
            self.chain = prompt | self.structured_llm
        except Exception as e:
            print(f"Failed to initialize model {model_name}: {e}")
            fallback_llm = get_llm("gpt-4o")
            self.structured_llm = fallback_llm.with_structured_output(MoveDecision, include_raw=True)
            self.chain = prompt | self.structured_llm

    async def get_move_async(self, game_engine):
        visual = game_engine.get_visual_board()
        textual = game_engine.get_textual_description()
        valid_moves = game_engine.get_valid_moves()

        try:
            result = await self.chain.ainvoke({
                "player_id": self.player_id,
                "opponent_id": self.opponent_id,
                "symbol": self.symbol,
                "opp_symbol": self.opp_symbol,
                "visual_board": visual,
                "textual_board": textual,
                "valid_moves": valid_moves
            })
            
            decision = result['parsed']
            raw_msg = result['raw']
            
            # Validate AI response - guard against empty/None responses
            if not decision:
                raise ValueError("Model returned empty decision")
            
            # Validate decision attributes exist
            if not hasattr(decision, 'column') or not hasattr(decision, 'reasoning'):
                raise ValueError("Model returned incomplete decision object")
            
            # Ensure column is a valid integer
            if not isinstance(decision.column, int):
                raise ValueError(f"Model returned non-integer column: {decision.column}")
            
            usage = getattr(raw_msg, 'usage_metadata', {}) or {"input_tokens": 0, "output_tokens": 0}
            
            # Validation: Column bounds
            if decision.column not in valid_moves:
                import random
                decision.column = random.choice(valid_moves)
                decision.reasoning += " [System: Original move invalid, corrected]"
            
            return {
                "decision": decision,
                "usage": usage
            }

        except Exception as e:
            print(f"AI Error ({self.model_name}): {e}")
            err_msg = str(e).lower()
            
            # Check for Rate Limit signatures (Generic + Provider Specific)
            is_rate_limit = any(x in err_msg for x in ["429", "rate_limit", "rate limit", "throttled", "quota exceeded", "too many requests"])
            
            if is_rate_limit:
                # DO NOT play random move. Escalate to the Service Layer.
                print(f"⚠️ Rate limit detected for {self.model_name}: {err_msg}")
                raise e 
            
            # Fallback to random move ONLY for parsing/logic errors
            import random
            
            # Capture specific exception message and truncate if too long
            error_msg = str(e)
            clean_error = (error_msg[:100] + '..') if len(error_msg) > 100 else error_msg
            
            fallback_reasoning = (
                f"⚠️ [SYSTEM ERROR] Model {self.model_name} failed. "
                f"Details: {clean_error}. "
                f"Action: Playing random valid move."
            )
            
            fallback = MoveDecision(
                reasoning=fallback_reasoning, 
                column=random.choice(valid_moves),
                is_fallback=True
            )
            return {
                "decision": fallback,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }