import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# --- 1. Define Structured Output ---
class MoveDecision(BaseModel):
    reasoning: str = Field(description="your reasoning that lead to the choice. step by step.")
    column: int = Field(description="Column index (0-6).")

# Pricing is in USD per 1 Million Tokens
MODEL_PROVIDERS = {
    # --- OpenAI (Frontier) ---
    "gpt-5.1": {
        "provider": "openai", 
        "label": "GPT-5.1", 
        "context": 128000,
        "pricing": {"input": 1.25, "output": 10.00}
    },
    "gpt-5-mini": {
        "provider": "openai", 
        "label": "GPT-5 Mini", 
        "context": 128000,
        "pricing": {"input": 0.25, "output": 2.00}
    },
    "gpt-5-nano": {
        "provider": "openai", 
        "label": "GPT-5 Nano", 
        "context": 128000,
        "pricing": {"input": 0.05, "output": 0.40}
    },
    "gpt-5": {
        "provider": "openai", 
        "label": "GPT-5", 
        "context": 128000,
        "pricing": {"input":1.25, "output": 10.00}
    },
    "gpt-4.1": {
        "provider": "openai", 
        "label": "GPT-4.1", 
        "context": 128000,
        "pricing": {"input": 2.00, "output": 8.00}
    },
    "gpt-4o": {
        "provider": "openai", 
        "label": "GPT-4o", 
        "context": 128000,
        "pricing": {"input": 2.50, "output": 10.00}
    },

    # --- Google (Gemini) ---
    "gemini-3-pro-preview": {
        "provider": "google", "label": "Gemini 3 Pro", "context": 1048576,
        "pricing": {"input": 2.00, "output": 12.00}
    },
    "gemini-2.5-pro": {
        "provider": "google", "label": "Gemini 2.5 Pro", "context": 1048576,
        "pricing": {"input": 1.25, "output": 10.00}
    },
    "gemini-2.5-flash": {
        "provider": "google", "label": "Gemini 2.5 Flash", "context": 1048576,
        "pricing": {"input": 0.3, "output": 2.5}
    },
    "gemini-2.5-flash-lite": {
        "provider": "google", "label": "Gemini 2.5 Flash-Lite", "context": 1048576,
        "pricing": {"input": 0.1, "output": 0.4}
    },

    # --- Anthropic (Claude) ---
    "claude-sonnet-4.5": {
        "provider": "anthropic", 
        "label": "Claude 4.5 Sonnet", 
        "context": 200000,
        "model_id": "claude-sonnet-4-5-20250929",
        "pricing": {"input": 3.00, "output": 15.00}
    },
    "claude-opus-4.5": {
        "provider": "anthropic", 
        "label": "Claude 4.5 Opus", 
        "context": 200000,
        "model_id": "claude-opus-4-5-20251101",
        "pricing": {"input": 15.00, "output": 75.00}
    },
    "claude-haiku-4.5": {
        "provider": "anthropic", 
        "label": "Claude 4.5 Haiku", 
        "context": 200000,
        "model_id": "claude-haiku-4-5-20251001",
        "pricing": {"input": 0.25, "output": 1.25}
    },

    # --- DeepSeek ---
    "deepseek-v3.2-chat": {
        "provider": "deepseek",
        "label": "DeepSeek V3.2 (Chat)",
        "context": 128000,
        "model_id": "deepseek-chat",
        "api_config": {
            "base_url": "https://api.deepseek.com"
        },
        "pricing": {"input": 0.14, "output": 0.28}  # Very cheap
    },
}

def get_llm(model_key: str, temperature: float = 0.2):
    """
    Factory function to return the correct LangChain Chat Model.
    """
    # 1. Get Config
    config = MODEL_PROVIDERS.get(model_key)
    
    # Auto-detection/Fallback for unknown models
    if not config:
        if "gpt" in model_key:
            provider = "openai"
        elif "claude" in model_key:
            provider = "anthropic"
        elif "gemini" in model_key:
            provider = "google"
        elif "deepseek" in model_key:
            provider = "deepseek"
        else:
            print(f"Warning: Unknown model {model_key}, defaulting to gpt-4o")
            provider = "openai"
            model_key = "gpt-4o"
        
        # Create a dummy config for defaults
        config = {"provider": provider, "model_id": model_key}
    else:
        provider = config["provider"]

    # 2. Resolve Actual API Model ID
    # Use 'model_id' if present (for overrides), otherwise use the dict key
    api_model_name = config.get("model_id", model_key)
    api_flags = config.get("api_config", {})

    # 3. Instantiate Provider Class
    if provider == "openai":
        # Check for custom base_url (future proofing)
        if api_flags.get("base_url"):
            return ChatOpenAI(
                model=api_model_name,
                temperature=temperature,
                base_url=api_flags.get("base_url")
            )
        else:
            return ChatOpenAI(
                model=api_model_name, 
                temperature=temperature
            )

    elif provider == "anthropic":
        return ChatAnthropic(
            model=api_model_name, 
            temperature=temperature
        )

    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=api_model_name,
            temperature=temperature,
            convert_system_message_to_human=True
        )

    elif provider == "deepseek":
        return ChatOpenAI(
            model=api_model_name,
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=api_flags.get("base_url", "https://api.deepseek.com")
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

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
                column=random.choice(valid_moves)
            )
            return {
                "decision": fallback,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }