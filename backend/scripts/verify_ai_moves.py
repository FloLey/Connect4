#!/usr/bin/env python3
"""
AI Model Structured Output Verification Script - Parallel Version

This script validates that all configured LLM providers can successfully generate
structured JSON outputs (MoveDecision) within the Connect Four game engine context.

Features:
- Parallel execution with configurable concurrency limits
- Rate limiting via asyncio.Semaphore to prevent API 429 errors
- Real-time result output as tests complete
- Comprehensive validation of structured outputs

Exit Codes:
  0: All models passed
  1: One or more models failed
"""

import asyncio
import sys
import os
import logging
from typing import Tuple

# Add project root to path so we can import from backend.app
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# Suppress HTTP request logs for cleaner output
logging.getLogger("httpx").setLevel(logging.WARNING)

from backend.app.engine.ai import MODEL_PROVIDERS, ConnectFourAI
from backend.app.engine.game import ConnectFour

# --- Configuration ---
CONCURRENCY_LIMIT = 50  # Adjust based on your API tier limits


async def test_model_generation(model_key: str) -> Tuple[str, bool, str]:
    """
    Test a single model's ability to generate structured outputs in a game context.
    
    Args:
        model_key: The model identifier (e.g., "gpt-4o", "deepseek-v3.2-chat")
        
    Returns:
        Tuple of (Model Name, Success Boolean, Error Message)
    """
    # 1. Setup Board - Simulate mid-game state (not empty board)
    engine = ConnectFour()
    engine.drop_piece(3)  # Player 1 move
    engine.drop_piece(2)  # Player 2 move  
    engine.drop_piece(3)  # Player 1 move
    
    try:
        # 2. Initialize AI Agent
        ai = ConnectFourAI(player_id=2, model_name=model_key)
        
        # 3. Execute AI Move Generation
        result = await ai.get_move_async(engine)
        decision = result["decision"]
        
        # 4. Critical Validation: Check for Internal Fallback
        if hasattr(decision, 'reasoning') and decision.reasoning and decision.reasoning.startswith("⚠️ [SYSTEM ERROR]"):
            return (model_key, False, f"Fallback Triggered: {decision.reasoning[:100]}...")
        
        # 5. Validate Decision Structure
        if not hasattr(decision, 'column'):
            return (model_key, False, "Response missing 'column' attribute")
            
        if not isinstance(decision.column, int):
            return (model_key, False, f"Invalid column type: {type(decision.column)}")
        
        # 6. Validate Move Range
        if not (0 <= decision.column <= 6):
            return (model_key, False, f"Column out of bounds: {decision.column}")
        
        # 7. Validate Move is Legal on Current Board
        valid_moves = engine.get_valid_moves()
        if decision.column not in valid_moves:
            return (model_key, False, f"Illegal move: column {decision.column} not in valid moves {valid_moves}")
        
        # 8. Success - All validations passed
        reasoning_preview = (decision.reasoning[:30] + "...") if hasattr(decision, 'reasoning') and decision.reasoning else "N/A"
        return (model_key, True, f"Column: {decision.column}, Reasoning: {reasoning_preview}")
        
    except Exception as e:
        return (model_key, False, f"Exception: {str(e)[:80]}...")


async def main():
    """Main execution function with parallel processing."""
    print(f"🚀 Starting Parallel Verification (Limit: {CONCURRENCY_LIMIT} concurrent requests)...")
    print("-" * 80)
    print(f"{'MODEL ID':<30} | {'STATUS':<8} | {'DETAILS'}")
    print("-" * 80)

    # Semaphore to control concurrency and prevent rate limiting
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def protected_test(model_key: str) -> Tuple[str, bool, str]:
        """Wrapper function that respects the semaphore for rate limiting."""
        async with sem:
            return await test_model_generation(model_key)

    # Create tasks for all models
    model_keys = sorted(MODEL_PROVIDERS.keys())
    tasks = [protected_test(model_key) for model_key in model_keys]
    
    results = {}
    
    # Execute tasks and print results as they complete
    for coro in asyncio.as_completed(tasks):
        model, success, msg = await coro
        results[model] = success
        
        status_icon = "✅ PASS" if success else "❌ FAIL"
        # Truncate message if too long for table formatting
        clean_msg = (msg[:35] + '..') if len(msg) > 35 else msg
        
        print(f"{model:<30} | {status_icon:<8} | {clean_msg}")

    print("-" * 80)
    
    # Print Summary
    passed_count = sum(results.values())
    total = len(results)
    print(f"🏁 Summary: {passed_count}/{total} Passed")
    
    if passed_count < total:
        print("\n❌ Failed Models:")
        failed_models = [model for model, success in results.items() if not success]
        for model in failed_models:
            provider = MODEL_PROVIDERS.get(model, {}).get('provider', 'unknown')
            print(f"   • {model} [{provider}]")
        print(f"\n⚠️  {len(failed_models)} model(s) cannot generate valid structured outputs")
        print("   These models will fallback to random moves in actual gameplay.")
        sys.exit(1)
    else:
        print("✅ All models passed! Structured output generation is working correctly.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())