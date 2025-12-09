import asyncio
import os
import sys

# Add project root to path so we can import from backend.app
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from backend.app.engine.ai import MODEL_PROVIDERS, get_llm

async def test_model(name, config):
    print(f"Testing {name} [{config['provider']}]...", end=" ", flush=True)
    try:
        llm = get_llm(name)
        # Send a tiny prompt to verify connectivity
        await llm.ainvoke("Hi")
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED")
        print(f"   Error: {str(e)}")
        return False

async def main():
    print("--- Verifying Model Configurations ---")
    results = []
    
    # Sort keys for consistent output
    sorted_models = sorted(MODEL_PROVIDERS.items())
    
    for name, config in sorted_models:
        success = await test_model(name, config)
        results.append(success)
        
    print("-" * 30)
    print(f"Passed: {sum(results)} / {len(results)}")

if __name__ == "__main__":
    asyncio.run(main())