# 🧠 AI Model Registry & Configuration

This document outlines the Large Language Models (LLMs) supported by the Connect Four Arena, how they are configured, and how to extend the list.

## 🏗 Architecture: Single Source of Truth

We use a **Backend-Driven Architecture** for model management. 

1.  **Master Config**: All models are defined in `backend/app/engine/ai.py`.
2.  **API Exposure**: The backend exposes `GET /models`.
3.  **Dynamic UI**: The Frontend (`NewGame.jsx`) fetches this list on load to populate the dropdowns. 

**🚫 Do not hardcode models in the Frontend.**

---

## 📋 Supported Models

As of the latest configuration, the following models are registered in the backend.

### 🟢 OpenAI (Frontier)
| Model ID | Label | Context Window | Notes |
| :--- | :--- | :--- | :--- |
| `gpt-5.1` | GPT-5.1 | 128k | Best for coding/agentic tasks |
| `gpt-5-pro` | GPT-5 Pro | 128k | High-precision reasoning |
| `gpt-5` | GPT-5 | 128k | Base frontier model |
| `gpt-5-mini` | GPT-5 Mini | 128k | Cost-efficient reasoning |
| `gpt-5-nano` | GPT-5 Nano | 128k | Fastest reasoning model |
| `gpt-4.1` | GPT-4.1 | 128k | Smartest non-reasoning model |
| `gpt-4o` | GPT-4o | 128k | Fallback / Standard |

### 🔵 Google (Gemini)
| Model ID | Label | Context Window | Notes |
| :--- | :--- | :--- | :--- |
| `gemini-3-pro-preview` | Gemini 3 Pro | 1M | Multimodal agentic power |
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1M | Advanced thinking model |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M | Low latency, high volume |
| `gemini-2.5-flash-lite`| Gemini 2.5 Flash-Lite | 1M | Ultra-fast cost optimized |

### 🟠 Anthropic (Claude)
| Model ID | Label | Context Window | Notes |
| :--- | :--- | :--- | :--- |
| `claude-sonnet-4.5` | Claude 4.5 Sonnet | 200k | Balanced intelligence |
| `claude-opus-4.5` | Claude 4.5 Opus | 200k | Maximum capability |
| `claude-haiku-4.5` | Claude 4.5 Haiku | 200k | Speed & efficiency |
| `claude-3-5-sonnet-20240620` | Claude 3.5 Sonnet | 200k | Legacy stable |

---

## ⚙️ How to Add a New Model

To add a new model (e.g., `gpt-6` or `gemini-4`), you **only** need to touch the backend.

1.  Open **`backend/app/engine/ai.py`**.
2.  Locate the `MODEL_PROVIDERS` dictionary.
3.  Add a new entry following this format:

```python
"new-model-id": {
    "provider": "openai",  # or "anthropic" or "google"
    "label": "New Model Name",
    "context": 128000
},
```

4.  Restart the Backend container to apply changes:

```bash
docker compose restart backend
```

The Frontend will automatically display the new model in the dropdown.

## 🧪 Verification

We have a script to verify that the Backend can successfully initialize these models and that your API keys are valid.

Run this command from the project root:

```bash
docker compose exec backend python backend/scripts/test_models.py
```

- ✅ **OK**: Model is reachable.
- ❌ **FAILED**: Check your .env API keys or the model ID.

## 🔑 Environment Variables

Ensure your .env file contains keys for all providers you intend to use:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```