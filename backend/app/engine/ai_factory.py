"""
AI Provider Strategy Pattern

Replaces the massive if/else block in ai.py with a provider registry.
"""

from abc import ABC, abstractmethod
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from backend.app.core.model_registry import registry


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        """Build and return a LangChain chat model instance"""
        pass


class OpenAIProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        if config.get("base_url"):
            return ChatOpenAI(
                model=model_id,
                temperature=temperature,
                base_url=config.get("base_url")
            )
        else:
            return ChatOpenAI(
                model=model_id,
                temperature=temperature
            )


class AnthropicProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        return ChatAnthropic(
            model=model_id,
            temperature=temperature
        )


class GoogleProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        return ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            convert_system_message_to_human=True
        )


class DeepSeekProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=config.get("base_url", "https://api.deepseek.com")
        )


class MistralProvider(AIProvider):
    """
    Provider for Mistral AI models.
    Supports native tool calling for structured output.
    """
    def build(self, model_id: str, config: dict, temperature: float = 0.2):
        return ChatMistralAI(
            model=model_id,
            temperature=temperature,
            api_key=os.getenv("MISTRAL_API_KEY")
        )


# Provider Registry
PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "deepseek": DeepSeekProvider(),
    "mistral": MistralProvider()
}


def get_llm(model_key: str, temperature: float = 0.2):
    """
    Factory function to return the correct LangChain Chat Model using provider strategy.
    """
    # 1. Get Config from ModelRegistry
    config = registry.get(model_key)
    
    # Auto-detection/Fallback for unknown models
    if not config:
        if "gpt" in model_key:
            provider_key = "openai"
        elif "claude" in model_key:
            provider_key = "anthropic"
        elif "gemini" in model_key:
            provider_key = "google"
        elif "deepseek" in model_key:
            provider_key = "deepseek"
        elif "mistral" in model_key or "ministral" in model_key:
            provider_key = "mistral"
        else:
            print(f"Warning: Unknown model {model_key}, defaulting to gpt-4o")
            provider_key = "openai"
            model_key = "gpt-4o"
        
        # Create a dummy config for defaults
        api_model_name = model_key
        api_flags = {}
    else:
        provider_key = config.provider
        # Resolve Actual API Model ID
        # Use 'model_id' if present (for overrides), otherwise use the dict key
        api_model_name = config.model_id or model_key
        api_flags = config.api_config or {}
    
    # 2. Get provider from registry
    provider = PROVIDERS.get(provider_key)
    if not provider:
        raise ValueError(f"Unsupported provider: {provider_key}")
    
    # 3. Build model using provider strategy
    return provider.build(api_model_name, api_flags, temperature)