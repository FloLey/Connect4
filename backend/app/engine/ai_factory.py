"""
AI Provider Strategy Pattern

Replaces the massive if/else block in ai.py with a provider registry.
"""

from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

from backend.app.core.config import settings
from backend.app.core.logging import get_logger
from backend.app.core.model_registry import registry
from backend.app.services.runtime_settings import runtime_settings

logger = get_logger(__name__)


def _default_temperature() -> float:
    return settings.default_temperature


def _api_key(provider: str) -> str | None:
    """Centralised provider key lookup: runtime override -> env var."""
    return runtime_settings.get_api_key(provider)


class AIProvider(ABC):
    """Abstract base class for AI providers"""

    @abstractmethod
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        """Build and return a LangChain chat model instance"""
        pass


class OpenAIProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        temperature = _default_temperature() if temperature is None else temperature
        kwargs = {"model": model_id, "temperature": temperature}
        key = _api_key("openai")
        if key:
            kwargs["api_key"] = key
        if config.get("base_url"):
            kwargs["base_url"] = config["base_url"]
        return ChatOpenAI(**kwargs)


class AnthropicProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        temperature = _default_temperature() if temperature is None else temperature
        kwargs = {"model": model_id, "temperature": temperature}
        key = _api_key("anthropic")
        if key:
            kwargs["api_key"] = key
        return ChatAnthropic(**kwargs)


class GoogleProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        temperature = _default_temperature() if temperature is None else temperature
        kwargs = {
            "model": model_id,
            "temperature": temperature,
            "convert_system_message_to_human": True,
        }
        key = _api_key("google")
        if key:
            kwargs["google_api_key"] = key
        return ChatGoogleGenerativeAI(**kwargs)


class DeepSeekProvider(AIProvider):
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        temperature = _default_temperature() if temperature is None else temperature
        return ChatOpenAI(
            model=model_id,
            temperature=temperature,
            api_key=_api_key("deepseek"),
            base_url=config.get("base_url", "https://api.deepseek.com"),
        )


class MistralProvider(AIProvider):
    """
    Provider for Mistral AI models.
    Supports native tool calling for structured output.
    """
    def build(self, model_id: str, config: dict, temperature: float | None = None):
        temperature = _default_temperature() if temperature is None else temperature
        return ChatMistralAI(
            model=model_id,
            temperature=temperature,
            api_key=_api_key("mistral"),
        )


# Provider Registry
PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "deepseek": DeepSeekProvider(),
    "mistral": MistralProvider()
}


def get_llm(model_key: str, temperature: float | None = None):
    """
    Factory function to return the correct LangChain Chat Model using provider strategy.
    """
    if temperature is None:
        temperature = settings.default_temperature

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
            logger.warning(
                "unknown_model_falling_back",
                requested=model_key,
                fallback=settings.fallback_model,
            )
            provider_key = "openai"
            model_key = settings.fallback_model

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