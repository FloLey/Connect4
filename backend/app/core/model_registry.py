import yaml
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional

class ModelConfig(BaseModel):
    # `model_id` collides with Pydantic's reserved `model_` namespace; opt out
    # of the protection (we don't shadow any framework method here).
    model_config = ConfigDict(protected_namespaces=())

    provider: str
    label: str
    context: int
    pricing: Dict[str, float]
    model_id: Optional[str] = None  # Optional override
    api_config: Optional[Dict[str, Any]] = None  # Optional API-specific config

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "models.yaml"
)


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            config_path = _DEFAULT_CONFIG_PATH
        self.models: Dict[str, ModelConfig] = {}
        self._load(config_path)

    def _load(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            for key, val in data.get("models", {}).items():
                self.models[key] = ModelConfig(**val)

    def get(self, model_key: str) -> Optional[ModelConfig]:
        return self.models.get(model_key)

    def list_all(self) -> Dict[str, ModelConfig]:
        return self.models

# Singleton instance
registry = ModelRegistry()