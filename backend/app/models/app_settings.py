"""Single-row table holding runtime-editable settings.

Two JSONB blobs keep the schema flat:
- ``api_keys``: ``{"openai": "...", "anthropic": "..."}`` — overrides env vars
  when set, used by ai_factory provider builders.
- ``tunables``: ``{"elo_k_factor": 24, "default_temperature": 0.0, ...}`` —
  override the pydantic-settings defaults via the ``settings`` facade.

Convention: there is exactly one row, with ``id = 1``. Use the helpers in
``services/runtime_settings.py`` to read / write rather than touching this
model directly.
"""

from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from backend.app.core.database import Base


class AppSettings(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True)  # always 1
    api_keys = Column(JSONB, nullable=False, default=dict)
    tunables = Column(JSONB, nullable=False, default=dict)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
