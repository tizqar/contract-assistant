"""Application settings, read from the environment with defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    extractor_model_id: str = field(
        default_factory=lambda: _env("EXTRACTOR_MODEL_ID", "models/ettin-150m-cuad-seed3407")
    )
    device: str = field(default_factory=lambda: _env("EXTRACTOR_DEVICE", "auto"))
    batch_size: int = field(default_factory=lambda: int(_env("EXTRACTOR_BATCH_SIZE", "16")))

    # Must match the settings the extractor was trained and evaluated with.
    max_seq_len: int = 1024
    doc_stride: int = 256
    max_answer_len: int = 512
    top_k: int = 20
    n_best_size: int = 20

    # found: conf >= tau_high; check: tau_low <= conf < tau_high; not found: conf < tau_low
    tau_low: float = field(default_factory=lambda: float(_env("TAU_LOW", "0.06")))
    tau_high: float = field(default_factory=lambda: float(_env("TAU_HIGH", "0.91")))
    tau_source: str = field(default_factory=lambda: _env("TAU_SOURCE", "validation, seed 3407"))

    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:1.7b"))
    openai_base_url: str | None = field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    llm_timeout_s: float = 10.0

    answer_style: str = field(default_factory=lambda: _env("ANSWER_STYLE", "template"))
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", "cache")))
    diplomski_dir: Path = field(
        default_factory=lambda: Path(_env("DIPLOMSKI_DIR", "../diplomski"))
    )
    clause_types_path: Path = ROOT / "data" / "clause_types.json"

    def decode_settings(self) -> dict:
        """The settings a cache must have been written with to be reused."""
        return {
            "id": self.extractor_model_id,
            "max_seq_len": self.max_seq_len,
            "doc_stride": self.doc_stride,
            "max_answer_len": self.max_answer_len,
            "top_k": self.top_k,
            "n_best_size": self.n_best_size,
        }


def load_settings() -> Settings:
    return Settings()
