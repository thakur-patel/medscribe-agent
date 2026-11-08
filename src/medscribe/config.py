"""Loads configs/models.yaml: model IDs, provider settings and prompt versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    id: str
    temperature: float
    max_tokens: int
    price_in_per_mtok: float
    price_out_per_mtok: float


@dataclass(frozen=True)
class PromptConfig:
    key: str
    version: str
    text: str


@dataclass(frozen=True)
class AppConfig:
    version: int
    base_url: str
    api_key_env: str
    models: dict[str, ModelConfig]
    prompts: dict[str, PromptConfig]

    def model(self, key: str = "default") -> ModelConfig:
        return self.models[key]

    def prompt(self, key: str) -> PromptConfig:
        return self.prompts[key]


def load_config(path: str | Path = DEFAULT_MODELS_CONFIG) -> AppConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    models = {
        key: ModelConfig(
            key=key,
            id=m["id"],
            temperature=float(m.get("temperature", 0)),
            max_tokens=int(m.get("max_tokens", 4000)),
            price_in_per_mtok=float(m.get("price_per_mtok", {}).get("input", 0)),
            price_out_per_mtok=float(m.get("price_per_mtok", {}).get("output", 0)),
        )
        for key, m in raw["models"].items()
    }
    prompts = {
        key: PromptConfig(
            key=key,
            version=p["version"],
            text=(REPO_ROOT / p["file"]).read_text(),
        )
        for key, p in raw["prompts"].items()
    }
    return AppConfig(
        version=int(raw["version"]),
        base_url=raw["provider"]["base_url"],
        api_key_env=raw["provider"]["api_key_env"],
        models=models,
        prompts=prompts,
    )
