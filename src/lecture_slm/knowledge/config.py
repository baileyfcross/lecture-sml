"""Validated configuration for local knowledge indexing and retrieval."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator


class IndexingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recursive: bool = True
    include_extensions: list[str] = Field(
        default_factory=lambda: [".md", ".pdf", ".txt", ".docx", ".pptx"]
    )
    ignore_directories: list[str] = Field(
        default_factory=lambda: [
            ".obsidian",
            ".git",
            ".trash",
            "node_modules",
            "evals",
            "artifacts",
            "logs",
            "generated",
            "outputs",
            "data",
            "models",
            "checkpoints",
            "adapters",
            "merged",
        ]
    )

    @model_validator(mode="after")
    def validate_extensions(self) -> "IndexingConfig":
        self.include_extensions = [extension.lower() for extension in self.include_extensions]
        if any(not extension.startswith(".") for extension in self.include_extensions):
            raise ValueError("included file extensions must start with a dot")
        return self


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: int = Field(default=450, gt=0)
    max_tokens: int = Field(default=650, gt=0)
    overlap_tokens: int = Field(default=60, ge=0)
    preserve_sections: bool = True

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "ChunkingConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if self.overlap_tokens >= min(self.target_tokens, self.max_tokens):
            raise ValueError("overlap_tokens must be smaller than target_tokens and max_tokens")
        return self


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="fastembed", min_length=1)
    model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    batch_size: int = Field(default=64, gt=0)
    cache_dir: Path | None = None

    @model_validator(mode="after")
    def validate_provider(self) -> "EmbeddingsConfig":
        if self.provider != "fastembed":
            raise ValueError("only the local FastEmbed provider is currently supported")
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lexical_candidates: int = Field(default=30, gt=0)
    semantic_candidates: int = Field(default=30, gt=0)
    fused_candidates: int = Field(default=20, gt=0)
    final_results: int = Field(default=8, gt=0)
    neighbor_expansion: int = Field(default=1, ge=0)
    reranker_enabled: bool = False
    rrf_constant: int = Field(default=60, gt=0)
    exact_title_boost: float = Field(default=0.15, ge=0.0)
    exact_section_boost: float = Field(default=0.15, ge=0.0)
    context_budgets: dict[str, int] = Field(
        default_factory=lambda: {"quick": 1500, "standard": 3500, "deep": 7000}
    )

    @model_validator(mode="after")
    def validate_candidate_counts(self) -> "RetrievalConfig":
        if self.final_results > self.fused_candidates:
            raise ValueError("final_results must not exceed fused_candidates")
        if not self.context_budgets or any(budget <= 0 for budget in self.context_budgets.values()):
            raise ValueError("context_budgets must contain positive token limits")
        if not {"quick", "standard", "deep"}.issubset(self.context_budgets):
            raise ValueError("context_budgets must define quick, standard, and deep")
        return self


class KnowledgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault_path: Path | None = None
    data_dir: Path = Path("data/knowledge")
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


def load_knowledge_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> KnowledgeConfig:
    """Load knowledge YAML and apply the explicit vault-path environment override."""

    if not path.is_file():
        raise FileNotFoundError(f"Knowledge configuration not found: {path}")
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid knowledge configuration YAML in {path}: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("knowledge"), dict):
        raise ValueError(f"Expected a 'knowledge' mapping in {path}")
    config = KnowledgeConfig.model_validate(raw["knowledge"])
    if environ is None:
        load_dotenv()
        environment: Mapping[str, str] = os.environ
    else:
        environment = environ
    vault_override = environment.get("LECTURE_SLM_VAULT_PATH")
    if vault_override:
        config.vault_path = Path(vault_override).expanduser()
    return config
