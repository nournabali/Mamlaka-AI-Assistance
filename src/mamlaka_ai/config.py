"""Environment-based application settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is a hard requirement, but keep import defensive for tests
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


PROJECT_ROOT = Path(
    os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR", PROJECT_ROOT / "data" / "approved")
        )
    )
    index_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("INDEX_DIR", PROJECT_ROOT / "artifacts" / "index")
        )
    )
    assets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "assets" / "images")

    # Language model
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").strip().lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "qwen/qwen3.6-27b").strip()
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        ).rstrip("/")
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
    )
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_num_ctx: int = field(default_factory=lambda: _env_int("LLM_NUM_CTX", 8192))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 900))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 240))
    # Grounded answers do not need visible model reasoning.
    llm_disable_thinking: bool = field(
        default_factory=lambda: _env_bool("LLM_DISABLE_THINKING", True)
    )

    @property
    def active_llm_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.llm_model

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    )
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    embedding_batch_size: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 16))

    # Chunking
    chunk_max_chars: int = field(default_factory=lambda: _env_int("CHUNK_MAX_CHARS", 900))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_CHARS", 150))

    # Retrieval
    top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 6))
    dense_candidates: int = field(default_factory=lambda: _env_int("RETRIEVAL_DENSE_CANDIDATES", 12))
    lexical_candidates: int = field(
        default_factory=lambda: _env_int("RETRIEVAL_LEXICAL_CANDIDATES", 12)
    )
    # Refuse before generation when no chunk reaches this cosine score.
    min_similarity: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SIMILARITY", 0.75))
    # Include nearby revision and typed-value chunks for conflict checks.
    revision_sweep_delta: float = field(
        default_factory=lambda: _env_float("REVISION_SWEEP_DELTA", 0.06)
    )
    revision_sweep_max: int = field(default_factory=lambda: _env_int("REVISION_SWEEP_MAX", 3))
    value_sweep_max: int = field(default_factory=lambda: _env_int("VALUE_SWEEP_MAX", 3))
    value_sweep_delta: float = field(
        default_factory=lambda: _env_float("VALUE_SWEEP_DELTA", 0.08)
    )
    # Add an English query for cross-language retrieval.
    cross_lingual_expansion: bool = field(
        default_factory=lambda: _env_bool("CROSS_LINGUAL_EXPANSION", True)
    )

    # Behavior
    debug_retrieval: bool = field(default_factory=lambda: _env_bool("DEBUG_RETRIEVAL", False))
    max_history_turns: int = field(default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 6))
    enable_query_rewrite: bool = field(
        default_factory=lambda: _env_bool("ENABLE_QUERY_REWRITE", True)
    )
    # Reject citations that are not supported by retrieved chunks.
    enforce_citation_validation: bool = field(
        default_factory=lambda: _env_bool("ENFORCE_CITATION_VALIDATION", True)
    )

    # Interface
    app_title: str = field(
        default_factory=lambda: os.getenv(
            "APP_TITLE", "Mamlaka AI — مساعد المملكة الذكي"
        )
    )
    @property
    def index_path(self) -> Path:
        return self.index_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"


settings = Settings()

# Only these PDFs may enter the knowledge base.
KNOWLEDGE_BASE_FILES = (
    "Almamlaka_Project_Overview.pdf",
    "Almamlaka_Team_Governance.pdf",
    "Almamlaka_Budget_Timeline.pdf",
)
