"""Agent configuration from environment variables.

Compatible with personal-os PERSONAL_OS_* env vars, with MATRIX_* as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


# ---- Environment variable names ----

# Matrix-native env vars (fallback when personal-os vars are unset)
ENV_AGENT_ADDR = "MATRIX_AGENT_ADDR"
ENV_CACHE_PATH = "MATRIX_CACHE_PATH"
ENV_TRACE_PATH = "MATRIX_TRACE_PATH"

# personal-os compatibility env vars (higher priority)
ENV_OS_AGENT_ADDR = "PERSONAL_OS_AGENT_ADDR"
ENV_OS_CACHE_PATH = "PERSONAL_OS_CACHE_PATH"
ENV_OS_TRACE_PATH = "PERSONAL_OS_AGENT_TRACE_PATH"

# Legacy env vars
ENV_AGENT_HOST = "AGENT_HOST"
ENV_AGENT_PORT = "AGENT_PORT"

# LLM provider env vars
ENV_AGENT_PROVIDER = "AGENT_PROVIDER"
ENV_DEEPSEEK_API_KEY = "DEEPSEEK_API_KEY"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_AGNES_API_KEY = "AGNES_API_KEY"
ENV_AGENT_MODEL = "AGENT_MODEL"
ENV_AGENT_MAX_TOKENS = "AGENT_MAX_TOKENS"
ENV_AGENT_MODEL_TIMEOUT_SEC = "AGENT_MODEL_TIMEOUT_SEC"
ENV_DEEPSEEK_BASE_URL = "DEEPSEEK_BASE_URL"
ENV_AGNES_BASE_URL = "AGNES_BASE_URL"
ENV_MEMORY_MAX_TURNS = "MEMORY_MAX_TURNS"
ENV_STORE_PATH = "MATRIX_STORE_PATH"
ENV_CHECKPOINT_PATH = "MATRIX_CHECKPOINT_PATH"
ENV_SKILLS_BASE_DIR = "MATRIX_SKILLS_BASE_DIR"  # root dir for skills/{common,investment,general}
ENV_PIPELINE_MODEL = "PIPELINE_MODEL"
ENV_RATE_LIMIT_PER_SEC = "RATE_LIMIT_PER_SEC"
ENV_MAX_MESSAGE_CHARS = "MAX_MESSAGE_CHARS"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_LOG_DIR = "MATRIX_LOG_DIR"
ENV_MEMORY_SYNC_PATH = "MATRIX_MEMORY_SYNC_PATH"
ENV_JWT_SECRET = "JWT_SECRET"
ENV_ADMIN_PASSWORD = "ADMIN_PASSWORD"
ENV_RAG_DOCS_PATH = "MATRIX_RAG_DOCS_PATH"
ENV_RAG_PERSIST_DIR = "MATRIX_RAG_PERSIST_DIR"
ENV_RAG_EMBED_MODEL = "MATRIX_RAG_EMBED_MODEL"
ENV_RAG_ALLOWED_DIRS = "MATRIX_RAG_ALLOWED_DIRS"

# MCP (Model Context Protocol) client config
ENV_MCP_CONFIG_PATH = "MCP_CONFIG_PATH"
ENV_REFLEXION_MAX_ATTEMPTS = "REFLEXION_MAX_ATTEMPTS"
ENV_OTEL_EXPORTER_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
ENV_OTEL_EXPORT = "OTEL_EXPORT"

# Code sandbox config
ENV_CODE_SANDBOX_ENABLED = "MATRIX_CODE_SANDBOX_ENABLED"
ENV_CODE_SANDBOX_TIMEOUT_SEC = "MATRIX_CODE_SANDBOX_TIMEOUT_SEC"
ENV_CODE_SANDBOX_MAX_MEMORY_MB = "MATRIX_CODE_SANDBOX_MAX_MEMORY_MB"
ENV_CODE_SANDBOX_MAX_OUTPUT_CHARS = "MATRIX_CODE_SANDBOX_MAX_OUTPUT_CHARS"
ENV_CODE_SANDBOX_NETWORK = "MATRIX_CODE_SANDBOX_NETWORK"

# ---- Defaults ----

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# Known models per provider (text/chat models only)
KNOWN_MODELS: dict[str, list[dict[str, str]]] = {
    "deepseek": [
        {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "desc": "快速 · 1M上下文"},
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "desc": "高质量 · 1M上下文"},
    ],
}

# Agnes base URL (used for image/video generation only, not text LLM)
DEFAULT_AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"

# RAG defaults are deliberately allowlisted. Personal assets contains private
# life, attachments, and runtime/system data that must not reach an LLM prompt.
DEFAULT_RAG_ALLOWED_DIRS = (
    "13-财富",
    "20-资料",
    "21-知识",
    "30-项目",
    "31-技能",
)

# Image generation models
IMAGE_MODELS: dict[str, list[dict[str, str]]] = {
    "agnes": [
        {"id": "agnes-image-2.0-flash", "name": "Agnes Image 2.0 Flash", "desc": "免费"},
    ],
}

# Video generation models
VIDEO_MODELS: dict[str, list[dict[str, str]]] = {
    "agnes": [
        {"id": "agnes-video-v2.0", "name": "Agnes Video V2.0", "desc": "免费"},
    ],
}

# Pipeline model: used for internal tasks (classify, plan, reflection, aggregate).
# The default is configured independently from the main execution provider so
# planning and evaluation can be tuned without changing the Runtime contract.
ENV_PIPELINE_PROVIDER = "PIPELINE_PROVIDER"
ENV_PIPELINE_MODEL = "PIPELINE_MODEL"
DEFAULT_PIPELINE_PROVIDER = "deepseek"
DEFAULT_PIPELINE_MODEL = DEFAULT_DEEPSEEK_MODEL

# Providers served by the OpenAI-compatible chat client. Agnes was retired as
# a chat provider — it survives only as the image/video generation tool, which
# is why AGNES_API_KEY is still read in the tools layer.
_API_PROVIDERS = frozenset({"deepseek"})
_API_PROVIDER_KEY_ENVS = {
    "deepseek": ENV_DEEPSEEK_API_KEY,
}


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration loaded from environment variables."""

    root_path: Path
    cache_path: Path
    trace_path: Path
    store_path: Path
    checkpoint_path: str
    skills_base_dir: Path  # root dir for skills/{common,investment,general}
    host: str
    port: int
    agent_provider: str = "deepseek"
    agent_model: str = DEFAULT_DEEPSEEK_MODEL
    agent_max_tokens: int = 8192
    agent_model_timeout_sec: float = 45.0
    deepseek_api_key: str = ""
    anthropic_api_key: str = ""
    agnes_api_key: str = ""
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    agnes_base_url: str = DEFAULT_AGNES_BASE_URL
    memory_max_turns: int = 8
    # Long-term memory retrieval (Phase 1)
    memory_prompt_budget_tokens: int = 800
    memory_retrieval_top_k: int = 8
    memory_full_dump: bool = False  # escape hatch: restore pre-Phase-1 behaviour
    memory_semantic_enabled: bool = True
    # Memory write policy (Phase 2)
    memory_write_decisions: bool = True
    memory_evolution_min_interval_sec: float = 3600.0
    rate_limit_per_sec: float = 5.0
    max_message_chars: int = 8000
    pipeline_provider: str = DEFAULT_PIPELINE_PROVIDER
    pipeline_model: str = DEFAULT_PIPELINE_MODEL
    log_level: int = 20  # INFO
    log_dir: str = ""
    memory_sync_path: str = ""
    jwt_secret: str = ""
    admin_password_hash: str = ""
    rag_docs_path: str = ""
    rag_persist_dir: str = ""
    rag_embed_model: str = "BAAI/bge-small-zh-v1.5"
    rag_allowed_dirs: tuple[str, ...] = DEFAULT_RAG_ALLOWED_DIRS
    mcp_config_path: str = ""
    reflexion_max_attempts: int = 2  # 0 disables Reflexion loop
    otel_exporter_endpoint: str = ""  # OTLP endpoint (e.g. http://localhost:4318/v1/traces)
    otel_export: bool = False  # Enable OTLP export
    code_sandbox_enabled: bool = False  # Code execution sandbox (default off)
    code_sandbox_timeout_sec: int = 30  # Execution timeout
    code_sandbox_max_memory_mb: int = 512  # Memory limit
    code_sandbox_max_output_chars: int = 10000  # Output truncation
    code_sandbox_network: bool = False  # Network access in sandbox
    @property
    def active_api_key(self) -> str:
        return self.deepseek_api_key

    @property
    def llm_available(self) -> bool:
        return self.llm_unavailable_reason == ""

    def _provider_unavailable_reason(self, provider: str) -> str:
        """Why a given provider cannot serve requests, or "" if it can."""
        if provider not in _API_PROVIDERS:
            return (
                f"unsupported provider: {provider} "
                f"(supported: {', '.join(sorted(_API_PROVIDERS))})"
            )
        if not self.active_api_key:
            return f"missing {_API_PROVIDER_KEY_ENVS.get(provider, 'API key')}"
        return ""

    @property
    def llm_unavailable_reason(self) -> str:
        return self._provider_unavailable_reason(self.agent_provider)

    @property
    def pipeline_llm_available(self) -> bool:
        """Whether the pipeline LLM *after fallback* can serve requests."""
        return self.resolved_pipeline_unavailable_reason == ""

    @property
    def pipeline_unavailable_reason(self) -> str:
        """Why the configured pipeline provider cannot run (pre-fallback)."""
        return self._provider_unavailable_reason(self.pipeline_provider)

    @property
    def resolved_pipeline_unavailable_reason(self) -> str:
        return self._provider_unavailable_reason(self.resolved_pipeline_provider)

    @property
    def resolved_pipeline_provider(self) -> str:
        """Pipeline provider actually used, after availability fallback.

        Internal pipeline tasks — notably memory extraction — use the API
        provider when the configured provider is unavailable.
        """
        if self.pipeline_unavailable_reason == "":
            return self.pipeline_provider
        return "deepseek"


def load_config() -> AgentConfig:
    """Load agent configuration from environment variables."""
    root = find_root(Path.cwd())

    # Cache path: PERSONAL_OS_CACHE_PATH > MATRIX_CACHE_PATH > default
    cache_path = _resolve_path(
        [ENV_OS_CACHE_PATH, ENV_CACHE_PATH],
        root / "var" / "cache" / "finance.sqlite",
    )

    # The legacy .jsonl input remains accepted; TraceLogger replaces its
    # suffix with .db before opening the SQLite store.
    # Trace path: PERSONAL_OS_AGENT_TRACE_PATH > MATRIX_TRACE_PATH > default
    trace_path = _resolve_path(
        [ENV_OS_TRACE_PATH, ENV_TRACE_PATH],
        root / "var" / "agent" / "tool-calls.jsonl",
    )

    # Store path: MATRIX_STORE_PATH > default
    store_path = _resolve_path(
        [ENV_STORE_PATH],
        root / "var" / "agent" / "sessions.db",
    )

    # Checkpoint path: MATRIX_CHECKPOINT_PATH > default
    checkpoint_path = os.environ.get(
        ENV_CHECKPOINT_PATH,
        str(root / "var" / "agent" / "checkpoints.db"),
    ).strip() or str(root / "var" / "agent" / "checkpoints.db")

    # Skills base dir: MATRIX_SKILLS_BASE_DIR > default
    skills_base_raw = os.environ.get(ENV_SKILLS_BASE_DIR, "").strip()
    if skills_base_raw:
        skills_base_path = Path(skills_base_raw).expanduser()
        skills_base_dir = skills_base_path if skills_base_path.is_absolute() else root / skills_base_path
    else:
        skills_base_dir = root / ".." / "personal-assets" / "31-技能"

    host, port = load_bind_addr()
    provider = os.environ.get(ENV_AGENT_PROVIDER, "deepseek").strip().lower() or "deepseek"
    model = os.environ.get(ENV_AGENT_MODEL, default_model(provider)).strip() or default_model(provider)

    # Log level: map string to int
    level_str = os.environ.get(ENV_LOG_LEVEL, "INFO").strip().upper()
    level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "WARN": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    log_level = level_map.get(level_str, 20)

    # Log dir: MATRIX_LOG_DIR > default
    log_dir = os.environ.get(ENV_LOG_DIR, "").strip()
    if not log_dir:
        log_dir = str(root / "var" / "log")

    # Memory sync path: MATRIX_MEMORY_SYNC_PATH > default
    memory_sync_path = os.environ.get(ENV_MEMORY_SYNC_PATH, "").strip()
    if not memory_sync_path:
        memory_sync_path = str(root / ".." / "personal-assets" / "92-系统" / "memory")

    # JWT secret: required
    jwt_secret = os.environ.get(ENV_JWT_SECRET, "").strip()
    if not jwt_secret:
        raise ValueError(
            "JWT_SECRET is required. Set it in .env (e.g. run: openssl rand -hex 32)"
        )

    # Admin password: optional, only used for first-run bootstrap (auto-create admin user).
    # If not set and no users exist, the app will log a warning.
    admin_password_hash = ""
    raw_admin = os.environ.get(ENV_ADMIN_PASSWORD, "").strip()
    if raw_admin:
        from .auth import hash_password
        admin_password_hash = hash_password(raw_admin)

    # RAG config
    rag_docs_path = str(_resolve_path(
        [ENV_RAG_DOCS_PATH],
        root / ".." / "personal-assets",
    ))
    rag_persist_dir = os.environ.get(ENV_RAG_PERSIST_DIR, "").strip()
    if not rag_persist_dir:
        rag_persist_dir = str(root / "var" / "rag")
    rag_embed_model = os.environ.get(ENV_RAG_EMBED_MODEL, "BAAI/bge-small-zh-v1.5").strip() or "BAAI/bge-small-zh-v1.5"
    rag_allowed_dirs = _parse_csv_env(
        ENV_RAG_ALLOWED_DIRS,
        DEFAULT_RAG_ALLOWED_DIRS,
    )

    # MCP config path: MCP_CONFIG_PATH > default
    mcp_config_path = os.environ.get(ENV_MCP_CONFIG_PATH, "").strip()
    if not mcp_config_path:
        mcp_config_path = str(root / "config" / "mcp_servers.json")

    reflexion_max = clamp_int_env(ENV_REFLEXION_MAX_ATTEMPTS, 2, 0, 5)

    # OTel export
    otel_endpoint = os.environ.get(ENV_OTEL_EXPORTER_ENDPOINT, "").strip()
    otel_export = os.environ.get(ENV_OTEL_EXPORT, "").strip().lower() in ("1", "true", "yes")

    # Code sandbox config
    code_sandbox_enabled = os.environ.get(
        ENV_CODE_SANDBOX_ENABLED, ""
    ).strip().lower() in ("1", "true", "yes")
    code_sandbox_timeout_sec = clamp_int_env(
        ENV_CODE_SANDBOX_TIMEOUT_SEC, 30, 5, 120
    )
    code_sandbox_max_memory_mb = clamp_int_env(
        ENV_CODE_SANDBOX_MAX_MEMORY_MB, 512, 128, 4096
    )
    code_sandbox_max_output_chars = clamp_int_env(
        ENV_CODE_SANDBOX_MAX_OUTPUT_CHARS, 10000, 1000, 50000
    )
    code_sandbox_network = os.environ.get(
        ENV_CODE_SANDBOX_NETWORK, ""
    ).strip().lower() in ("1", "true", "yes")

    return AgentConfig(
        root_path=root,
        cache_path=cache_path,
        trace_path=trace_path,
        store_path=store_path,
        checkpoint_path=checkpoint_path,
        skills_base_dir=skills_base_dir,
        host=host,
        port=port,
        agent_provider=provider,
        agent_model=model,
        agent_max_tokens=clamp_int_env(ENV_AGENT_MAX_TOKENS, 8192, 128, 8192),
        agent_model_timeout_sec=clamp_float_env(ENV_AGENT_MODEL_TIMEOUT_SEC, 45.0, 5.0, 180.0),
        deepseek_api_key=os.environ.get(ENV_DEEPSEEK_API_KEY, "").strip(),
        anthropic_api_key=os.environ.get(ENV_ANTHROPIC_API_KEY, "").strip(),
        agnes_api_key=os.environ.get(ENV_AGNES_API_KEY, "").strip(),
        deepseek_base_url=os.environ.get(ENV_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_BASE_URL).strip()
        or DEFAULT_DEEPSEEK_BASE_URL,
        agnes_base_url=os.environ.get(ENV_AGNES_BASE_URL, DEFAULT_AGNES_BASE_URL).strip()
        or DEFAULT_AGNES_BASE_URL,
        memory_max_turns=clamp_int_env(ENV_MEMORY_MAX_TURNS, 8, 1, 30),
        memory_prompt_budget_tokens=clamp_int_env(
            "MATRIX_MEMORY_PROMPT_BUDGET", 800, 100, 4000,
        ),
        memory_retrieval_top_k=clamp_int_env(
            "MATRIX_MEMORY_TOP_K", 8, 1, 30,
        ),
        memory_full_dump=_env_flag("MATRIX_MEMORY_FULL_DUMP", False),
        memory_semantic_enabled=_env_flag("MATRIX_MEMORY_SEMANTIC", True),
        memory_write_decisions=_env_flag("MATRIX_MEMORY_WRITE_DECISIONS", True),
        memory_evolution_min_interval_sec=clamp_float_env(
            "MATRIX_MEMORY_EVOLUTION_INTERVAL", 3600.0, 0.0, 86400.0,
        ),
        rate_limit_per_sec=clamp_float_env(ENV_RATE_LIMIT_PER_SEC, 5.0, 0.5, 60.0),
        max_message_chars=clamp_int_env(ENV_MAX_MESSAGE_CHARS, 8000, 500, 50000),
        pipeline_provider=os.environ.get(ENV_PIPELINE_PROVIDER, DEFAULT_PIPELINE_PROVIDER).strip().lower()
        or DEFAULT_PIPELINE_PROVIDER,
        pipeline_model=os.environ.get(ENV_PIPELINE_MODEL, DEFAULT_PIPELINE_MODEL).strip()
        or DEFAULT_PIPELINE_MODEL,
        log_level=log_level,
        log_dir=log_dir,
        memory_sync_path=memory_sync_path,
        jwt_secret=jwt_secret,
        admin_password_hash=admin_password_hash,
        rag_docs_path=rag_docs_path,
        rag_persist_dir=rag_persist_dir,
        rag_embed_model=rag_embed_model,
        rag_allowed_dirs=rag_allowed_dirs,
        mcp_config_path=mcp_config_path,
        reflexion_max_attempts=reflexion_max,
        otel_exporter_endpoint=otel_endpoint,
        otel_export=otel_export,
        code_sandbox_enabled=code_sandbox_enabled,
        code_sandbox_timeout_sec=code_sandbox_timeout_sec,
        code_sandbox_max_memory_mb=code_sandbox_max_memory_mb,
        code_sandbox_max_output_chars=code_sandbox_max_output_chars,
        code_sandbox_network=code_sandbox_network,
    )


def find_root(start: Path) -> Path:
    """Find the project root by locating pyproject.toml."""
    current = start.resolve()
    for path in (current, *current.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("matrix root not found: run from inside the repository")


def load_bind_addr() -> tuple[str, int]:
    """Load bind address: MATRIX_AGENT_ADDR > PERSONAL_OS_AGENT_ADDR > AGENT_HOST:AGENT_PORT > 127.0.0.1:7101."""
    raw_addr = os.environ.get(ENV_OS_AGENT_ADDR) or os.environ.get(ENV_AGENT_ADDR)
    if raw_addr:
        return parse_addr(raw_addr, ENV_OS_AGENT_ADDR if os.environ.get(ENV_OS_AGENT_ADDR) else ENV_AGENT_ADDR)
    host = os.environ.get(ENV_AGENT_HOST, "127.0.0.1").strip() or "127.0.0.1"
    port_raw = os.environ.get(ENV_AGENT_PORT, "7101").strip() or "7101"
    return parse_addr(f"{host}:{port_raw}", f"{ENV_AGENT_HOST}/{ENV_AGENT_PORT}")


def default_model(provider: str) -> str:
    return DEFAULT_DEEPSEEK_MODEL


def clamp_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def clamp_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _parse_csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated allowlist while keeping a safe default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


def parse_addr(raw: str, env_name: str = ENV_AGENT_ADDR) -> tuple[str, int]:
    if ":" not in raw:
        raise ValueError(f"{env_name} must be host:port")
    host, port_raw = raw.rsplit(":", 1)
    if not host:
        raise ValueError(f"{env_name} host is required")
    try:
        port = int(port_raw)
    except ValueError as err:
        raise ValueError(f"{env_name} port must be an integer") from err
    if port < 0 or port > 65535:
        raise ValueError(f"{env_name} port out of range")
    return host, port


def _resolve_path(env_names: list[str], default: Path) -> Path:
    """Resolve a path from env vars with priority order, falling back to default."""
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    return default.expanduser().resolve()
