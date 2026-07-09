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
ENV_SKILLS_DIR = "MATRIX_SKILLS_DIR"
ENV_SKILLS_BASE_DIR = "MATRIX_SKILLS_BASE_DIR"  # root dir for skills/{common,investment,general}
ENV_RATE_LIMIT_PER_SEC = "RATE_LIMIT_PER_SEC"
ENV_MAX_MESSAGE_CHARS = "MAX_MESSAGE_CHARS"

# ---- Defaults ----

DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
DEFAULT_AGNES_MODEL = "agnes-2.0-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"

# Known models per provider (text/chat models only)
KNOWN_MODELS: dict[str, list[dict[str, str]]] = {
    "deepseek": [
        {"id": "deepseek-v4-flash", "name": "DeepSeek V4 Flash", "desc": "快速 · 1M上下文"},
        {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "desc": "高质量 · 1M上下文"},
    ],
    "agnes": [
        {"id": "agnes-2.0-flash", "name": "Agnes 2.0 Flash", "desc": "推荐 · 最新"},
    ],
    "anthropic": [
        {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet", "desc": "推荐"},
        {"id": "claude-3-5-haiku-latest", "name": "Claude 3.5 Haiku", "desc": "快速"},
    ],
}

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

# Pipeline model: used for internal tasks (classify, plan, reflection)
# Defaults to Agnes 2.0 Flash for speed and cost efficiency
ENV_PIPELINE_PROVIDER = "PIPELINE_PROVIDER"
ENV_PIPELINE_MODEL = "PIPELINE_MODEL"
DEFAULT_PIPELINE_PROVIDER = "agnes"
DEFAULT_PIPELINE_MODEL = "agnes-2.0-flash"


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration loaded from environment variables."""

    root_path: Path
    cache_path: Path
    trace_path: Path
    store_path: Path
    checkpoint_path: str
    skills_dir: Path  # deprecated, kept for backward compat; use skills_base_dir
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
    rate_limit_per_sec: float = 5.0
    max_message_chars: int = 8000
    pipeline_provider: str = DEFAULT_PIPELINE_PROVIDER
    pipeline_model: str = DEFAULT_PIPELINE_MODEL

    @property
    def active_api_key(self) -> str:
        if self.agent_provider == "deepseek":
            return self.deepseek_api_key
        if self.agent_provider == "anthropic":
            return self.anthropic_api_key
        if self.agent_provider == "agnes":
            return self.agnes_api_key
        return ""

    @property
    def llm_available(self) -> bool:
        return self.llm_unavailable_reason == ""

    @property
    def llm_unavailable_reason(self) -> str:
        if self.agent_provider not in {"deepseek", "anthropic", "agnes"}:
            return f"unsupported AGENT_PROVIDER: {self.agent_provider}"
        if not self.active_api_key:
            key_map = {
                "deepseek": ENV_DEEPSEEK_API_KEY,
                "anthropic": ENV_ANTHROPIC_API_KEY,
                "agnes": ENV_AGNES_API_KEY,
            }
            key_name = key_map.get(self.agent_provider, "API key")
            return f"missing {key_name}"
        return ""


def load_config() -> AgentConfig:
    """Load agent configuration from environment variables."""
    root = find_root(Path.cwd())

    # Cache path: PERSONAL_OS_CACHE_PATH > MATRIX_CACHE_PATH > default
    cache_path = _resolve_path(
        [ENV_OS_CACHE_PATH, ENV_CACHE_PATH],
        root / "var" / "cache" / "finance.sqlite",
    )

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

    # Skills dir: MATRIX_SKILLS_DIR > default
    skills_raw = os.environ.get(ENV_SKILLS_DIR, "").strip()
    if skills_raw:
        skills_path = Path(skills_raw).expanduser()
        skills_dir = skills_path if skills_path.is_absolute() else root / skills_path
    else:
        skills_dir = root / "skills" / "investment"

    # Skills base dir: MATRIX_SKILLS_BASE_DIR > default
    skills_base_raw = os.environ.get(ENV_SKILLS_BASE_DIR, "").strip()
    if skills_base_raw:
        skills_base_path = Path(skills_base_raw).expanduser()
        skills_base_dir = skills_base_path if skills_base_path.is_absolute() else root / skills_base_path
    else:
        skills_base_dir = root / ".." / "personal-assets" / "技能"

    host, port = load_bind_addr()
    provider = os.environ.get(ENV_AGENT_PROVIDER, "deepseek").strip().lower() or "deepseek"
    model = os.environ.get(ENV_AGENT_MODEL, default_model(provider)).strip() or default_model(provider)

    return AgentConfig(
        root_path=root,
        cache_path=cache_path,
        trace_path=trace_path,
        store_path=store_path,
        checkpoint_path=checkpoint_path,
        skills_dir=skills_dir,
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
        rate_limit_per_sec=clamp_float_env(ENV_RATE_LIMIT_PER_SEC, 5.0, 0.5, 60.0),
        max_message_chars=clamp_int_env(ENV_MAX_MESSAGE_CHARS, 8000, 500, 50000),
        pipeline_provider=os.environ.get(ENV_PIPELINE_PROVIDER, DEFAULT_PIPELINE_PROVIDER).strip().lower()
        or DEFAULT_PIPELINE_PROVIDER,
        pipeline_model=os.environ.get(ENV_PIPELINE_MODEL, DEFAULT_PIPELINE_MODEL).strip()
        or DEFAULT_PIPELINE_MODEL,
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
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "agnes":
        return DEFAULT_AGNES_MODEL
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