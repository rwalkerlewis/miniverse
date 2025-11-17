"""
Miniverse Configuration

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # LLM Provider Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-nano")

    # API Keys
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # Local LLM Configuration (for Ollama, LM Studio, vLLM, etc.)
    # Set base URL for OpenAI-compatible local servers
    # Example: http://localhost:11434/v1 for Ollama
    LOCAL_LLM_BASE_URL: str | None = os.getenv("LOCAL_LLM_BASE_URL")
    # Optional API key for local servers that require authentication
    LOCAL_LLM_API_KEY: str | None = os.getenv("LOCAL_LLM_API_KEY", "not-needed")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://localhost/miniverse")

    # Simulation Configuration
    DEFAULT_TICK_COUNT: int = int(os.getenv("DEFAULT_TICK_COUNT", "50"))
    TICK_DURATION_SECONDS: int = int(os.getenv("TICK_DURATION_SECONDS", "10"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    PROMPTS_DIR: Path = PROJECT_ROOT / "examples" / "prompts"

    @classmethod
    def validate(cls) -> None:
        """Validate configuration and raise errors if required values are missing."""
        # For local LLM providers, check if base URL is set
        if cls.LLM_PROVIDER == "local" and not cls.LOCAL_LLM_BASE_URL:
            raise ValueError(
                "LOCAL_LLM_BASE_URL is required when using the 'local' provider. "
                "Set it to your local LLM server endpoint (e.g., http://localhost:11434/v1 for Ollama)"
            )

        # For cloud providers, check API keys (but skip if using local with base URL)
        is_using_local = cls.LOCAL_LLM_BASE_URL is not None
        
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY and not is_using_local:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using the 'anthropic' provider"
            )

        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY and not is_using_local:
            raise ValueError(
                "OPENAI_API_KEY is required when using the 'openai' provider. "
                "For local LLMs, set LOCAL_LLM_BASE_URL instead."
            )

    @classmethod
    def display(cls) -> str:
        """Return a formatted string showing current configuration."""
        lines = [
            "Miniverse Configuration:",
            f"  LLM Provider: {cls.LLM_PROVIDER}",
            f"  LLM Model: {cls.LLM_MODEL}",
            f"  Database: {cls.DATABASE_URL}",
            f"  Default Ticks: {cls.DEFAULT_TICK_COUNT}",
            f"  Tick Duration: {cls.TICK_DURATION_SECONDS}s",
        ]
        return "\n".join(lines)
