"""Configuration for Communication Health Analysis."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class CommunicationHealthConfig(BaseSettings):
    """Configuration settings for communication health analysis."""
    
    # LLM Configuration
    llm_provider: str = "nvidia"
    nvidia_api_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: Optional[str] = None
    nvidia_model: str = "deepseek-ai/deepseek-v3.1-terminus"
    nvidia_temperature: float = 0.2
    nvidia_top_p: float = 0.7
    nvidia_max_tokens: int = 8192
    nvidia_enable_thinking: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


def get_config() -> CommunicationHealthConfig:
    """Get configuration instance."""
    return CommunicationHealthConfig()

