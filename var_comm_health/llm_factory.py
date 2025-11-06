"""Factory for creating LLM instances based on configuration."""

from langchain_core.language_models import BaseChatModel

from .config import get_config


def create_llm() -> BaseChatModel:
    """
    Create an LLM instance based on configuration.
    
    Returns:
        LLM instance (ChatNVIDIA)
    """
    config = get_config()
    
    if config.llm_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        
        if not config.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required when using NVIDIA provider")
        
        return ChatNVIDIA(
            model=config.nvidia_model,
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_api_url,
            temperature=config.nvidia_temperature,
            top_p=config.nvidia_top_p,
            max_tokens=config.nvidia_max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": config.nvidia_enable_thinking}},
        )
    
    
    
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")

