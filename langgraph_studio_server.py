"""LangGraph Studio server file for Communication Health Analysis workflow."""

import warnings

# Suppress DeepSeek model type warning - the model works fine despite the warning
warnings.filterwarnings(
    "ignore",
    message="Found deepseek-ai/deepseek-v3.1-terminus in available_models, but type is unknown",
    category=UserWarning,
    module="langchain_nvidia_ai_endpoints"
)

from workflow import create_communication_health_workflow
from llm_factory import create_llm

# Create LLM instance
llm = create_llm()

# Create and compile the workflow
graph = create_communication_health_workflow(llm)

