"""
FinSight Configuration Module

Contains configuration management for the FinSight system including:
- Environment variable handling
- LLM configurations
- Agent configurations
- System settings
"""

from .config import FinSightConfig, get_llm_config, AGENT_CONFIGS

__all__ = [
    "FinSightConfig",
    "get_llm_config",
    "AGENT_CONFIGS",
] 