"""
FinSight Multi-Agent Financial Research System

An advanced multi-agent system for comprehensive equity research that enhances FinRobot
capabilities with modular agents for data collection, multimodal analysis, reasoning,
report writing, and verification.
"""

__version__ = "1.0.0"
__author__ = "FinSight Team"

# Import main components for easy access
from .finsight_system import FinSightOrchestrator
from .config.config import FinSightConfig

__all__ = [
    "FinSightOrchestrator",
    "FinSightConfig",
] 