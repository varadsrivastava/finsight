import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Only load .env if it exists (don't require it)
# if os.path.exists(".env"):
#     load_dotenv()

class FinSightConfig(BaseModel):
    """Configuration for the FinSight system"""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    finnhub_api_key: Optional[str] = Field(default=None)
    sec_api_key: Optional[str] = Field(default=None)
    newsapi_key: Optional[str] = Field(default=None)
    
    # Model Configurations
    primary_model: str = Field(default="gpt-4-0125-preview")
    multimodal_model: str = Field(default="gpt-4-vision-preview")
    reasoning_model: str = Field(default="gpt-4-0125-preview")
    verification_model: str = Field(default="claude-3-sonnet-20240229")
    
    # SEC EDGAR settings
    edgar_company_name: str = Field(default="FinSight")
    edgar_email: str = Field(default="user@finsight.ai")
    
    # Output paths
    charts_output_path: str = Field(default="outputs/charts")
    sec_filings_path: str = Field(default="outputs/sec_filings")
    reports_output_path: str = Field(default="outputs/reports")
    vector_db_path: str = Field(default="data/vectordb")
    shared_memory_path: str = Field(default="data/shared_memory.json")
    
    # Agent settings
    max_consecutive_auto_reply: int = Field(default=10)
    temperature: float = Field(default=0.3)
    timeout: int = Field(default=300)
    
    # Data Source settings
    yahoo_finance_period: str = Field(default="2y")
    news_lookback_days: int = Field(default=30)
    
    # Verification Settings
    fact_check_threshold: float = Field(default=0.8)
    verification_rounds: int = Field(default=2)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        # Load environment variables first
        load_dotenv()
        
        # Get values from environment
        env_data = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "finnhub_api_key": os.getenv("FINNHUB_API_KEY"),
            "sec_api_key": os.getenv("SEC_API_KEY"),
            "newsapi_key": os.getenv("NEWSAPI_KEY"),
            "edgar_company_name": os.getenv("EDGAR_COMPANY_NAME", "FinSight"),
            "edgar_email": os.getenv("EDGAR_EMAIL", "user@finsight.ai"),
            "charts_output_path": os.getenv("CHARTS_OUTPUT_PATH", "outputs/charts"),
            "max_consecutive_auto_reply": int(os.getenv("MAX_CONSECUTIVE_AUTO_REPLY", "10")),
            "temperature": float(os.getenv("TEMPERATURE", "0.3")),
            "timeout": int(os.getenv("TIMEOUT", "300"))
        }
        
        # Update with any provided data
        env_data.update(data)
        
        # Initialize the model
        super().__init__(**env_data)
        
        # Create output directories
        os.makedirs(self.charts_output_path, exist_ok=True)
        os.makedirs(self.sec_filings_path, exist_ok=True)
        os.makedirs(self.reports_output_path, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.shared_memory_path), exist_ok=True)

def get_llm_config(model_name: str = None) -> dict:
    """Get LLM configuration for AutoGen"""
    config = FinSightConfig()
    
    if not model_name:
        model_name = config.primary_model
        
    return {
        "config_list": [
            {
                "model": model_name,
                "api_key": config.openai_api_key if "gpt" in model_name else config.anthropic_api_key,
                "api_type": "openai" if "gpt" in model_name else "anthropic",
            }
        ],
        "temperature": config.temperature,
        "timeout": config.timeout,
    }

# Specialized configs for different agents
def _get_agent_configs():
    """Get agent configurations - called dynamically to use current config"""
    config = FinSightConfig()
    return {
        "data_collector": get_llm_config("gpt-4-0125-preview"),
        "multimodal_analyzer": get_llm_config("gpt-4-vision-preview"),
        "reasoning_agent": get_llm_config("gpt-4-0125-preview"),
        "report_writer": get_llm_config("gpt-4-0125-preview"),
        "verifier": {
            "config_list": [
                {
                    "model": "claude-3-sonnet-20240229",
                    "api_key": config.anthropic_api_key,
                    "api_type": "anthropic",
                }
            ],
            "temperature": 0.1,
            "timeout": 300,
        }
    }

# For backward compatibility
AGENT_CONFIGS = _get_agent_configs() 