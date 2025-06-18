# FinSight Multi-Agent Financial Research System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An advanced multi-agent system for comprehensive equity research that enhances FinRobot capabilities with modular agents for data collection, multimodal analysis, reasoning, report writing, and verification.

## 🚀 Quick Start with Virtual Environment

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Automated Setup (Recommended)

We provide automated setup scripts for easy virtual environment creation:

#### Windows
```cmd
setup.bat
```

#### macOS/Linux
```bash
chmod +x setup.sh
./setup.sh
```

#### Manual Setup (Cross-Platform)
```bash
python setup_venv.py
```

### Manual Virtual Environment Setup

If you prefer to set up the virtual environment manually:

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd finsight
```

#### 2. Create Virtual Environment

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 4. Configure Environment (Optional)

The system works with sensible defaults out of the box. To enable enhanced features, set environment variables:

```bash
# Option 1: Create a .env file (recommended for development)
# Create .env file with your API keys:
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
EDGAR_USER_AGENT="Your Company (email@domain.com)"

# Option 2: Set system environment variables
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ... etc
```

#### 5. Create Output Directories
```bash
mkdir -p data outputs/reports outputs/charts
```

## 🏗️ System Architecture

The FinSight system consists of five specialized agents:

1. **DataCollectorAgent**: Multi-source financial data collection (Yahoo Finance, SEC EDGAR, ESG, news)
2. **MultimodalAnalyzerAgent**: Chart creation and visual analysis using FinTral capabilities
3. **ReasoningAgent (FinR1)**: Advanced financial analysis and investment insights
4. **ReportWriterAgent**: Professional equity research report generation
5. **VerifierAgent**: Fact-checking and validation

## 📁 Project Structure

```
finsight/
├── agents/                 # Agent modules
│   ├── data_collector.py
│   ├── multimodal_analyzer.py
│   ├── reasoning_agent.py
│   ├── report_writer.py
│   └── verifier.py
├── config/                 # Configuration management
│   └── config.py
├── shared_memory/          # Shared memory system
│   └── memory_manager.py
├── finsight_system.py      # Main orchestrator
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── setup_venv.py          # Virtual environment setup

└── README.md              # This file
```

## 🔧 Configuration

FinSight works out of the box with sensible defaults. All API keys and configurations are optional - the system will use fallback methods when external services aren't available.

### API Keys (All Optional)

1. **OpenAI API Key** (Enables GPT-4 features)
   - Get from: https://platform.openai.com/api-keys
   - Used for: Enhanced reasoning and multimodal analysis

2. **Anthropic API Key** (Enables Claude verification)
   - Get from: https://console.anthropic.com/account/keys
   - Used for: Advanced fact-checking and verification

3. **Finnhub API Key** (Enhances financial data)
   - Get from: https://finnhub.io/register
   - Used for: Real-time financial data and news

4. **NewsAPI Key** (Enhances news collection)
   - Get from: https://newsapi.org/register
   - Used for: Additional news sources

### Environment Variables (Optional Overrides)

You can override defaults by setting environment variables. Create a `.env` file or set them in your system:

```bash
# API Keys (optional)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
NEWSAPI_KEY=your_newsapi_key_here

# SEC EDGAR User Agent (optional - has sensible default)
EDGAR_USER_AGENT="Your Company (email@domain.com)"

# Other settings (optional - have sensible defaults)
TEMPERATURE=0.3
MAX_CONSECUTIVE_AUTO_REPLY=10
```

## 🚀 Usage

### Activate Virtual Environment

**Windows:**
```cmd
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### Run FinSight

#### Option 1: Direct Python Execution
```bash
python finsight_system.py
```

#### Option 2: Command Line Tool (if installed)
```bash
finsight
```

#### Option 3: Python Import
```python
from finsight import FinSightOrchestrator

# Initialize the system
orchestrator = FinSightOrchestrator()

# Conduct research
result = orchestrator.conduct_comprehensive_research("AAPL")
```

### Example Research

```python
import asyncio
from finsight import FinSightOrchestrator

async def analyze_stock():
    orchestrator = FinSightOrchestrator()
    
    # Single stock analysis
    result = await orchestrator.conduct_comprehensive_research("AAPL")
    
    # Batch analysis
    batch_result = await orchestrator.conduct_batch_research(["AAPL", "MSFT", "GOOGL"])
    
    return result, batch_result

# Run the analysis
result, batch_result = asyncio.run(analyze_stock())
```

## 🧪 Development and Testing

### Install Development Dependencies
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Code Formatting
```bash
black .
flake8 .
```

### Jupyter Notebook Development
```bash
jupyter notebook test_data_collector.ipynb
```

## 📊 Output Files

The system generates various output files:

- **Reports**: `outputs/reports/` - PDF equity research reports
- **Charts**: `outputs/charts/` - Financial visualizations
- **Data**: `data/` - Cached financial data and vector database
- **Logs**: `finsight.log` - System operation logs

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and all dependencies are installed
2. **API Key Errors**: Check `.env` file configuration and API key validity
3. **Permission Errors**: Ensure write permissions for output directories
4. **Version Conflicts**: Use the exact dependency versions in `requirements.txt`

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf .venv  # or rmdir /s .venv on Windows
python setup_venv.py
```

### Verify Installation
```bash
python -c "import finsight; print('FinSight installed successfully!')"
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the logs in `finsight.log`
3. Open an issue on GitHub with detailed error information
4. Include your Python version, OS, and the full error traceback

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)

---

**Happy Financial Research! 📈** 