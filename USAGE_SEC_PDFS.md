# SEC Filings as PDFs - Usage Guide

The enhanced `get_sec_filings` function now supports downloading SEC filings as PDFs using [sec-api.io](https://sec-api.io/resources/download-sec-filings-as-pdfs) for superior multimodal analysis.

## Setup

### 1. Install Required Package
```bash
pip install sec-api
```

### 2. Get SEC API Key
1. Visit [sec-api.io](https://sec-api.io)
2. Sign up for a free account
3. Get your API key from the dashboard

### 3. Configure API Key
Set your SEC API key as an environment variable:
```bash
export SEC_API_KEY="your_sec_api_key_here"
```

Or add it to your `.env` file:
```
SEC_API_KEY=your_sec_api_key_here
```

## Features

### PDF Generation Benefits
According to [sec-api.io documentation](https://sec-api.io/resources/download-sec-filings-as-pdfs):

✅ **Optimized images**: Images are scaled and optimized for high-quality printing  
✅ **File size management**: Invisible inline XBRL tags are removed to reduce file size  
✅ **Preservation of original content**: All original content is retained without alteration  
✅ **Links and Fonts**: Links are standardized, fonts processed for visibility  
✅ **Shareable and printable PDFs**: Perfect for archiving and multimodal analysis  

### Supported Filing Types
- **10-K**: Annual reports (converted from HTML to optimized PDF)
- **10-Q**: Quarterly reports  
- **8-K**: Current reports
- **DEF 14A**: Proxy statements with properly scaled voting cards
- **Form 4**: Insider trading reports (converted from XML)
- All other SEC filing types

## Usage

### Basic Usage
```python
from agents.data_collector import DataCollectorTools
from config.config import FinSightConfig

# Configure with SEC API key
config = FinSightConfig(sec_api_key="your_api_key")
collector = DataCollectorTools(config)

# Download filings as PDFs
result = collector.get_sec_filings(
    symbol="AAPL",
    filing_types=["10-K", "10-Q"],
    max_filings=2
)

# Access PDF files
for filing in result["data"]["filings_downloaded"]:
    print(f"PDF saved to: {filing['pdf_path']}")
    print(f"File size: {filing['file_size_mb']} MB")
    print(f"Original URL: {filing['original_url']}")
```

### Fallback Behavior
If SEC API is not available or not configured:
- ✅ **Automatic fallback** to HTML parsing (original functionality)
- ✅ **No breaking changes** - existing code continues to work
- ⚠️ Limited table structure preservation in HTML mode

### Output Structure
```python
{
    "success": True,
    "data": {
        "symbol": "AAPL",
        "format": "pdf",  # or "html" if fallback
        "filings_downloaded": [
            {
                "filing_type": "10-K",
                "symbol": "AAPL", 
                "filing_date": "2024-01-31",
                "pdf_path": "./outputs/sec_filings/AAPL/pdfs/AAPL_10-K_20240131.pdf",
                "original_url": "https://www.sec.gov/Archives/edgar/data/320193/...",
                "file_size_mb": 4.2,
                "format": "pdf"
            }
        ],
        "filing_summaries": [...],
        "download_path": "./outputs/sec_filings/AAPL"
    }
}
```

## Advantages Over HTML Parsing

| Feature | PDF Mode | HTML Mode |
|---------|----------|-----------|
| **Table Structure** | ✅ Preserved with formatting | ❌ Lost in text conversion |
| **Charts & Images** | ✅ High-quality, properly scaled | ❌ Often broken or missing |
| **Multimodal Analysis** | ✅ Perfect for GPT-4V/Claude | ⚠️ Limited visual content |
| **Print Quality** | ✅ Professional, shareable | ❌ Poor formatting |
| **File Size** | ✅ Optimized, XBRL cleaned | ❌ Bloated with metadata |
| **Voting Cards** | ✅ Properly scaled for A4 | ❌ Often cut off |

## Cost Considerations
- SEC API has usage-based pricing
- Free tier includes generous limits
- Automatic fallback prevents system failures
- PDFs are cached locally to avoid re-downloads

## Multimodal Analysis Ready
The PDF files are optimized for:
- **GPT-4 Vision**: Direct PDF analysis with preserved formatting
- **Claude**: Document understanding with visual elements
- **Manual Review**: Professional-quality printable documents
- **Archival**: Long-term storage with original formatting

## Troubleshooting

### Common Issues
1. **API Key not configured**: Falls back to HTML mode automatically
2. **Network timeouts**: Large filings may take time to convert
3. **URL extraction failed**: Some older filings may not convert

### Debug Logging
Enable debug logging to see PDF conversion progress:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Support
- [SEC API Documentation](https://sec-api.io/resources/download-sec-filings-as-pdfs)
- [GitHub Issues](your-repo-issues-link) 