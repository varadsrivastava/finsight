import os
import asyncio
import yfinance as yf
from sec_edgar_downloader import Downloader
import finnhub
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import json
import autogen
from autogen import AssistantAgent
from dataclasses import dataclass
import logging
import re
from transformers import pipeline
import torch
from pathlib import Path
import zipfile
from bs4 import BeautifulSoup


from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig

logger = logging.getLogger(__name__)

@dataclass
class DataRequest:
    """Structure for data requests"""
    symbol: str
    company_name: Optional[str] = None
    data_types: List[str] = None  # ['financial', 'news', 'esg', 'trends', 'sec', 'earnings', 'competitors']
    lookback_days: int = 30
    include_fundamentals: bool = True
    competitors: List[str] = None
    filing_types: List[str] = None  # ['10-K', '10-Q', '8-K', 'DEF 14A']

class DataCollectorTools:
    """Tools for the DataCollector agent"""
    
    def __init__(self, config: FinSightConfig):
        self.config = config
        self.finnhub_client = finnhub.Client(api_key=config.finnhub_api_key) if config.finnhub_api_key else None
        # Note: edgar_downloader will be created per-company in get_sec_filings()
        # since each company needs its own download path
        # self.sec_filings_path = os.path.join(config.charts_output_path, "sec_filings")
        self.sec_filings_path = config.sec_filings_path
        os.makedirs(self.sec_filings_path, exist_ok=True)
        
        # Initialize sentiment analysis model
        self.sentiment_model = None
        self._init_sentiment_model()
    
    def _init_sentiment_model(self):
        """Initialize HuggingFace sentiment analysis model"""
        try:
            # Use FinBERT for financial sentiment analysis
            model_name = "ProsusAI/finbert"
            device = 0 if torch.cuda.is_available() else -1
            
            self.sentiment_model = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=device,
                return_all_scores=True
            )
            logger.info(f"Initialized sentiment model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Could not initialize FinBERT, falling back to simple model: {e}")
            self.sentiment_model = None
    
    def get_stock_data(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """Get stock price and volume data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period)
            info = ticker.info
            
            price_data = {
                "symbol": symbol,
                "current_price": float(info.get("currentPrice", 0)),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": float(info.get("fiftyTwoWeekHigh", 0)),
                "52_week_low": float(info.get("fiftyTwoWeekLow", 0)),
                "beta": info.get("beta"),
                "historical_data": {
                    "dates": [str(date) for date in hist_data.index.strftime("%Y-%m-%d")],
                    "prices": hist_data["Close"].tolist(),
                    "volumes": hist_data["Volume"].tolist(),
                    "highs": hist_data["High"].tolist(),
                    "lows": hist_data["Low"].tolist()
                }
            }
            
            return {"success": True, "data": price_data}
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_company_financials(self, symbol: str) -> Dict[str, Any]:
        """Get company financial statements"""
        try:
            ticker = yf.Ticker(symbol)
            
            financials = {
                "symbol": symbol,
                "income_statement": {},
                "balance_sheet": {},
                "cash_flow": {},
                "key_metrics": {}
            }
            
            # Income Statement
            if hasattr(ticker, 'financials') and ticker.financials is not None:
                inc_stmt = ticker.financials.fillna(0)
                financials["income_statement"] = {
                    str(col): inc_stmt[col].to_dict() for col in inc_stmt.columns
                }
            
            # Balance Sheet
            if hasattr(ticker, 'balance_sheet') and ticker.balance_sheet is not None:
                balance = ticker.balance_sheet.fillna(0)
                financials["balance_sheet"] = {
                    str(col): balance[col].to_dict() for col in balance.columns
                }
            
            # Cash Flow
            if hasattr(ticker, 'cashflow') and ticker.cashflow is not None:
                cf = ticker.cashflow.fillna(0)
                financials["cash_flow"] = {
                    str(col): cf[col].to_dict() for col in cf.columns
                }
            
            # Key metrics
            info = ticker.info
            financials["key_metrics"] = {
                "revenue_growth": info.get("revenueGrowth"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio")
            }
            
            return {"success": True, "data": financials}
            
        except Exception as e:
            logger.error(f"Error getting financials for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_company_news(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """Get recent company news with enhanced sentiment analysis"""
        try:
            news_data = {"symbol": symbol, "articles": []}
            
            if self.finnhub_client:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                to_date = datetime.now().strftime("%Y-%m-%d")
                
                news = self.finnhub_client.company_news(symbol, _from=from_date, to=to_date)
                
                for article in news[:20]:
                    headline = article.get("headline", "")
                    summary = article.get("summary", "")
                    combined_text = f"{headline} {summary}"
                    
                    sentiment_result = self._analyze_sentiment(combined_text)
                    
                    news_data["articles"].append({
                        "headline": headline,
                        "summary": summary,
                        "url": article.get("url", ""),
                        "source": article.get("source", ""),
                        "datetime": article.get("datetime", 0),
                        "sentiment": sentiment_result["label"],
                        "sentiment_score": sentiment_result["score"],
                        "sentiment_confidence": sentiment_result["confidence"],
                        "sentiment_model": sentiment_result.get("model", "unknown")
                    })
            
            return {"success": True, "data": news_data}
            
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_esg_data(self, symbol: str) -> Dict[str, Any]:
        """Get ESG data (placeholder - would integrate with actual ESG providers)"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            esg_data = {
                "symbol": symbol,
                "sustainability_score": None,
                "environment_score": None,
                "social_score": None,
                "governance_score": None,
                "esg_risk_rating": None,
                "controversies": [],
                "carbon_footprint": None,
                "employee_count": info.get("fullTimeEmployees"),
                "board_diversity": None,
                "executive_compensation": None
            }
            
            return {"success": True, "data": esg_data, "note": "Placeholder ESG data - integrate with ESG providers"}
            
        except Exception as e:
            logger.error(f"Error getting ESG data for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_market_trends(self, symbol: str) -> Dict[str, Any]:
        """Get market trends and sentiment indicators"""
        try:
            trends_data = {
                "symbol": symbol,
                "analyst_recommendations": {},
                "price_targets": {},
                "institutional_ownership": {},
                "insider_trading": {},
                "short_interest": {}
            }
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Analyst recommendations
            if hasattr(ticker, 'recommendations') and ticker.recommendations is not None:
                recs = ticker.recommendations.tail(10)
                trends_data["analyst_recommendations"] = {
                    "recent_recommendations": recs.to_dict('records') if not recs.empty else []
                }
            
            # Basic market data
            trends_data["market_indicators"] = {
                "short_ratio": info.get("shortRatio"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "shares_short": info.get("sharesShort"),
                "institutional_ownership_pct": info.get("heldByInstitutions"),
                "insider_ownership_pct": info.get("heldByInsiders")
            }
            
            return {"success": True, "data": trends_data}
            
        except Exception as e:
            logger.error(f"Error getting market trends for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_existing_filing(self, filing_type_dir: str, symbol: str, filing_type: str, accession_number: str) -> Optional[str]:
        """Check if a filing already exists locally and verify its integrity"""
        try:
            # Check for exact file match
            pdf_filename = f"{symbol}_{filing_type}_{accession_number}.pdf"
            pdf_path = os.path.join(filing_type_dir, pdf_filename)
            
            if os.path.exists(pdf_path):
                # Verify file integrity (check if it's a valid PDF and not empty)
                if os.path.getsize(pdf_path) > 1000:  # Basic size check (>1KB)
                    try:
                        with open(pdf_path, 'rb') as f:
                            header = f.read(4)
                            if header.startswith(b'%PDF'):  # Check PDF signature
                                logger.info(f"Found valid existing PDF for {symbol} {filing_type} ({accession_number})")
                                return pdf_path
                    except Exception:
                        pass
                
                # If file exists but is invalid, remove it
                try:
                    os.remove(pdf_path)
                    logger.warning(f"Removed invalid PDF file: {pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not remove invalid PDF: {e}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Error checking existing filing: {e}")
            return None

    def _get_sec_filings_from_api(self, symbol: str, filing_types: List[str] = None, max_filings: int = 1) -> Dict[str, Any]:
        """Get SEC filings directly as PDFs using SEC-API.IO"""
        try:
            if not self.config.sec_api_key:
                return {"success": False, "error": "SEC-API.IO API key not configured"}

            if filing_types is None:
                filing_types = ["10-K", "10-Q", "8-K"]
            
            company_filings_path = os.path.join(self.sec_filings_path, symbol)
            os.makedirs(company_filings_path, exist_ok=True)
            
            filings_data = {
                "symbol": symbol,
                "filings_downloaded": [],
                "filing_summaries": [],
                "download_path": company_filings_path,
                "source": "sec-api.io",
                "api_calls_made": 0,
                "files_reused": 0
            }

            # Initialize SEC-API.IO client (only if we'll need it)
            pdf_generator = None

            # Download filings for each type
            for filing_type in filing_types:
                try:
                    logger.info(f"Processing {filing_type} filings for {symbol}")
                    
                    # Determine max filings per type
                    if filing_type == "10-K":
                        max_filings = 1
                    elif filing_type == "10-Q":
                        max_filings = 4
                    elif filing_type == "8-K":
                        max_filings = 1

                    # Create filing type directory
                    filing_type_dir = os.path.join(company_filings_path, filing_type)
                    os.makedirs(filing_type_dir, exist_ok=True)

                    # First, download filings using the standard EDGAR downloader to get the file structure
                    temp_download_path = os.path.join(company_filings_path, "temp_edgar")
                    edgar_downloader = Downloader(
                        company_name=self.config.edgar_company_name,
                        email_address=self.config.edgar_email,
                        download_folder=temp_download_path
                    )
                    
                    # Download filings to get the metadata and file structure
                    filings_downloaded = edgar_downloader.get(
                        filing_type,
                        symbol,
                        limit=max_filings
                    )
                    
                    logger.info(f"EDGAR downloader returned {filings_downloaded} filings for {symbol} {filing_type}")
                    
                    if filings_downloaded == 0:
                        logger.warning(f"No {filing_type} filings found for {symbol}")
                        continue

                    # Now process the downloaded filings to get URLs for SEC-API.IO
                    temp_filing_folder = os.path.join(temp_download_path, "sec-edgar-filings", symbol, filing_type)
                    
                    logger.info(f"Checking temp filing folder: {temp_filing_folder}")
                    logger.info(f"Temp filing folder exists: {os.path.exists(temp_filing_folder)}")
                    
                    if os.path.exists(temp_filing_folder):
                        folder_contents = os.listdir(temp_filing_folder)
                        logger.info(f"Temp filing folder contents: {folder_contents}")
                        
                        for filing_dir in os.listdir(temp_filing_folder)[:max_filings]:
                            filing_path = os.path.join(temp_filing_folder, filing_dir)
                            logger.info(f"Processing filing directory: {filing_dir} at {filing_path}")
                            logger.info(f"Is directory: {os.path.isdir(filing_path)}")
                            
                            if os.path.isdir(filing_path):
                                try:
                                    # Extract accession number from directory name
                                    accession_number = filing_dir
                                    
                                    # Get the CIK for the symbol
                                    cik = self._get_cik_for_symbol(symbol)
                                    if not cik:
                                        logger.warning(f"Could not get CIK for {symbol}")
                                        continue
                                    
                                    # For SEC-API.IO, we can use the accession number directly
                                    # Format: https://www.sec.gov/Archives/edgar/data/CIK/ACCESSION_NUMBER_NO_DASHES/ACCESSION_NUMBER.txt
                                    # But SEC-API.IO also accepts direct accession number format
                                    accession_no_dashes = accession_number.replace('-', '')
                                    
                                    # Try to find the primary document filename from the full-submission.txt
                                    primary_doc_filename = None
                                    full_submission_path = os.path.join(filing_path, "full-submission.txt")
                                    
                                    if os.path.exists(full_submission_path):
                                        try:
                                            with open(full_submission_path, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read(50000)  # Read first 50KB to find primary document
                                                lines = content.split('\n')
                                                
                                                # Look for the primary document filename in SGML format
                                                # SEC EDGAR format uses <FILENAME>filename</FILENAME> or <FILENAME>filename
                                                for line in lines:
                                                    # Look for lines that contain <FILENAME> and end with .htm
                                                    if '<FILENAME>' in line and '.htm' in line:
                                                        # Extract filename from SGML tag
                                                        if '<FILENAME>' in line and '</FILENAME>' in line:
                                                            # Full tag format: <FILENAME>filename.htm</FILENAME>
                                                            start = line.find('<FILENAME>') + len('<FILENAME>')
                                                            end = line.find('</FILENAME>')
                                                            if start < end:
                                                                filename = line[start:end].strip()
                                                                if filename.endswith('.htm'):
                                                                    primary_doc_filename = filename
                                                                    break
                                                        elif line.strip().startswith('<FILENAME>') and line.strip().endswith('.htm'):
                                                            # Simple format: <FILENAME>filename.htm
                                                            filename = line.strip().replace('<FILENAME>', '').strip()
                                                            if filename.endswith('.htm'):
                                                                primary_doc_filename = filename
                                                                break
                                                
                                                # If no primary document found, look for any .htm file
                                                if not primary_doc_filename:
                                                    for line in lines:
                                                        if '<FILENAME>' in line and '.htm' in line:
                                                            # Extract any .htm filename
                                                            parts = line.split('<FILENAME>')
                                                            if len(parts) > 1:
                                                                filename_part = parts[1].split('</FILENAME>')[0] if '</FILENAME>' in parts[1] else parts[1].split()[0]
                                                                filename_part = filename_part.strip()
                                                                if filename_part.endswith('.htm') and not filename_part.startswith('R'):
                                                                    primary_doc_filename = filename_part
                                                                    break
                                                        
                                        except Exception as e:
                                            logger.warning(f"Could not parse full-submission.txt: {e}")
                                    
                                    # If we found a primary document filename, use it; otherwise use a standard format
                                    if primary_doc_filename:
                                        edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc_filename}"
                                        logger.info(f"Using primary document: {primary_doc_filename}")
                                    else:
                                        # Fallback: construct a likely filename based on common SEC patterns
                                        symbol_lower = symbol.lower()
                                        # Common patterns: aapl-20240928.htm, aapl10k_20240928.htm, etc.
                                        likely_filename = f"{symbol_lower}-{accession_number[-8:]}.htm"  # Use last 8 chars as date-like
                                        edgar_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{likely_filename}"
                                        logger.info(f"Using fallback filename: {likely_filename}")
                                    
                                    # Check if we already have this filing as PDF
                                    existing_pdf = self._check_existing_filing(
                                        filing_type_dir, 
                                        symbol, 
                                        filing_type, 
                                        accession_number
                                    )
                                    
                                    if existing_pdf:
                                        filings_data["files_reused"] += 1
                                        filing_info = {
                                            "filing_type": filing_type,
                                            "symbol": symbol,
                                            "filing_path": existing_pdf,
                                            "accession_number": accession_number,
                                            "source_url": edgar_url,
                                            "download_method": "reused_existing"
                                        }
                                    else:
                                        # Initialize SEC-API.IO client if we haven't yet
                                        if not pdf_generator:
                                            try:
                                                from sec_api import PdfGeneratorApi
                                                pdf_generator = PdfGeneratorApi(self.config.sec_api_key)
                                            except ImportError:
                                                return {"success": False, "error": "sec-api package not installed"}
                                            except Exception as e:
                                                return {"success": False, "error": f"Failed to initialize SEC-API.IO client: {e}"}

                                        # Download PDF using SEC-API.IO
                                        pdf_filename = f"{symbol}_{filing_type}_{accession_number}.pdf"
                                        pdf_path = os.path.join(filing_type_dir, pdf_filename)
                                        
                                        try:
                                            logger.info(f"Downloading PDF from SEC-API.IO: {edgar_url}")
                                            pdf_content = pdf_generator.get_pdf(edgar_url)
                                            filings_data["api_calls_made"] += 1
                                            
                                            with open(pdf_path, "wb") as pdf_file:
                                                pdf_file.write(pdf_content)

                                            filing_info = {
                                                "filing_type": filing_type,
                                                "symbol": symbol,
                                                "filing_path": pdf_path,
                                                "accession_number": accession_number,
                                                "source_url": edgar_url,
                                                "download_method": "sec-api.io"
                                            }
                                        except Exception as e:
                                            logger.warning(f"Failed to download PDF for {edgar_url}: {e}")
                                            continue
                                    
                                    filings_data["filings_downloaded"].append(filing_info)
                                    filings_data["filing_summaries"].append({
                                        "filing_type": filing_type,
                                        "file_path": filing_info["filing_path"],
                                        "accession_number": accession_number,
                                        "download_method": filing_info["download_method"]
                                    })

                                except Exception as e:
                                    logger.warning(f"Failed to process filing directory {filing_path}: {e}")
                                    continue
                    
                    # Clean up temporary download directory
                    try:
                        import shutil
                        if os.path.exists(temp_download_path):
                            shutil.rmtree(temp_download_path)
                    except Exception as e:
                        logger.warning(f"Could not clean up temp directory: {e}")

                except Exception as e:
                    logger.warning(f"Error processing {filing_type} filings: {e}")
                    filings_data["filing_summaries"].append({
                        "filing_type": filing_type,
                        "status": "error",
                        "error": str(e)
                    })

            # Log usage statistics
            logger.info(f"SEC-API.IO usage for {symbol}: {filings_data['api_calls_made']} API calls, {filings_data['files_reused']} files reused")
            return {"success": True, "data": filings_data}

        except Exception as e:
            logger.error(f"Error in SEC-API.IO filing retrieval: {e}")
            return {"success": False, "error": str(e)}

    def _get_cik_for_symbol(self, symbol: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a given stock symbol"""
        try:
            # Use the ticker_to_cik_mapping from sec_edgar_downloader
            from sec_edgar_downloader import Downloader
            temp_downloader = Downloader("temp", "temp@temp.com")
            
            if hasattr(temp_downloader, 'ticker_to_cik_mapping') and symbol in temp_downloader.ticker_to_cik_mapping:
                cik = temp_downloader.ticker_to_cik_mapping[symbol]
                return str(cik).zfill(10)  # Pad with zeros to 10 digits
            
            # Fallback: try to get CIK from SEC API
            url = f"https://data.sec.gov/submissions/CIK{symbol}.json"
            headers = {"User-Agent": f"{self.config.edgar_company_name} {self.config.edgar_email}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return str(data.get('cik', '')).zfill(10)
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not get CIK for {symbol}: {e}")
            return None

    def get_sec_filings(self, symbol: str, filing_types: List[str] = None, max_filings: int = 1) -> Dict[str, Any]:
        """Download and parse SEC filings for the company using SEC-API.IO with fallback to direct EDGAR download"""
        try:
            # First try SEC-API.IO
            api_result = self._get_sec_filings_from_api(symbol, filing_types, max_filings)
            
            if api_result["success"]:
                logger.info(f"Successfully retrieved SEC filings using SEC-API.IO for {symbol}")
                return api_result
            else:
                logger.warning(f"SEC-API.IO retrieval failed: {api_result['error']}, falling back to direct EDGAR download")
            
            # Fallback to original method
            company_filings_path = os.path.join(self.sec_filings_path, symbol)
            os.makedirs(company_filings_path, exist_ok=True)
            
            filings_data = {
                "symbol": symbol,
                "filings_downloaded": [],
                "filing_summaries": [],
                "download_path": company_filings_path,
                "source": "edgar_direct"
            }
            
            if filing_types is None:
                filing_types = ["10-K", "10-Q", "8-K"]
            
            # Create a downloader instance for this company
            edgar_downloader = Downloader(
                company_name=self.config.edgar_company_name,
                email_address=self.config.edgar_email,
                download_folder=company_filings_path
            )
            
            # Download filings for each type
            for filing_type in filing_types:
                try:
                    logger.info(f"Downloading {filing_type} filings for {symbol} using direct EDGAR download")
                    
                    if filing_type == "10-K":
                        max_filings = 1
                    elif filing_type == "10-Q":
                        max_filings = 4
                    elif filing_type == "8-K":
                        max_filings = 1

                    edgar_downloader.get(
                        filing_type,
                        symbol,
                        limit=max_filings
                    )
                    
                    # Parse downloaded filings
                    filing_folder = os.path.join(company_filings_path, symbol, filing_type)
                    
                    if os.path.exists(filing_folder):
                        for filing_dir in os.listdir(filing_folder)[:max_filings]:
                            filing_path = os.path.join(filing_folder, filing_dir)
                            if os.path.isdir(filing_path):
                                parsed_filing = self._parse_sec_filing(filing_path, filing_type, symbol)
                                if parsed_filing:
                                    filings_data["filings_downloaded"].append(parsed_filing)
                                    
                                    summary = {
                                        "filing_type": filing_type,
                                        "date": parsed_filing.get("filing_date"),
                                        "file_path": parsed_filing.get("file_path"),
                                        "key_metrics": parsed_filing.get("key_metrics", {}),
                                        "text_summary": parsed_filing.get("text_summary", "")[:500],
                                        "download_method": "edgar_direct"
                                    }
                                    filings_data["filing_summaries"].append(summary)
                    
                except Exception as e:
                    logger.warning(f"Could not download/parse {filing_type} for {symbol}: {e}")
                    filings_data["filing_summaries"].append({
                        "filing_type": filing_type,
                        "status": "error",
                        "error": str(e)
                    })
            
            return {"success": True, "data": filings_data}
            
        except Exception as e:
            logger.error(f"Error getting SEC filings for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _parse_sec_filing(self, filing_path: str, filing_type: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Parse a downloaded SEC filing and extract key information"""
        try:
            filing_data = {
                "filing_type": filing_type,
                "symbol": symbol,
                "filing_path": filing_path,
                "file_path": None,
                "filing_date": None,
                "key_metrics": {},
                "text_summary": "",
                "sections": {},
                "tables_found": []
            }
            
            # Find the main filing document
            main_file = None
            for file in os.listdir(filing_path):
                if file.endswith(('.htm', '.html', '.txt')):
                    main_file = os.path.join(filing_path, file)
                    filing_data["file_path"] = main_file
                    break
            
            if not main_file or not os.path.exists(main_file):
                return None
            
            # Read and parse the filing
            with open(main_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract filing date
            date_patterns = [
                r'CONFORMED PERIOD OF REPORT:\s*(\d{8})',
                r'FILED AS OF DATE:\s*(\d{8})',
                r'Filing Date[\s\S]*?(\d{4}-\d{2}-\d{2})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    if len(date_str) == 8:
                        filing_data["filing_date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        filing_data["filing_date"] = date_str
                    break
            
            # Parse HTML content if available
            if main_file.endswith(('.htm', '.html')):
                soup = BeautifulSoup(content, 'html.parser')
                text_content = soup.get_text()
                filing_data["text_summary"] = text_content[:2000]
                
                # Look for financial tables
                tables = soup.find_all('table')
                for i, table in enumerate(tables[:5]):
                    table_text = table.get_text().strip()
                    if any(keyword in table_text.lower() for keyword in ['revenue', 'income', 'balance', 'cash flow']):
                        filing_data["tables_found"].append({
                            "table_index": i,
                            "table_preview": table_text[:300],
                            "contains_financials": True
                        })
            else:
                filing_data["text_summary"] = content[:2000]
            
            return filing_data
            
        except Exception as e:
            logger.error(f"Error parsing SEC filing {filing_path}: {e}")
            return None
    
    def get_earnings_call_data(self, symbol: str, quarters: int = 4) -> Dict[str, Any]:
        """Get earnings conference call transcripts and data"""
        try:
            earnings_data = {
                "symbol": symbol,
                "earnings_calls": [],
                "earnings_calendar": [],
                "earnings_surprises": []
            }
            
            # Get earnings calendar from Finnhub
            if self.finnhub_client:
                try:
                    from_date = datetime.now().strftime("%Y-%m-%d")
                    to_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
                    
                    calendar = self.finnhub_client.earnings_calendar(
                        _from=from_date, to=to_date, symbol=symbol
                    )
                    earnings_data["earnings_calendar"] = calendar.get("earningsCalendar", [])
                    
                except Exception as e:
                    logger.warning(f"Could not get earnings calendar: {e}")
                
                try:
                    surprises = self.finnhub_client.earnings_surprise(symbol)
                    earnings_data["earnings_surprises"] = surprises
                except Exception as e:
                    logger.warning(f"Could not get earnings surprises: {e}")
            
            # If Finnhub doesn't have transcripts, use alternative approach
            earnings_data["earnings_calls"] = self._get_alternative_earnings_data(symbol)
            
            return {"success": True, "data": earnings_data}
            
        except Exception as e:
            logger.error(f"Error getting earnings call data for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_alternative_earnings_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get earnings data from alternative sources when primary source fails"""
        try:
            ticker = yf.Ticker(symbol)
            earnings_data = []
            
            if hasattr(ticker, 'quarterly_earnings') and ticker.quarterly_earnings is not None:
                quarterly = ticker.quarterly_earnings
                
                for index, row in quarterly.iterrows():
                    earnings_entry = {
                        "date": str(index),
                        "quarter": f"Q{((index.month - 1) // 3) + 1}",
                        "year": index.year,
                        "revenue": row.get("Revenue", 0),
                        "earnings": row.get("Earnings", 0),
                        "source": "yfinance",
                        "transcript": "Transcript not available from this source"
                    }
                    earnings_data.append(earnings_entry)
            
            return earnings_data[:4]
            
        except Exception as e:
            logger.warning(f"Alternative earnings data collection failed: {e}")
            return []
    
    def get_competitor_sec_filings(self, symbol: str, competitors: List[str] = None, filing_types: List[str] = None) -> Dict[str, Any]:
        """Download SEC filings for competitors to enable comparative analysis"""
        try:
            if competitors is None:
                competitors = self._identify_competitors(symbol)
            
            if filing_types is None:
                filing_types = ["10-K", "10-Q"]
            
            competitor_data = {
                "primary_symbol": symbol,
                "competitors_analyzed": competitors,
                "competitor_filings": {},
                "comparative_metrics": {}
            }
            
            for competitor in competitors:
                logger.info(f"Downloading SEC filings for competitor: {competitor}")
                
                try:
                    competitor_filings = self.get_sec_filings(
                        symbol=competitor,
                        filing_types=filing_types,
                        max_filings=2
                    )
                    
                    if competitor_filings["success"]:
                        competitor_data["competitor_filings"][competitor] = competitor_filings["data"]
                    else:
                        competitor_data["competitor_filings"][competitor] = {"error": competitor_filings["error"]}
                        
                except Exception as e:
                    logger.warning(f"Failed to get filings for competitor {competitor}: {e}")
                    competitor_data["competitor_filings"][competitor] = {"error": str(e)}
            
            return {"success": True, "data": competitor_data}
            
        except Exception as e:
            logger.error(f"Error getting competitor SEC filings: {e}")
            return {"success": False, "error": str(e)}
    
    def _identify_competitors(self, symbol: str) -> List[str]:
        """Identify competitors for a given symbol"""
        try:
            competitor_map = {
                "AAPL": ["MSFT", "GOOGL"],
                "MSFT": ["AAPL", "GOOGL"],
                "GOOGL": ["AAPL", "MSFT"],
                "AMZN": ["AAPL", "MSFT"],
                "TSLA": ["F", "GM"],
                "JPM": ["BAC", "WFC"],
                "BAC": ["JPM", "WFC"]
            }
            
            if symbol in competitor_map:
                return competitor_map[symbol]
            
            # Try to get sector info and make educated guesses
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector = info.get("sector", "")
                
                if "Technology" in sector:
                    return ["AAPL", "MSFT"][:2]
                elif "Financial" in sector:
                    return ["JPM", "BAC"][:2]
                else:
                    return []
                    
            except Exception:
                return []
            
        except Exception as e:
            logger.warning(f"Could not identify competitors for {symbol}: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis using HuggingFace model"""
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0, "confidence": 0.0, "model": "fallback"}
        
        try:
            if self.sentiment_model is not None:
                results = self.sentiment_model(text) # [:512]
                
                if isinstance(results[0], list):
                    best_result = max(results[0], key=lambda x: x['score'])
                else:
                    best_result = results[0]
                
                label_mapping = {
                    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
                    'POSITIVE': 'positive', 'NEGATIVE': 'negative', 'NEUTRAL': 'neutral',
                    'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'
                }
                
                sentiment_label = label_mapping.get(best_result['label'], 'neutral')
                confidence = best_result['score']
                
                return {
                    "label": sentiment_label,
                    "score": confidence,
                    "confidence": confidence,
                    "model": "huggingface"
                }
            else:
                return self._simple_sentiment_analysis(text)
                
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
            return self._simple_sentiment_analysis(text)
    
    # def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
    #     """Fallback simple sentiment analysis"""
    #     positive_words = ["growth", "increase", "profit", "beat", "strong", "bullish", "buy", "gains", "up", "rise"]
    #     negative_words = ["decline", "decrease", "loss", "miss", "weak", "bearish", "sell", "fall", "down", "drop"]
        
    #     text_lower = text.lower()
    #     positive_count = sum(1 for word in positive_words if word in text_lower)
    #     negative_count = sum(1 for word in negative_words if word in text_lower)
        
    #     total_words = positive_count + negative_count
    #     if total_words == 0:
    #         return {"label": "neutral", "score": 0.5, "confidence": 0.3, "model": "rule_based"}
        
    #     if positive_count > negative_count:
    #         confidence = positive_count / total_words
    #         return {"label": "positive", "score": confidence, "confidence": confidence, "model": "rule_based"}
    #     elif negative_count > positive_count:
    #         confidence = negative_count / total_words
    #         return {"label": "negative", "score": confidence, "confidence": confidence, "model": "rule_based"}
    #     else:
    #         return {"label": "neutral", "score": 0.5, "confidence": 0.5, "model": "rule_based"}

class DataCollectorAgent:
    """Multi-source financial data collection agent"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.tools = DataCollectorTools(self.config)
        
        # Create the Autogen agent
        self.agent = AssistantAgent(
            name="DataCollector",
            system_message=self._get_system_message(),
            llm_config=AGENT_CONFIGS["data_collector"],
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
        )
        
        # Register tools
        self._register_tools()
    
    def _get_system_message(self) -> str:
        return """You are a DataCollector agent specialized in gathering comprehensive financial data.

Your responsibilities:
1. Collect stock price and trading data from Yahoo Finance
2. Retrieve company financial statements and key metrics
3. Gather recent news and market sentiment using HuggingFace models
4. Download and parse actual SEC filings (10-K, 10-Q, 8-K)
5. Collect earnings call data and transcripts
6. Download competitor SEC filings for comparative analysis
7. Obtain ESG data and market trends

For each data request:
- Use appropriate tools to gather multi-source data
- Structure the data consistently
- Store results in shared memory with proper metadata
- Provide data quality assessments
- Handle errors gracefully and report data limitations

Always prioritize data accuracy and completeness. If data is missing or uncertain, clearly indicate this in your response.

Reply TERMINATE when all requested data has been collected and stored."""
    
    def _register_tools(self):
        """Register data collection tools with the agent"""
        
        def collect_stock_data(symbol: str, period: str = "2y") -> str:
            result = self.tools.get_stock_data(symbol, period)
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="stock_data",
                    content=result["data"],
                    metadata={"symbol": symbol, "period": period, "source": "yahoo_finance"},
                    tags=["stock_data", "price", symbol.lower()]
                )
                return f"Stock data collected and stored (ID: {entry_id}). Current price: ${result['data']['current_price']:.2f}"
            else:
                return f"Failed to collect stock data: {result['error']}"
        
        def collect_financials(symbol: str) -> str:
            result = self.tools.get_company_financials(symbol)
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="financial_statements",
                    content=result["data"],
                    metadata={"symbol": symbol, "source": "yahoo_finance"},
                    tags=["financials", "statements", symbol.lower()]
                )
                return f"Financial statements collected and stored (ID: {entry_id})"
            else:
                return f"Failed to collect financials: {result['error']}"
        
        def collect_news(symbol: str, days_back: int = 30) -> str:
            result = self.tools.get_company_news(symbol, days_back)
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="news",
                    content=result["data"],
                    metadata={"symbol": symbol, "days_back": days_back, "source": "finnhub"},
                    tags=["news", "sentiment", symbol.lower()]
                )
                article_count = len(result["data"]["articles"])
                return f"News data collected and stored (ID: {entry_id}). Found {article_count} articles with HuggingFace sentiment analysis"
            else:
                return f"Failed to collect news: {result['error']}"
        
        def collect_esg_data(symbol: str) -> str:
            result = self.tools.get_esg_data(symbol)
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="esg_data",
                    content=result["data"],
                    metadata={"symbol": symbol, "source": "placeholder"},
                    tags=["esg", "sustainability", symbol.lower()]
                )
                return f"ESG data collected and stored (ID: {entry_id}). Note: {result.get('note', '')}"
            else:
                return f"Failed to collect ESG data: {result['error']}"
        
        def collect_market_trends(symbol: str) -> str:
            result = self.tools.get_market_trends(symbol)
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="market_trends",
                    content=result["data"],
                    metadata={"symbol": symbol, "source": "yahoo_finance"},
                    tags=["trends", "analysts", "market", symbol.lower()]
                )
                return f"Market trends collected and stored (ID: {entry_id})"
            else:
                return f"Failed to collect market trends: {result['error']}"
        
        def collect_sec_filings(symbol: str, filing_types: str = "10-K,10-Q,8-K") -> str:
            types_list = [t.strip() for t in filing_types.split(",")]
            result = self.tools.get_sec_filings(symbol, types_list)
            
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="sec_filings",
                    content=result["data"],
                    metadata={"symbol": symbol, "filing_types": types_list, "source": "sec_edgar"},
                    tags=["sec", "filings", symbol.lower()]
                )
                filings_count = len(result["data"]["filings_downloaded"])
                return f"SEC filings downloaded and stored (ID: {entry_id}). Downloaded {filings_count} actual filings for multimodal analysis"
            else:
                return f"Failed to collect SEC filings: {result['error']}"
        
        def collect_earnings_data(symbol: str, quarters: int = 4) -> str:
            result = self.tools.get_earnings_call_data(symbol, quarters)
            
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="earnings_data",
                    content=result["data"],
                    metadata={"symbol": symbol, "quarters": quarters, "source": "finnhub_yfinance"},
                    tags=["earnings", "calls", "transcripts", symbol.lower()]
                )
                calls_count = len(result["data"]["earnings_calls"])
                calendar_count = len(result["data"]["earnings_calendar"])
                return f"Earnings data collected and stored (ID: {entry_id}). Found {calls_count} earnings calls and {calendar_count} calendar events"
            else:
                return f"Failed to collect earnings data: {result['error']}"
        
        def collect_competitor_filings(symbol: str, competitors: str = None, filing_types: str = "10-K,10-Q") -> str:
            competitors_list = None
            if competitors:
                competitors_list = [c.strip() for c in competitors.split(",")]
            
            types_list = [t.strip() for t in filing_types.split(",")]
            
            result = self.tools.get_competitor_sec_filings(symbol, competitors_list, types_list)
            
            if result["success"]:
                entry_id = self.memory_manager.store_entry(
                    agent_name="DataCollector",
                    content_type="competitor_filings",
                    content=result["data"],
                    metadata={"symbol": symbol, "competitors": competitors_list, "source": "sec_edgar"},
                    tags=["competitors", "sec", "filings", symbol.lower()]
                )
                competitors_analyzed = len(result["data"]["competitors_analyzed"])
                return f"Competitor filings collected and stored (ID: {entry_id}). Analyzed {competitors_analyzed} competitors for comparative analysis"
            else:
                return f"Failed to collect competitor filings: {result['error']}"
        
        # Register functions with autogen
        self.agent.register_for_execution(name="collect_stock_data")(collect_stock_data)
        self.agent.register_for_execution(name="collect_financials")(collect_financials)
        self.agent.register_for_execution(name="collect_news")(collect_news)
        self.agent.register_for_execution(name="collect_esg_data")(collect_esg_data)
        self.agent.register_for_execution(name="collect_market_trends")(collect_market_trends)
        self.agent.register_for_execution(name="collect_sec_filings")(collect_sec_filings)
        self.agent.register_for_execution(name="collect_earnings_data")(collect_earnings_data)
        self.agent.register_for_execution(name="collect_competitor_filings")(collect_competitor_filings)
    
    def process_data_request(self, request: DataRequest) -> Dict[str, Any]:
        """Process a data collection request"""
        try:
            results = {
                "symbol": request.symbol,
                "request_processed": True,
                "data_types_completed": [],
                "errors": []
            }
            
            if not request.data_types:
                request.data_types = ["financial", "news", "esg", "trends"]
            
            for data_type in request.data_types:
                try:
                    if data_type == "financial":
                        self.tools.get_stock_data(request.symbol)
                        self.tools.get_company_financials(request.symbol)
                        results["data_types_completed"].append("financial")
                    
                    elif data_type == "news":
                        self.tools.get_company_news(request.symbol, request.lookback_days)
                        results["data_types_completed"].append("news")
                    
                    elif data_type == "esg":
                        self.tools.get_esg_data(request.symbol)
                        results["data_types_completed"].append("esg")
                    
                    elif data_type == "trends":
                        self.tools.get_market_trends(request.symbol)
                        results["data_types_completed"].append("trends")
                    
                    elif data_type == "sec":
                        self.tools.get_sec_filings(request.symbol, request.filing_types)
                        results["data_types_completed"].append("sec")
                    
                    elif data_type == "earnings":
                        self.tools.get_earnings_call_data(request.symbol)
                        results["data_types_completed"].append("earnings")
                    
                    elif data_type == "competitors":
                        self.tools.get_competitor_sec_filings(request.symbol, request.competitors, request.filing_types)
                        results["data_types_completed"].append("competitors")
                        
                except Exception as e:
                    results["errors"].append(f"{data_type}: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing data request: {e}")
            return {"symbol": request.symbol, "request_processed": False, "error": str(e)} 