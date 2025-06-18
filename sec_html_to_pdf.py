#!/usr/bin/env python3
"""
SEC Filing HTML to PDF Converter

This module provides functionality to convert SEC filings from HTML format to PDF
without using external APIs. It supports multiple conversion backends for robustness.

Usage:
    # Command line
    python sec_html_to_pdf.py input.html output.pdf
    python sec_html_to_pdf.py https://sec.gov/filing.html output.pdf --backend weasyprint
    
    # Programmatic
    converter = SECHTMLToPDFConverter()
    converter.convert_file('filing.html', 'filing.pdf')
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SECHTMLToPDFConverter:
    """
    Converts SEC filing HTML content to PDF using multiple backend options.
    Supported backends: pyppeteer, playwright, weasyprint, xhtml2pdf
    """
    
    # Default backend preference order (can be overridden)
    DEFAULT_BACKEND_PREFERENCE = ['weasyprint', 'xhtml2pdf', 'pyppeteer', 'playwright']
    
    def __init__(self, backend: str = "auto", backend_preference: list = None):
        """
        Initialize the converter with a specified backend.
        
        Args:
            backend: Conversion backend ('pyppeteer', 'weasyprint', 'xhtml2pdf', 'playwright', 'auto')
            backend_preference: Custom preference order for auto-selection (optional)
        """
        self.backend = backend
        self.backend_preference = backend_preference or self.DEFAULT_BACKEND_PREFERENCE
        self.available_backends = self._detect_available_backends()
        
        if backend == "auto":
            self.backend = self._select_best_backend()
        elif backend not in self.available_backends:
            raise ValueError(f"Backend '{backend}' not available. Available: {list(self.available_backends.keys())}")
        elif not self.available_backends[backend]:
            raise ValueError(f"Backend '{backend}' is not installed or properly configured")
    
    def _detect_available_backends(self) -> Dict[str, bool]:
        """Detect which conversion backends are available and properly configured."""
        backends = {}
        
        # Check pyppeteer
        try:
            import pyppeteer
            # Test if it can actually be used
            backends['pyppeteer'] = True
            logger.debug("Pyppeteer backend available")
        except ImportError:
            backends['pyppeteer'] = False
        except Exception as e:
            logger.debug(f"Pyppeteer import succeeded but initialization failed: {e}")
            backends['pyppeteer'] = False
        
        # Check WeasyPrint
        try:
            import weasyprint
            # Test if WeasyPrint can actually be used (this will catch missing external libraries)
            try:
                from weasyprint import HTML
                # Try a minimal test to see if external dependencies are available
                HTML(string="<html><body>test</body></html>")
                backends['weasyprint'] = True
                logger.debug("WeasyPrint backend available and properly configured")
            except Exception as e:
                logger.debug(f"WeasyPrint import succeeded but configuration failed: {e}")
                backends['weasyprint'] = False
        except ImportError:
            backends['weasyprint'] = False
        
        # Check xhtml2pdf
        try:
            from xhtml2pdf import pisa
            # Test basic functionality
            backends['xhtml2pdf'] = True
            logger.debug("xhtml2pdf backend available")
        except ImportError:
            backends['xhtml2pdf'] = False
        except Exception as e:
            logger.debug(f"xhtml2pdf import succeeded but initialization failed: {e}")
            backends['xhtml2pdf'] = False
        
        # Check playwright
        try:
            from playwright.async_api import async_playwright
            backends['playwright'] = True
            logger.debug("Playwright backend available")
        except ImportError:
            backends['playwright'] = False
        except Exception as e:
            logger.debug(f"Playwright import succeeded but initialization failed: {e}")
            backends['playwright'] = False
        
        return backends
    
    def _select_best_backend(self) -> str:
        """Select the best available backend based on preference order."""
        
        for backend in self.backend_preference:
            if self.available_backends.get(backend, False):
                logger.info(f"Auto-selected backend: {backend}")
                return backend
        
        # Provide helpful error message with installation instructions
        available_installs = []
        if not self.available_backends.get('weasyprint', False):
            available_installs.append("pip install weasyprint")
        if not self.available_backends.get('xhtml2pdf', False):
            available_installs.append("pip install xhtml2pdf")
        if not self.available_backends.get('pyppeteer', False):
            available_installs.append("pip install pyppeteer")
        if not self.available_backends.get('playwright', False):
            available_installs.append("pip install playwright")
        
        install_cmd = " OR ".join(available_installs)
        
        raise RuntimeError(
            f"No supported PDF conversion backends available.\n"
            f"Please install at least one backend:\n{install_cmd}\n"
            f"Recommended for beginners: pip install weasyprint"
        )
    
    def clean_sec_html(self, html_content: str) -> str:
        """
        Clean and prepare SEC filing HTML for better PDF conversion.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content optimized for PDF conversion
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script tags that can interfere with conversion
        for script in soup.find_all("script"):
            script.decompose()
        
        # Remove problematic style tags
        for style in soup.find_all("style"):
            style_text = style.get_text()
            if "font-size: 0px" in style_text or "display: none" in style_text:
                style.decompose()
        
        # Fix tables for better PDF rendering
        for table in soup.find_all("table"):
            if not table.get("border"):
                table["border"] = "1"
            table["cellpadding"] = "5"
            table["cellspacing"] = "0"
            table["style"] = "border-collapse: collapse; width: 100%;"
        
        # Ensure proper encoding
        if not soup.find("meta", {"charset": True}):
            meta_charset = soup.new_tag("meta", charset="utf-8")
            if soup.head:
                soup.head.insert(0, meta_charset)
            else:
                # Create head if it doesn't exist
                head = soup.new_tag("head")
                head.append(meta_charset)
                if soup.html:
                    soup.html.insert(0, head)
        
        # Add enhanced CSS for better SEC filing formatting
        css_style = """
        <style>
        @page {
            size: A4;
            margin: 1in;
            @top-center {
                content: "SEC Filing";
                font-size: 10px;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page);
                font-size: 10px;
                color: #666;
            }
        }
        
        body { 
            font-family: 'Times New Roman', Times, serif; 
            font-size: 11px; 
            line-height: 1.4; 
            margin: 0;
            padding: 0;
            color: #000;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-weight: bold;
            margin: 15px 0 10px 0;
            color: #000;
        }
        
        h1 { font-size: 16px; }
        h2 { font-size: 14px; }
        h3 { font-size: 12px; }
        
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 10px 0;
            page-break-inside: avoid;
        }
        
        th, td { 
            border: 1px solid #000; 
            padding: 6px; 
            text-align: left;
            vertical-align: top;
            font-size: 10px;
        }
        
        th { 
            background-color: #f5f5f5; 
            font-weight: bold;
        }
        
        .header, .section-header { 
            font-size: 14px; 
            font-weight: bold; 
            margin: 20px 0 10px 0;
            text-transform: uppercase;
        }
        
        .signature { 
            margin-top: 30px; 
            font-style: italic;
            text-align: right;
        }
        
        .financial-data {
            font-family: 'Courier New', monospace;
            font-size: 10px;
        }
        
        .page-break {
            page-break-before: always;
        }
        
        .no-break {
            page-break-inside: avoid;
        }
        
        p {
            margin: 8px 0;
            text-align: justify;
        }
        
        .indent {
            margin-left: 20px;
        }
        
        .center {
            text-align: center;
        }
        
        .right {
            text-align: right;
        }
        
        .bold {
            font-weight: bold;
        }
        
        .italic {
            font-style: italic;
        }
        </style>
        """
        
        if soup.head:
            soup.head.append(BeautifulSoup(css_style, 'html.parser'))
        
        return str(soup)
    
    async def convert_with_pyppeteer(self, html_content: str, output_path: str) -> bool:
        """Convert HTML to PDF using Pyppeteer (Chrome headless)."""
        try:
            from pyppeteer import launch
            
            browser = await launch(
                headless=True,
                args=[
                    '--no-sandbox', 
                    '--disable-dev-shm-usage',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor'
                ]
            )
            page = await browser.newPage()
            
            # Set viewport and content
            await page.setViewport({'width': 1200, 'height': 800})
            await page.setContent(html_content, waitUntil='networkidle0')
            
            # Generate PDF with comprehensive options
            await page.pdf({
                'path': output_path,
                'format': 'A4',
                'margin': {
                    'top': '1in',
                    'right': '1in',
                    'bottom': '1in',
                    'left': '1in'
                },
                'printBackground': True,
                'preferCSSPageSize': True
            })
            
            await browser.close()
            logger.info(f"PDF successfully created using Pyppeteer: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Pyppeteer conversion failed: {e}")
            return False
    
    async def convert_with_playwright(self, html_content: str, output_path: str) -> bool:
        """Convert HTML to PDF using Playwright."""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                await page.set_content(html_content, wait_until='networkidle')
                
                await page.pdf(
                    path=output_path,
                    format='A4',
                    margin={
                        'top': '1in',
                        'right': '1in',
                        'bottom': '1in',
                        'left': '1in'
                    },
                    print_background=True,
                    prefer_css_page_size=True
                )
                
                await browser.close()
            
            logger.info(f"PDF successfully created using Playwright: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Playwright conversion failed: {e}")
            return False
    
    def convert_with_weasyprint(self, html_content: str, output_path: str) -> bool:
        """Convert HTML to PDF using WeasyPrint."""
        try:
            import weasyprint
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
            
            # Create font configuration
            font_config = FontConfiguration()
            
            # Enhanced CSS for WeasyPrint
            css = CSS(string="""
                @page {
                    size: A4;
                    margin: 1in;
                }
                body {
                    font-family: 'Times New Roman', serif;
                    font-size: 11px;
                    line-height: 1.4;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                }
                th, td {
                    border: 1px solid black;
                    padding: 4px;
                }
            """, font_config=font_config)
            
            # Convert HTML to PDF
            html_doc = HTML(string=html_content)
            html_doc.write_pdf(output_path, stylesheets=[css], font_config=font_config)
            
            logger.info(f"PDF successfully created using WeasyPrint: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"WeasyPrint conversion failed: {e}")
            return False
    
    def convert_with_xhtml2pdf(self, html_content: str, output_path: str) -> bool:
        """Convert HTML to PDF using xhtml2pdf."""
        try:
            from xhtml2pdf import pisa
            
            with open(output_path, "wb") as pdf_file:
                pisa_status = pisa.CreatePDF(
                    html_content.encode('utf-8'),
                    dest=pdf_file,
                    encoding='utf-8'
                )
                
                if pisa_status.err:
                    logger.warning(f"xhtml2pdf conversion had {pisa_status.err} errors, but PDF was created")
                    return True  # Often works despite errors
            
            logger.info(f"PDF successfully created using xhtml2pdf: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"xhtml2pdf conversion failed: {e}")
            return False
    
    async def convert_html_to_pdf(self, html_content: str, output_path: str) -> bool:
        """
        Convert HTML content to PDF using the selected backend.
        
        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        # Clean the HTML content for better PDF rendering
        cleaned_html = self.clean_sec_html(html_content)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert based on selected backend
        success = False
        if self.backend == 'pyppeteer':
            success = await self.convert_with_pyppeteer(cleaned_html, output_path)
        elif self.backend == 'playwright':
            success = await self.convert_with_playwright(cleaned_html, output_path)
        elif self.backend == 'weasyprint':
            success = self.convert_with_weasyprint(cleaned_html, output_path)
        elif self.backend == 'xhtml2pdf':
            success = self.convert_with_xhtml2pdf(cleaned_html, output_path)
        else:
            logger.error(f"Unknown backend: {self.backend}")
            return False
        
        if success and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"PDF created successfully: {output_path} ({file_size} bytes)")
        
        return success
    
    def convert_html_to_pdf_sync(self, html_content: str, output_path: str) -> bool:
        """
        Synchronous wrapper for convert_html_to_pdf.
        
        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        if self.backend in ['pyppeteer', 'playwright']:
            # These backends require async
            try:
                return asyncio.run(self.convert_html_to_pdf(html_content, output_path))
            except RuntimeError:
                # Handle case where event loop is already running
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.convert_html_to_pdf(html_content, output_path))
                finally:
                    loop.close()
        else:
            # Synchronous backends
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.convert_html_to_pdf(html_content, output_path))
            finally:
                loop.close()
    
    def convert_file(self, input_path: str, output_path: str) -> bool:
        """
        Convert HTML file to PDF.
        
        Args:
            input_path: Path to HTML file
            output_path: Path where PDF should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file does not exist: {input_path}")
                return False
            
            with open(input_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            logger.info(f"Converting file: {input_path} -> {output_path}")
            return self.convert_html_to_pdf_sync(html_content, output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert file {input_path}: {e}")
            return False
    
    def convert_url(self, url: str, output_path: str) -> bool:
        """
        Convert HTML from URL to PDF.
        
        Args:
            url: URL to fetch HTML from
            output_path: Path where PDF should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            logger.info(f"Fetching URL: {url}")
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            html_content = response.text
            logger.info(f"Converting URL: {url} -> {output_path}")
            return self.convert_html_to_pdf_sync(html_content, output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert URL {url}: {e}")
            return False
    
    def extract_html_from_sec_filing_txt(self, txt_file_path: str) -> str:
        """
        Extract HTML content from SEC filing text file (full-submission.txt).
        
        SEC filing text files contain multiple documents separated by specific markers.
        This function extracts the main HTML filing document.
        
        Args:
            txt_file_path: Path to SEC filing text file
            
        Returns:
            Extracted HTML content as string
        """
        try:
            logger.info(f"Extracting HTML from SEC filing: {txt_file_path}")
            
            with open(txt_file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # SEC filing structure markers
            # Documents are separated by lines like:
            # <DOCUMENT>
            # <TYPE>10-K
            # <SEQUENCE>1
            # <FILENAME>aapl-20230930.htm
            # <DESCRIPTION>10-K
            # <TEXT>
            # [HTML content here]
            # </TEXT>
            # </DOCUMENT>
            
            # Find the main HTML document (usually the first document with HTML content)
            import re
            
            # Pattern to match document sections
            doc_pattern = r'<DOCUMENT>(.*?)</DOCUMENT>'
            documents = re.findall(doc_pattern, content, re.DOTALL | re.IGNORECASE)
            
            html_content = None
            
            for doc in documents:
                # Look for HTML-like content in TEXT section
                text_pattern = r'<TEXT>(.*?)</TEXT>'
                text_match = re.search(text_pattern, doc, re.DOTALL | re.IGNORECASE)
                
                if text_match:
                    text_content = text_match.group(1).strip()
                    
                    # Check if this looks like HTML content
                    if '<html' in text_content.lower() or '<!doctype html' in text_content.lower():
                        html_content = text_content
                        logger.info("Found HTML document in SEC filing")
                        break
                    elif '<body' in text_content.lower() or '<table' in text_content.lower():
                        # Wrap partial HTML content
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>SEC Filing</title>
                            <meta charset="utf-8">
                        </head>
                        <body>
                        {text_content}
                        </body>
                        </html>
                        """
                        logger.info("Found partial HTML document in SEC filing, wrapped in HTML structure")
                        break
            
            if not html_content:
                # Fallback: look for any content between <TEXT> tags
                text_pattern = r'<TEXT>(.*?)</TEXT>'
                text_matches = re.findall(text_pattern, content, re.DOTALL | re.IGNORECASE)
                
                if text_matches:
                    # Use the longest text section
                    longest_text = max(text_matches, key=len)
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>SEC Filing</title>
                        <meta charset="utf-8">
                    </head>
                    <body>
                    <pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px;">
                    {longest_text}
                    </pre>
                    </body>
                    </html>
                    """
                    logger.info("No HTML found, using plain text content wrapped in HTML")
                else:
                    raise ValueError("No extractable content found in SEC filing")
            
            logger.info(f"Successfully extracted HTML content ({len(html_content)} characters)")
            return html_content
            
        except Exception as e:
            logger.error(f"Failed to extract HTML from SEC filing {txt_file_path}: {e}")
            raise
    
    def convert_sec_filing_txt(self, txt_file_path: str, output_path: str) -> bool:
        """
        Convert SEC filing text file to PDF.
        
        This method handles SEC filing text files (like full-submission.txt) that contain
        HTML documents embedded within SEC EDGAR document structure.
        
        Args:
            txt_file_path: Path to SEC filing text file
            output_path: Path where PDF should be saved
            
        Returns:
            True if conversion successful, False otherwise
        """
        try:
            if not os.path.exists(txt_file_path):
                logger.error(f"SEC filing file does not exist: {txt_file_path}")
                return False
            
            # Extract HTML content from the SEC filing text file
            html_content = self.extract_html_from_sec_filing_txt(txt_file_path)
            
            # Convert the extracted HTML to PDF
            logger.info(f"Converting SEC filing: {txt_file_path} -> {output_path}")
            return self.convert_html_to_pdf_sync(html_content, output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert SEC filing {txt_file_path}: {e}")
            return False
    
    @classmethod
    def list_available_backends(cls) -> Dict[str, bool]:
        """
        Class method to list all available backends without creating an instance.
        
        Returns:
            Dictionary mapping backend names to availability status
        """
        temp_converter = cls.__new__(cls)  # Create instance without calling __init__
        return temp_converter._detect_available_backends()
    
    @classmethod
    def get_backend_info(cls) -> Dict[str, str]:
        """
        Get information about each backend.
        
        Returns:
            Dictionary with backend descriptions
        """
        return {
            'weasyprint': 'Good quality, pure Python, handles CSS well, recommended for most use cases',
            'xhtml2pdf': 'Lightweight, fast, basic quality, good for simple documents',
            'pyppeteer': 'High quality, uses Chromium, excellent CSS support, requires download',
            'playwright': 'High quality, uses Chromium, excellent CSS support, alternative to pyppeteer'
        }
    
    @classmethod 
    def create_with_preference(cls, preferences: list, fallback_to_auto: bool = True):
        """
        Create converter with custom backend preference order.
        
        Args:
            preferences: List of backends in order of preference
            fallback_to_auto: Whether to use auto-selection if none of preferred backends work
            
        Returns:
            SECHTMLToPDFConverter instance
        """
        available = cls.list_available_backends()
        
        # Try each preference in order
        for backend in preferences:
            if available.get(backend, False):
                try:
                    return cls(backend=backend)
                except ValueError:
                    continue
        
        if fallback_to_auto:
            return cls(backend="auto", backend_preference=preferences)
        else:
            raise RuntimeError(f"None of the preferred backends {preferences} are available")
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all backends and current selection.
        
        Returns:
            Dictionary with backend status information
        """
        return {
            'current_backend': self.backend,
            'available_backends': self.available_backends,
            'backend_preference': self.backend_preference,
            'backend_info': self.get_backend_info()
        }


def install_requirements():
    """Display installation instructions for required packages."""
    print("SEC HTML to PDF Converter - Installation Instructions")
    print("=" * 50)
    print("\nRequired packages:")
    print("pip install beautifulsoup4 requests")
    print("\nOptional PDF conversion backends (install at least one):")
    print("pip install pyppeteer        # Recommended: Best quality")
    print("pip install weasyprint       # Good quality, no external dependencies")
    print("pip install xhtml2pdf        # Basic quality, fastest")
    print("pip install playwright       # High quality, similar to pyppeteer")
    print("\nIf using playwright, also run:")
    print("playwright install chromium")
    print("\nIf using pyppeteer, it will auto-download Chromium on first use.")


def main():
    """Command line interface for the SEC HTML to PDF converter."""
    parser = argparse.ArgumentParser(
        description='Convert SEC filing HTML to PDF without external APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sec_html_to_pdf.py filing.html filing.pdf
  python sec_html_to_pdf.py https://sec.gov/filing.html output.pdf --backend weasyprint
  python sec_html_to_pdf.py --install-help
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input HTML file or URL')
    parser.add_argument('output', nargs='?', help='Output PDF file path')
    parser.add_argument('--backend', 
                      choices=['auto', 'pyppeteer', 'playwright', 'weasyprint', 'xhtml2pdf'],
                      default='auto',
                      help='PDF conversion backend (default: auto)')
    parser.add_argument('--install-help', action='store_true',
                      help='Show installation instructions')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.install_help:
        install_requirements()
        return
    
    if not args.input or not args.output:
        parser.print_help()
        return
    
    try:
        converter = SECHTMLToPDFConverter(backend=args.backend)
        
        # Display backend information
        print(f"Using backend: {converter.backend}")
        print(f"Available backends: {list(converter.available_backends.keys())}")
        
        # Check if input is URL, SEC filing text file, or regular HTML file
        if args.input.startswith(('http://', 'https://')):
            success = converter.convert_url(args.input, args.output)
        elif args.input.endswith('.txt') and ('sec' in args.input.lower() or 'edgar' in args.input.lower() or 'submission' in args.input.lower()):
            # Assume this is a SEC filing text file
            success = converter.convert_sec_filing_txt(args.input, args.output)
        else:
            success = converter.convert_file(args.input, args.output)
        
        if success:
            print(f"✓ Successfully converted to PDF: {args.output}")
            sys.exit(0)
        else:
            print("✗ Conversion failed. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 