#!/usr/bin/env python3
"""
Simple SEC Filing to PDF Converter
Converts a local SEC filing text file to PDF using xhtml2pdf.
"""

import os
import re
from xhtml2pdf import pisa

def extract_html_from_sec_filing(txt_file_path):
    """Extract HTML content from SEC filing text file."""
    try:
        with open(txt_file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Find HTML content between <TEXT> tags
        text_pattern = r'<TEXT>(.*?)</TEXT>'
        text_matches = re.findall(text_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not text_matches:
            raise ValueError("No HTML content found in SEC filing")
        
        # Use the longest text section (likely the main filing)
        html_content = max(text_matches, key=len)
        
        # If it's not already HTML, wrap it
        if '<html' not in html_content.lower():
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>SEC Filing</title>
                <meta charset="utf-8">
                <style>
                    body {{ 
                        font-family: 'Times New Roman', Times, serif;
                        font-size: 12px;
                        line-height: 1.4;
                    }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%;
                        margin: 10px 0;
                    }}
                    th, td {{ 
                        border: 1px solid black;
                        padding: 4px;
                        text-align: left;
                    }}
                </style>
            </head>
            <body>
            {html_content}
            </body>
            </html>
            """
        
        return html_content
    except Exception as e:
        print(f"Error extracting HTML: {e}")
        return None

def convert_to_pdf(html_content, output_pdf_path):
    """Convert HTML content to PDF using xhtml2pdf."""
    try:
        with open(output_pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_content,
                dest=pdf_file,
                encoding='utf-8'
            )
        
        if pisa_status.err:
            print(f"Error converting to PDF: {pisa_status.err}")
            return False
        return True
    except Exception as e:
        print(f"Error during PDF conversion: {e}")
        return False

def main():
    # File paths
    sec_filing_path = r"outputs\charts\sec_filings\AAPL\sec-edgar-filings\AAPL\10-Q\0000320193-24-000069\full-submission.txt"
    output_pdf_path = r"outputs\charts\sec_filings\AAPL\sec-edgar-filings\AAPL\10-Q\0000320193-24-000069\AAPL_10Q1_Filing.pdf"
    
    print("SEC Filing to PDF Converter")
    print("=" * 40)
    
    # Check if input file exists
    if not os.path.exists(sec_filing_path):
        print(f"Error: SEC filing not found at: {sec_filing_path}")
        return
    
    # Extract HTML from SEC filing
    print("Extracting HTML from SEC filing...")
    html_content = extract_html_from_sec_filing(sec_filing_path)
    
    if not html_content:
        print("Failed to extract HTML content")
        return
    
    # Convert to PDF
    print("Converting to PDF...")
    if convert_to_pdf(html_content, output_pdf_path):
        print(f"✅ Successfully created PDF: {output_pdf_path}")
    else:
        print("❌ Failed to create PDF")

if __name__ == "__main__":
    main()