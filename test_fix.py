#!/usr/bin/env python3
"""
Test the SEC filings fix
"""

import logging
from agents.data_collector import DataCollectorTools
from config.config import FinSightConfig

# Set up logging to see debug messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_fix():
    print("=== Testing SEC Filings Fix ===")
    
    # Initialize tools
    config = FinSightConfig()
    tools = DataCollectorTools(config)
    
    # Test SEC filings
    result = tools.get_sec_filings("AAPL", filing_types=["10-K"], max_filings=1)
    
    print(f"Success: {result['success']}")
    if result['success']:
        data = result['data']
        print(f"Source: {data.get('source')}")
        print(f"API calls made: {data.get('api_calls_made', 0)}")
        print(f"Files reused: {data.get('files_reused', 0)}")
        print(f"Filings downloaded: {len(data.get('filings_downloaded', []))}")
        print(f"Filing summaries: {len(data.get('filing_summaries', []))}")
        
        # Show first filing if available
        if data.get('filings_downloaded'):
            first_filing = data['filings_downloaded'][0]
            print(f"\nFirst filing:")
            print(f"  Path: {first_filing.get('filing_path')}")
            print(f"  Method: {first_filing.get('download_method')}")
            print(f"  URL: {first_filing.get('source_url')}")
    else:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_fix()