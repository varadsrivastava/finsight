from sec_edgar_downloader import Downloader
import os
import argparse

def download_sec_filings(ticker, filing_type="10-K", limit=1, output_dir="sec_filings_txt"):
    """
    Download SEC filings for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., AAPL, MSFT)
        filing_type (str): Type of filing to download (e.g., "10-K", "10-Q", "8-K")
        limit (int): Maximum number of filings to download
        output_dir (str): Directory to save the downloaded filings
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the downloader
        # Note: Replace with your company name and email
        dl = Downloader(
            company_name="YourCompanyName",  # Replace with your company name
            email_address="your.email@example.com",  # Replace with your email
            download_folder=output_dir
        )
        
        print(f"Downloading {filing_type} filings for {ticker}...")
        
        # Download the filings
        dl.get(filing_type, ticker, limit=limit)
        
        print(f"Files have been downloaded to: {output_dir}/sec-edgar-filings/{ticker}/{filing_type}/")
        
    except Exception as e:
        print(f"Error downloading filings: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SEC filings for a ticker")
    parser.add_argument("ticker", default="AAPL", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--type", default="10-K", help="Filing type (default: 10-K)")
    parser.add_argument("--limit", type=int, default=1, help="Maximum number of filings to download (default: 1)")
    parser.add_argument("--output", default="sec_filings_txt", help="Output directory (default: sec_filings)")
    
    args = parser.parse_args()
    
    download_sec_filings(
        ticker=args.ticker,
        filing_type=args.type,
        limit=args.limit,
        output_dir=args.output
    ) 