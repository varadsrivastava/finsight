import os
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import autogen
from autogen import AssistantAgent
import requests
import json
import logging
from datetime import datetime, timedelta
import openai
from tqdm import tqdm

from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig
from agents.pdf_analyzer import PDFAnalyzer

logger = logging.getLogger(__name__)

class FinancialChartGenerator:
    """Generate financial charts and visualizations"""
    
    def __init__(self, output_path: str = "./outputs/charts"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Set style for professional-looking charts
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Flag to control image display
        self.show_plots = True
    
    def display_image(self, fig):
        """Display the matplotlib figure if show_plots is True"""
        if self.show_plots:
            plt.show()
    
    def create_price_chart(self, price_data: Dict[str, Any], symbol: str) -> str:
        """Create a price and volume chart"""
        try:
            hist_data = price_data.get("historical_data", {})
            dates = pd.to_datetime(hist_data.get("dates", []))
            pct_changes = hist_data.get("pct_changes", [])
            spy_pct_changes = hist_data.get("spy_pct_changes", [])
            
            if not dates.empty and pct_changes and spy_pct_changes:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot percentage changes
                ax.plot(dates, pct_changes, linewidth=2, color='#2E86AB', label=f'{symbol} Change %')
                ax.plot(dates, spy_pct_changes, linewidth=2, color='#F18F01', label='S&P 500 Change %')
                
                ax.set_title(f'{symbol} vs S&P 500 - Change % Over the Past Year', fontsize=16, fontweight='bold')
                ax.set_ylabel('Change %', fontsize=12)
                ax.set_xlabel('Date', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                plt.tight_layout()

                # Save chart
                chart_path = os.path.join(self.output_path, f"{symbol}_vs_spy_chart.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                
                # Display the chart
                self.display_image(fig)
                
                plt.close()
                
                return chart_path
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return None
    
    def create_financial_ratios_chart(self, financial_data: Dict[str, Any], symbol: str) -> str:
        """Create a financial ratios visualization"""
        try:
            metrics = financial_data.get("key_metrics", {})
            
            # Extract key ratios
            ratios = {
                'Profit Margin': metrics.get('profit_margin', 0) or 0,
                'Operating Margin': metrics.get('operating_margin', 0) or 0,
                'ROE': metrics.get('return_on_equity', 0) or 0,
                'ROA': metrics.get('return_on_assets', 0) or 0,
                'Current Ratio': metrics.get('current_ratio', 0) or 0,
            }
            
            # Filter out None values and convert to percentages where appropriate
            clean_ratios = {}
            for key, value in ratios.items():
                if value is not None and value != 0:
                    if key in ['Profit Margin', 'Operating Margin', 'ROE', 'ROA']:
                        clean_ratios[key] = value * 100  # Convert to percentage
                    else:
                        clean_ratios[key] = value
            
            if clean_ratios:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                keys = list(clean_ratios.keys())
                values = list(clean_ratios.values())
                
                bars = ax.bar(keys, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
                ax.set_title(f'{symbol} Key Financial Ratios', fontsize=16, fontweight='bold')
                ax.set_ylabel('Value (%)', fontsize=12)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%' if value < 100 else f'{value:.0f}%',
                           ha='center', va='bottom', fontsize=10)
                
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Display the chart
                self.display_image(fig)
                
                chart_path = os.path.join(self.output_path, f"{symbol}_ratios_chart.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return chart_path
                
        except Exception as e:
            logger.error(f"Error creating ratios chart: {e}")
            return None
    
    def create_news_sentiment_chart(self, news_data: Dict[str, Any], symbol: str) -> str:
        """Create a news sentiment visualization"""
        try:
            articles = news_data.get("articles", [])
            if not articles:
                return None
            
            # Count sentiment
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for article in articles:
                sentiment = article.get("sentiment", "neutral")
                sentiment_counts[sentiment] += 1
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = list(sentiment_counts.keys())
            sizes = list(sentiment_counts.values())
            colors = ['#6A994E', '#C73E1D', '#A8DADC']
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 12})
            
            ax.set_title(f'{symbol} News Sentiment Analysis', fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Display the chart
            self.display_image(fig)
            
            chart_path = os.path.join(self.output_path, f"{symbol}_sentiment_chart.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {e}")
            return None
    
    def create_pe_eps_chart(self, financial_data: Dict[str, Any], symbol: str) -> str:
        """Create a chart showing P/E ratio and EPS over time"""
        try:
            # Extract income statement data
            income_stmt = financial_data.get("income_statement", {})
            max_prices = financial_data.get("max_prices", {})
            info = financial_data.get("key_metrics", {})
            
            if not income_stmt:
                logger.error("No income statement data available")
                return None
            
            if not max_prices:
                logger.error("No historical stock price data available")
                return None
                
            # Convert dates and sort chronologically
            dates = []
            eps_values = []
            pe_values = []
            
            # Get EPS from income statement
            for date_str, data in income_stmt.items():
                try:
                    date = pd.to_datetime(date_str)
                    # Calculate EPS from net income and shares outstanding if available
                    net_income = data.get("Net Income", 0)
                    shares = data.get("Diluted Average Shares", None)
                    if net_income and shares:
                        eps = net_income / shares
                        eps_values.append(eps)
                        dates.append(date)
                        
                    # Calculate P/E using historical stock prices if available
                    if max_prices[str(date_str)] and eps != 0:
                        pe = max_prices[str(date_str)] / eps
                        pe_values.append(pe)
                    else:
                        pe_values.append(None)

                except Exception as e:
                    logger.warning(f"Error processing date {date_str}: {e}")
                    continue
            
            if not dates or not eps_values:
                logger.error("No valid EPS data found")
                return None
            
            # Create the plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot EPS
            color1 = '#2E86AB'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('EPS ($)', color=color1)
            line1 = ax1.plot(dates, eps_values, color=color1, linewidth=2, label='EPS')
            ax1.tick_params(axis='y', labelcolor=color1)
            
            # Plot P/E ratio on secondary axis if available
            if any(pe_values):
                ax2 = ax1.twinx()
                color2 = '#F18F01'
                ax2.set_ylabel('P/E Ratio', color=color2)
                line2 = ax2.plot(dates, pe_values, color=color2, linewidth=2, linestyle='--', label='P/E Ratio')
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Add both lines to legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
            else:
                ax1.legend(loc='upper left')
            
            ax1.grid(True, alpha=0.3)
            plt.title(f'{symbol} - EPS and P/E Ratio Over Time', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(self.output_path, f"{symbol}_pe_eps_chart.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            
            # Display the chart
            self.display_image(fig)
            
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating P/E and EPS chart: {e}")
            return None
        

class AnnualReportAnalyzer:
    """Agent for analyzing annual report PDF"""
    
    def __init__(self, config: FinSightConfig, memory_manager: SharedMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.pdf_analyzer = PDFAnalyzer(self.config, memory_manager)
        
    def analyze_annual_report_pdf(self, symbol: str, year: Optional[str] = None) -> str:
        """Analyze annual report PDF using ColPali retrieval and vision models"""
        try:

            # check if retrieved images json exists
            retrieved_images_path = os.path.join(self.config.sec_filings_path, symbol, "10-K", "retrieved_images.json")
            
            if not os.path.exists(retrieved_images_path):
                print(f"Retrieved images not found for {symbol}. Analyzing annual report...")
                retrieved_images = self.analyze_annual_report(symbol, year)
                return retrieved_images
            
            # load retrieved images
            with open(retrieved_images_path, "r") as f:
                retrieved_images = json.load(f)

            analysis_result = self.report_analysis_gpt(retrieved_images)

            return {"success": True, "data": analysis_result}

                                    
        except Exception as e:
            return f"Error analyzing annual report for {symbol}: {e}"
            
        
    def analyze_annual_report(self, symbol: str, year: Optional[str] = None) -> str:
        """Analyze annual report PDF"""
        try:

            # Load model
            self.pdf_analyzer.load_model()
            
            # Find PDF
            selected_pdf = self.pdf_analyzer.find_annual_report(symbol, year)
            if not selected_pdf:
                return f"Annual report PDF not found for {symbol}"
            
            # Extract pages as images
            images, page_texts = self.pdf_analyzer.extract_pages_as_images(selected_pdf)
            
            # Create index if not exists
            pdf_images, page_embeddings = self.pdf_analyzer.create_retrieval_index(symbol, images)
            # return f"Failed to create retrieval index for {symbol}"

            # Create query embeddings
            queries, query_embeddings = self.pdf_analyzer.create_query_embeddings(self.pdf_analyzer.analysis_aspects)
            
            # Retrieve pages for aspect
            retrieved_images_for_gpt = self.pdf_analyzer.retrieve_pages_for_aspect(selected_pdf, queries, pdf_images, symbol, query_embeddings, page_embeddings, k=5)
            
            return retrieved_images_for_gpt            
            
                
        except Exception as e:
            return f"Error analyzing annual report for {symbol}: {e}"
            
    
    def report_analysis_gpt(self, retrieved_images):
        """Analyze annual report PDF using GPT-4o mini"""
        try:
            # Initialize the OpenAI client
            client = openai.OpenAI()

            # List to store the results
            gpt_analysis_results = []

            # Get aspects and instructions from pdf_analyzer
            queries = self.pdf_analyzer.analysis_aspects
            # instructions = [self.pdf_analyzer.analysis_aspects[query]["instruction"] for query in queries]

            # Iterate through the retrieved images and make API calls


            # Define the instructions for each query
            for item in tqdm(retrieved_images):
                try:
                    query_instruction = self.pdf_analyzer.analysis_aspects_instructions[item["query"]]

                    page_number = item["page_number"]
                    base64_image = item["base64_image"]

                    # Construct the messages list
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a Financial MultiModal Analyzer that analyzes images from the company's annual report based on the query. If the image does not contain any relevant information, only mention that no relevant information is present."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query_instruction},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        },
                    ]

                    # Make the API call
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=300,
                    )

                    # Store the response with the corresponding query and page number
                    gpt_analysis_results.append({
                        "query": item["query"],
                        "page_number": page_number,
                        "image": base64_image,
                        "gpt_response": response.choices[0].message.content
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing image for query '{item['query']}', page {item['page_number']}: {e}")
                    gpt_analysis_results.append({
                        "query": item["query"],
                        "page_number": page_number,
                        "image": base64_image,
                        "error": str(e)
                    })

            return gpt_analysis_results
                        
        except Exception as e:
            logger.error(f"Error analyzing annual report PDF using GPT-4o mini: {e}")
            return []



class MultimodalAnalyzerAgent:
    """Agent for analyzing financial images, charts, and multimodal content"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.chart_generator = FinancialChartGenerator(self.config.charts_output_path)
        self.pdf_analyzer = PDFAnalyzer(self.config, memory_manager)
        self.client = openai.OpenAI(api_key=self.config.openai_api_key)
        self.annual_report_analyzer = AnnualReportAnalyzer(self.config, memory_manager)
        
        # Create the Autogen agent
        self.agent = AssistantAgent(
            name="MultimodalAnalyzer",
            system_message=self._get_system_message(),
            llm_config=AGENT_CONFIGS["multimodal_analyzer"],
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
        )
        
        # Register tools
        self._register_tools()
    
    def _get_system_message(self) -> str:
        return """You are a MultimodalAnalyzer agent specialized in financial text, image and chart analysis.

Your responsibilities:
1. Generate professional financial charts and visualizations
2. Analyze financial images, charts, and tables using vision models
3. Analyze annual report PDFs using ColPali retrieval and vision models
4. Extract insights from visual financial data and PDF documents
5. Assess data quality and reliability in visual materials
6. Provide investment-relevant conclusions from multimodal content

For each analysis:
- Create clear, professional visualizations when generating charts
- Use vision models for image understanding and PDF analysis
- Extract quantitative data from visual sources and PDF documents
- Provide actionable financial insights
- Store results in shared memory with proper metadata

PDF Analysis capabilities:
- Find and analyze annual report PDFs for companies
- Extract and analyze specific financial aspects: income statement, balance sheet, cash flow, business segments, risk factors, business summary, company description
- Use ColPali for intelligent page retrieval
- Analyze retrieved pages with vision models using specialized financial prompts

Always prioritize data accuracy and completeness. If data is missing or uncertain, clearly indicate this in your response.

Reply TERMINATE when all requested analysis is complete."""
    
    def _register_tools(self):
        """Register multimodal analysis tools with the agent"""
        
        def generate_price_chart(symbol: str) -> str:
            """Generate price and volume chart from stored data"""
            try:
                # Search for stock data in memory
                entries = self.memory_manager.search_entries(
                    query=f"stock data {symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                
                if not entries:
                    return f"No stock data found for {symbol}. Please collect data first."
                
                # Get the full entry
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if not entry:
                    return "Could not retrieve stock data entry."
                
                # Generate chart
                chart_path = self.chart_generator.create_price_chart(
                    entry["content"], symbol
                )
                
                if chart_path:
                    # Store chart information in memory
                    self.memory_manager.store_entry(
                        agent_name="MultimodalAnalyzer",
                        content_type="chart",
                        content={
                            "chart_type": "price_volume",
                            "symbol": symbol,
                            "file_path": chart_path,
                            "description": f"Price and volume chart for {symbol}"
                        },
                        metadata={"symbol": symbol, "chart_type": "price_volume"},
                        tags=["chart", "price", "volume", symbol.lower()]
                    )
                    
                    return f"Price chart generated and saved: {chart_path}"
                else:
                    return "Failed to generate price chart"
                    
            except Exception as e:
                return f"Error generating price chart: {e}"
        
        def generate_ratios_chart(symbol: str) -> str:
            """Generate financial ratios chart from stored data"""
            try:
                # Search for financial data in memory
                entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                
                if not entries:
                    return f"No financial data found for {symbol}. Please collect data first."
                
                # Get the full entry
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if not entry:
                    return "Could not retrieve financial data entry."
                
                # Generate chart
                chart_path = self.chart_generator.create_financial_ratios_chart(
                    entry["content"], symbol
                )
                
                if chart_path:
                    # Store chart information in memory
                    self.memory_manager.store_entry(
                        agent_name="MultimodalAnalyzer",
                        content_type="chart",
                        content={
                            "chart_type": "financial_ratios",
                            "symbol": symbol,
                            "file_path": chart_path,
                            "description": f"Financial ratios chart for {symbol}"
                        },
                        metadata={"symbol": symbol, "chart_type": "financial_ratios"},
                        tags=["chart", "ratios", "financial", symbol.lower()]
                    )
                    
                    return f"Financial ratios chart generated and saved: {chart_path}"
                else:
                    return "Failed to generate ratios chart"
                    
            except Exception as e:
                return f"Error generating ratios chart: {e}"
        
        def generate_pe_eps_chart(symbol: str) -> str:
            """Generate P/E ratio and EPS chart from stored data"""
            try:
                # Search for financial data in memory
                entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                
                if not entries:
                    return f"No financial data found for {symbol}. Please collect data first."
                
                # Get the full entry
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if not entry:
                    return "Could not retrieve financial data entry."
                
                # Generate chart
                chart_path = self.chart_generator.create_pe_eps_chart(
                    entry["content"], symbol
                )
                
                if chart_path:
                    # Store chart information in memory
                    self.memory_manager.store_entry(
                        agent_name="MultimodalAnalyzer",
                        content_type="chart",
                        content={
                            "chart_type": "pe_eps",
                            "symbol": symbol,
                            "file_path": chart_path,
                            "description": f"P/E ratio and EPS chart for {symbol}"
                        },
                        metadata={"symbol": symbol, "chart_type": "pe_eps"},
                        tags=["chart", "pe_ratio", "eps", symbol.lower()]
                    )
                    
                    return f"P/E and EPS chart generated and saved: {chart_path}"
                else:
                    return "Failed to generate P/E and EPS chart"
                    
            except Exception as e:
                return f"Error generating P/E and EPS chart: {e}"
        
        def generate_sentiment_chart(symbol: str) -> str:
            """Generate news sentiment chart from stored data"""
            try:
                # Search for news data in memory
                entries = self.memory_manager.search_entries(
                    query=f"news {symbol}",
                    content_type_filter="news",
                    n_results=1
                )
                
                if not entries:
                    return f"No news data found for {symbol}. Please collect data first."
                
                # Get the full entry
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if not entry:
                    return "Could not retrieve news data entry."
                
                # Generate chart
                chart_path = self.chart_generator.create_news_sentiment_chart(
                    entry["content"], symbol
                )
                
                if chart_path:
                    # Store chart information in memory
                    self.memory_manager.store_entry(
                        agent_name="MultimodalAnalyzer",
                        content_type="chart",
                        content={
                            "chart_type": "news_sentiment",
                            "symbol": symbol,
                            "file_path": chart_path,
                            "description": f"News sentiment chart for {symbol}"
                        },
                        metadata={"symbol": symbol, "chart_type": "news_sentiment"},
                        tags=["chart", "sentiment", "news", symbol.lower()]
                    )
                    
                    return f"News sentiment chart generated and saved: {chart_path}"
                else:
                    return "Failed to generate sentiment chart"
                    
            except Exception as e:
                return f"Error generating sentiment chart: {e}"
    
        
        def create_comprehensive_visualization(symbol: str) -> str:
            """Create a comprehensive set of visualizations for a symbol"""
            try:
                results = []
                
                # Generate all available charts
                price_result = generate_price_chart(symbol)
                results.append(f"Price Chart: {price_result}")
                
                ratios_result = generate_ratios_chart(symbol)
                results.append(f"Ratios Chart: {ratios_result}")
                
                pe_eps_result = generate_pe_eps_chart(symbol)
                results.append(f"P/E and EPS Chart: {pe_eps_result}")
                
                sentiment_result = generate_sentiment_chart(symbol)
                results.append(f"Sentiment Chart: {sentiment_result}")
                
                # Store comprehensive visualization summary
                self.memory_manager.store_entry(
                    agent_name="MultimodalAnalyzer",
                    content_type="visualization_summary",
                    content={
                        "symbol": symbol,
                        "charts_generated": results,
                        "timestamp": datetime.now().isoformat()
                    },
                    metadata={"symbol": symbol, "visualization_type": "comprehensive"},
                    tags=["visualization", "comprehensive", symbol.lower()]
                )
                
                return f"Comprehensive visualization created for {symbol}. Results: " + "; ".join(results)
                
            except Exception as e:
                return f"Error creating comprehensive visualization: {e}"
            
        
        def collect_annual_report_analysis(symbol: str, year: Optional[str] = None) -> str:
            """Analyze annual report PDF using ColPali retrieval and vision models"""
            try:
                # First check if we already have PDF analysis for this symbol in memory
                existing_entries = self.memory_manager.search_entries(
                    query=f"pdf analysis {symbol} annual report",
                    content_type_filter=None,  
                    n_results=7
                )
                
                # Filter for PDF analysis entries specifically
                pdf_analysis_entries = []
                for entry in existing_entries:
                    entry_data = self.memory_manager.get_entry_by_id(entry["id"])
                    if (entry_data and 
                        entry_data.get("agent_name", "").startswith("MultimodalAnalyzer") and
                        entry_data.get("content_type", "").startswith("pdf_analysis_") and
                        entry_data.get("metadata", {}).get("symbol") == symbol):
                        pdf_analysis_entries.append(entry_data)
                
                if pdf_analysis_entries:
                    # Return summary of existing analysis
                    aspects_found = [entry["metadata"].get("aspect", "unknown") for entry in pdf_analysis_entries]
                    entry_ids = [entry["id"] for entry in pdf_analysis_entries]
                    return f"PDF analysis already exists for {symbol}. Found analysis for aspects: {aspects_found}. Entry IDs: {entry_ids}. Use existing data or delete entries to reanalyze."
                
                # If no existing analysis found, proceed with new analysis
                result = self.annual_report_analyzer.analyze_annual_report_pdf(symbol, year)

                # Store in memory
                # combine results for each query
                if result["success"]:
                    combined_results = {}
                    analysis_result = result["data"]
                    for item in analysis_result:
                        if item["query"] not in combined_results:
                            combined_results[item["query"]] = f"Analysis for Page number: {item["page_number"]} \n {item["gpt_response"]}"
                        else:
                            combined_results[item["query"]] += f"\n\nAnalysis for Page number: {item["page_number"]} \n {item["gpt_response"]}"
                    
                    entry_ids = []
                    for aspect in combined_results.keys():
                        entry_id = self.memory_manager.store_entry(
                            agent_name="MultimodalAnalyzer",
                            content_type=f"pdf_analysis_{aspect.strip().replace(' ', '_').lower()}",
                            content=combined_results[aspect],
                            metadata={
                                "symbol": symbol,
                                "aspect": aspect,
                                # "pdf_path": os.path.dirname(retrieved_images_path),
                                # "pages_analyzed": item["page_number"]
                            },
                            tags=["pdf_analysis", aspect.lower(), symbol.lower(), "annual_report"]
                        )
                        entry_ids.append(entry_id)
                    
                    return f"Successfully analyzed Report for {symbol}. Analysis stored with IDs: {entry_ids}."
                else:
                    return f"Failed to analyze annual report for {symbol}: {result.get('error', 'Unknown error')}"
                    
            except Exception as e:
                return f"Error analyzing annual report PDF: {e}"

        

        # Register functions with autogen
        self.agent.register_for_execution(name="generate_price_chart")(generate_price_chart)
        self.agent.register_for_execution(name="generate_ratios_chart")(generate_ratios_chart)
        self.agent.register_for_execution(name="generate_pe_eps_chart")(generate_pe_eps_chart)
        self.agent.register_for_execution(name="generate_sentiment_chart")(generate_sentiment_chart)
        self.agent.register_for_execution(name="create_comprehensive_visualization")(create_comprehensive_visualization)
        self.agent.register_for_execution(name="collect_annual_report_analysis")(collect_annual_report_analysis)