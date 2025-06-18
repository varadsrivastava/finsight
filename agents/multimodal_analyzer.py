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
from datetime import datetime

from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig

logger = logging.getLogger(__name__)

class FinancialChartGenerator:
    """Generate financial charts and visualizations"""
    
    def __init__(self, output_path: str = "./outputs/charts"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Set style for professional-looking charts
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_price_chart(self, price_data: Dict[str, Any], symbol: str) -> str:
        """Create a price and volume chart"""
        try:
            hist_data = price_data.get("historical_data", {})
            dates = pd.to_datetime(hist_data.get("dates", []))
            prices = hist_data.get("prices", [])
            volumes = hist_data.get("volumes", [])
            
            if not dates.empty and prices:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
                
                # Price chart
                ax1.plot(dates, prices, linewidth=2, color='#2E86AB', label='Close Price')
                ax1.set_title(f'{symbol} Stock Price', fontsize=16, fontweight='bold')
                ax1.set_ylabel('Price ($)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Volume chart
                ax2.bar(dates, volumes, color='#A23B72', alpha=0.7, width=1)
                ax2.set_title('Trading Volume', fontsize=14)
                ax2.set_ylabel('Volume', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save chart
                chart_path = os.path.join(self.output_path, f"{symbol}_price_chart.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
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
                           f'{value:.1f}%' if abs(value) < 10 else f'{value:.1f}',
                           ha='center', va='bottom', fontsize=10)
                
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
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
            
            chart_path = os.path.join(self.output_path, f"{symbol}_sentiment_chart.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.error(f"Error creating sentiment chart: {e}")
            return None

class FinTralImageAnalyzer:
    """Placeholder for FinTral multimodal analysis"""
    
    def __init__(self, config: FinSightConfig):
        self.config = config
        # In a real implementation, this would connect to FinTral API
        self.fintral_available = False
    
    def analyze_financial_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze financial charts, tables, or documents using FinTral"""
        try:
            # Placeholder implementation - in production, integrate with actual FinTral
            
            # For now, use GPT-4V as a substitute
            if self.config.openai_api_key:
                return self._analyze_with_gpt4v(image_path)
            else:
                return {
                    "success": False,
                    "error": "No multimodal analysis available. Please configure FinTral or GPT-4V."
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"success": False, "error": str(e)}
    
    def _analyze_with_gpt4v(self, image_path: str) -> Dict[str, Any]:
        """Use GPT-4V as substitute for FinTral analysis"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this financial image/chart/table. Provide:
                                1. Type of visualization (chart, table, document, etc.)
                                2. Key data points and trends
                                3. Financial insights and implications
                                4. Data quality assessment
                                5. Investment-relevant conclusions
                                
                                Structure your response as JSON with these sections."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post("https://api.openai.com/v1/chat/completions", 
                                   headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result["choices"][0]["message"]["content"]
                
                return {
                    "success": True,
                    "analysis_type": "gpt4v_substitute",
                    "analysis": analysis_text,
                    "image_path": image_path,
                    "model": "gpt-4-vision-preview"
                }
            else:
                return {
                    "success": False,
                    "error": f"API call failed: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error in GPT-4V analysis: {e}")
            return {"success": False, "error": str(e)}

class MultimodalAnalyzerAgent:
    """Agent for analyzing financial images, charts, and multimodal content"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.chart_generator = FinancialChartGenerator(self.config.charts_output_path)
        self.image_analyzer = FinTralImageAnalyzer(self.config)
        
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
        return """You are a MultimodalAnalyzer agent specialized in financial image and chart analysis.

Your responsibilities:
1. Generate professional financial charts and visualizations
2. Analyze financial images, charts, and tables using FinTral capabilities
3. Extract insights from visual financial data
4. Assess data quality and reliability in visual materials
5. Provide investment-relevant conclusions from multimodal content

For each analysis:
- Create clear, professional visualizations when generating charts
- Use FinTral (or GPT-4V substitute) for image understanding
- Extract quantitative data from visual sources
- Provide actionable financial insights
- Store results in shared memory with proper metadata

Focus on accuracy and clear communication of visual financial information.

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
        
        def analyze_financial_image(image_path: str, analysis_context: str = "general") -> str:
            """Analyze financial image using FinTral capabilities"""
            try:
                if not os.path.exists(image_path):
                    return f"Image file not found: {image_path}"
                
                # Analyze image
                result = self.image_analyzer.analyze_financial_image(image_path)
                
                if result["success"]:
                    # Store analysis in memory
                    entry_id = self.memory_manager.store_entry(
                        agent_name="MultimodalAnalyzer",
                        content_type="image_analysis",
                        content={
                            "image_path": image_path,
                            "analysis_context": analysis_context,
                            "analysis_result": result,
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={
                            "image_path": image_path,
                            "analysis_type": result.get("analysis_type", "unknown"),
                            "model": result.get("model", "unknown")
                        },
                        tags=["image_analysis", "multimodal", analysis_context]
                    )
                    
                    return f"Image analysis completed and stored (ID: {entry_id}). Analysis: {result['analysis'][:200]}..."
                else:
                    return f"Image analysis failed: {result['error']}"
                    
            except Exception as e:
                return f"Error analyzing image: {e}"
        
        def create_comprehensive_visualization(symbol: str) -> str:
            """Create a comprehensive set of visualizations for a symbol"""
            try:
                results = []
                
                # Generate all available charts
                price_result = generate_price_chart(symbol)
                results.append(f"Price Chart: {price_result}")
                
                ratios_result = generate_ratios_chart(symbol)
                results.append(f"Ratios Chart: {ratios_result}")
                
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
        
        # Register functions with autogen
        self.agent.register_for_execution(name="generate_price_chart")(generate_price_chart)
        self.agent.register_for_execution(name="generate_ratios_chart")(generate_ratios_chart)
        self.agent.register_for_execution(name="generate_sentiment_chart")(generate_sentiment_chart)
        self.agent.register_for_execution(name="analyze_financial_image")(analyze_financial_image)
        self.agent.register_for_execution(name="create_comprehensive_visualization")(create_comprehensive_visualization)