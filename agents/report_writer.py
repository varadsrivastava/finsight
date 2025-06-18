import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import autogen
from autogen import AssistantAgent
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import logging

from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig

logger = logging.getLogger(__name__)

class EquityReportGenerator:
    """Generate professional equity research reports"""
    
    def __init__(self, output_path: str = "./outputs/reports"):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceAfter=12,
            spaceBefore=20
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkgreen,
            spaceAfter=8,
            spaceBefore=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=12,
            backgroundColor=colors.lightgrey,
            borderPadding=10,
            spaceAfter=12
        ))
    
    def generate_equity_report(self, symbol: str, company_data: Dict[str, Any], 
                             analyses: Dict[str, Any], charts: List[str] = None) -> str:
        """Generate a comprehensive equity research report"""
        try:
            report_filename = f"{symbol}_equity_report_{datetime.now().strftime('%Y%m%d')}.pdf"
            report_path = os.path.join(self.output_path, report_filename)
            
            # Create document
            doc = SimpleDocTemplate(report_path, pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Build story (content)
            story = []
            
            # Title page
            story.extend(self._create_title_page(symbol, company_data))
            
            # Executive summary
            story.extend(self._create_executive_summary(symbol, analyses))
            
            # Company overview
            story.extend(self._create_company_overview(symbol, company_data))
            
            # Financial analysis
            story.extend(self._create_financial_analysis(analyses))
            
            # Valuation analysis
            story.extend(self._create_valuation_section(analyses))
            
            # Market position and trends
            story.extend(self._create_market_analysis(analyses))
            
            # Charts and visuals
            if charts:
                story.extend(self._create_charts_section(charts))
            
            # Investment recommendation
            story.extend(self._create_recommendation_section(analyses))
            
            # Risk analysis
            story.extend(self._create_risk_section(analyses))
            
            # Disclaimer
            story.extend(self._create_disclaimer())
            
            # Build PDF
            doc.build(story)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating equity report: {e}")
            return None
    
    def _create_title_page(self, symbol: str, company_data: Dict[str, Any]) -> List:
        """Create title page"""
        story = []
        
        # Main title
        story.append(Paragraph(f"EQUITY RESEARCH REPORT", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))
        
        # Company info
        story.append(Paragraph(f"<b>{symbol}</b>", self.styles['Heading1']))
        
        current_price = company_data.get('stock_data', {}).get('current_price', 'N/A')
        market_cap = company_data.get('stock_data', {}).get('market_cap', 'N/A')
        
        if market_cap != 'N/A' and market_cap:
            market_cap_formatted = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
        else:
            market_cap_formatted = 'N/A'
        
        story.append(Paragraph(f"Current Price: ${current_price}", self.styles['Normal']))
        story.append(Paragraph(f"Market Cap: {market_cap_formatted}", self.styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Report metadata
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        story.append(Paragraph("Generated by FinSight AI Research System", self.styles['Normal']))
        
        story.append(PageBreak())
        return story
    
    def _create_executive_summary(self, symbol: str, analyses: Dict[str, Any]) -> List:
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        
        # Get comprehensive insight if available
        insight = analyses.get('comprehensive_insight', {})
        if insight:
            recommendation = insight.get('overall_recommendation', 'NEUTRAL')
            confidence = insight.get('confidence_level', 'MEDIUM')
            
            # Recommendation box
            rec_text = f"<b>INVESTMENT RECOMMENDATION: {recommendation}</b><br/>Confidence Level: {confidence}"
            story.append(Paragraph(rec_text, self.styles['HighlightBox']))
            
            # Key points
            strengths = insight.get('key_strengths', [])
            weaknesses = insight.get('key_weaknesses', [])
            
            if strengths:
                story.append(Paragraph("Key Strengths:", self.styles['SubHeader']))
                for strength in strengths[:3]:  # Top 3
                    story.append(Paragraph(f"• {strength}", self.styles['Normal']))
                story.append(Spacer(1, 6))
            
            if weaknesses:
                story.append(Paragraph("Key Concerns:", self.styles['SubHeader']))
                for weakness in weaknesses[:3]:  # Top 3
                    story.append(Paragraph(f"• {weakness}", self.styles['Normal']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("Comprehensive analysis data not available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_company_overview(self, symbol: str, company_data: Dict[str, Any]) -> List:
        """Create company overview section"""
        story = []
        
        story.append(Paragraph("COMPANY OVERVIEW", self.styles['SectionHeader']))
        
        stock_data = company_data.get('stock_data', {})
        
        # Basic metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Symbol', symbol],
            ['Current Price', f"${stock_data.get('current_price', 'N/A')}"],
            ['52-Week High', f"${stock_data.get('52_week_high', 'N/A')}"],
            ['52-Week Low', f"${stock_data.get('52_week_low', 'N/A')}"],
            ['P/E Ratio', f"{stock_data.get('pe_ratio', 'N/A')}"],
            ['Beta', f"{stock_data.get('beta', 'N/A')}"],
            ['Dividend Yield', f"{stock_data.get('dividend_yield', 'N/A')}%"]
        ]
        
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        return story
    
    def _create_financial_analysis(self, analyses: Dict[str, Any]) -> List:
        """Create financial analysis section"""
        story = []
        
        story.append(Paragraph("FINANCIAL ANALYSIS", self.styles['SectionHeader']))
        
        financial_health = analyses.get('financial_health', {})
        if financial_health:
            overall_score = financial_health.get('overall_score', 0)
            
            story.append(Paragraph(f"Financial Health Score: {overall_score}/100", self.styles['SubHeader']))
            
            # Individual category scores
            categories = ['profitability', 'liquidity', 'leverage', 'efficiency', 'growth']
            for category in categories:
                if category in financial_health:
                    cat_data = financial_health[category]
                    score = cat_data.get('score', 0)
                    analysis = cat_data.get('analysis', '')
                    
                    story.append(Paragraph(f"<b>{category.title()}</b> (Score: {score})", self.styles['Normal']))
                    story.append(Paragraph(analysis, self.styles['Normal']))
                    
                    concerns = cat_data.get('concerns', [])
                    if concerns:
                        for concern in concerns:
                            story.append(Paragraph(f"⚠ {concern}", self.styles['Normal']))
                    
                    story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("Financial health analysis not available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_valuation_section(self, analyses: Dict[str, Any]) -> List:
        """Create valuation analysis section"""
        story = []
        
        story.append(Paragraph("VALUATION ANALYSIS", self.styles['SectionHeader']))
        
        valuation = analyses.get('valuation', {})
        if valuation:
            conclusion = valuation.get('valuation_conclusion', 'Unknown')
            story.append(Paragraph(f"<b>Valuation Conclusion:</b> {conclusion}", self.styles['HighlightBox']))
            
            # Relative valuation
            rel_val = valuation.get('relative_valuation', {})
            if rel_val:
                story.append(Paragraph("Relative Valuation Metrics:", self.styles['SubHeader']))
                for metric, assessment in rel_val.items():
                    story.append(Paragraph(f"• <b>{metric}:</b> {assessment}", self.styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Risk factors
            risk_factors = valuation.get('risk_factors', [])
            if risk_factors:
                story.append(Paragraph("Valuation Risk Factors:", self.styles['SubHeader']))
                for risk in risk_factors:
                    story.append(Paragraph(f"• {risk}", self.styles['Normal']))
        else:
            story.append(Paragraph("Valuation analysis not available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_market_analysis(self, analyses: Dict[str, Any]) -> List:
        """Create market analysis section"""
        story = []
        
        story.append(Paragraph("MARKET POSITION & TRENDS", self.styles['SectionHeader']))
        
        market_position = analyses.get('market_position', {})
        if market_position:
            # Market performance
            market_perf = market_position.get('market_performance', {})
            if market_perf:
                range_pos = market_perf.get('52_week_range_position', 0)
                volatility = market_perf.get('volatility_assessment', 'Unknown')
                
                story.append(Paragraph(f"52-Week Range Position: {range_pos:.1f}%", self.styles['Normal']))
                story.append(Paragraph(f"Volatility Assessment: {volatility}", self.styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Sentiment analysis
            sentiment = market_position.get('sentiment_analysis', {})
            if sentiment:
                overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
                story.append(Paragraph(f"Market Sentiment: {overall_sentiment.title()}", self.styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Investment thesis
            thesis = market_position.get('investment_thesis', [])
            if thesis:
                story.append(Paragraph("Investment Thesis:", self.styles['SubHeader']))
                for point in thesis:
                    story.append(Paragraph(f"• {point}", self.styles['Normal']))
        else:
            story.append(Paragraph("Market position analysis not available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_charts_section(self, charts: List[str]) -> List:
        """Create charts and visuals section"""
        story = []
        
        story.append(Paragraph("CHARTS & VISUALIZATIONS", self.styles['SectionHeader']))
        
        for chart_path in charts:
            if os.path.exists(chart_path):
                try:
                    # Add chart image
                    img = Image(chart_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 12))
                    
                    # Add caption
                    chart_name = os.path.basename(chart_path).replace('.png', '').replace('_', ' ').title()
                    story.append(Paragraph(f"Figure: {chart_name}", self.styles['Normal']))
                    story.append(Spacer(1, 12))
                except Exception as e:
                    logger.warning(f"Could not add chart {chart_path}: {e}")
        
        return story
    
    def _create_recommendation_section(self, analyses: Dict[str, Any]) -> List:
        """Create investment recommendation section"""
        story = []
        
        story.append(Paragraph("INVESTMENT RECOMMENDATION", self.styles['SectionHeader']))
        
        insight = analyses.get('comprehensive_insight', {})
        if insight:
            recommendation = insight.get('overall_recommendation', 'NEUTRAL')
            confidence = insight.get('confidence_level', 'MEDIUM')
            
            # Recommendation details
            rec_text = f"""
            <b>Recommendation:</b> {recommendation}<br/>
            <b>Confidence Level:</b> {confidence}<br/>
            <b>Time Horizon:</b> {insight.get('time_horizon', 'Medium-term')}<br/>
            <b>Target Investor:</b> {insight.get('target_investor_profile', 'Balanced investors')}
            """
            
            story.append(Paragraph(rec_text, self.styles['HighlightBox']))
            
            # Investment rationale
            rationale = insight.get('investment_rationale', '')
            if rationale:
                story.append(Paragraph("Investment Rationale:", self.styles['SubHeader']))
                story.append(Paragraph(rationale, self.styles['Normal']))
        else:
            story.append(Paragraph("Investment recommendation not available.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_risk_section(self, analyses: Dict[str, Any]) -> List:
        """Create risk analysis section"""
        story = []
        
        story.append(Paragraph("RISK ANALYSIS", self.styles['SectionHeader']))
        
        # Collect all risk factors
        all_risks = []
        
        # From comprehensive insight
        insight = analyses.get('comprehensive_insight', {})
        if insight:
            all_risks.extend(insight.get('key_weaknesses', []))
        
        # From valuation analysis
        valuation = analyses.get('valuation', {})
        if valuation:
            all_risks.extend(valuation.get('risk_factors', []))
        
        # From market position
        market_position = analyses.get('market_position', {})
        if market_position:
            all_risks.extend(market_position.get('risks', []))
        
        if all_risks:
            # Remove duplicates
            unique_risks = list(set(all_risks))
            
            for risk in unique_risks:
                story.append(Paragraph(f"• {risk}", self.styles['Normal']))
        else:
            story.append(Paragraph("No specific risk factors identified.", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _create_disclaimer(self) -> List:
        """Create disclaimer section"""
        story = []
        
        story.append(Paragraph("IMPORTANT DISCLAIMER", self.styles['SectionHeader']))
        
        disclaimer_text = """
        This report is generated by an AI-powered financial analysis system and is for informational 
        purposes only. It should not be considered as investment advice, recommendation to buy or sell 
        securities, or financial planning guidance. 
        
        The analysis is based on publicly available data and may contain errors or omissions. Past 
        performance does not guarantee future results. All investments carry risk of loss.
        
        Please consult with a qualified financial advisor before making investment decisions.
        
        Generated by FinSight AI Research System.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        return story

class ReportWriterAgent:
    """Agent for generating comprehensive equity research reports"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.report_generator = EquityReportGenerator(self.config.reports_output_path)
        
        # Create the Autogen agent
        self.agent = AssistantAgent(
            name="ReportWriter",
            system_message=self._get_system_message(),
            llm_config=AGENT_CONFIGS["report_writer"],
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
        )
        
        # Register tools
        self._register_tools()
    
    def _get_system_message(self) -> str:
        return """You are a ReportWriter agent specialized in creating professional equity research reports.

Your responsibilities:
1. Compile comprehensive equity research reports from analysis data
2. Format reports with professional layouts and visualizations
3. Include executive summaries, detailed analysis, and recommendations
4. Ensure reports follow institutional research standards
5. Integrate charts and visual elements effectively
6. Provide clear investment recommendations with supporting rationale

Report structure should include:
- Executive Summary with key recommendation
- Company Overview with basic metrics
- Financial Health Analysis
- Valuation Analysis
- Market Position Assessment
- Risk Analysis
- Investment Recommendation
- Professional disclaimers

Generate clear, actionable reports that support investment decision-making.

Reply TERMINATE when the report is successfully generated."""
    
    def _register_tools(self):
        """Register report generation tools with the agent"""
        
        def generate_equity_report(symbol: str) -> str:
            """Generate a comprehensive equity research report"""
            try:
                # Collect all data and analyses for the symbol
                company_data = {}
                analyses = {}
                charts = []
                
                # Get stock data
                stock_entries = self.memory_manager.search_entries(
                    query=f"stock data {symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                if stock_entries:
                    stock_entry = self.memory_manager.get_entry_by_id(stock_entries[0]["id"])
                    if stock_entry:
                        company_data['stock_data'] = stock_entry["content"]
                
                # Get financial data
                financial_entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                if financial_entries:
                    financial_entry = self.memory_manager.get_entry_by_id(financial_entries[0]["id"])
                    if financial_entry:
                        company_data['financial_data'] = financial_entry["content"]
                
                # Get analyses
                analysis_types = [
                    ("financial_health_analysis", "financial_health"),
                    ("valuation_analysis", "valuation"),
                    ("market_position_analysis", "market_position"),
                    ("comprehensive_insight", "comprehensive_insight")
                ]
                
                for content_type, key in analysis_types:
                    entries = self.memory_manager.search_entries(
                        query=f"{key} {symbol}",
                        content_type_filter=content_type,
                        n_results=1
                    )
                    if entries:
                        entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                        if entry:
                            if content_type == "comprehensive_insight":
                                analyses[key] = entry["content"]["insight"]
                            elif content_type == "financial_health_analysis":
                                analyses[key] = entry["content"]["analysis"]
                            elif content_type == "valuation_analysis":
                                analyses[key] = entry["content"]["valuation"]
                            elif content_type == "market_position_analysis":
                                analyses[key] = entry["content"]["position_analysis"]
                
                # Get charts
                chart_entries = self.memory_manager.search_entries(
                    query=f"chart {symbol}",
                    content_type_filter="chart",
                    n_results=10
                )
                for entry_summary in chart_entries:
                    entry = self.memory_manager.get_entry_by_id(entry_summary["id"])
                    if entry:
                        chart_path = entry["content"].get("file_path")
                        if chart_path and os.path.exists(chart_path):
                            charts.append(chart_path)
                
                if not company_data and not analyses:
                    return f"Insufficient data to generate report for {symbol}. Please collect data and run analyses first."
                
                # Generate the report
                report_path = self.report_generator.generate_equity_report(
                    symbol, company_data, analyses, charts
                )
                
                if report_path:
                    # Store report metadata
                    entry_id = self.memory_manager.store_entry(
                        agent_name="ReportWriter",
                        content_type="equity_report",
                        content={
                            "symbol": symbol,
                            "report_path": report_path,
                            "sections_included": list(analyses.keys()),
                            "charts_included": len(charts),
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={"symbol": symbol, "report_type": "equity_research"},
                        tags=["report", "equity", "pdf", symbol.lower()]
                    )
                    
                    return f"Equity research report generated successfully (ID: {entry_id}). Report saved to: {report_path}"
                else:
                    return "Failed to generate equity research report."
                    
            except Exception as e:
                logger.error(f"Error generating equity report: {e}")
                return f"Error generating equity report: {e}"
        
        def create_summary_report(symbols: List[str]) -> str:
            """Create a summary report for multiple symbols"""
            try:
                if isinstance(symbols, str):
                    symbols = [s.strip() for s in symbols.split(',')]
                
                summary_data = {
                    "report_date": datetime.now().isoformat(),
                    "symbols_analyzed": symbols,
                    "symbol_summaries": {}
                }
                
                for symbol in symbols:
                    # Get comprehensive insight for each symbol
                    insight_entries = self.memory_manager.search_entries(
                        query=f"comprehensive insight {symbol}",
                        content_type_filter="comprehensive_insight",
                        n_results=1
                    )
                    
                    if insight_entries:
                        entry = self.memory_manager.get_entry_by_id(insight_entries[0]["id"])
                        if entry:
                            insight = entry["content"]["insight"]
                            summary_data["symbol_summaries"][symbol] = {
                                "recommendation": insight.get("overall_recommendation", "NEUTRAL"),
                                "confidence": insight.get("confidence_level", "MEDIUM"),
                                "key_strengths": insight.get("key_strengths", [])[:2],
                                "key_weaknesses": insight.get("key_weaknesses", [])[:2]
                            }
                    else:
                        summary_data["symbol_summaries"][symbol] = {
                            "recommendation": "NO DATA",
                            "confidence": "N/A",
                            "key_strengths": [],
                            "key_weaknesses": []
                        }
                
                # Store summary report
                entry_id = self.memory_manager.store_entry(
                    agent_name="ReportWriter",
                    content_type="summary_report",
                    content=summary_data,
                    metadata={"symbols": symbols, "report_type": "multi_symbol_summary"},
                    tags=["report", "summary", "multi_symbol"]
                )
                
                return f"Summary report created for {len(symbols)} symbols (ID: {entry_id}). Analyzed: {', '.join(symbols)}"
                
            except Exception as e:
                return f"Error creating summary report: {e}"
        
        # Register functions with autogen
        self.agent.register_for_execution(name="generate_equity_report")(generate_equity_report)
        self.agent.register_for_execution(name="create_summary_report")(create_summary_report)