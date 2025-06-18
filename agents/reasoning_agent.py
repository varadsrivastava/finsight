import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import autogen
from autogen import AssistantAgent
import logging
import pandas as pd
import numpy as np

from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig

logger = logging.getLogger(__name__)

class FinancialReasoningEngine:
    """Advanced financial reasoning and analysis engine (FinR1 style)"""
    
    def __init__(self, config: FinSightConfig):
        self.config = config
    
    def analyze_financial_health(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive financial health analysis"""
        try:
            metrics = financial_data.get("key_metrics", {})
            
            health_score = 0
            max_score = 100
            analysis = {
                "overall_score": 0,
                "profitability": {"score": 0, "analysis": "", "concerns": []},
                "liquidity": {"score": 0, "analysis": "", "concerns": []},
                "leverage": {"score": 0, "analysis": "", "concerns": []},
                "efficiency": {"score": 0, "analysis": "", "concerns": []},
                "growth": {"score": 0, "analysis": "", "concerns": []},
                "recommendations": []
            }
            
            # Profitability Analysis (25 points)
            profit_margin = metrics.get("profit_margin", 0) or 0
            operating_margin = metrics.get("operating_margin", 0) or 0
            roe = metrics.get("return_on_equity", 0) or 0
            
            profitability_score = 0
            if profit_margin > 0.15:  # 15%+ is excellent
                profitability_score += 10
            elif profit_margin > 0.05:  # 5-15% is good
                profitability_score += 7
            elif profit_margin > 0:  # Positive is okay
                profitability_score += 4
            
            if operating_margin > 0.20:  # 20%+ is excellent
                profitability_score += 8
            elif operating_margin > 0.10:  # 10-20% is good
                profitability_score += 5
            elif operating_margin > 0:  # Positive is okay
                profitability_score += 2
            
            if roe > 0.15:  # 15%+ ROE is excellent
                profitability_score += 7
            elif roe > 0.10:  # 10-15% is good
                profitability_score += 4
            elif roe > 0:  # Positive is okay
                profitability_score += 2
            
            analysis["profitability"]["score"] = profitability_score
            analysis["profitability"]["analysis"] = f"Profit margin: {profit_margin*100:.1f}%, Operating margin: {operating_margin*100:.1f}%, ROE: {roe*100:.1f}%"
            
            if profit_margin < 0:
                analysis["profitability"]["concerns"].append("Negative profit margins indicate losses")
            if operating_margin < 0.05:
                analysis["profitability"]["concerns"].append("Low operating margins may indicate operational inefficiency")
            
            health_score += profitability_score
            
            # Liquidity Analysis (20 points)
            current_ratio = metrics.get("current_ratio", 0) or 0
            quick_ratio = metrics.get("quick_ratio", 0) or 0
            
            liquidity_score = 0
            if current_ratio >= 2.0:
                liquidity_score += 10
            elif current_ratio >= 1.5:
                liquidity_score += 8
            elif current_ratio >= 1.0:
                liquidity_score += 5
            elif current_ratio >= 0.8:
                liquidity_score += 2
            
            if quick_ratio >= 1.0:
                liquidity_score += 10
            elif quick_ratio >= 0.8:
                liquidity_score += 7
            elif quick_ratio >= 0.5:
                liquidity_score += 4
            
            analysis["liquidity"]["score"] = liquidity_score
            analysis["liquidity"]["analysis"] = f"Current ratio: {current_ratio:.2f}, Quick ratio: {quick_ratio:.2f}"
            
            if current_ratio < 1.0:
                analysis["liquidity"]["concerns"].append("Current ratio below 1.0 indicates potential liquidity issues")
            if quick_ratio < 0.5:
                analysis["liquidity"]["concerns"].append("Low quick ratio may indicate difficulty meeting short-term obligations")
            
            health_score += liquidity_score
            
            # Leverage Analysis (20 points)
            debt_to_equity = metrics.get("debt_to_equity", 0) or 0
            
            leverage_score = 0
            if debt_to_equity <= 0.3:  # Low debt is good
                leverage_score += 20
            elif debt_to_equity <= 0.6:  # Moderate debt is acceptable
                leverage_score += 15
            elif debt_to_equity <= 1.0:  # High but manageable
                leverage_score += 10
            elif debt_to_equity <= 2.0:  # Very high debt
                leverage_score += 5
            
            analysis["leverage"]["score"] = leverage_score
            analysis["leverage"]["analysis"] = f"Debt-to-equity ratio: {debt_to_equity:.2f}"
            
            if debt_to_equity > 1.0:
                analysis["leverage"]["concerns"].append("High debt levels may increase financial risk")
            if debt_to_equity > 2.0:
                analysis["leverage"]["concerns"].append("Very high leverage poses significant risk")
            
            health_score += leverage_score
            
            # Efficiency Analysis (20 points)
            roa = metrics.get("return_on_assets", 0) or 0
            
            efficiency_score = 0
            if roa > 0.10:  # 10%+ ROA is excellent
                efficiency_score += 20
            elif roa > 0.05:  # 5-10% is good
                efficiency_score += 15
            elif roa > 0.02:  # 2-5% is fair
                efficiency_score += 10
            elif roa > 0:  # Positive is minimal
                efficiency_score += 5
            
            analysis["efficiency"]["score"] = efficiency_score
            analysis["efficiency"]["analysis"] = f"Return on assets: {roa*100:.1f}%"
            
            if roa < 0.02:
                analysis["efficiency"]["concerns"].append("Low ROA indicates poor asset utilization")
            
            health_score += efficiency_score
            
            # Growth Analysis (15 points) - placeholder
            revenue_growth = metrics.get("revenue_growth", 0) or 0
            
            growth_score = 0
            if revenue_growth > 0.15:  # 15%+ growth is excellent
                growth_score += 15
            elif revenue_growth > 0.05:  # 5-15% is good
                growth_score += 10
            elif revenue_growth > 0:  # Positive growth
                growth_score += 7
            elif revenue_growth > -0.05:  # Slight decline
                growth_score += 3
            
            analysis["growth"]["score"] = growth_score
            analysis["growth"]["analysis"] = f"Revenue growth: {revenue_growth*100:.1f}%"
            
            if revenue_growth < 0:
                analysis["growth"]["concerns"].append("Negative revenue growth indicates declining business")
            
            health_score += growth_score
            
            # Overall assessment
            analysis["overall_score"] = min(health_score, max_score)
            
            # Generate recommendations
            if analysis["overall_score"] >= 80:
                analysis["recommendations"].append("Strong financial position - consider for investment")
            elif analysis["overall_score"] >= 60:
                analysis["recommendations"].append("Solid financials with some areas for improvement")
            elif analysis["overall_score"] >= 40:
                analysis["recommendations"].append("Mixed financial health - requires careful analysis")
            else:
                analysis["recommendations"].append("Poor financial health - high risk investment")
            
            return {"success": True, "analysis": analysis}
            
        except Exception as e:
            logger.error(f"Error in financial health analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def generate_valuation_insights(self, stock_data: Dict[str, Any], financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valuation insights and fair value estimates"""
        try:
            current_price = stock_data.get("current_price", 0)
            pe_ratio = stock_data.get("pe_ratio", 0)
            market_cap = stock_data.get("market_cap", 0)
            
            metrics = financial_data.get("key_metrics", {})
            
            valuation = {
                "current_valuation": {
                    "price": current_price,
                    "pe_ratio": pe_ratio,
                    "market_cap": market_cap
                },
                "relative_valuation": {},
                "intrinsic_value_estimates": {},
                "valuation_conclusion": "",
                "risk_factors": []
            }
            
            # P/E Analysis
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 15:
                    pe_assessment = "Potentially undervalued based on P/E"
                elif pe_ratio < 25:
                    pe_assessment = "Fairly valued based on P/E"
                elif pe_ratio < 40:
                    pe_assessment = "Potentially overvalued based on P/E"
                else:
                    pe_assessment = "Significantly overvalued based on P/E"
                
                valuation["relative_valuation"]["pe_analysis"] = pe_assessment
            
            # ROE-based valuation
            roe = metrics.get("return_on_equity", 0) or 0
            if roe > 0 and pe_ratio > 0:
                peg_proxy = pe_ratio / (roe * 100)  # Simplified PEG-like ratio
                if peg_proxy < 1.0:
                    valuation["relative_valuation"]["roe_based"] = "Attractive based on ROE relative to P/E"
                elif peg_proxy < 1.5:
                    valuation["relative_valuation"]["roe_based"] = "Fair based on ROE relative to P/E"
                else:
                    valuation["relative_valuation"]["roe_based"] = "Expensive based on ROE relative to P/E"
            
            # Simple DCF estimate (very basic)
            profit_margin = metrics.get("profit_margin", 0) or 0
            revenue_growth = metrics.get("revenue_growth", 0) or 0
            
            if market_cap and profit_margin > 0:
                # Rough revenue estimate
                estimated_revenue = market_cap / (profit_margin * 15)  # Assuming P/S of 15 * profit margin
                
                # Simple growth-adjusted value
                if revenue_growth > 0:
                    growth_multiplier = 1 + min(revenue_growth, 0.20)  # Cap growth assumption at 20%
                    estimated_fair_value = current_price * growth_multiplier
                    
                    valuation["intrinsic_value_estimates"]["simple_growth_model"] = {
                        "estimated_fair_value": estimated_fair_value,
                        "upside_downside": ((estimated_fair_value - current_price) / current_price) * 100
                    }
            
            # Overall valuation conclusion
            assessments = list(valuation["relative_valuation"].values())
            if assessments:
                if any("undervalued" in assessment for assessment in assessments):
                    valuation["valuation_conclusion"] = "Potentially undervalued"
                elif any("overvalued" in assessment for assessment in assessments):
                    valuation["valuation_conclusion"] = "Potentially overvalued"
                else:
                    valuation["valuation_conclusion"] = "Fairly valued"
            
            # Risk factors
            if pe_ratio > 30:
                valuation["risk_factors"].append("High P/E ratio indicates elevated valuation risk")
            if roe < 0.10:
                valuation["risk_factors"].append("Low ROE may not justify current valuation")
            if revenue_growth < 0:
                valuation["risk_factors"].append("Declining revenue poses valuation risk")
            
            return {"success": True, "valuation": valuation}
            
        except Exception as e:
            logger.error(f"Error in valuation analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_market_position(self, stock_data: Dict[str, Any], news_data: Dict[str, Any], trends_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market position and competitive dynamics"""
        try:
            position_analysis = {
                "market_performance": {},
                "sentiment_analysis": {},
                "institutional_interest": {},
                "competitive_position": "Unknown",
                "investment_thesis": [],
                "risks": []
            }
            
            # Market performance analysis
            week_52_high = stock_data.get("52_week_high", 0)
            week_52_low = stock_data.get("52_week_low", 0)
            current_price = stock_data.get("current_price", 0)
            beta = stock_data.get("beta", 1.0)
            
            if week_52_high and week_52_low and current_price:
                range_position = (current_price - week_52_low) / (week_52_high - week_52_low)
                
                position_analysis["market_performance"] = {
                    "52_week_range_position": range_position * 100,
                    "distance_from_high": ((week_52_high - current_price) / week_52_high) * 100,
                    "distance_from_low": ((current_price - week_52_low) / week_52_low) * 100,
                    "beta": beta,
                    "volatility_assessment": "High" if beta > 1.5 else "Moderate" if beta > 0.8 else "Low"
                }
            
            # Sentiment analysis from news
            articles = news_data.get("articles", [])
            if articles:
                sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
                for article in articles:
                    sentiment = article.get("sentiment", "neutral")
                    sentiment_counts[sentiment] += 1
                
                total_articles = len(articles)
                position_analysis["sentiment_analysis"] = {
                    "positive_ratio": sentiment_counts["positive"] / total_articles,
                    "negative_ratio": sentiment_counts["negative"] / total_articles,
                    "overall_sentiment": max(sentiment_counts, key=sentiment_counts.get),
                    "sentiment_strength": max(sentiment_counts.values()) / total_articles
                }
            
            # Institutional interest
            market_indicators = trends_data.get("market_indicators", {})
            institutional_pct = market_indicators.get("institutional_ownership_pct", 0)
            insider_pct = market_indicators.get("insider_ownership_pct", 0)
            
            position_analysis["institutional_interest"] = {
                "institutional_ownership": institutional_pct or 0,
                "insider_ownership": insider_pct or 0,
                "institutional_confidence": "High" if institutional_pct > 0.7 else "Moderate" if institutional_pct > 0.4 else "Low"
            }
            
            # Investment thesis generation
            if position_analysis["market_performance"].get("52_week_range_position", 0) > 80:
                position_analysis["investment_thesis"].append("Trading near 52-week highs - momentum play")
            elif position_analysis["market_performance"].get("52_week_range_position", 0) < 20:
                position_analysis["investment_thesis"].append("Trading near 52-week lows - potential value opportunity")
            
            if position_analysis["sentiment_analysis"].get("positive_ratio", 0) > 0.6:
                position_analysis["investment_thesis"].append("Strong positive sentiment momentum")
            
            if institutional_pct > 0.6:
                position_analysis["investment_thesis"].append("Strong institutional backing")
            
            # Risk identification
            if beta > 1.5:
                position_analysis["risks"].append("High volatility stock - significant price swings expected")
            
            if position_analysis["sentiment_analysis"].get("negative_ratio", 0) > 0.5:
                position_analysis["risks"].append("Negative sentiment may pressure stock price")
            
            if institutional_pct < 0.2:
                position_analysis["risks"].append("Low institutional interest may indicate concerns")
            
            return {"success": True, "position_analysis": position_analysis}
            
        except Exception as e:
            logger.error(f"Error in market position analysis: {e}")
            return {"success": False, "error": str(e)}

class ReasoningAgent:
    """Advanced financial reasoning agent (FinR1 style)"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.reasoning_engine = FinancialReasoningEngine(self.config)
        
        # Create the Autogen agent
        self.agent = AssistantAgent(
            name="ReasoningAgent",
            system_message=self._get_system_message(),
            llm_config=AGENT_CONFIGS["reasoning_agent"],
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
        )
        
        # Register tools
        self._register_tools()
    
    def _get_system_message(self) -> str:
        return """You are a ReasoningAgent (FinR1) specialized in advanced financial analysis and investment reasoning.

Your responsibilities:
1. Conduct comprehensive financial health analysis
2. Generate valuation insights and fair value estimates
3. Analyze market position and competitive dynamics
4. Synthesize multi-source data into actionable investment insights
5. Identify key risks and opportunities
6. Provide evidence-based investment recommendations

Your analysis approach:
- Use quantitative metrics and qualitative factors
- Consider multiple valuation methodologies
- Assess risk-adjusted returns
- Evaluate market sentiment and momentum
- Account for macroeconomic factors
- Generate clear, actionable insights

For each analysis:
- Base conclusions on data from memory
- Provide detailed reasoning for recommendations
- Quantify risks and potential returns
- Consider various investment scenarios
- Store comprehensive analysis results

Focus on delivering institutional-quality financial analysis and investment insights.

Reply TERMINATE when comprehensive analysis is complete."""
    
    def _register_tools(self):
        """Register reasoning and analysis tools with the agent"""
        
        def analyze_financial_health(symbol: str) -> str:
            """Conduct comprehensive financial health analysis"""
            try:
                # Get financial data from memory
                entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                
                if not entries:
                    return f"No financial data found for {symbol}. Please collect financial data first."
                
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if not entry:
                    return "Could not retrieve financial data."
                
                # Perform analysis
                result = self.reasoning_engine.analyze_financial_health(entry["content"])
                
                if result["success"]:
                    # Store analysis result
                    entry_id = self.memory_manager.store_entry(
                        agent_name="ReasoningAgent",
                        content_type="financial_health_analysis",
                        content={
                            "symbol": symbol,
                            "analysis": result["analysis"],
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={"symbol": symbol, "analysis_type": "financial_health"},
                        tags=["analysis", "financial_health", symbol.lower()]
                    )
                    
                    analysis = result["analysis"]
                    return f"Financial health analysis completed (ID: {entry_id}). Overall score: {analysis['overall_score']}/100. Key concerns: {len([c for section in analysis.values() if isinstance(section, dict) and 'concerns' in section for c in section['concerns']])}"
                else:
                    return f"Financial health analysis failed: {result['error']}"
                    
            except Exception as e:
                return f"Error in financial health analysis: {e}"
        
        def generate_valuation_analysis(symbol: str) -> str:
            """Generate comprehensive valuation analysis"""
            try:
                # Get stock and financial data
                stock_entries = self.memory_manager.search_entries(
                    query=f"stock data {symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                
                financial_entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                
                if not stock_entries or not financial_entries:
                    return f"Insufficient data for valuation analysis of {symbol}. Need both stock and financial data."
                
                stock_entry = self.memory_manager.get_entry_by_id(stock_entries[0]["id"])
                financial_entry = self.memory_manager.get_entry_by_id(financial_entries[0]["id"])
                
                if not stock_entry or not financial_entry:
                    return "Could not retrieve required data entries."
                
                # Perform valuation analysis
                result = self.reasoning_engine.generate_valuation_insights(
                    stock_entry["content"], financial_entry["content"]
                )
                
                if result["success"]:
                    # Store analysis result
                    entry_id = self.memory_manager.store_entry(
                        agent_name="ReasoningAgent",
                        content_type="valuation_analysis",
                        content={
                            "symbol": symbol,
                            "valuation": result["valuation"],
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={"symbol": symbol, "analysis_type": "valuation"},
                        tags=["analysis", "valuation", symbol.lower()]
                    )
                    
                    valuation = result["valuation"]
                    conclusion = valuation.get("valuation_conclusion", "Unknown")
                    return f"Valuation analysis completed (ID: {entry_id}). Conclusion: {conclusion}. Risk factors: {len(valuation.get('risk_factors', []))}"
                else:
                    return f"Valuation analysis failed: {result['error']}"
                    
            except Exception as e:
                return f"Error in valuation analysis: {e}"
        
        def analyze_market_position(symbol: str) -> str:
            """Analyze market position and competitive dynamics"""
            try:
                # Get required data from memory
                stock_entries = self.memory_manager.search_entries(
                    query=f"stock data {symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                
                news_entries = self.memory_manager.search_entries(
                    query=f"news {symbol}",
                    content_type_filter="news",
                    n_results=1
                )
                
                trends_entries = self.memory_manager.search_entries(
                    query=f"market trends {symbol}",
                    content_type_filter="market_trends",
                    n_results=1
                )
                
                if not stock_entries:
                    return f"No stock data found for {symbol}. Please collect stock data first."
                
                stock_entry = self.memory_manager.get_entry_by_id(stock_entries[0]["id"])
                news_entry = self.memory_manager.get_entry_by_id(news_entries[0]["id"]) if news_entries else None
                trends_entry = self.memory_manager.get_entry_by_id(trends_entries[0]["id"]) if trends_entries else None
                
                # Perform market position analysis
                result = self.reasoning_engine.analyze_market_position(
                    stock_entry["content"],
                    news_entry["content"] if news_entry else {"articles": []},
                    trends_entry["content"] if trends_entry else {"market_indicators": {}}
                )
                
                if result["success"]:
                    # Store analysis result
                    entry_id = self.memory_manager.store_entry(
                        agent_name="ReasoningAgent",
                        content_type="market_position_analysis",
                        content={
                            "symbol": symbol,
                            "position_analysis": result["position_analysis"],
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={"symbol": symbol, "analysis_type": "market_position"},
                        tags=["analysis", "market_position", symbol.lower()]
                    )
                    
                    analysis = result["position_analysis"]
                    thesis_count = len(analysis.get("investment_thesis", []))
                    risk_count = len(analysis.get("risks", []))
                    return f"Market position analysis completed (ID: {entry_id}). Investment thesis points: {thesis_count}, Risk factors: {risk_count}"
                else:
                    return f"Market position analysis failed: {result['error']}"
                    
            except Exception as e:
                return f"Error in market position analysis: {e}"
        
        def generate_comprehensive_insight(symbol: str) -> str:
            """Generate comprehensive investment insight combining all analyses"""
            try:
                # Search for all analysis types
                analyses = {}
                
                # Get financial health analysis
                health_entries = self.memory_manager.search_entries(
                    query=f"financial health {symbol}",
                    content_type_filter="financial_health_analysis",
                    n_results=1
                )
                if health_entries:
                    health_entry = self.memory_manager.get_entry_by_id(health_entries[0]["id"])
                    if health_entry:
                        analyses["financial_health"] = health_entry["content"]["analysis"]
                
                # Get valuation analysis
                valuation_entries = self.memory_manager.search_entries(
                    query=f"valuation {symbol}",
                    content_type_filter="valuation_analysis",
                    n_results=1
                )
                if valuation_entries:
                    valuation_entry = self.memory_manager.get_entry_by_id(valuation_entries[0]["id"])
                    if valuation_entry:
                        analyses["valuation"] = valuation_entry["content"]["valuation"]
                
                # Get market position analysis
                position_entries = self.memory_manager.search_entries(
                    query=f"market position {symbol}",
                    content_type_filter="market_position_analysis",
                    n_results=1
                )
                if position_entries:
                    position_entry = self.memory_manager.get_entry_by_id(position_entries[0]["id"])
                    if position_entry:
                        analyses["market_position"] = position_entry["content"]["position_analysis"]
                
                if not analyses:
                    return f"No analysis data found for {symbol}. Please run individual analyses first."
                
                # Generate comprehensive insight
                insight = {
                    "symbol": symbol,
                    "overall_recommendation": "NEUTRAL",
                    "confidence_level": "MEDIUM",
                    "key_strengths": [],
                    "key_weaknesses": [],
                    "investment_rationale": "",
                    "risk_assessment": "",
                    "target_investor_profile": "",
                    "time_horizon": "MEDIUM_TERM"
                }
                
                # Analyze financial health
                if "financial_health" in analyses:
                    health = analyses["financial_health"]
                    health_score = health.get("overall_score", 0)
                    
                    if health_score >= 80:
                        insight["key_strengths"].append("Strong financial health")
                    elif health_score < 50:
                        insight["key_weaknesses"].append("Poor financial health")
                
                # Analyze valuation
                if "valuation" in analyses:
                    valuation = analyses["valuation"]
                    conclusion = valuation.get("valuation_conclusion", "")
                    
                    if "undervalued" in conclusion.lower():
                        insight["key_strengths"].append("Potentially undervalued")
                    elif "overvalued" in conclusion.lower():
                        insight["key_weaknesses"].append("Potentially overvalued")
                
                # Analyze market position
                if "market_position" in analyses:
                    position = analyses["market_position"]
                    thesis = position.get("investment_thesis", [])
                    risks = position.get("risks", [])
                    
                    insight["key_strengths"].extend(thesis[:2])  # Top 2 thesis points
                    insight["key_weaknesses"].extend(risks[:2])  # Top 2 risks
                
                # Generate overall recommendation
                strength_count = len(insight["key_strengths"])
                weakness_count = len(insight["key_weaknesses"])
                
                if strength_count > weakness_count + 1:
                    insight["overall_recommendation"] = "BUY"
                    insight["confidence_level"] = "HIGH" if strength_count >= 3 else "MEDIUM"
                elif weakness_count > strength_count + 1:
                    insight["overall_recommendation"] = "SELL"
                    insight["confidence_level"] = "HIGH" if weakness_count >= 3 else "MEDIUM"
                else:
                    insight["overall_recommendation"] = "HOLD"
                
                # Store comprehensive insight
                entry_id = self.memory_manager.store_entry(
                    agent_name="ReasoningAgent",
                    content_type="comprehensive_insight",
                    content={
                        "symbol": symbol,
                        "insight": insight,
                        "supporting_analyses": analyses,
                        "timestamp": datetime.now().isoformat()
                    },
                    metadata={"symbol": symbol, "analysis_type": "comprehensive"},
                    tags=["insight", "comprehensive", "recommendation", symbol.lower()]
                )
                
                return f"Comprehensive insight generated (ID: {entry_id}). Recommendation: {insight['overall_recommendation']} with {insight['confidence_level']} confidence. Strengths: {len(insight['key_strengths'])}, Weaknesses: {len(insight['key_weaknesses'])}"
                
            except Exception as e:
                return f"Error generating comprehensive insight: {e}"
        
        # Register functions with autogen
        self.agent.register_for_execution(name="analyze_financial_health")(analyze_financial_health)
        self.agent.register_for_execution(name="generate_valuation_analysis")(generate_valuation_analysis)
        self.agent.register_for_execution(name="analyze_market_position")(analyze_market_position)
        self.agent.register_for_execution(name="generate_comprehensive_insight")(generate_comprehensive_insight)