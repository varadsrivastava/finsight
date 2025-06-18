import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import autogen
from autogen import AssistantAgent
import requests
import logging
import re
from dataclasses import dataclass

from shared_memory.memory_manager import SharedMemoryManager
from config.config import AGENT_CONFIGS, FinSightConfig

logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Result of fact verification"""
    claim: str
    verification_status: str  # "VERIFIED", "DISPUTED", "UNVERIFIABLE", "ERROR"
    confidence_score: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    verification_reasoning: str
    data_sources_checked: List[str]

class FactChecker:
    """Advanced fact checking and verification engine"""
    
    def __init__(self, config: FinSightConfig):
        self.config = config
        self.verification_threshold = config.fact_check_threshold
    
    def verify_financial_claim(self, claim: str, context_data: Dict[str, Any]) -> VerificationResult:
        """Verify a financial claim against available data"""
        try:
            # Extract numerical claims from the statement
            numerical_claims = self._extract_numerical_claims(claim)
            
            verification_results = []
            supporting_evidence = []
            contradicting_evidence = []
            data_sources = []
            
            # Verify each numerical claim
            for claim_type, claimed_value, tolerance in numerical_claims:
                result = self._verify_numerical_claim(claim_type, claimed_value, tolerance, context_data)
                verification_results.append(result)
                
                if result["verified"]:
                    supporting_evidence.append(result["evidence"])
                else:
                    contradicting_evidence.append(result["evidence"])
                
                data_sources.extend(result["sources"])
            
            # Determine overall verification status
            if not verification_results:
                # No numerical claims to verify - check for qualitative consistency
                qualitative_result = self._verify_qualitative_claim(claim, context_data)
                return qualitative_result
            
            verified_count = sum(1 for r in verification_results if r["verified"])
            total_count = len(verification_results)
            confidence_score = verified_count / total_count if total_count > 0 else 0.0
            
            if confidence_score >= 0.8:
                status = "VERIFIED"
            elif confidence_score >= 0.6:
                status = "PARTIALLY_VERIFIED"
            elif confidence_score >= 0.3:
                status = "DISPUTED"
            else:
                status = "DISPUTED"
            
            reasoning = f"Verified {verified_count}/{total_count} numerical claims. "
            if supporting_evidence:
                reasoning += f"Supporting evidence: {'; '.join(supporting_evidence[:3])}. "
            if contradicting_evidence:
                reasoning += f"Contradicting evidence: {'; '.join(contradicting_evidence[:3])}."
            
            return VerificationResult(
                claim=claim,
                verification_status=status,
                confidence_score=confidence_score,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                verification_reasoning=reasoning,
                data_sources_checked=list(set(data_sources))
            )
            
        except Exception as e:
            logger.error(f"Error verifying claim: {e}")
            return VerificationResult(
                claim=claim,
                verification_status="ERROR",
                confidence_score=0.0,
                supporting_evidence=[],
                contradicting_evidence=[],
                verification_reasoning=f"Verification error: {str(e)}",
                data_sources_checked=[]
            )
    
    def _extract_numerical_claims(self, text: str) -> List[Tuple[str, float, float]]:
        """Extract numerical financial claims from text"""
        claims = []
        
        # Patterns for different types of financial claims
        patterns = {
            "price": r"(?:price|trading|costs?)\s*(?:of|at|is|was)?\s*\$?(\d+(?:\.\d+)?)",
            "percentage": r"(\d+(?:\.\d+)?)\s*%",
            "ratio": r"(?:ratio|multiple)\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)",
            "margin": r"margin\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*%?",
            "growth": r"growth\s*(?:of|is|was)?\s*(\d+(?:\.\d+)?)\s*%?",
            "billion": r"\$?(\d+(?:\.\d+)?)\s*(?:billion|B)",
            "million": r"\$?(\d+(?:\.\d+)?)\s*(?:million|M)",
        }
        
        for claim_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value = float(match.group(1))
                
                # Adjust for units
                if claim_type == "billion":
                    value *= 1e9
                elif claim_type == "million":
                    value *= 1e6
                
                # Set tolerance based on claim type
                if claim_type in ["percentage", "margin", "growth"]:
                    tolerance = 0.1  # 0.1% tolerance
                elif claim_type == "ratio":
                    tolerance = 0.05  # 5% tolerance
                else:
                    tolerance = 0.02  # 2% tolerance for prices and large numbers
                
                claims.append((claim_type, value, tolerance))
        
        return claims
    
    def _verify_numerical_claim(self, claim_type: str, claimed_value: float, 
                               tolerance: float, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a specific numerical claim against context data"""
        
        # Look for relevant data based on claim type
        actual_value = None
        source = "unknown"
        
        if claim_type == "price" and "stock_data" in context_data:
            actual_value = context_data["stock_data"].get("current_price")
            source = "stock_data"
        
        elif claim_type in ["percentage", "margin"] and "financial_data" in context_data:
            metrics = context_data["financial_data"].get("key_metrics", {})
            if "profit" in claim_type.lower():
                actual_value = metrics.get("profit_margin", 0) * 100  # Convert to percentage
            elif "operating" in claim_type.lower():
                actual_value = metrics.get("operating_margin", 0) * 100
            source = "financial_statements"
        
        elif claim_type == "ratio" and "stock_data" in context_data:
            actual_value = context_data["stock_data"].get("pe_ratio")
            source = "stock_data"
        
        elif claim_type in ["billion", "million"] and "stock_data" in context_data:
            actual_value = context_data["stock_data"].get("market_cap")
            source = "stock_data"
        
        # Verify the claim
        if actual_value is not None:
            difference = abs(actual_value - claimed_value) / actual_value if actual_value != 0 else float('inf')
            verified = difference <= tolerance
            
            evidence = f"Claimed: {claimed_value}, Actual: {actual_value}, Difference: {difference:.1%}"
            
            return {
                "verified": verified,
                "evidence": evidence,
                "sources": [source],
                "actual_value": actual_value,
                "claimed_value": claimed_value,
                "difference": difference
            }
        else:
            return {
                "verified": False,
                "evidence": f"Could not find data to verify claim of {claimed_value}",
                "sources": [],
                "actual_value": None,
                "claimed_value": claimed_value,
                "difference": None
            }
    
    def _verify_qualitative_claim(self, claim: str, context_data: Dict[str, Any]) -> VerificationResult:
        """Verify qualitative claims using pattern matching and consistency checks"""
        
        # Common qualitative financial claims
        bullish_indicators = ["strong", "growing", "profitable", "increasing", "outperforming", "bullish"]
        bearish_indicators = ["weak", "declining", "losing", "decreasing", "underperforming", "bearish"]
        
        claim_lower = claim.lower()
        
        # Determine claim sentiment
        claim_sentiment = "neutral"
        if any(indicator in claim_lower for indicator in bullish_indicators):
            claim_sentiment = "bullish"
        elif any(indicator in claim_lower for indicator in bearish_indicators):
            claim_sentiment = "bearish"
        
        # Check consistency with data
        supporting_evidence = []
        contradicting_evidence = []
        
        # Check profitability claims
        if "profitable" in claim_lower or "profit" in claim_lower:
            if "financial_data" in context_data:
                profit_margin = context_data["financial_data"].get("key_metrics", {}).get("profit_margin", 0)
                if profit_margin > 0 and claim_sentiment == "bullish":
                    supporting_evidence.append(f"Company is profitable with {profit_margin*100:.1f}% profit margin")
                elif profit_margin <= 0 and claim_sentiment == "bearish":
                    supporting_evidence.append(f"Company has negative profit margin of {profit_margin*100:.1f}%")
                else:
                    contradicting_evidence.append(f"Profit margin of {profit_margin*100:.1f}% doesn't align with claim sentiment")
        
        # Check growth claims
        if "growth" in claim_lower or "growing" in claim_lower:
            if "financial_data" in context_data:
                revenue_growth = context_data["financial_data"].get("key_metrics", {}).get("revenue_growth", 0)
                if revenue_growth > 0 and claim_sentiment == "bullish":
                    supporting_evidence.append(f"Revenue growth of {revenue_growth*100:.1f}% supports growth claims")
                elif revenue_growth <= 0 and claim_sentiment == "bearish":
                    supporting_evidence.append(f"Negative revenue growth of {revenue_growth*100:.1f}% supports decline claims")
                else:
                    contradicting_evidence.append(f"Revenue growth of {revenue_growth*100:.1f}% doesn't align with claim")
        
        # Determine verification status
        evidence_score = len(supporting_evidence) / max(len(supporting_evidence) + len(contradicting_evidence), 1)
        
        if evidence_score >= 0.7:
            status = "VERIFIED"
        elif evidence_score >= 0.5:
            status = "PARTIALLY_VERIFIED"
        elif evidence_score >= 0.3:
            status = "DISPUTED"
        else:
            status = "UNVERIFIABLE"
        
        reasoning = f"Qualitative claim sentiment: {claim_sentiment}. Found {len(supporting_evidence)} supporting and {len(contradicting_evidence)} contradicting evidence points."
        
        return VerificationResult(
            claim=claim,
            verification_status=status,
            confidence_score=evidence_score,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            verification_reasoning=reasoning,
            data_sources_checked=["financial_data", "stock_data"]
        )

class LLMCritic:
    """LLM-based critical analysis of financial reports and claims"""
    
    def __init__(self, config: FinSightConfig):
        self.config = config
    
    def critique_analysis(self, analysis_content: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to critically analyze and validate analysis content"""
        try:
            if not self.config.anthropic_api_key:
                return {"success": False, "error": "No Anthropic API key configured for LLM critic"}
            
            # Prepare critique prompt
            critique_prompt = f"""
            You are a senior financial analyst and fact-checker. Please critically review the following financial analysis:

            ANALYSIS TO REVIEW:
            {analysis_content}

            AVAILABLE DATA CONTEXT:
            {json.dumps(context_data, indent=2)[:2000]}...

            Please provide a detailed critique covering:
            1. Factual accuracy - are the numbers and claims correct?
            2. Logical consistency - do the conclusions follow from the data?
            3. Completeness - what important factors might be missing?
            4. Bias detection - are there signs of overly optimistic/pessimistic analysis?
            5. Risk assessment - are key risks properly identified?

            Provide your response in JSON format with these fields:
            - "accuracy_score": 0-100 rating
            - "consistency_score": 0-100 rating  
            - "completeness_score": 0-100 rating
            - "bias_assessment": "none", "bullish_bias", "bearish_bias"
            - "key_issues": list of specific problems found
            - "missing_elements": list of important missing analysis points
            - "overall_quality": "excellent", "good", "fair", "poor"
            - "recommendations": list of improvement suggestions
            """
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1500,
                "messages": [
                    {
                        "role": "user",
                        "content": critique_prompt
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                critique_text = result["content"][0]["text"]
                
                # Try to parse JSON response
                try:
                    critique_data = json.loads(critique_text)
                    return {"success": True, "critique": critique_data}
                except json.JSONDecodeError:
                    # If JSON parsing fails, return raw text
                    return {
                        "success": True,
                        "critique": {
                            "raw_critique": critique_text,
                            "overall_quality": "unknown",
                            "parsing_error": "Could not parse structured critique"
                        }
                    }
            else:
                return {"success": False, "error": f"API call failed: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error in LLM critique: {e}")
            return {"success": False, "error": str(e)}

class VerifierAgent:
    """Agent for fact-checking and quality verification of financial analysis"""
    
    def __init__(self, memory_manager: SharedMemoryManager):
        self.config = FinSightConfig()
        self.memory_manager = memory_manager
        self.fact_checker = FactChecker(self.config)
        self.llm_critic = LLMCritic(self.config)
        
        # Create the Autogen agent
        self.agent = AssistantAgent(
            name="Verifier",
            system_message=self._get_system_message(),
            llm_config=AGENT_CONFIGS["verifier"],
            max_consecutive_auto_reply=self.config.max_consecutive_auto_reply,
        )
        
        # Register tools
        self._register_tools()
    
    def _get_system_message(self) -> str:
        return """You are a Verifier agent specialized in fact-checking and quality assurance for financial analysis.

Your responsibilities:
1. Verify factual accuracy of financial claims and analyses
2. Check consistency between conclusions and supporting data
3. Identify potential biases or errors in reasoning
4. Validate numerical calculations and ratios
5. Ensure completeness of risk assessment
6. Provide quality scores and improvement recommendations

Verification approach:
- Cross-reference claims with source data
- Check mathematical accuracy of calculations
- Validate logical consistency of conclusions
- Identify missing critical factors
- Assess potential biases or overconfidence
- Evaluate appropriateness of recommendations

For each verification:
- Provide specific factual assessments
- Identify discrepancies or concerns
- Suggest improvements or corrections
- Rate overall analysis quality
- Store verification results in memory

Maintain high standards for accuracy and objectivity in all financial analysis.

Reply TERMINATE when verification is complete."""
    
    def _register_tools(self):
        """Register verification and fact-checking tools with the agent"""
        
        def verify_financial_claims(symbol: str, analysis_text: str) -> str:
            """Verify specific financial claims in analysis text"""
            try:
                # Get context data for verification
                context_data = {}
                
                # Get stock data
                stock_entries = self.memory_manager.search_entries(
                    query=f"stock data {symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                if stock_entries:
                    stock_entry = self.memory_manager.get_entry_by_id(stock_entries[0]["id"])
                    if stock_entry:
                        context_data["stock_data"] = stock_entry["content"]
                
                # Get financial data
                financial_entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                if financial_entries:
                    financial_entry = self.memory_manager.get_entry_by_id(financial_entries[0]["id"])
                    if financial_entry:
                        context_data["financial_data"] = financial_entry["content"]
                
                # Split analysis into individual claims/sentences
                claims = [s.strip() for s in analysis_text.split('.') if s.strip()]
                
                verification_results = []
                for claim in claims[:10]:  # Limit to first 10 claims
                    if len(claim) > 20:  # Skip very short fragments
                        result = self.fact_checker.verify_financial_claim(claim, context_data)
                        verification_results.append(result)
                
                # Calculate overall verification statistics
                total_claims = len(verification_results)
                verified_claims = sum(1 for r in verification_results if r.verification_status == "VERIFIED")
                disputed_claims = sum(1 for r in verification_results if r.verification_status == "DISPUTED")
                
                verification_summary = {
                    "symbol": symbol,
                    "total_claims_checked": total_claims,
                    "verified_claims": verified_claims,
                    "disputed_claims": disputed_claims,
                    "verification_rate": verified_claims / total_claims if total_claims > 0 else 0,
                    "individual_results": [
                        {
                            "claim": r.claim[:100] + "..." if len(r.claim) > 100 else r.claim,
                            "status": r.verification_status,
                            "confidence": r.confidence_score,
                            "reasoning": r.verification_reasoning[:200] + "..." if len(r.verification_reasoning) > 200 else r.verification_reasoning
                        }
                        for r in verification_results
                    ]
                }
                
                # Store verification results
                entry_id = self.memory_manager.store_entry(
                    agent_name="Verifier",
                    content_type="fact_verification",
                    content=verification_summary,
                    metadata={"symbol": symbol, "verification_type": "claims"},
                    tags=["verification", "fact_check", symbol.lower()]
                )
                
                return f"Fact verification completed (ID: {entry_id}). Verified {verified_claims}/{total_claims} claims ({verification_summary['verification_rate']:.1%} accuracy). Disputed: {disputed_claims}"
                
            except Exception as e:
                return f"Error in fact verification: {e}"
        
        def critique_comprehensive_analysis(symbol: str) -> str:
            """Perform comprehensive quality critique of all analyses for a symbol"""
            try:
                # Get comprehensive insight
                insight_entries = self.memory_manager.search_entries(
                    query=f"comprehensive insight {symbol}",
                    content_type_filter="comprehensive_insight",
                    n_results=1
                )
                
                if not insight_entries:
                    return f"No comprehensive analysis found for {symbol} to critique."
                
                insight_entry = self.memory_manager.get_entry_by_id(insight_entries[0]["id"])
                if not insight_entry:
                    return "Could not retrieve comprehensive analysis."
                
                # Get supporting data for context
                context_data = {}
                
                # Collect all relevant data
                data_types = [
                    ("stock_data", "stock data"),
                    ("financial_statements", "financial statements"),
                    ("news", "news"),
                    ("market_trends", "market trends")
                ]
                
                for content_type, query_term in data_types:
                    entries = self.memory_manager.search_entries(
                        query=f"{query_term} {symbol}",
                        content_type_filter=content_type,
                        n_results=1
                    )
                    if entries:
                        entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                        if entry:
                            context_data[content_type] = entry["content"]
                
                # Prepare analysis content for critique
                insight_data = insight_entry["content"]["insight"]
                analysis_content = f"""
                Symbol: {symbol}
                Recommendation: {insight_data.get('overall_recommendation', 'N/A')}
                Confidence: {insight_data.get('confidence_level', 'N/A')}
                
                Key Strengths:
                {chr(10).join(f"- {s}" for s in insight_data.get('key_strengths', []))}
                
                Key Weaknesses:
                {chr(10).join(f"- {w}" for w in insight_data.get('key_weaknesses', []))}
                
                Investment Rationale: {insight_data.get('investment_rationale', 'Not provided')}
                Risk Assessment: {insight_data.get('risk_assessment', 'Not provided')}
                """
                
                # Perform LLM critique
                critique_result = self.llm_critic.critique_analysis(analysis_content, context_data)
                
                if critique_result["success"]:
                    # Store critique results
                    entry_id = self.memory_manager.store_entry(
                        agent_name="Verifier",
                        content_type="quality_critique",
                        content={
                            "symbol": symbol,
                            "critique": critique_result["critique"],
                            "analysis_reviewed": analysis_content,
                            "timestamp": datetime.now().isoformat()
                        },
                        metadata={"symbol": symbol, "verification_type": "quality_critique"},
                        tags=["verification", "critique", "quality", symbol.lower()]
                    )
                    
                    critique = critique_result["critique"]
                    overall_quality = critique.get("overall_quality", "unknown")
                    key_issues_count = len(critique.get("key_issues", []))
                    
                    return f"Quality critique completed (ID: {entry_id}). Overall quality: {overall_quality}. Key issues identified: {key_issues_count}"
                else:
                    return f"Quality critique failed: {critique_result['error']}"
                    
            except Exception as e:
                return f"Error in quality critique: {e}"
        
        def verify_calculation_accuracy(symbol: str) -> str:
            """Verify mathematical accuracy of financial calculations"""
            try:
                verification_results = []
                
                # Get financial data
                financial_entries = self.memory_manager.search_entries(
                    query=f"financial statements {symbol}",
                    content_type_filter="financial_statements",
                    n_results=1
                )
                
                if not financial_entries:
                    return f"No financial data found for {symbol} to verify calculations."
                
                financial_entry = self.memory_manager.get_entry_by_id(financial_entries[0]["id"])
                if not financial_entry:
                    return "Could not retrieve financial data."
                
                metrics = financial_entry["content"].get("key_metrics", {})
                
                # Verify key ratio calculations
                calculations_to_verify = [
                    ("profit_margin", "Profit Margin"),
                    ("operating_margin", "Operating Margin"),
                    ("return_on_equity", "Return on Equity"),
                    ("return_on_assets", "Return on Assets"),
                    ("current_ratio", "Current Ratio"),
                    ("debt_to_equity", "Debt to Equity")
                ]
                
                verified_calculations = 0
                total_calculations = 0
                
                for metric_key, metric_name in calculations_to_verify:
                    value = metrics.get(metric_key)
                    if value is not None:
                        total_calculations += 1
                        
                        # Basic sanity checks
                        is_reasonable = True
                        issues = []
                        
                        if metric_key in ["profit_margin", "operating_margin"]:
                            if value < -1.0 or value > 1.0:  # -100% to 100% is reasonable range
                                is_reasonable = False
                                issues.append(f"{metric_name} of {value*100:.1f}% seems unreasonable")
                        
                        elif metric_key in ["return_on_equity", "return_on_assets"]:
                            if value < -2.0 or value > 2.0:  # -200% to 200% is reasonable range
                                is_reasonable = False
                                issues.append(f"{metric_name} of {value*100:.1f}% seems unreasonable")
                        
                        elif metric_key == "current_ratio":
                            if value < 0 or value > 50:  # Negative or extremely high ratios are suspicious
                                is_reasonable = False
                                issues.append(f"{metric_name} of {value:.2f} seems unreasonable")
                        
                        elif metric_key == "debt_to_equity":
                            if value < 0 or value > 50:  # Negative or extremely high D/E is suspicious
                                is_reasonable = False
                                issues.append(f"{metric_name} of {value:.2f} seems unreasonable")
                        
                        if is_reasonable:
                            verified_calculations += 1
                        
                        verification_results.append({
                            "metric": metric_name,
                            "value": value,
                            "reasonable": is_reasonable,
                            "issues": issues
                        })
                
                # Calculate verification statistics
                verification_rate = verified_calculations / total_calculations if total_calculations > 0 else 0
                
                calculation_verification = {
                    "symbol": symbol,
                    "total_calculations": total_calculations,
                    "verified_calculations": verified_calculations,
                    "verification_rate": verification_rate,
                    "calculation_results": verification_results
                }
                
                # Store verification results
                entry_id = self.memory_manager.store_entry(
                    agent_name="Verifier",
                    content_type="calculation_verification",
                    content=calculation_verification,
                    metadata={"symbol": symbol, "verification_type": "calculations"},
                    tags=["verification", "calculations", symbol.lower()]
                )
                
                issues_count = sum(len(r["issues"]) for r in verification_results)
                
                return f"Calculation verification completed (ID: {entry_id}). Verified {verified_calculations}/{total_calculations} calculations ({verification_rate:.1%}). Issues found: {issues_count}"
                
            except Exception as e:
                return f"Error in calculation verification: {e}"
        
        def generate_verification_report(symbol: str) -> str:
            """Generate comprehensive verification report for a symbol"""
            try:
                # Collect all verification results
                verification_types = [
                    ("fact_verification", "Fact Verification"),
                    ("quality_critique", "Quality Critique"),
                    ("calculation_verification", "Calculation Verification")
                ]
                
                verification_summary = {
                    "symbol": symbol,
                    "verification_timestamp": datetime.now().isoformat(),
                    "verification_sections": {}
                }
                
                for content_type, section_name in verification_types:
                    entries = self.memory_manager.search_entries(
                        query=f"verification {symbol}",
                        content_type_filter=content_type,
                        n_results=1
                    )
                    
                    if entries:
                        entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                        if entry:
                            verification_summary["verification_sections"][section_name] = entry["content"]
                
                # Generate overall quality assessment
                overall_scores = []
                
                # Extract scores from different verification types
                if "Fact Verification" in verification_summary["verification_sections"]:
                    fact_rate = verification_summary["verification_sections"]["Fact Verification"].get("verification_rate", 0)
                    overall_scores.append(fact_rate * 100)
                
                if "Calculation Verification" in verification_summary["verification_sections"]:
                    calc_rate = verification_summary["verification_sections"]["Calculation Verification"].get("verification_rate", 0)
                    overall_scores.append(calc_rate * 100)
                
                if "Quality Critique" in verification_summary["verification_sections"]:
                    critique = verification_summary["verification_sections"]["Quality Critique"].get("critique", {})
                    if "accuracy_score" in critique:
                        overall_scores.append(critique["accuracy_score"])
                
                overall_quality_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
                
                if overall_quality_score >= 80:
                    quality_rating = "HIGH"
                elif overall_quality_score >= 60:
                    quality_rating = "MEDIUM"
                else:
                    quality_rating = "LOW"
                
                verification_summary["overall_quality_score"] = overall_quality_score
                verification_summary["quality_rating"] = quality_rating
                
                # Store comprehensive verification report
                entry_id = self.memory_manager.store_entry(
                    agent_name="Verifier",
                    content_type="verification_report",
                    content=verification_summary,
                    metadata={"symbol": symbol, "verification_type": "comprehensive"},
                    tags=["verification", "report", "comprehensive", symbol.lower()]
                )
                
                sections_count = len(verification_summary["verification_sections"])
                
                return f"Comprehensive verification report generated (ID: {entry_id}). Overall quality: {quality_rating} ({overall_quality_score:.1f}/100). Sections included: {sections_count}"
                
            except Exception as e:
                return f"Error generating verification report: {e}"
        
        # Register functions with autogen
        self.agent.register_for_execution(name="verify_financial_claims")(verify_financial_claims)
        self.agent.register_for_execution(name="critique_comprehensive_analysis")(critique_comprehensive_analysis)
        self.agent.register_for_execution(name="verify_calculation_accuracy")(verify_calculation_accuracy)
        self.agent.register_for_execution(name="generate_verification_report")(generate_verification_report)