#!/usr/bin/env python3
"""
FinSight Multi-Agent Financial Research System

An advanced multi-agent system for comprehensive equity research that enhances FinRobot
capabilities with modular agents for data collection, multimodal analysis, reasoning,
report writing, and verification.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import autogen
from autogen import GroupChat, GroupChatManager

# Import our custom agents
from agents.data_collector import DataCollectorAgent, DataRequest
from agents.multimodal_analyzer import MultimodalAnalyzerAgent
from agents.reasoning_agent import ReasoningAgent
from agents.report_writer import ReportWriterAgent
from agents.verifier import VerifierAgent
from shared_memory.memory_manager import SharedMemoryManager
from config.config import FinSightConfig, get_llm_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finsight.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinSightOrchestrator:
    """Main orchestrator for the FinSight multi-agent research system"""
    
    def __init__(self):
        self.config = FinSightConfig()
        self.memory_manager = SharedMemoryManager(
            self.config.vector_db_path,
            self.config.shared_memory_path
        )
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Create coordinator agent
        self.coordinator = autogen.AssistantAgent(
            name="Coordinator",
            system_message=self._get_coordinator_system_message(),
            llm_config=get_llm_config(),
            max_consecutive_auto_reply=5,
        )
        
        # Setup directories
        self._setup_directories()
        
        logger.info("FinSight system initialized successfully")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents in the system"""
        logger.info("Initializing FinSight agents...")
        
        agents = {
            'data_collector': DataCollectorAgent(self.memory_manager),
            'multimodal_analyzer': MultimodalAnalyzerAgent(self.memory_manager),
            'reasoning_agent': ReasoningAgent(self.memory_manager),
            'report_writer': ReportWriterAgent(self.memory_manager),
            'verifier': VerifierAgent(self.memory_manager)
        }
        
        logger.info(f"Initialized {len(agents)} agents: {', '.join(agents.keys())}")
        return agents
    
    def _setup_directories(self):
        """Setup required directories"""
        directories = [
            self.config.reports_output_path,
            self.config.charts_output_path,
            os.path.dirname(self.config.vector_db_path),
            os.path.dirname(self.config.shared_memory_path)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Directory structure setup complete")
    
    def _get_coordinator_system_message(self) -> str:
        return """You are the Coordinator for the FinSight multi-agent financial research system.

Your role is to orchestrate comprehensive equity research by coordinating between specialized agents:

1. DataCollector: Gathers multi-source financial data (Yahoo Finance, EDGAR, ESG, news, trends)
2. MultimodalAnalyzer: Creates charts and analyzes financial images/tables using FinTral capabilities
3. ReasoningAgent (FinR1): Performs advanced financial analysis and generates investment insights
4. ReportWriter: Creates professional equity research reports with text and visuals
5. Verifier: Fact-checks and validates all analysis for accuracy and quality

Research Process:
1. Define research scope and objectives
2. Coordinate data collection across all relevant sources
3. Direct multimodal analysis and visualization creation
4. Guide comprehensive reasoning and insight generation
5. Orchestrate report compilation and formatting
6. Ensure thorough verification and quality assurance

Coordination Principles:
- Ensure comprehensive data coverage before analysis
- Maintain logical flow from data → analysis → insights → recommendations
- Verify accuracy at each stage
- Generate institutional-quality research outputs
- Store all results in shared memory for transparency

Begin by understanding the research request, then systematically coordinate agent activities."""
    
    async def conduct_comprehensive_research(self, symbol: str, research_scope: str = "full") -> Dict[str, Any]:
        """Conduct comprehensive equity research for a given symbol"""
        logger.info(f"Starting comprehensive research for {symbol}")
        
        research_results = {
            "symbol": symbol,
            "research_scope": research_scope,
            "start_time": datetime.now().isoformat(),
            "stages_completed": [],
            "final_outputs": {},
            "memory_entries": []
        }
        
        try:
            # Stage 1: Data Collection
            logger.info(f"Stage 1: Data Collection for {symbol}")
            data_request = DataRequest(
                symbol=symbol,
                data_types=["financial", "news", "esg", "trends", "sec","earnings","competitors"],
                lookback_days=30,
                include_fundamentals=True
            )
            
            collection_result = self.agents['data_collector'].process_data_request(data_request)
            research_results["stages_completed"].append("data_collection")
            logger.info(f"Data collection completed for {symbol}")
            
            # Stage 2: Multimodal Analysis and Visualization
            logger.info(f"Stage 2: Creating visualizations for {symbol}")
            
            # Create comprehensive visualizations
            vis_agent = self.agents['multimodal_analyzer'].agent
            vis_result = vis_agent.generate_reply(
                f"Create comprehensive visualizations for {symbol} including price charts, financial ratios, and sentiment analysis."
            )
            
            research_results["stages_completed"].append("visualization")
            logger.info(f"Visualization creation completed for {symbol}")
            
            # Stage 3: Advanced Reasoning and Analysis
            logger.info(f"Stage 3: Advanced reasoning and analysis for {symbol}")
            
            reasoning_agent = self.agents['reasoning_agent'].agent
            
            # Perform all reasoning analyses
            analyses_to_run = [
                f"analyze_financial_health('{symbol}')",
                f"generate_valuation_analysis('{symbol}')",
                f"analyze_market_position('{symbol}')",
                f"generate_comprehensive_insight('{symbol}')"
            ]
            
            for analysis in analyses_to_run:
                reasoning_result = reasoning_agent.generate_reply(f"Please run: {analysis}")
                logger.info(f"Completed reasoning analysis: {analysis}")
            
            research_results["stages_completed"].append("reasoning_analysis")
            logger.info(f"Advanced reasoning completed for {symbol}")
            
            # Stage 4: Report Generation
            logger.info(f"Stage 4: Generating equity research report for {symbol}")
            
            report_agent = self.agents['report_writer'].agent
            report_result = report_agent.generate_reply(
                f"generate_equity_report('{symbol}')"
            )
            
            research_results["stages_completed"].append("report_generation")
            research_results["final_outputs"]["report"] = report_result
            logger.info(f"Report generation completed for {symbol}")
            
            # Stage 5: Verification and Quality Assurance
            logger.info(f"Stage 5: Verification and quality assurance for {symbol}")
            
            verifier_agent = self.agents['verifier'].agent
            
            # Run all verification processes
            verification_tasks = [
                f"verify_calculation_accuracy('{symbol}')",
                f"critique_comprehensive_analysis('{symbol}')",
                f"generate_verification_report('{symbol}')"
            ]
            
            for task in verification_tasks:
                verification_result = verifier_agent.generate_reply(f"Please run: {task}")
                logger.info(f"Completed verification task: {task}")
            
            research_results["stages_completed"].append("verification")
            logger.info(f"Verification completed for {symbol}")
            
            # Collect final memory statistics
            memory_stats = self.memory_manager.get_memory_stats()
            research_results["memory_stats"] = memory_stats
            research_results["end_time"] = datetime.now().isoformat()
            
            logger.info(f"Comprehensive research completed for {symbol}")
            return research_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive research for {symbol}: {e}")
            research_results["error"] = str(e)
            research_results["end_time"] = datetime.now().isoformat()
            return research_results
    
    def conduct_batch_research(self, symbols: List[str], research_scope: str = "full") -> Dict[str, Any]:
        """Conduct research on multiple symbols"""
        logger.info(f"Starting batch research for {len(symbols)} symbols: {', '.join(symbols)}")
        
        batch_results = {
            "symbols": symbols,
            "research_scope": research_scope,
            "start_time": datetime.now().isoformat(),
            "individual_results": {},
            "summary_report": None
        }
        
        try:
            # Process each symbol individually
            for symbol in symbols:
                logger.info(f"Processing symbol {symbol} in batch")
                result = asyncio.run(self.conduct_comprehensive_research(symbol, research_scope))
                batch_results["individual_results"][symbol] = result
            
            # Generate summary report
            logger.info("Generating batch summary report")
            report_agent = self.agents['report_writer'].agent
            symbol_list = ', '.join([f"'{s}'" for s in symbols])
            summary_result = report_agent.generate_reply(
                f"create_summary_report([{symbol_list}])"
                )
            
            batch_results["summary_report"] = summary_result
            
            batch_results["end_time"] = datetime.now().isoformat()
            logger.info(f"Batch research completed for {len(symbols)} symbols")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"Error in batch research: {e}")
            batch_results["error"] = str(e)
            batch_results["end_time"] = datetime.now().isoformat()
            return batch_results
    
    def get_research_status(self, symbol: str) -> Dict[str, Any]:
        """Get current research status for a symbol"""
        try:
            # Search for recent entries related to the symbol
            recent_entries = self.memory_manager.search_entries(
                query=symbol,
                n_results=20
            )
            
            # Categorize entries by agent and content type
            status = {
                "symbol": symbol,
                "last_activity": None,
                "agents_active": {},
                "content_types": {},
                "recent_activities": []
            }
            
            for entry in recent_entries:
                # Get full entry details
                full_entry = self.memory_manager.get_entry_by_id(entry["id"])
                if full_entry:
                    agent_name = full_entry["agent_name"]
                    content_type = full_entry["content_type"]
                    timestamp = full_entry["timestamp"]
                    
                    # Track agent activity
                    if agent_name not in status["agents_active"]:
                        status["agents_active"][agent_name] = 0
                    status["agents_active"][agent_name] += 1
                    
                    # Track content types
                    if content_type not in status["content_types"]:
                        status["content_types"][content_type] = 0
                    status["content_types"][content_type] += 1
                    
                    # Track recent activities
                    status["recent_activities"].append({
                        "timestamp": timestamp,
                        "agent": agent_name,
                        "content_type": content_type,
                        "id": entry["id"]
                    })
                    
                    # Update last activity
                    if not status["last_activity"] or timestamp > status["last_activity"]:
                        status["last_activity"] = timestamp
            
            # Sort recent activities by timestamp
            status["recent_activities"].sort(key=lambda x: x["timestamp"], reverse=True)
            status["recent_activities"] = status["recent_activities"][:10]  # Keep top 10
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting research status for {symbol}: {e}")
            return {"error": str(e)}
    
    def cleanup_memory(self, confirm: bool = False):
        """Clean up memory (use with caution)"""
        if confirm:
            self.memory_manager.clear_memory(confirm=True)
            logger.warning("Memory cleared")
        else:
            logger.warning("Memory cleanup requires confirm=True")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        try:
            memory_stats = self.memory_manager.get_memory_stats()
            
            system_stats = {
                "system_version": "1.0.0",
                "agents_available": list(self.agents.keys()),
                "memory_statistics": memory_stats,
                "configuration": {
                    "vector_db_path": self.config.vector_db_path,
                    "reports_output_path": self.config.reports_output_path,
                    "charts_output_path": self.config.charts_output_path
                }
            }
            
            return system_stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

# CLI Interface
def main():
    """Main CLI interface for FinSight"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FinSight Multi-Agent Financial Research System")
    parser.add_argument("action", choices=["research", "batch", "status", "stats", "cleanup"],
                      help="Action to perform")
    parser.add_argument("--symbol", "-s", help="Stock symbol for research")
    parser.add_argument("--symbols", nargs="+", help="Multiple stock symbols for batch research")
    parser.add_argument("--scope", default="full", choices=["full", "basic"],
                      help="Research scope")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive actions")
    
    args = parser.parse_args()
    
    # Initialize system
    finsight = FinSightOrchestrator()
    
    try:
        if args.action == "research":
            if not args.symbol:
                print("Error: --symbol required for research action")
                sys.exit(1)
            
            print(f"Starting comprehensive research for {args.symbol}...")
            result = asyncio.run(finsight.conduct_comprehensive_research(args.symbol, args.scope))
            
            print(f"\nResearch completed for {args.symbol}")
            print(f"Stages completed: {', '.join(result.get('stages_completed', []))}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("Research completed successfully!")
        
        elif args.action == "batch":
            if not args.symbols:
                print("Error: --symbols required for batch action")
                sys.exit(1)
            
            print(f"Starting batch research for {len(args.symbols)} symbols...")
            result = finsight.conduct_batch_research(args.symbols, args.scope)
            
            print(f"\nBatch research completed")
            print(f"Symbols processed: {', '.join(result.get('symbols', []))}")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print("Batch research completed successfully!")
        
        elif args.action == "status":
            if not args.symbol:
                print("Error: --symbol required for status action")
                sys.exit(1)
            
            status = finsight.get_research_status(args.symbol)
            
            print(f"\nResearch status for {args.symbol}:")
            print(f"Last activity: {status.get('last_activity', 'None')}")
            print(f"Active agents: {', '.join(status.get('agents_active', {}).keys())}")
            print(f"Content types: {', '.join(status.get('content_types', {}).keys())}")
        
        elif args.action == "stats":
            stats = finsight.get_system_stats()
            
            print("\nFinSight System Statistics:")
            print(f"Available agents: {', '.join(stats.get('agents_available', []))}")
            
            memory_stats = stats.get('memory_statistics', {})
            print(f"Total memory entries: {memory_stats.get('total_entries', 0)}")
            print(f"Vector DB size: {memory_stats.get('vector_db_size', 0)}")
        
        elif args.action == "cleanup":
            if args.confirm:
                finsight.cleanup_memory(confirm=True)
                print("Memory cleared successfully")
            else:
                print("Add --confirm flag to clean up memory")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()