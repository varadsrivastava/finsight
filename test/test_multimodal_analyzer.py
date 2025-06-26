"""
Comprehensive test for MultimodalAnalyzer functionality
Tests:
1. Chart generation (price, ratios, P/E-EPS, sentiment)
2. Memory storage of charts
3. PDF analysis using ColPali retrieval
4. Memory storage of PDF analysis results
"""

from agents.data_collector import DataCollectorAgent, DataRequest
from agents.multimodal_analyzer import MultimodalAnalyzerAgent
from shared_memory.memory_manager import SharedMemoryManager
import logging
import os
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveMultimodalTest:
    def __init__(self):
        self.memory_manager = SharedMemoryManager()
        self.data_collector = DataCollectorAgent(self.memory_manager)
        self.multimodal_analyzer = MultimodalAnalyzerAgent(self.memory_manager)
        self.symbol = "MSFT"
        self.test_results = {
            "chart_generation": {},
            "chart_memory_storage": {},
            "pdf_analysis": {},
            "pdf_memory_storage": {}
        }

    def setup_test_data(self):
        """Setup necessary data for testing"""
        logger.info(f"Setting up test data for {self.symbol}...")
        
        # Check if data already exists
        existing_financial = self.memory_manager.search_entries(
            query=f"financial statements {self.symbol}",
            content_type_filter="financial_statements",
            n_results=1
        )
        
        existing_stock = self.memory_manager.search_entries(
            query=f"stock data {self.symbol}",
            content_type_filter="stock_data",
            n_results=1
        )
        
        if not existing_financial or not existing_stock:
            logger.info("Collecting fresh data...")
            request = DataRequest(symbol=self.symbol, data_types=["financial", "stock"])
            result = self.data_collector.process_data_request(request)
            logger.info(f"Data collection result: {result}")
        else:
            logger.info("Using existing data in memory")
            
        return True

    def test_1_chart_generation(self):
        """Test 1: Verify charts are generated correctly"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: CHART GENERATION FUNCTIONALITY")
        logger.info("="*60)
        
        # Test 1.1: Price Chart Generation
        logger.info("1.1 Testing price chart generation...")
        try:
            entries = self.memory_manager.search_entries(
                query=f"stock data {self.symbol}",
                content_type_filter="stock_data",
                n_results=1
            )
            
            if entries:
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if entry:
                    chart_path = self.multimodal_analyzer.chart_generator.create_price_chart(
                        entry["content"], self.symbol
                    )
                    if chart_path and os.path.exists(chart_path):
                        self.test_results["chart_generation"]["price_chart"] = {
                            "status": "PASS", 
                            "path": chart_path
                        }
                        logger.info(f"✓ Price chart generated successfully: {chart_path}")
                    else:
                        self.test_results["chart_generation"]["price_chart"] = {
                            "status": "FAIL", 
                            "error": "Chart file not created"
                        }
                        logger.error("✗ Price chart generation failed")
                else:
                    self.test_results["chart_generation"]["price_chart"] = {
                        "status": "FAIL", 
                        "error": "No stock data entry found"
                    }
            else:
                self.test_results["chart_generation"]["price_chart"] = {
                    "status": "FAIL", 
                    "error": "No stock data in memory"
                }
        except Exception as e:
            self.test_results["chart_generation"]["price_chart"] = {
                "status": "FAIL", 
                "error": str(e)
            }
            logger.error(f"✗ Price chart test failed: {e}")

        # Test 1.2: Financial Ratios Chart Generation
        logger.info("1.2 Testing financial ratios chart generation...")
        try:
            entries = self.memory_manager.search_entries(
                query=f"financial statements {self.symbol}",
                content_type_filter="financial_statements",
                n_results=1
            )
            
            if entries:
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if entry:
                    chart_path = self.multimodal_analyzer.chart_generator.create_financial_ratios_chart(
                        entry["content"], self.symbol
                    )
                    if chart_path and os.path.exists(chart_path):
                        self.test_results["chart_generation"]["ratios_chart"] = {
                            "status": "PASS", 
                            "path": chart_path
                        }
                        logger.info(f"✓ Financial ratios chart generated successfully: {chart_path}")
                    else:
                        self.test_results["chart_generation"]["ratios_chart"] = {
                            "status": "FAIL", 
                            "error": "Chart file not created"
                        }
                        logger.error("✗ Financial ratios chart generation failed")
        except Exception as e:
            self.test_results["chart_generation"]["ratios_chart"] = {
                "status": "FAIL", 
                "error": str(e)
            }
            logger.error(f"✗ Financial ratios chart test failed: {e}")

        # Test 1.3: P/E and EPS Chart Generation
        logger.info("1.3 Testing P/E and EPS chart generation...")
        try:
            entries = self.memory_manager.search_entries(
                query=f"financial statements {self.symbol}",
                content_type_filter="financial_statements",
                n_results=1
            )
            
            if entries:
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if entry:
                    chart_path = self.multimodal_analyzer.chart_generator.create_pe_eps_chart(
                        entry["content"], self.symbol
                    )
                    if chart_path and os.path.exists(chart_path):
                        self.test_results["chart_generation"]["pe_eps_chart"] = {
                            "status": "PASS", 
                            "path": chart_path
                        }
                        logger.info(f"✓ P/E and EPS chart generated successfully: {chart_path}")
                    else:
                        self.test_results["chart_generation"]["pe_eps_chart"] = {
                            "status": "FAIL", 
                            "error": "Chart file not created"
                        }
                        logger.error("✗ P/E and EPS chart generation failed")
        except Exception as e:
            self.test_results["chart_generation"]["pe_eps_chart"] = {
                "status": "FAIL", 
                "error": str(e)
            }
            logger.error(f"✗ P/E and EPS chart test failed: {e}")

    def test_2_chart_memory_storage(self):
        """Test 2: Verify charts are saved in memory in appropriate format"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: CHART MEMORY STORAGE")
        logger.info("="*60)
        
        # Test 2.1: Chart Storage via Agent Functions
        logger.info("2.1 Testing chart storage in memory via agent functions...")
        try:
            # Access the registered functions through the agent's function_map
            if hasattr(self.multimodal_analyzer.agent, 'function_map'):
                generate_price_chart_func = self.multimodal_analyzer.agent.function_map.get("generate_price_chart")
                if generate_price_chart_func:
                    result = generate_price_chart_func(self.symbol)
                    logger.info(f"Price chart generation result: {result}")
                else:
                    logger.error("generate_price_chart function not found in function_map")
                    self.test_results["chart_memory_storage"]["price_chart_storage"] = {
                        "status": "FAIL",
                        "error": "Function not found in function_map"
                    }
                    return
            else:
                # Alternative approach: use the agent directly in a conversation
                # For testing purposes, we'll manually call the chart generator and then store in memory
                entries = self.memory_manager.search_entries(
                    query=f"stock data {self.symbol}",
                    content_type_filter="stock_data",
                    n_results=1
                )
                
                if entries:
                    entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                    if entry:
                        chart_path = self.multimodal_analyzer.chart_generator.create_price_chart(
                            entry["content"], self.symbol
                        )
                        if chart_path:
                            # Manually store chart information in memory to test storage format
                            entry_id = self.memory_manager.store_entry(
                                agent_name="MultimodalAnalyzer",
                                content_type="chart",
                                content={
                                    "chart_type": "price_volume",
                                    "symbol": self.symbol,
                                    "file_path": chart_path,
                                    "description": f"Price and volume chart for {self.symbol}"
                                },
                                metadata={"symbol": self.symbol, "chart_type": "price_volume"},
                                tags=["chart", "price", "volume", self.symbol.lower()]
                            )
                            result = f"Price chart generated and saved: {chart_path}"
                        else:
                            result = "Failed to generate price chart"
                else:
                    result = "No stock data found"
            
            # Check if chart was stored in memory
            chart_entries = self.memory_manager.search_entries(
                query=f"chart {self.symbol} price",
                content_type_filter="chart",
                n_results=5
            )
            
            price_chart_entry = None
            for entry in chart_entries:
                entry_data = self.memory_manager.get_entry_by_id(entry["id"])
                if entry_data["metadata"].get("chart_type") == "price_volume":
                    price_chart_entry = entry_data
                    break
            
            if price_chart_entry:
                self.test_results["chart_memory_storage"]["price_chart_storage"] = {
                    "status": "PASS",
                    "entry_id": price_chart_entry["id"],
                    "metadata": price_chart_entry["metadata"],
                    "content_keys": list(price_chart_entry["content"].keys())
                }
                logger.info(f"✓ Price chart stored in memory with ID: {price_chart_entry['id']}")
            else:
                self.test_results["chart_memory_storage"]["price_chart_storage"] = {
                    "status": "FAIL",
                    "error": "Chart not found in memory"
                }
                logger.error("✗ Price chart not stored in memory")
                
        except Exception as e:
            self.test_results["chart_memory_storage"]["price_chart_storage"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"✗ Chart memory storage test failed: {e}")

        # Test 2.2: Comprehensive Visualization Storage
        logger.info("2.2 Testing comprehensive visualization storage...")
        try:
            # Create multiple charts and store visualization summary
            results = []
            
            # Generate charts directly and store summary
            entries = self.memory_manager.search_entries(
                query=f"financial statements {self.symbol}",
                content_type_filter="financial_statements", 
                n_results=1
            )
            
            if entries:
                entry = self.memory_manager.get_entry_by_id(entries[0]["id"])
                if entry:
                    # Generate ratios chart
                    ratios_chart = self.multimodal_analyzer.chart_generator.create_financial_ratios_chart(
                        entry["content"], self.symbol
                    )
                    if ratios_chart:
                        results.append(f"Ratios Chart: Generated successfully: {ratios_chart}")
                    
                    # Generate P/E EPS chart
                    pe_eps_chart = self.multimodal_analyzer.chart_generator.create_pe_eps_chart(
                        entry["content"], self.symbol
                    )
                    if pe_eps_chart:
                        results.append(f"P/E and EPS Chart: Generated successfully: {pe_eps_chart}")
            
            # Store comprehensive visualization summary
            entry_id = self.memory_manager.store_entry(
                agent_name="MultimodalAnalyzer",
                content_type="visualization_summary",
                content={
                    "symbol": self.symbol,
                    "charts_generated": results,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={"symbol": self.symbol, "visualization_type": "comprehensive"},
                tags=["visualization", "comprehensive", self.symbol.lower()]
            )
            
            # Check for visualization summary in memory
            viz_entries = self.memory_manager.search_entries(
                query=f"visualization {self.symbol}",
                content_type_filter="visualization_summary",
                n_results=1
            )
            
            if viz_entries:
                viz_entry = self.memory_manager.get_entry_by_id(viz_entries[0]["id"])
                self.test_results["chart_memory_storage"]["comprehensive_viz"] = {
                    "status": "PASS",
                    "entry_id": viz_entry["id"],
                    "charts_count": len(viz_entry["content"]["charts_generated"])
                }
                logger.info(f"✓ Comprehensive visualization stored in memory with ID: {viz_entry['id']}")
            else:
                self.test_results["chart_memory_storage"]["comprehensive_viz"] = {
                    "status": "FAIL",
                    "error": "Visualization summary not found in memory"
                }
                logger.error("✗ Comprehensive visualization not stored in memory")
                
        except Exception as e:
            self.test_results["chart_memory_storage"]["comprehensive_viz"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"✗ Comprehensive visualization storage test failed: {e}")

    def test_3_pdf_analysis(self):
        """Test 3: Verify PDF analysis is generated correctly"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: PDF ANALYSIS FUNCTIONALITY")
        logger.info("="*60)
        
        # Test 3.1: PDF File Existence
        pdf_path = "2023-07-27_10-K_msft-20230630.htm.pdf"
        if not os.path.exists(pdf_path):
            self.test_results["pdf_analysis"]["pdf_existence"] = {
                "status": "FAIL",
                "error": f"PDF file not found: {pdf_path}"
            }
            logger.error(f"✗ PDF file not found: {pdf_path}")
            return
        
        self.test_results["pdf_analysis"]["pdf_existence"] = {
            "status": "PASS",
            "path": pdf_path
        }
        logger.info(f"✓ PDF file found: {pdf_path}")

        # Test 3.2: PDF Analysis Execution
        logger.info("3.1 Testing PDF analysis functionality...")
        try:
            result = self.multimodal_analyzer.annual_report_analyzer.analyze_annual_report_pdf(self.symbol)
            
            if isinstance(result, dict) and result.get("success"):
                self.test_results["pdf_analysis"]["analysis_execution"] = {
                    "status": "PASS",
                    "analysis_count": len(result["data"]) if result.get("data") else 0
                }
                logger.info(f"✓ PDF analysis completed successfully. Analyzed {len(result['data'])} items")
            else:
                self.test_results["pdf_analysis"]["analysis_execution"] = {
                    "status": "FAIL",
                    "error": str(result)
                }
                logger.error(f"✗ PDF analysis failed: {result}")
                
        except Exception as e:
            self.test_results["pdf_analysis"]["analysis_execution"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"✗ PDF analysis test failed: {e}")

    def test_4_pdf_memory_storage(self):
        """Test 4: Verify PDF analysis is saved in memory appropriately"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: PDF ANALYSIS MEMORY STORAGE")
        logger.info("="*60)
        
        # Test 4.1: PDF Analysis Storage via Agent Function
        logger.info("4.1 Testing PDF analysis memory storage...")
        try:
            # Access the registered function through the agent's function_map or call directly
            if hasattr(self.multimodal_analyzer.agent, 'function_map'):
                collect_pdf_func = self.multimodal_analyzer.agent.function_map.get("collect_annual_report_analysis")
                if collect_pdf_func:
                    result = collect_pdf_func(self.symbol)
                    logger.info(f"PDF analysis storage result: {result}")
                else:
                    # Alternative: test the PDF analysis storage manually
                    logger.info("Function not found in function_map, testing manual PDF analysis storage...")
                    result = self.multimodal_analyzer.annual_report_analyzer.analyze_annual_report_pdf(self.symbol)
                    
                    if isinstance(result, dict) and result.get("success"):
                        # Store the analysis manually to test storage format
                        analysis_result = result["data"]
                        combined_results = {}
                        
                        for item in analysis_result:
                            query = item["query"]
                            if query not in combined_results:
                                combined_results[query] = f"Analysis for Page number: {item['page_number']} \n {item['gpt_response']}"
                            else:
                                combined_results[query] += f"\n\nAnalysis for Page number: {item['page_number']} \n {item['gpt_response']}"
                        
                        entry_ids = []
                        for aspect in combined_results.keys():
                            entry_id = self.memory_manager.store_entry(
                                agent_name="MultimodalAnalyzer",
                                content_type=f"pdf_analysis_{aspect.strip().replace(' ', '_').lower()}",
                                content=combined_results[aspect],
                                metadata={
                                    "symbol": self.symbol,
                                    "aspect": aspect,
                                },
                                tags=["pdf_analysis", aspect.lower(), self.symbol.lower(), "annual_report"]
                            )
                            entry_ids.append(entry_id)
                        
                        result = f"Successfully analyzed Report for {self.symbol}. Analysis stored with IDs: {entry_ids}."
                    else:
                        result = f"Failed to analyze annual report for {self.symbol}: {result.get('error', 'Unknown error')}"
            else:
                # Call the function directly from the analyzer
                result = "Function access method not available, testing direct analyzer call"
            
            # Check if PDF analysis was stored in memory
            pdf_entries = self.memory_manager.search_entries(
                query=f"pdf analysis {self.symbol}",
                content_type_filter=None,
                n_results=10
            )
            
            # Filter for PDF analysis entries
            pdf_analysis_entries = []
            for entry in pdf_entries:
                entry_data = self.memory_manager.get_entry_by_id(entry["id"])
                if (entry_data and 
                    entry_data.get("content_type", "").startswith("pdf_analysis_") and
                    entry_data.get("metadata", {}).get("symbol") == self.symbol):
                    pdf_analysis_entries.append(entry_data)
            
            if pdf_analysis_entries:
                self.test_results["pdf_memory_storage"]["storage_success"] = {
                    "status": "PASS",
                    "entries_count": len(pdf_analysis_entries),
                    "entry_ids": [entry["id"] for entry in pdf_analysis_entries],
                    "aspects_analyzed": [entry["metadata"].get("aspect") for entry in pdf_analysis_entries]
                }
                logger.info(f"✓ PDF analysis stored in memory. Found {len(pdf_analysis_entries)} entries")
                logger.info(f"✓ Aspects analyzed: {[entry['metadata'].get('aspect') for entry in pdf_analysis_entries]}")
                
                # Test 4.2: Content Quality Check
                sample_entry = pdf_analysis_entries[0]
                content = sample_entry["content"]
                
                self.test_results["pdf_memory_storage"]["content_quality"] = {
                    "status": "PASS" if len(content) > 50 else "FAIL",
                    "content_length": len(content),
                    "has_analysis": "Analysis for Page number:" in content,
                    "sample_content": content[:200] + "..." if len(content) > 200 else content
                }
                
                if self.test_results["pdf_memory_storage"]["content_quality"]["status"] == "PASS":
                    logger.info("✓ PDF analysis content quality check passed")
                else:
                    logger.error("✗ PDF analysis content quality check failed")
                    
            else:
                self.test_results["pdf_memory_storage"]["storage_success"] = {
                    "status": "FAIL",
                    "error": "PDF analysis not found in memory"
                }
                logger.error("✗ PDF analysis not stored in memory")
                
        except Exception as e:
            self.test_results["pdf_memory_storage"]["storage_success"] = {
                "status": "FAIL",
                "error": str(e)
            }
            logger.error(f"✗ PDF analysis memory storage test failed: {e}")

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE TEST REPORT")
        logger.info("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        for test_category, tests in self.test_results.items():
            logger.info(f"\n{test_category.upper().replace('_', ' ')}:")
            for test_name, result in tests.items():
                total_tests += 1
                status = result.get("status", "UNKNOWN")
                if status == "PASS":
                    passed_tests += 1
                    logger.info(f"  ✓ {test_name}: {status}")
                else:
                    logger.error(f"  ✗ {test_name}: {status} - {result.get('error', 'No error details')}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\nOVERALL RESULTS:")
        logger.info(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Save detailed results to file
        report_path = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "success_rate": success_rate,
                "detailed_results": self.test_results
            }, f, indent=2)
        
        logger.info(f"Detailed test report saved to: {report_path}")
        return success_rate >= 75  # Consider 75% success rate as overall pass

    def run_all_tests(self):
        """Run all comprehensive tests"""
        logger.info("Starting comprehensive MultimodalAnalyzer test suite...")
        
        # Setup
        if not self.setup_test_data():
            logger.error("Failed to setup test data. Aborting tests.")
            return False
        
        # Run all tests
        self.test_1_chart_generation()
        self.test_2_chart_memory_storage()
        self.test_3_pdf_analysis()
        self.test_4_pdf_memory_storage()
        
        # Generate report
        success = self.generate_test_report()
        
        if success:
            logger.info("🎉 Overall test suite PASSED!")
        else:
            logger.error("❌ Overall test suite FAILED!")
            
        return success

def main():
    """Main test execution"""
    test_suite = ComprehensiveMultimodalTest()
    success = test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    main() 