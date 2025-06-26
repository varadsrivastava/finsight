import os
import base64
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import glob
from textwrap import dedent
from pathlib import Path
from PIL import Image
import io
import requests
from pdf2image import convert_from_path
from pypdf import PdfReader
import json

from shared_memory.memory_manager import SharedMemoryManager
from config.config import FinSightConfig    

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import BytesIO
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils.import_utils import is_flash_attn_2_available


# !sudo apt-get update && sudo apt-get install poppler-utils -y
# !pip3 install colpali-engine pdf2image pypdf pyvespa vespacli requests numpy tqdm

logger = logging.getLogger(__name__)

class PDFAnalyzer:
    """Analyzes annual report PDFs using ColPali retrieval and vision models"""
    
    def __init__(self, config: FinSightConfig, memory_manager: SharedMemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        self.model = None
        self._pdf_cache = {}
        
        # Check dependencies

        
        # Define analysis aspects
        self.analysis_aspects = [
                                "income statement",
                                "balance sheet",
                                "cash flow statement",
                                "business segments",
                                "risk factors",
                                "business overview",
                                "company description"
                            ]
        
        self.analysis_aspects_instructions = {
            "income statement": self._get_income_statement_instruction(),
            "balance sheet": self._get_balance_sheet_instruction(),
            "cash flow statement": self._get_cash_flow_instruction(),
            "business segments": self._get_segment_instruction(),
            "risk factors": self._get_risk_assessment_instruction(),
            "business summary": self._get_business_highlights_instruction(),
            "company description": self._get_company_description_instruction()
        }
    
    def load_model(self):
        self.model_name = "vidore/colqwen2-v0.1"

        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        logger.info("ColPali model loaded successfully")
    
    def find_annual_report(self, ticker: str, year: Optional[str] = None) -> Optional[str]:
        """Find the annual report PDF for a given ticker"""
        try:
            search_patterns = [
                f"*{ticker.upper()}*.pdf",
                f"*{ticker.lower()}*.pdf"
            ]
            
            search_dir = os.path.join(self.config.sec_filings_path, ticker, "10-K")
                        
            found_files = []
            # for search_dir in search_dir:
            if os.path.exists(search_dir):
                for pattern in search_patterns:
                    files = glob.glob(os.path.join(search_dir, "**", pattern), recursive=True)
                    found_files.extend(files)
                
                if not found_files:
                    logger.warning(f"No annual report PDF found for ticker {ticker}")
                    return None
            
            # if year:
            #     for file in found_files:
            #         if year in file:
            #             return file
            
            # selected_file = sorted(found_files)[-1]
            selected_file = sorted(found_files)[0]
            logger.info(f"Found annual report for {ticker}: {selected_file}")
            return selected_file
            
        except Exception as e:
            logger.error(f"Error finding annual report for {ticker}: {e}")
            return None
        
        
    def extract_pages_as_images(self, pdf_path: str):
        """Extract specified pages from PDF as images"""
        try:
            with open(pdf_path, "rb") as pdf_file:
                # Save the PDF temporarily to disk (pdf2image requires a file path)
                temp_file = "temp.pdf"
                with open(temp_file, "wb") as f:
                    f.write(pdf_file.read())
                
                reader = PdfReader(temp_file)
                page_texts = []
                for page_number in range(len(reader.pages)):
                    page = reader.pages[page_number]
                    text = page.extract_text()
                    page_texts.append(text)
                
                images = convert_from_path(temp_file)
                assert len(images) == len(page_texts)
                
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
                logger.info(f"Successfully extracted {len(images)} pages as images")
                return (images, page_texts)
            
        except Exception as e:
            logger.error(f"Error extracting pages as images: {e}")
            return []
    
    def create_retrieval_index(self, ticker, pdf_images):
        """Create a retrieval index for the PDF using ColPali"""
        try:
            if not self.model:
                logger.error("ColPali model not available")
                return False
            
            page_embeddings = []
            dataloader = DataLoader(
                pdf_images,
                batch_size=2,
                shuffle=False,
                collate_fn=lambda x: self.processor.process_images(x),
            )

            for batch_doc in tqdm(dataloader):
                with torch.no_grad():
                    batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                    embeddings_doc = self.model(**batch_doc)
                    page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
            # pdf_embeddings = page_embeddings
            
            # save embedding index
            index_name = os.path.join(self.config.sec_filings_path, ticker, "10-K", "pdf_index_embeddings.json")
            with open(index_name, "w") as f:
                json.dump(page_embeddings, f)
            
            # self._pdf_cache[pdf_path] = {
            #     "index_name": index_name,
            #     "indexed_at": datetime.now().isoformat()
            # }
            
            logger.info(f"Successfully created retrieval index for {index_name}")
            return pdf_images, page_embeddings
            
        except Exception as e:
            logger.error(f"Error creating retrieval index: {e}")
            return False
        

    def create_query_embeddings(self, analysis_aspects):
        """Create query embeddings for a given query"""
        try:
            queries = analysis_aspects
            dataloader = DataLoader(
            queries,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
            )
            qs = []
            for batch_query in dataloader:
                with torch.no_grad():
                    batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                    embeddings_query = self.model(**batch_query)
                    qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

            return queries,qs
        
        except Exception as e:
            logger.error(f"Error creating query embeddings: {e}")
            return None, None

    def get_base64_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return str(base64.b64encode(buffered.getvalue()), "utf-8")
    
    def resize_image(self, image, max_height=800):
        width, height = image.size
        if height > max_height:
            ratio = max_height / height
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height))
        return image

    
    def retrieve_pages_for_aspect(self, pdf_path: str, queries, pdf_images, ticker, qs, pdf_embeddings, k: int = 5) -> List[int]:
        """Retrieve relevant page numbers and images for analysis"""
        try:
            retrieved_images_for_gpt = []

            for query_idx, query in enumerate(queries):
                top_k_indices = torch.topk(self.processor.score_multi_vector(qs[query_idx].unsqueeze(0), pdf_embeddings), k).indices.tolist()[0]
                for i, page_index in enumerate(top_k_indices):
                    page_image = pdf_images[page_index]
                    base64_image = self.get_base64_image(self.resize_image(page_image))
                    retrieved_images_for_gpt.append({
                        "query": query,
                        "page_number": page_index + 1,
                        "base64_image": base64_image
                    })

            print(f"Prepared {len(retrieved_images_for_gpt)} images for GPT-4o mini analysis.")
                        
            logger.info(f"Retrieved {len(retrieved_images_for_gpt)} pages for all aspects.")

            # save retrieved_images_for_gpt to json
            with open(os.path.join(self.config.sec_filings_path, ticker, "10-K", "retrieved_images.json"), "w") as f:
                json.dump(retrieved_images_for_gpt, f)

            return retrieved_images_for_gpt
            
        except Exception as e:
            logger.error(f"Error retrieving pages for aspects.")
            return []
    
    # Instruction templates based on finrobot analyzer
    def _get_income_statement_instruction(self) -> str:
        return dedent("""
            Conduct a comprehensive analysis of the company's income statement for the current fiscal year. 
            Start with an overall revenue record, including Year-over-Year or Quarter-over-Quarter comparisons, 
            and break down revenue sources to identify primary contributors and trends. Examine the Cost of 
            Goods Sold for potential cost control issues. Review profit margins such as gross, operating, 
            and net profit margins to evaluate cost efficiency, operational effectiveness, and overall profitability. 
            Analyze Earnings Per Share to understand investor perspectives. Compare these metrics with historical 
            data and industry or competitor benchmarks to identify growth patterns, profitability trends, and 
            operational challenges. The output should be a strategic overview of the company's financial health 
            in a single paragraph, less than 130 words, summarizing the previous analysis into 4-5 key points under 
            respective subheadings with specific discussion and strong data support.
        """)
    
    def _get_balance_sheet_instruction(self) -> str:
        return dedent("""
            Delve into a detailed scrutiny of the company's balance sheet for the most recent fiscal year, pinpointing 
            the structure of assets, liabilities, and shareholders' equity to decode the firm's financial stability and 
            operational efficiency. Focus on evaluating the liquidity through current assets versus current liabilities, 
            the solvency via long-term debt ratios, and the equity position to gauge long-term investment potential. 
            Contrast these metrics with previous years' data to highlight financial trends, improvements, or deteriorations. 
            Finalize with a strategic assessment of the company's financial leverage, asset management, and capital structure, 
            providing insights into its fiscal health and future prospects in a single paragraph. Less than 130 words.
        """)
    
    def _get_cash_flow_instruction(self) -> str:
        return dedent("""
            Dive into a comprehensive evaluation of the company's cash flow for the latest fiscal year, focusing on cash inflows 
            and outflows across operating, investing, and financing activities. Examine the operational cash flow to assess the 
            core business profitability, scrutinize investing activities for insights into capital expenditures and investments, 
            and review financing activities to understand debt, equity movements, and dividend policies. Compare these cash movements 
            to prior periods to discern trends, sustainability, and liquidity risks. Conclude with an informed analysis of the company's 
            cash management effectiveness, liquidity position, and potential for future growth or financial challenges in a single paragraph. 
            Less than 130 words.
        """)
    
    def _get_segment_instruction(self) -> str:
        return dedent("""
            Identify the company's business segments and create a segment analysis using the Management's Discussion and Analysis 
            and the income statement, subdivided by segment with clear headings. Address revenue and net profit with specific data, 
            and calculate the changes. Detail strategic partnerships and their impacts, including details like the companies or organizations. 
            Describe product innovations and their effects on income growth. Quantify market share and its changes, or state market position 
            and its changes. Analyze market dynamics and profit challenges, noting any effects from national policy changes. Include the cost side, 
            detailing operational costs, innovation investments, and expenses from channel expansion, etc. Support each statement with evidence, 
            keeping each segment analysis concise and under 60 words, accurately sourcing information. For each segment, consolidate the most 
            significant findings into one clear, concise paragraph, excluding less critical or vaguely described aspects to ensure clarity and 
            reliance on evidence-backed information. For each segment, the output should be one single paragraph within 150 words.
        """)
    
    def _get_risk_assessment_instruction(self) -> str:
        return dedent("""
            According to the given information in the 10-k report, summarize the top 3 key risks of the company. 
            Then, for each key risk, break down the risk assessment into the following aspects:
            1. Industry Vertical Risk: How does this industry vertical compare with others in terms of risk? Consider factors such as regulation, market volatility, and competitive landscape.
            2. Cyclicality: How cyclical is this industry? Discuss the impact of economic cycles on the company's performance.
            3. Risk Quantification: Enumerate the key risk factors with supporting data if the company or segment is deemed risky.
            4. Downside Protections: If the company or segment is less risky, discuss the downside protections in place. Consider factors such as diversification, long-term contracts, and government regulation.

            Finally, provide a detailed and nuanced assessment that reflects the true risk landscape of the company. And Avoid any bullet points in your response.
        """)
    
    def _get_business_highlights_instruction(self) -> str:
        return dedent("""
            According to the given information, describe the performance highlights for each company's business line.
            Each business description should contain one sentence of a summarization and one sentence of explanation.
        """)
    
    def _get_company_description_instruction(self) -> str:
        return dedent("""
            According to the given information, 
            1. Briefly describe the company overview and company's industry, using the structure: "Founded in xxxx, 'company name' is a xxxx that provides .....
            2. Highlight core strengths and competitive advantages key products or services,
            3. Include topics about end market (geography), major customers (blue chip or not), market share for market position section,
            4. Identify current industry trends, opportunities, and challenges that influence the company's strategy,
            5. Outline recent strategic initiatives such as product launches, acquisitions, or new partnerships, and describe the company's response to market conditions. 
            Less than 300 words.
        """) 