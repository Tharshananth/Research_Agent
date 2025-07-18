

#   Extractor_agent

import os
import re

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    import fitz  # pymupdf
except ImportError:
    print("Please install: pip install PyPDF2 pdfplumber pymupdf")

# Assume LLM Manager is available
from core.llm_manager import LLMManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Structure for extraction results"""
    paper_id: str
    metadata: Dict[str, Any]
    sections: Dict[str, str]
    full_text: str
    extraction_quality: Dict[str, Any]
    llm_enhanced: Dict[str, Any]

class ExtractorAgent:
    """
    Agent responsible for extracting and processing content from PDF papers
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.processed_dir = Path("data/processed_text")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Common academic paper section patterns
        self.section_patterns = {
            'abstract': r'(?i)abstract\s*:?\s*',
            'introduction': r'(?i)(?:1\.?\s*)?introduction\s*:?\s*',
            'methodology': r'(?i)(?:2\.?\s*)?(?:methodology|methods?|approach)\s*:?\s*',
            'results': r'(?i)(?:3\.?\s*)?(?:results?|findings?|experiments?)\s*:?\s*',
            'discussion': r'(?i)(?:4\.?\s*)?discussion\s*:?\s*',
            'conclusion': r'(?i)(?:5\.?\s*)?(?:conclusion|summary)\s*:?\s*',
            'references': r'(?i)(?:references?|bibliography)\s*:?\s*'
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF using multiple fallback methods"""
        
        # Method 1: Try pdfplumber (best for complex layouts)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    logger.info(f"Successfully extracted text using pdfplumber: {len(text)} chars")
                    return text
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        # Method 2: Try PyPDF2 (fallback)
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    logger.info(f"Successfully extracted text using PyPDF2: {len(text)} chars")
                    return text
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")
        
        # Method 3: Try pymupdf (final fallback)
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            if text.strip():
                logger.info(f"Successfully extracted text using pymupdf: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"pymupdf failed: {e}")
        
        logger.error(f"All PDF extraction methods failed for {pdf_path}")
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing common PDF artifacts"""
        if not text:
            return ""
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'\n\s*Page\s+\d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\^\&\*\+\=\<\>\~\`]', '', text)
        
        # Fix broken words (common in PDF extraction)
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def extract_metadata(self, text: str, paper_data: Dict) -> Dict[str, Any]:
        """Extract metadata from paper text and existing data"""
        metadata = {
            "title": paper_data.get("title", "Unknown Title"),
            "authors": paper_data.get("authors", []),
            "doi": paper_data.get("doi", ""),
            "publication_year": None,
            "journal": None,
            "keywords": []
        }
        
        # Extract publication year
        year_match = re.search(r'(?:19|20)\d{2}', text[:2000])
        if year_match:
            metadata["publication_year"] = year_match.group()
        
        # Extract journal name (basic heuristic)
        journal_patterns = [
            r'(?i)published\s+in\s+([^\n]+)',
            r'(?i)journal\s+of\s+([^\n]+)',
            r'(?i)proceedings\s+of\s+([^\n]+)'
        ]
        for pattern in journal_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                metadata["journal"] = match.group(1).strip()
                break
        
        return metadata
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract paper sections using regex patterns"""
        sections = {}
        
        # Split text into potential sections
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            section_found = False
            for section_name, pattern in self.section_patterns.items():
                if re.match(pattern, line):
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def assess_extraction_quality(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Assess the quality of text extraction"""
        quality = {
            "status": "success",
            "confidence": 0.0,
            "issues": [],
            "text_length": len(text),
            "sections_found": len(sections)
        }
        
        # Basic quality checks
        if len(text) < 500:
            quality["issues"].append("Text too short - possible extraction failure")
            quality["confidence"] -= 0.3
        
        if not any(section in sections for section in ['abstract', 'introduction']):
            quality["issues"].append("Missing key sections (abstract/introduction)")
            quality["confidence"] -= 0.2
        
        # Check for common extraction artifacts
        if text.count('') > len(text) * 0.1:  # Too many special characters
            quality["issues"].append("High number of extraction artifacts")
            quality["confidence"] -= 0.1
        
        # Calculate confidence score
        base_confidence = 0.8
        if 'abstract' in sections:
            base_confidence += 0.1
        if 'introduction' in sections:
            base_confidence += 0.1
        if len(sections) >= 4:
            base_confidence += 0.1
        
        quality["confidence"] = max(0.0, min(1.0, base_confidence + quality["confidence"]))
        
        if quality["confidence"] < 0.5:
            quality["status"] = "low_quality"
        elif quality["confidence"] < 0.3:
            quality["status"] = "failed"
        
        return quality
    
    def enhance_with_llm(self, text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Use LLM to extract additional insights"""
        
        try:
            # Use the LLM Manager's specialized method for paper analysis
            return self.llm_manager.extract_insights(text, sections)
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return {
                "key_contributions": ["LLM enhancement failed"],
                "main_findings": ["LLM enhancement failed"],
                "technical_approach": "LLM enhancement failed",
                "research_gap": "LLM enhancement failed"
            }
    
    def save_extracted_data(self, result: ExtractionResult) -> str:
        """Save extraction result to file"""
        output_path = self.processed_dir / f"{result.paper_id}.json"
        
        # Convert dataclass to dict for JSON serialization
        result_dict = {
            "paper_id": result.paper_id,
            "metadata": result.metadata,
            "sections": result.sections,
            "full_text": result.full_text,
            "extraction_quality": result.extraction_quality,
            "llm_enhanced": result.llm_enhanced
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted data to {output_path}")
        return str(output_path)
    
    def process_paper(self, paper_data: Dict) -> Optional[ExtractionResult]:
        """Main method to process a paper"""
        
        pdf_path = paper_data.get("pdf_path")
        if not pdf_path or not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        logger.info(f"Processing paper: {paper_data.get('title', 'Unknown')}")
        
        # Step 1: Extract text from PDF
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            logger.error("Failed to extract text from PDF")
            return None
        
        # Step 2: Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Step 3: Extract metadata
        metadata = self.extract_metadata(cleaned_text, paper_data)
        
        # Step 4: Extract sections
        sections = self.extract_sections(cleaned_text)
        
        # Step 5: Assess quality
        quality = self.assess_extraction_quality(cleaned_text, sections)
        
        # Step 6: LLM enhancement
        llm_enhanced = self.enhance_with_llm(cleaned_text, sections)
        
        # Step 7: Create result
        paper_id = paper_data.get("doi", "").replace("/", "_").replace(":", "_")
        if not paper_id:
            paper_id = f"paper_{hash(paper_data.get('title', 'unknown'))}"
        
        result = ExtractionResult(
            paper_id=paper_id,
            metadata=metadata,
            sections=sections,
            full_text=cleaned_text,
            extraction_quality=quality,
            llm_enhanced=llm_enhanced
        )
        
        # Step 8: Save result
        self.save_extracted_data(result)
        
        return result

# Usage example
from agents.fetcher_agent import handle_user_input

if __name__ == "__main__":
    input_query = "Driver Drowsiness Detection Using ECG Signals"  # or a DOI or PDF URL

    paper_results = handle_user_input(input_query)

    llm_manager = LLMManager(api_key="YOUR_API_KEY")
    extractor = ExtractorAgent(llm_manager)

    for paper_data in paper_results:
        if paper_data.get("pdf_path"):
            result = extractor.process_paper(paper_data)
            if result:
                print(f"✅ Successfully processed: {result.metadata['title']}")
                print(f"📊 Quality: {result.extraction_quality['confidence']:.2f}")
                print(f"📄 Sections found: {list(result.sections.keys())}")
                print(f"🧠 LLM insights: {result.llm_enhanced['key_contributions']}")
            else:
                print("❌ Extraction failed")
        else:
            print("⚠️ No PDF found or download failed")


