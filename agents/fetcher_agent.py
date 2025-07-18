# Enhanced Fetcher Agent
import os
import requests
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
import time
import hashlib
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PAPER_DIR = Path("data/raw_papers/")
RAW_PAPER_DIR.mkdir(parents=True, exist_ok=True)

class FetcherAgent:
    def __init__(self, email: str = "tharshananth969@gmail.com", rate_limit: float = 1.0):
        self.email = email
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
    def _rate_limit_wait(self):
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()

    def is_doi(self, s: str) -> bool:
        """Check if string is a DOI"""
        return s.startswith("10.") and "/" in s

    def is_pdf_url(self, s: str) -> bool:
        """Check if string is a PDF URL"""
        return s.lower().endswith(".pdf") or "pdf" in s.lower()

    def is_local_file(self, s: str) -> bool:
        """Check if string is a local file path"""
        return os.path.exists(s)

    def is_topic(self, s: str) -> bool:
        """Check if string is a search topic"""
        return not (self.is_doi(s) or self.is_pdf_url(s) or self.is_local_file(s))

    def clean_filename(self, s: str) -> str:
        """Clean filename for safe file system storage"""
        # Remove invalid characters and limit length
        cleaned = "".join(c for c in s if c.isalnum() or c in "._-")
        return cleaned[:100]  # Limit length

    def get_paper_metadata_from_doi(self, doi: str) -> Optional[Dict]:
        """Get detailed metadata from CrossRef API using DOI"""
        try:
            self._rate_limit_wait()
            url = f"https://api.crossref.org/works/{doi}"
            headers = {'User-Agent': f'PaperFetcher/1.0 (mailto:{self.email})'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json().get("message", {})
            
            return {
                "title": data.get("title", ["Unknown"])[0],
                "authors": [
                    f"{author.get('given', '')} {author.get('family', '')}".strip()
                    for author in data.get("author", [])
                ],
                "doi": doi,
                "url": data.get("URL", f"https://doi.org/{doi}"),
                "journal": data.get("container-title", ["Unknown"])[0] if data.get("container-title") else "Unknown",
                "year": data.get("published-print", {}).get("date-parts", [[None]])[0][0] or 
                       data.get("published-online", {}).get("date-parts", [[None]])[0][0],
                "abstract": data.get("abstract", ""),
                "subject": data.get("subject", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get metadata for DOI {doi}: {e}")
            return None

    def search_crossref_by_topic(self, topic: str, max_results: int = 5) -> List[Dict]:
        """Search CrossRef API for papers by topic"""
        try:
            self._rate_limit_wait()
            query = quote(topic)
            url = f"https://api.crossref.org/works?query={query}&rows={max_results}&filter=type:journal-article&sort=relevance"
            headers = {'User-Agent': f'PaperFetcher/1.0 (mailto:{self.email})'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            items = response.json().get("message", {}).get("items", [])
            papers = []
            
            for item in items:
                doi = item.get("DOI", "")
                if not doi:
                    continue
                    
                # Get detailed metadata
                detailed_metadata = self.get_paper_metadata_from_doi(doi)
                if detailed_metadata:
                    papers.append(detailed_metadata)
            
            return papers

        except Exception as e:
            logger.error(f"CrossRef topic search failed: {e}")
            return []

    def resolve_doi_to_pdf_url(self, doi: str) -> Tuple[Optional[str], Dict]:
        """Resolve DOI to PDF URL with additional metadata"""
        try:
            self._rate_limit_wait()
            encoded_doi = quote(doi)
            api_url = f"https://api.unpaywall.org/v2/{encoded_doi}?email={self.email}"
            
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get best open access location
            best_oa = data.get("best_oa_location")
            if best_oa and best_oa.get("url_for_pdf"):
                return best_oa["url_for_pdf"], {
                    "is_open_access": True,
                    "oa_host": best_oa.get("host_type"),
                    "license": best_oa.get("license")
                }
            
            # Check other OA locations
            oa_locations = data.get("oa_locations", [])
            for location in oa_locations:
                if location.get("url_for_pdf"):
                    return location["url_for_pdf"], {
                        "is_open_access": True,
                        "oa_host": location.get("host_type"),
                        "license": location.get("license")
                    }
            
            return None, {"is_open_access": False, "reason": "No open access PDF found"}
            
        except Exception as e:
            logger.error(f"Unpaywall API failed for {doi}: {e}")
            return None, {"is_open_access": False, "reason": f"API error: {e}"}

    def validate_pdf_content(self, content: bytes) -> bool:
        """Validate that downloaded content is actually a PDF"""
        # Check PDF signature
        if content.startswith(b'%PDF-'):
            return True
        
        # Check for common HTML error pages
        content_str = content[:1000].decode('utf-8', errors='ignore').lower()
        if any(marker in content_str for marker in ['<html', '<body', 'error', 'not found']):
            return False
            
        return False

    def download_pdf(self, pdf_url: str, filename: str, max_retries: int = 3) -> Optional[str]:
        """Download PDF with retry logic and validation"""
        file_path = RAW_PAPER_DIR / filename
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(pdf_url, headers=headers, stream=True, 
                                      timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Download content
                content = response.content
                
                # Validate PDF content
                if not self.validate_pdf_content(content):
                    logger.warning(f"Downloaded content is not a valid PDF (attempt {attempt + 1})")
                    continue
                
                # Save file
                with open(file_path, "wb") as f:
                    f.write(content)
                
                logger.info(f"PDF saved to: {file_path} ({len(content)} bytes)")
                return str(file_path)
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        logger.error(f"Failed to download PDF after {max_retries} attempts")
        return None

    def handle_user_input(self, input_data: str, max_papers: int = 5) -> List[Dict]:
        """Main handler for different input types"""
        results = []

        if self.is_doi(input_data):
            logger.info("Input detected as DOI")
            metadata = self.get_paper_metadata_from_doi(input_data)
            if metadata:
                pdf_url, oa_info = self.resolve_doi_to_pdf_url(input_data)
                if pdf_url:
                    filename = self.clean_filename(input_data) + ".pdf"
                    pdf_path = self.download_pdf(pdf_url, filename)
                    metadata.update({
                        "pdf_url": pdf_url,
                        "pdf_path": pdf_path,
                        "open_access_info": oa_info
                    })
                else:
                    metadata.update({
                        "pdf_url": None,
                        "pdf_path": None,
                        "open_access_info": oa_info
                    })
                results.append(metadata)

        elif self.is_pdf_url(input_data):
            logger.info("Input detected as PDF URL")
            filename = self.clean_filename(os.path.basename(input_data))
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            pdf_path = self.download_pdf(input_data, filename)
            results.append({
                "title": "PDF from URL",
                "authors": [],
                "doi": None,
                "pdf_url": input_data,
                "pdf_path": pdf_path,
                "open_access_info": {"is_open_access": True, "source": "direct_url"}
            })

        elif self.is_local_file(input_data):
            logger.info("Input detected as local file")
            results.append({
                "title": "Local PDF",
                "authors": [],
                "doi": None,
                "pdf_url": None,
                "pdf_path": input_data,
                "open_access_info": {"is_open_access": True, "source": "local_file"}
            })

        elif self.is_topic(input_data):
            logger.info("Input detected as topic")
            papers = self.search_crossref_by_topic(input_data, max_results=max_papers)
            for paper in papers:
                pdf_url, oa_info = self.resolve_doi_to_pdf_url(paper["doi"])
                if pdf_url:
                    filename = self.clean_filename(paper["doi"]) + ".pdf"
                    pdf_path = self.download_pdf(pdf_url, filename)
                    paper.update({
                        "pdf_url": pdf_url,
                        "pdf_path": pdf_path,
                        "open_access_info": oa_info
                    })
                else:
                    paper.update({
                        "pdf_url": None,
                        "pdf_path": None,
                        "open_access_info": oa_info
                    })
                results.append(paper)

        return results

    def fetch_pdf_path(self, input_query: str = "artificial intelligence") -> str:
        """Wrapper to fetch the first available PDF path"""
        papers = self.handle_user_input(input_query)
        for paper in papers:
            if paper.get("pdf_path"):
                return paper["pdf_path"]
        raise ValueError("No valid PDF found")

# Create a global instance for backward compatibility
fetcher_agent = FetcherAgent()

# Function wrapper for backward compatibility - FIXED TO ACCEPT max_papers
def handle_user_input(input_data: str, max_papers: int = 5) -> List[Dict]:
    """Global function wrapper for backward compatibility"""
    return fetcher_agent.handle_user_input(input_data, max_papers)

# Usage
if __name__ == "__main__":
    results = handle_user_input("machine learning", max_papers=3)
    for result in results:
        print(f"Title: {result['title']}")
        print(f"PDF Available: {result['pdf_path'] is not None}")
        print("---")

