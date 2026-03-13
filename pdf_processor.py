import pdfplumber
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            logger.info(msg=f"Successfully extracted text from {pdf_path.name}")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
            return ""
    
    def process_resume_folder(self, folder_path: Path) -> List[Dict[str, str]]:
        resumes = []
        pdf_files = list(folder_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return resumes
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                resumes.append({
                    "filename": pdf_file.name,
                    "content": text,
                    "path": str(pdf_file)
                })
        
        logger.info(f"Successfully processed {len(resumes)} resumes")
        return resumes
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
