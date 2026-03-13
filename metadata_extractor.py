import ollama
import json
import logging
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetadataExtractor:
    
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
        logger.info(f"Initialized MetadataExtractor with model: {model_name}")
    
    def extract_metadata(self, resume_text: str, filename: str) -> Dict:
        # Use first 3000 characters for metadata extraction to avoid token limits
        text_sample = resume_text[:3000]
        
        prompt = f"""Analyze the following resume text and extract key information in JSON format.

Resume Text:
{text_sample}

Extract the following information and return ONLY a valid JSON object (no additional text):
{{
    "name": "Full name of the person",
    "current_company": "Current or most recent company",
    "companies": ["List of all companies mentioned"],
    "current_role": "Current or most recent job title",
    "total_experience_years": "Estimated years of experience as a number",
    "skills": ["List of technical skills, tools, and technologies"],
    "education": ["Degrees and institutions"],
    "location": "Location/city if mentioned"
}}

If any field is not found, use null for that field. Return ONLY the JSON object."""

        try:
            logger.info(f"Extracting metadata from {filename}")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for more consistent extraction
                }
            )
            
            # Parse the response
            response_text = response['response'].strip()
            
            # Try to extract JSON from the response
            metadata = self._parse_json_response(response_text)
            
            # Add filename to metadata
            metadata['filename'] = filename
            
            # Fallback extraction if LLM fails
            if not metadata.get('name'):
                metadata['name'] = self._extract_name_fallback(resume_text, filename)
            
            # Ensure lists are not None
            metadata['companies'] = metadata.get('companies') or []
            metadata['skills'] = metadata.get('skills') or []
            metadata['education'] = metadata.get('education') or []
            
            logger.info(f"Successfully extracted metadata for {filename}")
            logger.info(f"  Name: {metadata.get('name')}")
            logger.info(f"  Companies: {len(metadata.get('companies', []))}")
            logger.info(f"  Skills: {len(metadata.get('skills', []))}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {filename}: {str(e)}")
            return self._get_default_metadata(filename, resume_text)
    
    def _parse_json_response(self, response_text: str) -> Dict:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If direct parsing fails, try the whole response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {}
    
    def _extract_name_fallback(self, text: str, filename: str) -> Optional[str]:
        # Try to get name from filename (remove .pdf extension)
        name_from_file = filename.replace('.pdf', '').replace('_', ' ').strip()
        if name_from_file and len(name_from_file) > 2:
            return name_from_file
        
        # Try to find name in first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            # Look for lines that might be names (2-4 words, capitalized)
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                return line
        
        return None
    
    
    def _get_default_metadata(self, filename: str, text: str) -> Dict:
        return {
            'filename': filename,
            'name': self._extract_name_fallback(text, filename),
            'current_company': None,
            'companies': [],
            'current_role': None,
            'total_experience_years': None,
            'skills': [],
            'education': [],
            'location': None
        }