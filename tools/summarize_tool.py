from typing import Dict, Any
from .base_tool import BaseTool
import ollama
import logging

logger = logging.getLogger(__name__)


class SummarizeResumeTool(BaseTool):    
    def __init__(self, vector_store, llm_model: str):
        super().__init__()
        self.name = "summarize_resume"
        self.description = "Generate a concise summary of a specific candidate's resume. Use this when you need detailed information about a particular candidate."
        self.parameters = {
            "candidate_name": {
                "type": "string",
                "description": "Name of the candidate to summarize",
                "required": True
            }
        }
        self.vector_store = vector_store
        self.llm_model = llm_model
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # Extract parameters
        candidate_name = kwargs.get('candidate_name', '')
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(**kwargs)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "message": f"Invalid input: {error_msg}"
                }
            
            self.logger.info(f"Generating summary for candidate: {candidate_name}")
            
            # Search for candidate by name
            try:
                results = self.vector_store.collection.get(
                    where={"person_name": candidate_name}
                )
            except Exception as e:
                # Fallback: search by filename
                self.logger.warning(f"Could not find by person_name, trying filename: {e}")
                results = self.vector_store.collection.get(
                    where={"filename": {"$contains": candidate_name}}
                )
            
            if not results or not results.get('documents'):
                return {
                    "success": False,
                    "error": f"Candidate '{candidate_name}' not found",
                    "message": f"No resume found for candidate: {candidate_name}"
                }
            
            # Combine all chunks for this candidate
            full_text = "\n\n".join(results['documents'])
            metadata = results['metadatas'][0] if results['metadatas'] else {}
            
            # Generate summary using LLM
            summary_prompt = f"""Analyze this resume and provide a concise professional summary.

Resume Content:
{full_text[:4000]}

Generate a summary with:
1. Brief professional overview (2-3 sentences)
2. Key technical skills and expertise
3. Notable achievements or highlights
4. Current role and experience level

Keep the summary concise and professional."""

            try:
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=summary_prompt,
                    options={
                        "temperature": 0.3,
                    }
                )
                
                summary_text = response['response'].strip()
                
            except Exception as e:
                self.logger.error(f"Error generating summary with LLM: {e}")
                # Fallback to metadata-based summary
                summary_text = self._generate_fallback_summary(metadata, full_text)
            
            # Extract highlights from metadata
            highlights = []
            
            if metadata.get('current_role'):
                highlights.append(f"Current Role: {metadata['current_role']}")
            
            if metadata.get('current_company'):
                highlights.append(f"Current Company: {metadata['current_company']}")
            
            if metadata.get('total_experience_years'):
                highlights.append(f"Experience: {metadata['total_experience_years']} years")
            
            if metadata.get('location'):
                highlights.append(f"Location: {metadata['location']}")
            
            skills = metadata.get('skills', '').split(',')[:10] if metadata.get('skills') else []
            if skills:
                highlights.append(f"Top Skills: {', '.join(skills)}")
            
            self.logger.info(f"Successfully generated summary for {candidate_name}")
            
            return {
                "success": True,
                "data": {
                    "candidate": candidate_name,
                    "summary": summary_text,
                    "highlights": highlights,
                    "metadata": {
                        "name": metadata.get('person_name', candidate_name),
                        "email": metadata.get('email'),
                        "phone": metadata.get('phone'),
                        "current_company": metadata.get('current_company'),
                        "current_role": metadata.get('current_role'),
                        "experience_years": metadata.get('total_experience_years'),
                        "location": metadata.get('location'),
                        "skills": skills,
                        "companies": metadata.get('companies', '').split(',') if metadata.get('companies') else [],
                        "education": metadata.get('education', '').split(',') if metadata.get('education') else []
                    }
                },
                "message": f"Generated summary for {candidate_name}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in summarize_resume: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error generating summary: {str(e)}"
            }
    
    def _generate_fallback_summary(self, metadata: Dict, full_text: str) -> str:
        name = metadata.get('person_name', 'Unknown')
        role = metadata.get('current_role', 'Professional')
        company = metadata.get('current_company', '')
        experience = metadata.get('total_experience_years', 'N/A')
        location = metadata.get('location', '')
        
        summary = f"{name} is a {role}"
        if company:
            summary += f" at {company}"
        summary += f" with {experience} years of experience"
        if location:
            summary += f" based in {location}"
        summary += "."
        
        skills = metadata.get('skills', '').split(',')[:5] if metadata.get('skills') else []
        if skills:
            summary += f" Key skills include: {', '.join(skills)}."
        
        return summary