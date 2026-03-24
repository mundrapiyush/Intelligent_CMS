from typing import Dict, Any, List, Optional
from .base_tool import BaseTool
import logging

logger = logging.getLogger(__name__)


class FilterByMetadataTool(BaseTool):
    
    def __init__(self, vector_store):
        super().__init__()
        self.name = "filter_by_metadata"
        self.description = "Filter resumes by specific criteria like skills, experience, location, or company. Use this when you need to narrow down candidates based on structured requirements."
        self.parameters = {
            "skills": {
                "type": "array",
                "description": "List of required skills (e.g., ['Python', 'AWS'])",
                "required": False
            },
            "min_experience": {
                "type": "integer",
                "description": "Minimum years of experience required",
                "required": False
            },
            "location": {
                "type": "string",
                "description": "Candidate location/city",
                "required": False
            },
            "company": {
                "type": "string",
                "description": "Current or past company name",
                "required": False
            },
            "role": {
                "type": "string",
                "description": "Current or past job role/title",
                "required": False
            }
        }
        self.vector_store = vector_store
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # Extract parameters
        skills = kwargs.get('skills', [])
        min_experience = kwargs.get('min_experience')
        location = kwargs.get('location')
        company = kwargs.get('company')
        role = kwargs.get('role')
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(**kwargs)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "message": f"Invalid input: {error_msg}"
                }
            
            filters_applied = {}
            
            # Build filter criteria
            self.logger.info(f"Filtering with criteria: skills={skills}, min_exp={min_experience}, location={location}, company={company}, role={role}")
            
            # Get all documents first (we'll filter in memory for complex criteria)
            all_results = self.vector_store.collection.get()
            
            if not all_results or not all_results.get('metadatas'):
                return {
                    "success": True,
                    "data": {
                        "candidates": [],
                        "count": 0,
                        "filters_applied": filters_applied
                    },
                    "message": "No resumes in database"
                }
            
            # Filter candidates
            candidates_dict = {}
            
            for metadata in all_results['metadatas']:
                # Check if candidate passes all filters
                passes_filter = True
                
                # Skills filter
                if skills:
                    candidate_skills = metadata.get('skills', '').lower()
                    if not any(skill.lower() in candidate_skills for skill in skills):
                        passes_filter = False
                    filters_applied['skills'] = skills
                
                # Experience filter
                if min_experience is not None:
                    candidate_exp = metadata.get('total_experience_years')
                    if candidate_exp is None or int(candidate_exp) < min_experience:
                        passes_filter = False
                    filters_applied['min_experience'] = min_experience
                
                # Location filter
                if location:
                    candidate_location = metadata.get('location', '').lower()
                    if location.lower() not in candidate_location:
                        passes_filter = False
                    filters_applied['location'] = location
                
                # Company filter
                if company:
                    candidate_companies = metadata.get('companies', '').lower()
                    candidate_current = metadata.get('current_company', '').lower()
                    if company.lower() not in candidate_companies and company.lower() not in candidate_current:
                        passes_filter = False
                    filters_applied['company'] = company
                
                # Role filter
                if role:
                    candidate_role = metadata.get('current_role', '').lower()
                    if role.lower() not in candidate_role:
                        passes_filter = False
                    filters_applied['role'] = role
                
                if passes_filter:
                    filename = metadata.get('filename', 'Unknown')
                    
                    if filename not in candidates_dict:
                        candidates_dict[filename] = {
                            'filename': filename,
                            'name': metadata.get('person_name', 'Unknown'),
                            'email': metadata.get('email'),
                            'phone': metadata.get('phone'),
                            'current_company': metadata.get('current_company'),
                            'current_role': metadata.get('current_role'),
                            'experience_years': metadata.get('total_experience_years'),
                            'location': metadata.get('location'),
                            'skills': metadata.get('skills', '').split(',') if metadata.get('skills') else [],
                            'companies': metadata.get('companies', '').split(',') if metadata.get('companies') else [],
                            'education': metadata.get('education', '').split(',') if metadata.get('education') else []
                        }
            
            candidates = list(candidates_dict.values())
            
            # Sort by experience (descending)
            candidates.sort(key=lambda x: int(x.get('experience_years', 0) or 0), reverse=True)
            
            self.logger.info(f"Found {len(candidates)} candidates matching filters")
            
            filter_desc = ", ".join([f"{k}={v}" for k, v in filters_applied.items()])
            
            return {
                "success": True,
                "data": {
                    "candidates": candidates,
                    "count": len(candidates),
                    "filters_applied": filters_applied
                },
                "message": f"Found {len(candidates)} candidate(s) matching filters: {filter_desc}"
            }
            
        except Exception as e:
            self.logger.error(f"Error in filter_by_metadata: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error filtering resumes: {str(e)}"
            }