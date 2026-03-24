from typing import Dict, Any
from .base_tool import BaseTool
import logging

logger = logging.getLogger(__name__)


class SearchResumesTool(BaseTool):
    
    def __init__(self, vector_store):
        super().__init__()
        self.name = "search_resumes"
        self.description = "Search resumes using semantic similarity based on natural language query. Use this when you need to find candidates based on skills, experience, or job descriptions."
        self.parameters = {
            "query": {
                "type": "string",
                "description": "Natural language search query (e.g., 'Python developers', 'machine learning engineers')",
                "required": True
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 3)",
                "required": False,
                "default": 3
            }
        }
        self.vector_store = vector_store
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        # Extract parameters
        query = kwargs.get('query', '')
        top_k = kwargs.get('top_k', 3)
        
        try:
            # Validate input
            is_valid, error_msg = self.validate_input(**kwargs)
            if not is_valid:
                return {
                    "success": False,
                    "error": error_msg,
                    "message": f"Invalid input: {error_msg}"
                }
            
            self.logger.info(f"Searching resumes with query: '{query}', top_k={top_k}")
            
            # Perform vector search
            results = self.vector_store.query(query, n_results=top_k)
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return {
                    "success": True,
                    "data": {
                        "candidates": [],
                        "count": 0
                    },
                    "message": f"No resumes found matching query: '{query}'"
                }
            
            # Extract and format results
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [[]])[0]
            
            # Group by candidate (filename)
            candidates_dict = {}
            for doc, meta, dist in zip(documents, metadatas, distances):
                filename = meta.get('filename', 'Unknown')
                
                if filename not in candidates_dict:
                    candidates_dict[filename] = {
                        'filename': filename,
                        'name': meta.get('person_name', 'Unknown'),
                        'email': meta.get('email'),
                        'phone': meta.get('phone'),
                        'current_company': meta.get('current_company'),
                        'current_role': meta.get('current_role'),
                        'experience_years': meta.get('total_experience_years'),
                        'location': meta.get('location'),
                        'skills': meta.get('skills', '').split(',') if meta.get('skills') else [],
                        'companies': meta.get('companies', '').split(',') if meta.get('companies') else [],
                        'relevance_score': 1 - dist,  # Convert distance to similarity
                        'matching_content': []
                    }
                
                # Add matching content snippet
                candidates_dict[filename]['matching_content'].append(doc[:200] + "...")
            
            # Convert to list and sort by relevance
            candidates = list(candidates_dict.values())
            candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            self.logger.info(f"Found {len(candidates)} unique candidates")
            
            return {
                "success": True,
                "data": {
                    "candidates": candidates,
                    "count": len(candidates),
                    "query": query
                },
                "message": f"Found {len(candidates)} candidate(s) matching '{query}'"
            }
            
        except Exception as e:
            self.logger.error(f"Error in search_resumes: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error searching resumes: {str(e)}"
            }
