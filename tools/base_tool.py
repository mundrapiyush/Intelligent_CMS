from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTool(ABC):
    
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.parameters: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters
        
        Returns:
            Dict containing:
                - success: bool
                - data: Any (tool-specific result)
                - message: str (human-readable message)
                - error: str (if success=False)
        """
        pass
    
    def validate_input(self, **kwargs) -> Tuple[bool, str]:
        for param_name, param_info in self.parameters.items():
            if param_info.get('required', False) and param_name not in kwargs:
                return False, f"Missing required parameter: {param_name}"
            
            if param_name in kwargs:
                value = kwargs[param_name]
                expected_type = param_info.get('type')
                
                # Type validation
                if expected_type == 'string' and not isinstance(value, str):
                    return False, f"Parameter {param_name} must be a string"
                elif expected_type == 'integer' and not isinstance(value, int):
                    return False, f"Parameter {param_name} must be an integer"
                elif expected_type == 'array' and not isinstance(value, list):
                    return False, f"Parameter {param_name} must be a list"
        
        return True, ""
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    def format_for_llm(self) -> str:
        params_str = []
        for param_name, param_info in self.parameters.items():
            required = " (required)" if param_info.get('required', False) else " (optional)"
            params_str.append(f"  - {param_name}: {param_info.get('description', '')}{required}")
        
        return f"""
Tool: {self.name}
Description: {self.description}
Parameters:
{chr(10).join(params_str) if params_str else '  None'}
"""