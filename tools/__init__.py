from .base_tool import BaseTool
from .search_tool import SearchResumesTool
from .filter_tool import FilterByMetadataTool
from .summarize_tool import SummarizeResumeTool
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    
    def __init__(self, vector_store, llm_model: str):
        self.vector_store = vector_store
        self.llm_model = llm_model
        self.tools: Dict[str, BaseTool] = {}
        self._initialize_tools()
        logger.info(f"Initialized ToolRegistry with {len(self.tools)} tools")
    
    def _initialize_tools(self):
        self.tools = {
            "search_resumes": SearchResumesTool(self.vector_store),
            "filter_by_metadata": FilterByMetadataTool(self.vector_store),
            "summarize_resume": SummarizeResumeTool(self.vector_store, self.llm_model)
        }
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        return self.tools
    
    def get_tools_schema(self) -> list:
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_tools_description(self) -> str:
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.format_for_llm())
        
        return "\n".join(descriptions)
    
    def list_tool_names(self) -> list:
        return list(self.tools.keys())


__all__ = [
    'BaseTool',
    'SearchResumesTool',
    'FilterByMetadataTool',
    'SummarizeResumeTool',
    'ToolRegistry'
]