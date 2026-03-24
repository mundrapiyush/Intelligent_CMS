import ollama
import json
import re
from typing import Dict, Any, List, Optional
import logging
from tools import ToolRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentController:
    """Agent controller implementing ReAct (Reasoning + Acting) pattern"""
    
    def __init__(self, vector_store, model_name: str = "llama3.2:latest", max_iterations: int = 5):
        """
        Initialize agent controller
        
        Args:
            vector_store: VectorStore instance for database access
            model_name: Name of the Ollama model to use
            max_iterations: Maximum number of reasoning iterations
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.tool_registry = ToolRegistry(vector_store, model_name)
        logger.info(f"Initialized AgentController with model: {model_name}")
    
    def run(self, user_query: str, verbose: bool = False) -> Dict[str, Any]:
        logger.info(f"Agent processing query: {user_query}")
        
        conversation_history = []
        iteration = 0
        goal_achieved = False
        accumulated_info = []
        
        while iteration < self.max_iterations and not goal_achieved:
            if verbose:
                print(f"\n[Iteration {iteration + 1}]")
            
            # Generate thought
            thought = self._generate_thought(user_query, conversation_history, accumulated_info)
            
            if verbose:
                print(f"Thought: {thought}")
            
            # Check if agent thinks goal is achieved
            if "goal achieved" in thought.lower() or "task complete" in thought.lower():
                goal_achieved = True
                break
            
            # Select and parse action
            action = self._parse_action(thought)
            
            if not action or action.get('tool') == 'none':
                # No more actions needed
                goal_achieved = True
                break
            
            if verbose:
                print(f"Action: {action['tool']}({action.get('parameters', {})})")
            
            # Execute action
            observation = self._execute_action(action)
            
            if verbose:
                print(f"Observation: {observation.get('message', 'Action completed')}")
            
            # Store in history
            conversation_history.append({
                "iteration": iteration + 1,
                "thought": thought,
                "action": action,
                "observation": observation
            })
            
            # Accumulate information
            if observation.get('success'):
                accumulated_info.append(observation)
            
            iteration += 1
        
        # Generate final response
        final_response = self._generate_final_response(user_query, conversation_history, accumulated_info)
        
        logger.info(f"Agent completed in {iteration} iterations")
        
        return {
            "query": user_query,
            "iterations": iteration,
            "reasoning_trace": conversation_history,
            "response": final_response,
            "success": True
        }
    
    def _generate_thought(self, query: str, history: List, accumulated_info: List) -> str:
        # Build context from history
        history_text = ""
        if history:
            for entry in history[-3:]:  # Last 3 iterations for context
                history_text += f"\nPrevious Action: {entry['action']['tool']}"
                history_text += f"\nResult: {entry['observation'].get('message', 'Completed')}"
        
        # Build accumulated info summary
        info_summary = ""
        if accumulated_info:
            info_summary = "\nInformation gathered so far:\n"
            for i, info in enumerate(accumulated_info[-3:], 1):
                if info.get('data'):
                    data = info['data']
                    if 'count' in data:
                        info_summary += f"{i}. Found {data['count']} candidates\n"
                    if 'candidates' in data and data['candidates']:
                        info_summary += f"   Candidates: {', '.join([c.get('name', 'Unknown') for c in data['candidates'][:3]])}\n"
        
        # Get available tools
        tools_desc = self.tool_registry.get_tools_description()
        
        prompt = f"""You are an intelligent resume analysis agent. Analyze the user's query and decide what action to take next.

Available Tools:
{tools_desc}

User Query: {query}
{history_text}
{info_summary}

Think step by step:
1. What information do I need to answer this query?
2. What have I already gathered?
3. What should I do next?

If you have enough information to answer the query, say "Goal achieved" and explain what you found.
Otherwise, specify which tool to use next and why.

Your thought (be concise):"""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.3}
            )
            
            thought = response['response'].strip()
            return thought
            
        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            return "Error in reasoning. Will try to complete with available information."
    
    def _parse_action(self, thought: str) -> Optional[Dict[str, Any]]:
        """
        Parse action from thought
        
        Args:
            thought: Thought string from LLM
            
        Returns:
            Action dict with tool and parameters
        """
        # Check if goal achieved
        if "goal achieved" in thought.lower() or "enough information" in thought.lower():
            return {"tool": "none", "parameters": {}}
        
        # Try to extract tool name and parameters
        # Look for patterns like: search_resumes(query="Python developers")
        tool_pattern = r'(search_resumes|filter_by_metadata|summarize_resume)\s*\((.*?)\)'
        match = re.search(tool_pattern, thought, re.IGNORECASE)
        
        if match:
            tool_name = match.group(1).lower()
            params_str = match.group(2)
            
            # Parse parameters
            parameters = self._parse_parameters(params_str)
            
            return {
                "tool": tool_name,
                "parameters": parameters
            }
        
        # Fallback: try to infer tool from keywords
        thought_lower = thought.lower()
        
        if "search" in thought_lower or "find" in thought_lower:
            # Extract query from thought
            query = self._extract_query_from_thought(thought)
            return {
                "tool": "search_resumes",
                "parameters": {"query": query, "top_k": 3}
            }
        elif "filter" in thought_lower:
            # Try to extract filter criteria
            parameters = self._extract_filter_criteria(thought)
            return {
                "tool": "filter_by_metadata",
                "parameters": parameters
            }
        elif "summarize" in thought_lower or "details" in thought_lower:
            # Extract candidate name
            name = self._extract_candidate_name(thought)
            if name:
                return {
                    "tool": "summarize_resume",
                    "parameters": {"candidate_name": name}
                }
        
        return {"tool": "none", "parameters": {}}
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dict"""
        parameters = {}
        
        # Simple parameter parsing
        # Handle: query="value", top_k=3, skills=["Python", "AWS"]
        
        # String parameters
        string_matches = re.findall(r'(\w+)="([^"]*)"', params_str)
        for key, value in string_matches:
            parameters[key] = value
        
        # Integer parameters
        int_matches = re.findall(r'(\w+)=(\d+)', params_str)
        for key, value in int_matches:
            if key not in parameters:  # Don't override string params
                parameters[key] = int(value)
        
        # Array parameters
        array_matches = re.findall(r'(\w+)=\[(.*?)\]', params_str)
        for key, value in array_matches:
            items = [item.strip().strip('"\'') for item in value.split(',')]
            parameters[key] = items
        
        return parameters
    
    def _extract_query_from_thought(self, thought: str) -> str:
        """Extract search query from thought"""
        # Look for quoted strings
        match = re.search(r'"([^"]+)"', thought)
        if match:
            return match.group(1)
        
        # Fallback: use key phrases
        for phrase in ["find", "search for", "looking for"]:
            if phrase in thought.lower():
                parts = thought.lower().split(phrase)
                if len(parts) > 1:
                    query = parts[1].strip().split('.')[0].strip()
                    return query[:100]
        
        return "candidates"
    
    def _extract_filter_criteria(self, thought: str) -> Dict[str, Any]:
        """Extract filter criteria from thought"""
        criteria = {}
        
        # Extract skills
        skills_match = re.search(r'skills?[:\s]+\[([^\]]+)\]', thought, re.IGNORECASE)
        if skills_match:
            skills = [s.strip().strip('"\'') for s in skills_match.group(1).split(',')]
            criteria['skills'] = skills
        
        # Extract experience
        exp_match = re.search(r'(\d+)\+?\s*years?', thought, re.IGNORECASE)
        if exp_match:
            criteria['min_experience'] = int(exp_match.group(1))
        
        # Extract location
        location_match = re.search(r'(?:in|location|based in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', thought)
        if location_match:
            criteria['location'] = location_match.group(1)
        
        return criteria
    
    def _extract_candidate_name(self, thought: str) -> Optional[str]:
        """Extract candidate name from thought"""
        # Look for quoted names
        match = re.search(r'"([^"]+)"', thought)
        if match:
            return match.group(1)
        
        # Look for capitalized names (2-3 words)
        name_match = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', thought)
        if name_match:
            return name_match.group(1)
        
        return None
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute selected tool
        
        Args:
            action: Action dict with tool and parameters
            
        Returns:
            Observation dict with results
        """
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})
        
        if tool_name == "none":
            return {"success": True, "message": "No action needed"}
        
        tool = self.tool_registry.get_tool(tool_name)
        
        if not tool:
            logger.error(f"Tool '{tool_name}' not found")
            return {
                "success": False,
                "error": f"Tool {tool_name} not found",
                "message": f"Error: Tool '{tool_name}' does not exist"
            }
        
        try:
            result = tool.execute(**parameters)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error executing {tool_name}: {str(e)}"
            }
    
    def _generate_final_response(self, query: str, history: List, accumulated_info: List) -> str:
        """
        Generate final natural language response
        
        Args:
            query: Original user query
            history: Full conversation history
            accumulated_info: All gathered information
            
        Returns:
            Final response string
        """
        # Compile all observations
        observations_text = ""
        for entry in history:
            obs = entry['observation']
            if obs.get('success') and obs.get('data'):
                data = obs['data']
                
                if 'candidates' in data:
                    candidates = data['candidates']
                    if candidates:
                        observations_text += f"\nFound {len(candidates)} candidates:\n"
                        for candidate in candidates[:5]:  # Limit to top 5
                            observations_text += f"- {candidate.get('name', 'Unknown')}"
                            if candidate.get('current_role'):
                                observations_text += f" ({candidate['current_role']})"
                            if candidate.get('experience_years'):
                                observations_text += f" - {candidate['experience_years']} years exp"
                            observations_text += "\n"
                
                if 'summary' in data:
                    observations_text += f"\nCandidate Summary:\n{data['summary']}\n"
                    if data.get('highlights'):
                        observations_text += "Highlights:\n"
                        for highlight in data['highlights']:
                            observations_text += f"- {highlight}\n"
        
        if not observations_text:
            return "I couldn't find any relevant information to answer your query. Please try rephrasing or check if resumes are loaded in the database."
        
        # Generate response using LLM
        prompt = f"""Based on the information gathered, provide a clear and concise answer to the user's query.

User Query: {query}

Information Gathered:
{observations_text}

Generate a natural, helpful response that directly answers the user's question. Be specific and include relevant details."""

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.5}
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            # Fallback to simple response
            return f"Based on my analysis:\n{observations_text}"

# Made with Bob
