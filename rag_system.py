import ollama
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize RAG system
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        logger.info(f"Initialized RAG system with model: {model_name}")
    
    def create_context(self, query_results: Dict) -> str:

        if not query_results['documents'] or not query_results['documents'][0]:
            return "No relevant information found."
        
        context_parts = []
        documents = query_results['documents'][0]
        metadatas = query_results['metadatas'][0]
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            # Add metadata header
            header = f"--- Resume: {metadata.get('filename', 'Unknown')} ---"
            if metadata.get('person_name'):
                header += f"\nCandidate: {metadata['person_name']}"
            if metadata.get('current_company'):
                header += f"\nCurrent Company: {metadata['current_company']}"
            if metadata.get('current_role'):
                header += f"\nCurrent Role: {metadata['current_role']}"
            
            context_parts.append(header)
            context_parts.append(doc)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""You are an AI assistant helping to analyze resumes. Use the following context from resumes to answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        try:
            logger.info(f"Generating response for query: {query}")
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            
            answer = response['response']
            logger.info("Response generated successfully")
            return answer
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}"
    
    def query_with_rag(self, query: str, vector_store, n_results: int = 3, where: Dict = None) -> Dict:
        # Retrieve relevant documents with optional filtering
        query_results = vector_store.query(query, n_results=n_results, where=where)
        
        # Create context
        context = self.create_context(query_results)
        
        # Generate response
        response = self.generate_response(query, context)
        
        return {
            "query": query,
            "context": context,
            "response": response,
            "sources": [meta.get('filename', 'Unknown') for meta in query_results['metadatas'][0]],
            "metadata": query_results['metadatas'][0]
        }
    
    def chat(self, query: str, vector_store, n_results: int = 3, where: Dict = None):
        result = self.query_with_rag(query, vector_store, n_results, where)
        
        print("\n" + "="*80)
        print(f"QUERY: {result['query']}")
        print("="*80)
        
        # Show unique sources with metadata
        sources_info = []
        seen_files = set()
        for meta in result['metadata']:
            filename = meta.get('filename', 'Unknown')
            if filename not in seen_files:
                seen_files.add(filename)
                info = filename
                if meta.get('person_name'):
                    info += f" ({meta['person_name']})"
                sources_info.append(info)
        
        print(f"\nSOURCES: {', '.join(sources_info)}")
        print("\n" + "-"*80)
        print("RESPONSE:")
        print("-"*80)
        print(result['response'])
        print("="*80 + "\n")
