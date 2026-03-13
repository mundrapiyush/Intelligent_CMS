import chromadb
from chromadb.config import Settings
from typing import List, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    
    def __init__(self, db_path: Path, collection_name: str):

        self.db_path = db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Resume embeddings collection"}
        )
        
        logger.info(f"Initialized ChromaDB at {db_path}")
        logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} documents")
    
    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks

    def add_documents(self, documents: List[Dict[str, str]], chunk_size: int = 1000, overlap: int = 200):
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        doc_count = 0
        for doc in documents:
            chunks = self._chunk_text(doc['content'], chunk_size, overlap)
            
            # Get extracted metadata if available
            extracted_metadata = doc.get('metadata', {})
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['filename']}_chunk_{i}"
                all_ids.append(chunk_id)
                all_texts.append(chunk)
                
                # Combine basic metadata with extracted metadata
                chunk_metadata = {
                    "filename": doc['filename'],
                    "path": doc['path'],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "person_name": extracted_metadata.get('name'),
                    "email": extracted_metadata.get('email'),
                    "phone": extracted_metadata.get('phone'),
                    "current_company": extracted_metadata.get('current_company'),
                    "current_role": extracted_metadata.get('current_role'),
                    "total_experience_years": extracted_metadata.get('total_experience_years'),
                    "location": extracted_metadata.get('location'),
                    "companies": ','.join(extracted_metadata.get('companies', [])) if extracted_metadata.get('companies') else None,
                    "skills": ','.join(extracted_metadata.get('skills', [])[:20]) if extracted_metadata.get('skills') else None,  # Limit to 20 skills
                    "education": ','.join(extracted_metadata.get('education', [])) if extracted_metadata.get('education') else None,
                }
                
                # Remove None values to keep metadata clean
                chunk_metadata = {k: v for k, v in chunk_metadata.items() if v is not None}
                
                all_metadatas.append(chunk_metadata)
            
            doc_count += 1
        
        if all_texts:
            self.collection.add(
                documents=all_texts,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"Added {len(all_texts)} chunks from {doc_count} documents to vector store")
    
    def query(self, query_text: str, n_results: int = 3, where: Dict = None) -> Dict:
        query_params = {
            "query_texts": [query_text],
            "n_results": n_results
        }
        
        if where:
            query_params["where"] = where
            logger.info(f"Querying with filter: {where}")
        
        results = self.collection.query(**query_params)
        
        logger.info(f"Retrieved {len(results['documents'][0])} results for query")
        return results
    
    def check_duplicate(self, metadata: Dict) -> Dict:
        matches = []
        
        # Check by person name (primary identifier)
        if metadata.get('name'):
            try:
                results = self.collection.get(
                    where={"person_name": metadata['name']},
                    limit=10
                )
                
                if results['ids']:
                    logger.info(f"Found {len(results['ids'])} existing chunks for {metadata['name']}")
                    matches.extend(results['metadatas'])
            except Exception as e:
                logger.warning(f"Error checking by name: {e}")
        
        # Check by email if name check didn't find anything
        if not matches and metadata.get('email'):
            try:
                results = self.collection.get(
                    where={"email": metadata['email']},
                    limit=10
                )
                
                if results['ids']:
                    logger.info(f"Found {len(results['ids'])} existing chunks for email {metadata['email']}")
                    matches.extend(results['metadatas'])
            except Exception as e:
                logger.warning(f"Error checking by email: {e}")
        
        # Check by phone if previous checks didn't find anything
        if not matches and metadata.get('phone'):
            try:
                results = self.collection.get(
                    where={"phone": metadata['phone']},
                    limit=10
                )
                
                if results['ids']:
                    logger.info(f"Found {len(results['ids'])} existing chunks for phone {metadata['phone']}")
                    matches.extend(results['metadatas'])
            except Exception as e:
                logger.warning(f"Error checking by phone: {e}")
        
        is_duplicate = len(matches) > 0
        
        # Get unique filenames from matches
        unique_files = set()
        if matches:
            for match in matches:
                if match.get('filename'):
                    unique_files.add(match['filename'])
        
        return {
            'is_duplicate': is_duplicate,
            'matches': list(unique_files),
            'match_count': len(matches)
        }
    
    def remove_by_filename(self, filename: str) -> int:
        try:
            # Get all IDs for this filename
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} chunks for {filename}")
                return len(results['ids'])
            else:
                logger.info(f"No chunks found for {filename}")
                return 0
        except Exception as e:
            logger.error(f"Error removing {filename}: {e}")
            return 0
        return results
    
    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Resume embeddings collection"}
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
    
    def get_collection_stats(self) -> Dict:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name
        }
