import argparse
import sys
from pathlib import Path
import logging

from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_system import RAGSystem
from metadata_extractor import MetadataExtractor
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_resumes():
    logger.info("Starting resume loading process...")
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Process resumes
    resumes = pdf_processor.process_resume_folder(config.RESUMES_DIR)
    
    if not resumes:
        logger.error(f"No resumes found in {config.RESUMES_DIR}")
        logger.info(f"Please add PDF resumes to the '{config.RESUMES_DIR}' folder")
        return None
    
    # Initialize metadata extractor
    logger.info("Extracting metadata from resumes...")
    metadata_extractor = MetadataExtractor(model_name=config.OLLAMA_MODEL)
    
    # Extract metadata for each resume
    for resume in resumes:
        metadata = metadata_extractor.extract_metadata(
            resume['content'],
            resume['filename']
        )
        resume['metadata'] = metadata
        logger.info(f"Extracted metadata for {resume['filename']}: {metadata.get('name', 'Unknown')}")
    
    # Initialize vector store
    vector_store = VectorStore(config.CHROMA_DB_DIR, config.COLLECTION_NAME)
    
    # Clear existing data (optional - comment out if you want to keep adding)
    logger.info("Clearing existing data from vector store...")
    vector_store.clear_collection()
    
    # Add documents to vector store with metadata
    logger.info("Adding resumes to vector store with metadata...")
    vector_store.add_documents(
        resumes,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    
    # Show stats
    stats = vector_store.get_collection_stats()
    logger.info(f"Vector store stats: {stats}")
    
    return vector_store

def add_single_resume(resume_path: str, replace: bool = False):
    logger.info(f"Processing resume: {resume_path}")
    
    # Validate file path
    pdf_path = Path(resume_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {resume_path}")
        return False
    
    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"File must be a PDF: {resume_path}")
        return False
    
    # Initialize components
    pdf_processor = PDFProcessor()
    metadata_extractor = MetadataExtractor(model_name=config.OLLAMA_MODEL)
    vector_store = VectorStore(config.CHROMA_DB_DIR, config.COLLECTION_NAME)
    
    # Extract text from PDF
    logger.info("Extracting text from PDF...")
    text = pdf_processor.extract_text_from_pdf(pdf_path)
    
    if not text:
        logger.error("Failed to extract text from PDF")
        return False
    
    logger.info(f"Extracted {len(text)} characters from PDF")
    
    # Extract metadata
    logger.info("Extracting metadata...")
    metadata = metadata_extractor.extract_metadata(text, pdf_path.name)
    
    print("\n" + "="*80)
    print("EXTRACTED METADATA")
    print("="*80)
    print(f"Name: {metadata.get('name', 'Unknown')}")
    print(f"Email: {metadata.get('email', 'N/A')}")
    print(f"Phone: {metadata.get('phone', 'N/A')}")
    print(f"Current Company: {metadata.get('current_company', 'N/A')}")
    print(f"Current Role: {metadata.get('current_role', 'N/A')}")
    print(f"Experience: {metadata.get('total_experience_years', 'N/A')} years")
    print(f"Location: {metadata.get('location', 'N/A')}")
    if metadata.get('skills'):
        print(f"Skills: {', '.join(metadata['skills'][:5])}...")
    print("="*80 + "\n")
    
    # Check for duplicates
    logger.info("Checking for duplicates...")
    duplicate_check = vector_store.check_duplicate(metadata)
    
    if duplicate_check['is_duplicate']:
        print("DUPLICATE DETECTED!")
        print(f"Found {duplicate_check['match_count']} existing chunks matching this resume")
        print(f"Existing files: {', '.join(duplicate_check['matches'])}")
        
        if replace:
            print("\nReplacing existing resume(s)...")
            # Remove old versions
            for old_file in duplicate_check['matches']:
                removed = vector_store.remove_by_filename(old_file)
                logger.info(f"Removed {removed} chunks from {old_file}")
        else:
            print("\nResume NOT added to database (duplicate exists)")
            print("Use --replace flag to replace the existing resume")
            return False
    else:
        print("No duplicate found - proceeding with addition")
    
    # Prepare document for vector store
    document = {
        'filename': pdf_path.name,
        'content': text,
        'path': str(pdf_path),
        'metadata': metadata
    }
    
    # Add to vector store
    logger.info("Adding resume to vector store...")
    vector_store.add_documents(
        [document],
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP
    )
    
    print("\nResume successfully added to database!")
    
    # Show updated stats
    stats = vector_store.get_collection_stats()
    print(f"Total documents in database: {stats['total_documents']}")
    
    return True

def query_resumes(vector_store):
    logger.info("Starting interactive query mode...")
    logger.info(f"Using Ollama model: {config.OLLAMA_MODEL}")
    
    # Initialize RAG system
    rag_system = RAGSystem(model_name=config.OLLAMA_MODEL)
    
    print("\n" + "="*80)
    print("RESUME RAG SYSTEM - Interactive Query Mode")
    print("="*80)
    print(f"Model: {config.OLLAMA_MODEL}")
    print(f"Documents in database: {vector_store.get_collection_stats()['total_documents']}")
    print("\nType your questions about the resumes. Type 'quit' or 'exit' to stop.")
    print("="*80 + "\n")
    
    while True:
        try:
            query = input("Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            # Query with RAG
            rag_system.chat(query, vector_store, n_results=config.TOP_K_RESULTS)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"\nError: {str(e)}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Resume RAG System - Load resumes and query them using AI"
    )
    parser.add_argument(
        '--load',
        action='store_true',
        help='Load resumes from the resumes folder into the vector database'
    )
    parser.add_argument(
        '--query',
        action='store_true',
        help='Start interactive query mode'
    )
    parser.add_argument(
        '--load-and-query',
        action='store_true',
        help='Load resumes and then start query mode'
    )
    parser.add_argument(
        '--add-resume',
        type=str,
        metavar='PATH',
        help='Add a single resume file to the database (checks for duplicates)'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace existing resume if duplicate is found (use with --add-resume)'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not (args.load or args.query or args.load_and_query or args.add_resume):
        parser.print_help()
        return
    
    try:
        vector_store = None
        
        # Add single resume
        if args.add_resume:
            success = add_single_resume(args.add_resume, replace=args.replace)
            if not success:
                sys.exit(1)
            return
        
        # Load resumes
        if args.load or args.load_and_query:
            vector_store = load_resumes()
            if vector_store is None:
                return
        
        # Query mode
        if args.query or args.load_and_query:
            if vector_store is None:
                # Initialize vector store for query-only mode
                vector_store = VectorStore(config.CHROMA_DB_DIR, config.COLLECTION_NAME)
                
                # Check if there are documents
                stats = vector_store.get_collection_stats()
                if stats['total_documents'] == 0:
                    logger.error("No documents in vector store. Please run with --load first.")
                    return
            
            query_resumes(vector_store)
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
