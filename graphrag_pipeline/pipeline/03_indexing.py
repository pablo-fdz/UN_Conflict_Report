# Utilities
import sys
import os
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent  # Get the graphrag_pipeline directory
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

import asyncio
from dotenv import load_dotenv
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from library.kg_indexer import KGIndexer

# Neo4j and Neo4j GraphRAG imports
import neo4j

async def main():
    """Main function to run the indexing pipeline."""
    
    # ==================== 1. Setup ====================
    
    config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
    
    try:
        # Load environment variables from .env file
        load_dotenv(os.path.join(config_files_path, '.env'), override=True)
        
        # Load KG building configurations
        with open(os.path.join(config_files_path, 'kg_building_config.json'), 'r') as f:
            build_config = json.load(f)
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e.filename}. Please ensure the file exists in the config_files directory.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in configuration file: {e.msg}. Please check the file format.")
    
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    # Get the dimensions from the SentenceTransformer model for vector indexing
    try:
        model = SentenceTransformer(f'sentence-transformers/{build_config['embedder_config']['model_name']}')  # Load the model
        embedding_dim = model.get_sentence_embedding_dimension()  # Get the embedding dimension dynamically (only if using SentenceTransformer models!)
    except Exception as e:
        print(f"Error loading model: {e}. Try using a SentenceTransformer model.")

    vector_index_name = "embeddings_index"
    fulltext_index_name = "fulltext_index"

    # ==================== 2. Create indexer instance ====================
    
    with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
        
        indexer = KGIndexer(driver=driver)  # Initialize the KGIndexer
        
        # ==================== 3. Create vector and text indexes ====================
    
        print(f"Creating vector indexes...")
        try:
            indexer.create_vector_index(
                index_name=vector_index_name,  # Name of the index
                label="Chunk",  # Node label to index
                embedding_property="embedding",  # Name of the node specified in "label" containing the embeddings
                dimensions=embedding_dim,  # Dimensions of the embeddings, dynamically set from the model
            )
            print(f"Successfully created index for the embeddings.")
        except Exception as e:
            print(f"Error creating the vector embeddings: {e}")
            raise
        
        # Show the first vector index of the database
        print(f"Showing vector index information for the first vector index in the database...")
        indexer.retrieve_vector_index_info(
            index_name=vector_index_name,  # Name of the index to retrieve information about
            label_or_type="Chunk",  # Node label or relationship type to check for the index
            embedding_property="embedding"  # Name of the property containing the embeddings
        )

        print(f"Creating text indexes...")
        try:
            indexer.create_fulltext_index(
                index_name=fulltext_index_name,  # Name of the index
                label="Chunk",  # Node label to index
                node_properties=["text"]  # Name of the node specified in "label" containing the full text
            )
            print(f"Successfully created index for the text.")
        except Exception as e:
            print(f"Error creating the fulltext index: {e}")
            raise
        
        # Show the first fulltext index of the database
        print(f"Showing fulltext index information for the first fulltext index in the database...")
        indexer.retrieve_fulltext_index_info(
            index_name=fulltext_index_name,  # Name of the index to retrieve information about
            label_or_type="Chunk",  # Node label or relationship type to check for the index
            text_properties=["text"]  # Name of the property containing the full text
        )

    print(f"Successfully created all indexes in the Neo4j database.")

if __name__ == "__main__":
    # Execute main function when script is run directly
    try:
        result = asyncio.run(main())
        print(f"Indexing completed.")
    except Exception as e:
        print(f"Error during indexing: {e}")
        sys.exit(1)