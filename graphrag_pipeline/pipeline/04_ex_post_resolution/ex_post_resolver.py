import sys
import os
import asyncio
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if str(graphrag_pipeline_dir) not in sys.path:
    sys.path.append(str(graphrag_pipeline_dir))

# Utilities
from dotenv import load_dotenv
import os
import json
from library.kg_builder.utilities import ensure_spacy_model
from neo4j_graphrag.experimental.components.resolver import (
    SpaCySemanticMatchResolver, FuzzyMatchResolver, SinglePropertyExactMatchResolver
)

# Neo4j and Neo4j GraphRAG imports
import neo4j


async def main():
    """Main function to run entity resolution on the knowledge graph."""
    
    # ==================== 1. Setup ====================
    
    config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
    
    try:
        # Load environment variables from .env file
        load_dotenv(os.path.join(config_files_path, '.env'), override=True)
        
        # Load data and KG building configurations
        with open(os.path.join(config_files_path, 'data_ingestion_config.json'), 'r') as f:
            data_config = json.load(f)
        with open(os.path.join(config_files_path, 'kg_building_config.json'), 'r') as f:
            build_config = json.load(f)
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {e.filename}. Please ensure the file exists in the config_files directory.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON in configuration file: {e.msg}. Please check the file format.")
    
    neo4j_uri = os.getenv('NEO4J_URI')
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    # ==================== 2. Create resolver instance ====================
    
    entity_resolution_config = build_config['entity_resolution_config']
        
    accepted_resolvers = [
        'SinglePropertyExactMatchResolver',
        'FuzzyMatchResolver',
        'SpaCySemanticMatchResolver'
    ]
    
    resolver_type = entity_resolution_config['ex_post_resolver']
    
    with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:
    
        if resolver_type == 'SinglePropertyExactMatchResolver':
            config = entity_resolution_config['SinglePropertyExactMatchResolver_config']
            resolver = SinglePropertyExactMatchResolver(  # Merge nodes with same label and exact property
                driver,
                filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                resolve_property=config['resolve_property'],  # Property to use for resolution (default is "name")
                neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
            )
        elif resolver_type == 'FuzzyMatchResolver':
            config = entity_resolution_config['FuzzyMatchResolver_config']
            resolver = FuzzyMatchResolver(  # Merge nodes with same label and similar textual properties
                driver,
                filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                resolve_properties=config['resolve_properties'],  # Properties to use for resolution (default is "name")
                similarity_threshold=config['similarity_threshold'],
                neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
            )
        elif resolver_type == 'SpaCySemanticMatchResolver':
            config = entity_resolution_config['SpaCySemanticMatchResolver_config']
            ensure_spacy_model(config['spacy_model'])  # Ensure the spaCy model is installed, install if necessary
            resolver = SpaCySemanticMatchResolver(  # Merge nodes with same label and similar textual properties using SpaCy embeddings
                driver,
                filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                resolve_properties=config['resolve_properties'],  # Properties to use for resolution (default is "name")
                similarity_threshold=config['similarity_threshold'],  # The similarity threshold above which nodes are merged (default is 0.8). Higher threshold will result in less false positives, but may miss some matches. 
                spacy_model=config['spacy_model'],  # spaCy model to use for resolution (default is "en_core_web_lg")
                neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
            )
        else:
            raise ValueError(f"Unknown entity resolution resolver: {resolver_type}. Please set one of the following in the configuration files of the KG building: {accepted_resolvers}")
        
        # ==================== 3. Run the resolver ====================
    
        print(f"Running {resolver_type}...")
        try:
            result = await resolver.run()  # Execute the resolver
            print(f"Successfully ran {resolver_type}.")
            return result
        except Exception as e:
            print(f"Error running {resolver_type}: {e}")
            raise

if __name__ == "__main__":
    # Execute main function when script is run directly
    try:
        result = asyncio.run(main())
        print(f"Entity resolution completed with result: {result}")
    except Exception as e:
        print(f"Error during entity resolution: {e}")
        sys.exit(1)