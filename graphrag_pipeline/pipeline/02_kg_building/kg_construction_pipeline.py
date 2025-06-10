import sys
import os
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

# Utilities
from dotenv import load_dotenv
import os
import json
from google import genai
import polars as pl
from library.kg_builder import CustomKGPipeline, build_kg_from_df
from library.kg_builder.utilities import GeminiLLM, ensure_spacy_model
from neo4j_graphrag.experimental.components.resolver import (
    SpaCySemanticMatchResolver, FuzzyMatchResolver, SinglePropertyExactMatchResolver
)

# Neo4j and Neo4j GraphRAG imports
import neo4j
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings

class KGConstructionPipeline:
    """Main coordinator for the Knowledge Graph building pipeline."""
    
    def __init__(self):
    
        # Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
        # modules in parent directory)
        script_dir = Path(__file__).parent  # Get the directory where this script is located
        graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
        self.config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
        self._load_configs()
        self._setup_credentials()
        
    def _load_configs(self):
        """Load all configuration files."""

        try:
            # Load environment variables from .env file
            load_dotenv(os.path.join(self.config_files_path, '.env'), override=True)
            
            # Load data and KG building configurations
            with open(os.path.join(self.config_files_path, 'kg_building_config.json'), 'r') as f:
                self.build_config = json.load(f)
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {e.filename}. Please ensure the file exists in the config_files directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in configuration file: {e.msg}. Please check the file format.")
    
    def _setup_credentials(self):
        """Setup database and API credentials."""
        required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'GEMINI_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        self.neo4j_uri = os.getenv('NEO4J_URI')
        self.neo4j_username = os.getenv('NEO4J_USERNAME')
        self.neo4j_password = os.getenv('NEO4J_PASSWORD')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is not set. Please provide a valid API key.")
    
    def _create_resolver(self, driver: neo4j.Driver):
        """Create entity resolver based on configuration."""

        entity_resolution_config = self.build_config['entity_resolution_config']
        
        # Initialize entity resolver if entity resolution is enabled
        if entity_resolution_config['use_resolver'] == True:
            
            accepted_resolvers = [
                'SinglePropertyExactMatchResolver',
                'FuzzyMatchResolver',
                'SpaCySemanticMatchResolver'
            ]

            resolver_type = entity_resolution_config['resolver']
            
            if resolver_type == 'SinglePropertyExactMatchResolver':
                config = entity_resolution_config['SinglePropertyExactMatchResolver_config']
                return SinglePropertyExactMatchResolver(  # Merge nodes with same label and exact property
                    driver,
                    filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                    resolve_property=config['resolve_property'],  # Property to use for resolution (default is "name")
                    neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
                )
            elif resolver_type == 'FuzzyMatchResolver':
                config = entity_resolution_config['FuzzyMatchResolver_config']
                return FuzzyMatchResolver(  # Merge nodes with same label and similar textual properties
                    driver,
                    filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                    resolve_properties=config['resolve_properties'],  # Properties to use for resolution (default is "name")
                    similarity_threshold=config['similarity_threshold'],
                    neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
                )
            elif resolver_type == 'SpaCySemanticMatchResolver':
                config = entity_resolution_config['SpaCySemanticMatchResolver_config']
                ensure_spacy_model(config['spacy_model'])  # Ensure the spaCy model is installed, install if necessary
                return SpaCySemanticMatchResolver(  # Merge nodes with same label and similar textual properties using SpaCy embeddings
                    driver,
                    filter_query=config['filter_query'],  # Cypher query used to reduce the resolution scope to a specific document. If None, the resolver will consider all nodes in the database.
                    resolve_properties=config['resolve_properties'],  # Properties to use for resolution (default is "name")
                    similarity_threshold=config['similarity_threshold'],  # The similarity threshold above which nodes are merged (default is 0.8). Higher threshold will result in less false positives, but may miss some matches. 
                    spacy_model=config['spacy_model'],  # spaCy model to use for resolution (default is "en_core_web_lg")
                    neo4j_database='neo4j'  # Neo4j database to use for resolution (default is "neo4j")
                )
            else:
                raise ValueError(f"Unknown entity resolution resolver: {resolver_type}. Please set one of the following in the configuration files of the KG building: {accepted_resolvers}")

        else:
            # If entity resolution is not enabled, use a dummy resolver that does nothing
            return None

    def _create_kg_pipeline(self, driver: neo4j.Driver) -> CustomKGPipeline:
        """Create the main KG pipeline with all components."""
        # Initialize LLM
        llm = GeminiLLM(
            model_name=self.build_config['llm_config']['model_name'],
            google_api_key=self.gemini_api_key,
            model_params=self.build_config['llm_config']['model_params']
        )
        
        # Initialize embedder
        embedder = SentenceTransformerEmbeddings(
            model=self.build_config['embedder_config']['model_name']
        )
        
        # Create resolver
        resolver = self._create_resolver(driver)
        
        # Initialize the KG pipeline
        return CustomKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            schema_config=self.build_config['schema_config'],
            prompt_template=self.build_config['prompt_template_config']['template']
                if not self.build_config['prompt_template_config'].get('use_default', True) else None,  # Use custom template if specified, otherwise use default
            text_splitter_config=self.build_config['text_splitter_config'],
            resolver=resolver,
            examples_config=self.build_config['examples_config'],
            on_error=self.build_config.get('dev_settings', {}).get('on_error', 'IGNORE'),  # Default to 'IGNORE' if not specified (or invalid value)
            batch_size=self.build_config.get('dev_settings', {}).get('batch_size', 1000),  # Neo4j batch size for writing documents
            max_concurrency=self.build_config.get('dev_settings', {}).get('max_concurrency', 5)  # Maximum number of concurrent LLM requests
        )
    
    async def run_async(
            self, 
            df: pl.DataFrame,
            document_base_field: str = 'id',
            text_column: str = 'text',
            document_metadata_mapping: dict = None,
            document_id_column: str = None
        ):
        
        """
        Run the complete KG building pipeline for the input data frame.

        Args:
            df (pl.DataFrame): The input data frame containing documents to process.
            document_base_field (str): The field from which to create the document node in the lexical graph.
            text_column (str): The column containing the text to process.
            document_metadata_mapping (dict): Optional mapping of document property names to data frame columns.
            document_id_column (str): Optional column to use as document ID. A random UUID will be generated if not provided.
        
        Returns:
        """
        
        try:
            with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_username, self.neo4j_password)) as driver:
                
                # Create the KG pipeline
                kg_pipeline = self._create_kg_pipeline(driver)
                
                # Process the dataframe
                results = await build_kg_from_df(
                    kg_pipeline=kg_pipeline,
                    df=df,
                    document_base_field=document_base_field,
                    text_column=text_column,
                    document_metadata_mapping=document_metadata_mapping,
                    document_id_column=document_id_column  # Use default document ID generation
                )
        except Exception as e:
            raise RuntimeError(f"Error during KG construction pipeline execution: {e}")
        
        return results