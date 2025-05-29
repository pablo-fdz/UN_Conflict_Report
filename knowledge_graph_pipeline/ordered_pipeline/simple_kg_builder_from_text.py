"""This example illustrates how to get started easily with the SimpleKGPipeline
and ingest text into a Neo4j Knowledge Graph.

This example assumes a Neo4j db is up and running. Update the credentials below
if needed.

NB: when building a KG from text, no 'Document' node is created in the Knowledge Graph.
"""

# Utilities
import asyncio
import logging
from dotenv import load_dotenv
import os
import json
import time
from google import genai
import polars as pl

# LLM imports
from neo4j_graphrag.llm import LLMInterface
from gemini_llm import GeminiLLM  # Custom Gemini LLM class

# Neo4j and Neo4j GraphRAG imports
import neo4j
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter

# Configure logging to see debug messages
# logging.basicConfig()  # Set up basic logging configuration
# logging.getLogger("neo4j_graphrag").setLevel(logging.DEBUG)  # Set the logging level to DEBUG for the neo4j_graphrag module (detailed diagnostic information)

script_dir = os.path.dirname(os.path.abspath(__file__))  # Path to the directory where this script is located

# Load environment variables from a .env file
dotenv_path = os.path.join(script_dir, '.env')  # Path to the .env file
load_dotenv(dotenv_path, override=True)

# Open configuration file from JSON format
config_path = os.path.join(script_dir, 'config.json')  # Path to the configuration file
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# Neo4j DB infos
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize the LLM client if GEMINI_API_KEY is provided
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    raise ValueError("GEMINI_API_KEY is not set. Please provide a valid API key.")

# Load the configuration for the creation of the knowledge graph
ENTITIES = config['schema_config']['node_types'] if config['schema_config']['create_schema'] == True else None
RELATIONS = config['schema_config']['relationship_types'] if config['schema_config']['create_schema'] == True else None
PATTERNS = config['schema_config']['patterns'] if config['schema_config']['suggest_pattern'] == True else None
PROMPT_TEMPLATE = config['prompt_template_config']['template'] if config['prompt_template_config']['use_default'] == False else None
FIXED_SIZE_SPLITTER = FixedSizeSplitter(
    chunk_size=config['text_splitter_config']['chunk_size'],  # Increase chunk size for faster processing (at the cost of accuracy, remember that LLMs read mostly the first and last tokens). Consider LLM context window size.
    chunk_overlap=config['text_splitter_config']['chunk_overlap']  # Overlap between chunks to ensure context is preserved (must be less than chunk_size)
)

# Load the data to be processed
df_path = os.path.join(script_dir, '../FILTERED_DATAFRAME.parquet')  # Path to the Parquet file containing the data
df = pl.read_parquet(df_path)
n = 100  # Number of rows to sample
df_subset = df.head(n)  # Create a subset of the dataframe

async def load_embedder_async() -> SentenceTransformerEmbeddings:
    """Load the embedding model asynchronously to avoid blocking."""
    loop = asyncio.get_event_loop()
    # Run the synchronous embedding model loading in a thread pool
    embedder = await loop.run_in_executor(
        None, 
        lambda: SentenceTransformerEmbeddings(model=config['embedder_config']['model_name'])
    )
    return embedder

async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    embedder: SentenceTransformerEmbeddings
) -> PipelineResult:
    
    # Create an instance of the SimpleKGPipeline
    kg_builder = SimpleKGPipeline(
        llm=llm,  # LLM instance
        driver=neo4j_driver,  # Neo4j driver instance
        embedder=embedder,  # Embedding model instance
        entities=ENTITIES,  # List of entity types (nodes)
        relations=RELATIONS,  # List of relation types (edges)
        potential_schema=PATTERNS,  # List of patterns for potential schema (associations between nodes and edges)
        enforce_schema=config['schema_config']['enforce_schema'],  # Whether to enforce the schema ("STRICT") or not ("NONE")
        prompt_template=PROMPT_TEMPLATE,  # Use a prompt template for the creation of the knowledge graph
        text_splitter=FIXED_SIZE_SPLITTER,  # Increase chunk size for faster processing (at the cost of accuracy, remember that LLMs read mostly the first and last tokens)
        from_pdf=False,
        kg_writer=Neo4jWriter(),  # Default Neo4j writer to write the knowledge graph to the database
        perform_entity_resolution=True  # Merge nodes with the same label
    )

    results = []
    row_counter = 0
    start_time = time.time()
    
    # Iterate over all of the rows in the polars dataframe
    for row in df_subset.iter_rows(named=True):
        # Print the row number
        row_counter += 1
        print(f"Processing row {row_counter} of {df_subset.shape[0]}")

        # Get the text from the row
        text = row['full_text']
        
        # Only process if text is not None or empty
        if text and text.strip():
            # Process the text with the pipeline
            result = await kg_builder.run_async(text=text)
            results.append(result)
            # Print the result
            print(f"Result: {result}")
        else:
            print(f"Skipping row {row_counter} due to empty text")
            results.append(None)

        # Update time tracking
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {(elapsed_time / row_counter) * (df_subset.shape[0] - row_counter):.2f} seconds\n")
    
    return results  # Runs the pipeline asynchronously and returns the result

async def main() -> PipelineResult:
    
    # Start loading embedder and LLM concurrently
    embedder_task = asyncio.create_task(load_embedder_async())

    # Synchronous initialization of the LLM
    if GEMINI_API_KEY:
        llm = GeminiLLM(  # Synchronous initialization of the LLM
            model_name=config['llm_config']['model_name'],
            google_api_key=GEMINI_API_KEY,
            model_params=config['llm_config']['model_params']
        )
    else:
        raise ValueError("GEMINI_API_KEY is not set. Please provide a valid API key.")

    # Wait for embedder to finish loading
    embedder = await embedder_task

    with neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        res = await define_and_run_pipeline(driver, llm, embedder)  # Pauses execution until the pipeline is complete

    return res

if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)