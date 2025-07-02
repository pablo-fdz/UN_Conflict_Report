import os
import sys
import asyncio
from .graphrag_construction_pipeline import GraphRAGConstructionPipeline
import neo4j
import json
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

from library.kg_builder.utilities import GeminiLLM
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.schema import get_schema

async def main(country: str = None, output_directory: str = None):

    """
    Main function to run GraphRAG with Text2CypherRetriever for a specific country.
    
    Args:
        country (str): Country name to generate report for
        output_directory (str): Directory to save the report
    """

    # Get arguments from environment variables if not provided directly
    if country is None:
        country = os.getenv('GRAPHRAG_COUNTRY')
    if output_directory is None:
        output_directory = os.getenv('GRAPHRAG_OUTPUT_DIR')

    # Set retriever type
    retriever_type = 'Text2Cypher'

    # Load KG retrieval configurations
    config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
    with open(os.path.join(config_files_path, 'kg_retrieval_config.json'), 'r') as f:
        retrieval_config = json.load(f)

    # Initialize the pipeline
    graphrag_pipeline = GraphRAGConstructionPipeline()

    # Create LLM instance
    llm = GeminiLLM(
            model_name=retrieval_config[f'{retriever_type}Retriever']['llm_config']['model_name'],
            google_api_key=graphrag_pipeline.gemini_api_key,
            model_params=retrieval_config[f'{retriever_type}Retriever']['llm_config']['model_params']
        )
    
    # Define whether to use examples to generate the Cypher queries
    use_examples = retrieval_config[f'{retriever_type}Retriever']['examples_config'].get('include_examples', False)
    if use_examples == True:
        examples = retrieval_config[f'{retriever_type}Retriever']['examples_config'].get('examples', [])
    else:
        examples = None

    # Connect to Neo4j and create retriever
    with neo4j.GraphDatabase.driver(graphrag_pipeline.neo4j_uri,auth=(graphrag_pipeline.neo4j_username, graphrag_pipeline.neo4j_password)) as driver:
        
        # Get the schema of the Neo4j database
        schema = get_schema(
            driver=driver,
            is_enhanced=True,  # Whether to use the enhanced schema with additional information (e.g., include examples)
            sanitize=False
        )

        # Create Retriever instance
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,  # LLM to use for generating Cypher queries from the query text
            neo4j_schema=schema,  # Schema of the knowledge graph to use for generating Cypher queries
            examples=examples,  # Examples to use for generating Cypher queries
            custom_prompt=None  # Use the auto-generated prompt by th LLM for generating Cypher queries
        )
        
        # Run GraphRAG with this retriever
        answer, context = await graphrag_pipeline.run_async(
            retriever=retriever,
            retriever_search_params=retrieval_config[f'{retriever_type}Retriever'].get('search_params', None),  # Search parameters for the retriever (if not provided, default parameters will be used)
            country=country,  # Country to generate report for
            output_directory=output_directory  # Directory where the report will be saved (for including ACLED forecasts into prompt), can be None to use default output directory
        )
        
        # Save to markdown - if no output_directory provided, will use default structure
        filepath, context_filepath = graphrag_pipeline.save_report_to_markdown(
            answer=answer,
            context=context,  # Context used for generating the answer, can be None if not provided
            output_directory=output_directory,  # Can be None, in which case the report will be saved to the default output directory
            filename=None,  # If None, the filename will be generated based on the country and retriever type
            country=country,
            retriever_type=retriever_type,
            metadata={
                "search_params": retrieval_config[f'{retriever_type}Retriever'].get('search_params', {}),
                "graphrag_model": graphrag_pipeline.graphrag_config['llm_config']['model_name']
            }
        )
        print(f"Report saved to: {filepath}")
        
        print(f"{retriever_type}Retriever GraphRAG Answer for {country}:\n{answer}")
        return answer

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    result = asyncio.run(main(country=None, output_directory=None))  # Set to None for getting variables from environment variables