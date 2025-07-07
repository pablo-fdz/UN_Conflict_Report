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
if str(graphrag_pipeline_dir) not in sys.path:
    sys.path.append(str(graphrag_pipeline_dir))

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import HybridRetriever

async def main(country: str = None, output_directory: str = None):

    """
    Main function to run GraphRAG with HybridRetriever for a specific country.
    
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
    retriever_type = 'Hybrid'

    # Load KG retrieval configurations
    config_files_path = os.path.join(graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
    with open(os.path.join(config_files_path, 'kg_retrieval_config.json'), 'r') as f:
        retrieval_config = json.load(f)

    # Initialize the pipeline
    graphrag_pipeline = GraphRAGConstructionPipeline()

    # Create embedder
    embedder = SentenceTransformerEmbeddings(
        model=graphrag_pipeline.build_config['embedder_config']['model_name']
    )
    
    # Connect to Neo4j and create retriever
    with neo4j.GraphDatabase.driver(graphrag_pipeline.neo4j_uri,auth=(graphrag_pipeline.neo4j_username, graphrag_pipeline.neo4j_password)) as driver:
        
        # Get index names
        embeddings_index_name, fulltext_index_name = graphrag_pipeline._get_indexes(driver)

        # Create HybridRetriever
        retriever = HybridRetriever(
            driver=driver,
            vector_index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
            fulltext_index_name=fulltext_index_name,  # Name of the fulltext index that will be used for text search
            embedder=embedder,  # Embedder to use for embedding the query text when doing a vector search
            return_properties=retrieval_config[f'{retriever_type}Retriever'].get('return_properties', ['text'])  # Properties to return from the vector search results, apart from the similarity scores (cosine similarity scores by default). Returns the 'text' property by default, which is the text of the document in the knowledge graph.
        )
        
        # Run GraphRAG with this retriever
        answer, context  = await graphrag_pipeline.run_async(
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