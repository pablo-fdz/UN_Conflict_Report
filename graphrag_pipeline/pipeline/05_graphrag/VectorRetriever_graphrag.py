import os
import sys
import asyncio
from .graphrag_construction_pipeline import GraphRAGConstructionPipeline
import neo4j
import json

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where this script is located
graphrag_pipeline_dir = os.path.dirname(os.path.dirname(script_dir))  # Get the parent directory (graphrag_pipeline)
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorRetriever

async def main(country: str = None, output_directory: str = None):

    """
    Main function to run GraphRAG with VectorRetriever for a specific country.
    
    Args:
        country (str): Country name to generate report for
        output_directory (str): Directory to save the report
    """

    # Set retriever type
    retriever_type = 'Vector'

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
        embeddings_index_name, _ = graphrag_pipeline._get_indexes(driver)

        # Create Retriever
        retriever = VectorRetriever(
            driver=driver,
            index_name=embeddings_index_name,  # Name of the vector index that will be used for similarity search with the embedded query text
            embedder=embedder,  # Embedder to use for embedding the query text when doing a vector search
            return_properties=retrieval_config[f'{retriever_type}Retriever'].get('return_properties', ['text'])  # Properties to return from the vector search results, apart from the similarity scores (cosine similarity scores by default). Returns the 'text' property by default, which is the text of the document in the knowledge graph.
        )
        
        # Run GraphRAG with this retriever
        answer = await graphrag_pipeline.run_async(
            retriever=retriever,
            retriever_search_params=retrieval_config[f'{retriever_type}Retriever'].get('search_params', None),  # Search parameters for the retriever (if not provided, default parameters will be used)
            country=country  # Country to generate report for
        )
        
        # Save to markdown - if no output_directory provided, will use default structure
        filepath = graphrag_pipeline.save_report_to_markdown(
            answer=answer,
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
    import sys
    country = sys.argv[1] if len(sys.argv) > 1 else None  # Get country from command line argument, if provided
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None  # Get output directory from command line argument, if provided
    result = asyncio.run(main(country=country, output_directory=output_dir))