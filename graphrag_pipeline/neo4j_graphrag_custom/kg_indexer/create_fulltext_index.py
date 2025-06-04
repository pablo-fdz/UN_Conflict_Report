import neo4j
from neo4j_graphrag.indexes import create_fulltext_index, retrieve_fulltext_index_info
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

# Load configuration and setup
script_dir = os.path.dirname(os.path.abspath(__file__))  # Uncomment if running as a script

# Load environment variables from a .env file
dotenv_path = os.path.join(script_dir, '.env')
load_dotenv(dotenv_path, override=True)

# Neo4j connection
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = os.getenv('NEO4J_USERNAME')
neo4j_password = os.getenv('NEO4J_PASSWORD')
gemini_api_key = os.getenv('GEMINI_API_KEY')

index_name = "fulltext_index"

with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)) as driver:

    create_fulltext_index(
        driver=driver,
        name=index_name,  # Name of the index
        label="Chunk",  # Node label to index
        node_properties=["text"],  # Name of the node specified in "label" containing the full text
        fail_if_exists=False  # Set to True if you want to fail if the index already exists. Setting to False enables running the script multiple times (with additional nodes) without errors.
    )

    # Print success message
    print(f"Full text index '{index_name}' created successfully.")

    # Check if the index was created successfully
    first_index_info = retrieve_fulltext_index_info(
        driver=driver,
        index_name=index_name,  # Name of the index to retrieve information about
        label_or_type="Chunk", # Node label or relationship type to check for the index
        text_properties=["text"]  # Name of the property containing the full text
    )

    if first_index_info:
        print(f"Full text index '{index_name}' exists with the following details:")
        print(first_index_info)
    else:
        print(f"Full text index '{index_name}' does not exist or could not be retrieved. Index creation may have failed.")

    driver.close()