"""In this script, we set up a single pipeline with two Neo4j writers:
1. One for creating the lexical graph (Document and Chunks)
2. And another for creating the entity graph (entities and relations derived from the text).

Script based on the example from the Neo4j GraphRAG repository:
text_to_lexical_graph_to_entity_graph_single_pipeline.py
"""

# Utilities
import asyncio
from __future__ import annotations
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
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import (
    SchemaEnforcementMode
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, OpenAILLM


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
    embedder: Embedder,
    lexical_graph_config: LexicalGraphConfig,
    row_data,
) -> PipelineResult:
    """Define and run the pipeline with the following components (that are executed in order):

    1. Text Splitter: to split the text into manageable chunks of fixed size
    2. Chunk Embedder: to embed the chunks' text

    3. Lexical Graph Builder: to build the lexical graph, i.e. creating the chunk nodes and relationships between them
    4. LG KG writer: save the lexical graph to Neo4j

    5. Schema Builder: this component takes a list of entities, relationships and
        possible triplets as inputs, validate them and return a schema ready to use
        for the rest of the pipeline
    6. LLM Entity Relation Extractor is an LLM-based entity and relation extractor:
        based on the provided schema, the LLM will do its best to identity these
        entities and their relations within the provided text
    7. EG KG writer: once entities and relations are extracted, they can be writen
        to a Neo4j database

    8. Resolver: SpaCySemanticMatchResolver used to merge nodes with same label
        and similar textual properties (by default using the "name" property) based on spaCy
        embeddings and cosine similarities of embedding vectors.

    """

    pipe = Pipeline()  # Initialize the pipeline

    # 1. Define the components

    pipe.add_component(
        FixedSizeSplitter(
            chunk_size=config['text_splitter_config']['chunk_size'],  # Increase chunk size for faster processing (at the cost of accuracy, remember that LLMs read mostly the first and last tokens). Consider LLM context window size.
            chunk_overlap=config['text_splitter_config']['chunk_overlap'],  # Overlap between chunks to ensure context is preserved (must be less than chunk_size)
            approximate=True  # Avoid splitting words in the middle of chunk boundaries
        ),
        "splitter"  # Name of the component in the pipeline
    )

    pipe.add_component(
        TextChunkEmbedder(embedder=embedder), 
        "chunk_embedder"
    )

    pipe.add_component(
        LexicalGraphBuilder(lexical_graph_config),
        "lexical_graph_builder"
    )

    pipe.add_component(
        Neo4jWriter(neo4j_driver), 
        "lg_writer"  # Lexical Graph Writer to save the lexical graph to Neo4j
    )

    pipe.add_component(
        SchemaBuilder(),  # To enable the creation of a schema based on the provided entities, relationships and patterns 
        "schema"
    )


    enforce_schema_mapping = {
        "STRICT": SchemaEnforcementMode.STRICT,
        "NONE": SchemaEnforcementMode.NONE,
    }
    enforce_schema_param = enforce_schema_mapping.get(config['schema_config']['enforce_schema'])
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,  # LLM instance to extract entities and relations
            prompt_template=PROMPT_TEMPLATE,  # Use a prompt template for the creation of the knowledge graph
            create_lexical_graph=False,  # Do not create a lexical graph from the text chunks here, we already have one (more controlled in the pipeline)
            enforce_schema=enforce_schema_param,  # Whether to enforce the schema ("STRICT") or not ("NONE")
            on_error="RAISE",  # Raise an error if the extraction fails
            max_concurrency=5  # Maximum number of concurrent requests to the LLM
        ),
        "extractor"
    )

    pipe.add_component(
        Neo4jWriter(neo4j_driver), 
        "eg_writer"  # Entity Graph Writer to save the entity graph to Neo4j
    )

    pipe.add_component(
        SpaCySemanticMatchResolver(  # Merge nodes with same label and similar textual properties
            driver,
            filter_query="WHERE (entity)-[:FROM_CHUNK]->(:Chunk)-[:FROM_DOCUMENT]->(doc:"
            "Document {id = 'docId'}",
            resolve_properties=["name"],  # Properties to use for resolution (default is "name")
            similarity_threshold=0.8,  # The similarity threshold above which nodes are merged (default is 0.8)
            spacy_model="en_core_web_sm"  # spaCy model to use for resolution (default is "en_core_web_lg")
            ), 
        "resolver"
    )

    # 2. Define the execution order of component and how the output of previous 
    # components must be used

    # Steps 1 and 2: Split the text into chunks and embed them

    pipe.connect(
        start_component_name="splitter",  # Name of the starting component
        end_component_name="chunk_embedder",  # Name of the next component to connect to
        input_config={"text_chunks": "splitter"}  # Pass "splitter" output as "text_chunks" input to "chunk_embedder". E.g., TextChunkEmbedder.run() requires a `text_chunks` parameter.
    )

    # Steps 3 and 4: Build the lexical graph from the text chunks and save it to Neo4j

    pipe.connect(
        start_component_name="chunk_embedder",
        end_component_name="lexical_graph_builder",
        input_config={"text_chunks": "chunk_embedder"},  # Pass the output of "chunk_embedder" as input to "lexical_graph_builder", through the parameter `text_chunks`
    )

    pipe.connect(
        start_component_name="lexical_graph_builder",
        end_component_name="lg_writer",
        input_config={
            "graph": "lexical_graph_builder.graph",
            "lexical_graph_config": "lexical_graph_builder.config",
        },
    )

    # Connect outcome of step 2 to steps 5-7: extract entities and relations from the text chunks and save them to Neo4j

    pipe.connect(
        start_component_name="chunk_embedder", 
        end_component_name="extractor", 
        input_config={"chunks": "chunk_embedder"}
    )

    pipe.connect(
        start_component_name="schema", 
        end_component_name="extractor", 
        input_config={"schema": "schema"}
    )

    pipe.connect(
        start_component_name="extractor",
        end_component_name="eg_writer",
        input_config={"graph": "extractor"},
    )

    # Connect the lexical graph writer to the entity graph writer
    # This ensures that the lexical graph is created before the entity graph

    pipe.connect(
        start_component_name="lg_writer", 
        end_component_name="eg_writer", 
        input_config={}  # This was empty in the original code, but you can specify any additional input configuration if needed
    )

    # Finally, connect the entity graph writer to the resolver
    pipe.connect(
        start_component_name="eg_writer",
        end_component_name="resolver",
        input_config={}
    )

    # 3. Define the inputs to the pipeline

    pipe_inputs = {
        "splitter": {
            "text": row_data['full_text']  # Assuming 'full_text' is the column containing the text to process
        },
        "lexical_graph_builder": {
            "document_info": {
                # 'path' can be anything
                "path": "example/lexical_graph_from_text.py"
            },
        },
        "schema": {
            "node_types": [
                NodeType(
                    label="Person",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                        PropertyType(name="place_of_birth", type="STRING"),
                        PropertyType(name="date_of_birth", type="DATE"),
                    ],
                ),
                NodeType(
                    label="Organization",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                        PropertyType(name="country", type="STRING"),
                    ],
                ),
                NodeType(
                    label="Field",
                    properties=[
                        PropertyType(name="name", type="STRING"),
                    ],
                ),
            ],
            "relationship_types": [
                RelationshipType(
                    label="WORKED_ON",
                ),
                RelationshipType(
                    label="WORKED_FOR",
                ),
            ],
            "patterns": [
                ("Person", "WORKED_ON", "Field"),
                ("Person", "WORKED_FOR", "Organization"),
            ],
        },
        "extractor": {
            "lexical_graph_config": lexical_graph_config,
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


async def main(driver: neo4j.Driver) -> PipelineResult:
    # optional: define some custom node labels for the lexical graph:
    lexical_graph_config = LexicalGraphConfig()  # Use default configuration for the lexical graph
    text = """Albert Einstein was a German physicist born in 1879 who
            wrote many groundbreaking papers especially about general relativity
            and quantum mechanics. He worked for many different institutions, including
            the University of Bern in Switzerland and the University of Oxford."""
    llm = OpenAILLM(
        model_name="gpt-4o",
        model_params={
            "max_tokens": 1000,
            "response_format": {"type": "json_object"},
        },
    )
    res = await define_and_run_pipeline(
        driver,
        llm,
        lexical_graph_config,
        text,
    )
    await llm.async_client.close()
    return res


if __name__ == "__main__":
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password")
    ) as driver:
        print(asyncio.run(main(driver)))
