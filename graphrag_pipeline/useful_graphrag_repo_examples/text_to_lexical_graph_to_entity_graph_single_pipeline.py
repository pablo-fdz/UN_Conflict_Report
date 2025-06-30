"""In this example, we set up a single pipeline with two Neo4j writers:
one for creating the lexical graph (Document and Chunks)
and another for creating the entity graph (entities and relations derived from the text).
"""

from __future__ import annotations

import asyncio
import os
import sys
from dotenv import load_dotenv

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaRelation,
    SchemaProperty
)
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult
from neo4j_graphrag.llm import LLMInterface, OpenAILLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kg_builder.utilities import GeminiLLM
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
import neo4j

script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(script_dir, '.env'), override=True)
embedder = SentenceTransformerEmbeddings(model='all-MiniLM-L6-v2')


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver,
    llm: LLMInterface,
    lexical_graph_config: LexicalGraphConfig,
    text: str,
) -> PipelineResult:
    """Define and run the pipeline with the following components:

    - Text Splitter: to split the text into manageable chunks of fixed size
    - Chunk Embedder: to embed the chunks' text
    - Lexical Graph Builder: to build the lexical graph, ie creating the chunk nodes and relationships between them
    - LG KG writer: save the lexical graph to Neo4j

    - Schema Builder: this component takes a list of entities, relationships and
        possible triplets as inputs, validate them and return a schema ready to use
        for the rest of the pipeline
    - LLM Entity Relation Extractor is an LLM-based entity and relation extractor:
        based on the provided schema, the LLM will do its best to identity these
        entities and their relations within the provided text
    - EG KG writer: once entities and relations are extracted, they can be writen
        to a Neo4j database

    """
    pipe = Pipeline()
    # define the components
    pipe.add_component(
        FixedSizeSplitter(chunk_size=200, chunk_overlap=50, approximate=False),
        "splitter",
    )
    pipe.add_component(TextChunkEmbedder(embedder=embedder), "chunk_embedder")
    pipe.add_component(
        LexicalGraphBuilder(lexical_graph_config),
        "lexical_graph_builder",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "lg_writer")
    pipe.add_component(SchemaBuilder(), "schema")
    pipe.add_component(
        LLMEntityRelationExtractor(
            llm=llm,
            create_lexical_graph=False,
        ),
        "extractor",
    )
    pipe.add_component(Neo4jWriter(neo4j_driver), "eg_writer")
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect("splitter", "chunk_embedder", input_config={"text_chunks": "splitter"})
    pipe.connect(
        "chunk_embedder",
        "lexical_graph_builder",
        input_config={"text_chunks": "chunk_embedder"},
    )
    pipe.connect(
        "lexical_graph_builder",
        "lg_writer",
        input_config={
            "graph": "lexical_graph_builder.graph",
            "lexical_graph_config": "lexical_graph_builder.config",
        },
    )
    # define the execution order of component
    # and how the output of previous components must be used
    pipe.connect(
        "chunk_embedder", "extractor", input_config={"chunks": "chunk_embedder"}
    )
    pipe.connect("schema", "extractor", input_config={"schema": "schema"})
    pipe.connect(
        "extractor",
        "eg_writer",
        input_config={"graph": "extractor"},
    )
    # make sure the lexical graph is created before creating the entity graph:
    pipe.connect("lg_writer", "eg_writer", {})
    # user input:
    # the initial text
    # and the list of entities and relations we are looking for
    pipe_inputs = {
        "splitter": {
            "text": text,
        },
        "lexical_graph_builder": {
            "document_info": {
                # 'path' can be anything
                "path": "example/lexical_graph_from_text.py"
            },
        },
        "schema": {
            "entities": [
                SchemaEntity(
                    label="Person",
                    description="An individual human being.",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="place_of_birth", type="STRING"),
                        SchemaProperty(name="date_of_birth", type="DATE"),
                    ],
                ),
                SchemaEntity(
                    label="Organization",
                    description="A structured group of people with a common purpose.",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                        SchemaProperty(name="country", type="STRING"),
                    ],
                ),
                SchemaEntity(
                    label="Field",
                    properties=[
                        SchemaProperty(name="name", type="STRING"),
                    ],
                ),
            ],
            "relations": [
                SchemaRelation(
                    label="EMPLOYED_BY",
                    description="Indicates employment relationship.",
                ),
                SchemaRelation(
                    label="WORKED_ON",
                ),
            ],
            "potential_schema": [
                ("Person", "WORKED_ON", "Field"),
                ("Person", "EMPLOYED_BY", "Organization"),
            ],
        },
        "extractor": {
            "lexical_graph_config": lexical_graph_config,
        },
    }
    # run the pipeline
    return await pipe.run(pipe_inputs)


async def main(driver: neo4j.Driver, gemini_api_key) -> PipelineResult:

    # optional: define some custom node labels for the lexical graph:
    lexical_graph_config = LexicalGraphConfig(
        chunk_node_label="TextPart",
        document_node_label="Text",
    )
    text = """Albert Einstein was a German physicist born in 1879 who
            wrote many groundbreaking papers especially about general relativity
            and quantum mechanics. He worked for many different institutions, including
            the University of Bern in Switzerland and the University of Oxford."""
    llm = GeminiLLM(
        model_name="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        model_params={'temperature': 0.0}
    )
    res = await define_and_run_pipeline(
        driver,
        llm,
        lexical_graph_config,
        text,
    )
    return res


if __name__ == "__main__":
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    with neo4j.GraphDatabase.driver(
        neo4j_uri, auth=(neo4j_username, neo4j_password)
    ) as driver:
        print(asyncio.run(main(driver, gemini_api_key)))
