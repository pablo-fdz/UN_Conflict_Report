"""Custom Knowledge Graph Pipeline implementation that creates both lexical
and entity graphs from text data with document metadata support.

Compared to the SimpleKGPipeline, this pipeline allows for (but is not limited to):
1. Creating document nodes with metadata from dataframe columns
2. Entity resolution using SpaCy semantic matching
3. Full customization of pipeline components
"""

# Utilities
from typing import Dict, List, Tuple, Any, Optional, Union
from .utilities import (
    build_schema_from_config  # Function to build schema from JSON config file
)

# LLM imports
from neo4j_graphrag.llm import LLMInterface

# Neo4j and Neo4j GraphRAG imports
import neo4j
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.resolver import EntityResolver, BasePropertySimilarityResolver
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.types import (SchemaEnforcementMode, LexicalGraphConfig)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult


class CustomKGPipeline:
    """A custom Knowledge Graph pipeline that creates both lexical and entity graphs.
    
    This pipeline improves on SimpleKGPipeline by:
    1. Supporting document metadata from dataframe columns
    2. Using SpaCy semantic matching for entity resolution
    3. Allowing full customization of pipeline components and connections
    """
    
    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        schema_config: Optional[Dict[str, Any]] = None,
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        text_splitter_config: Optional[Dict[str, Any]] = None,
        resolver: Union[EntityResolver, BasePropertySimilarityResolver] = None,
        examples_config: Optional[Dict[str, Any]] = None,
        on_error: str = "IGNORE",
        batch_size: int = 1000,
        max_concurrency: int = 5
    ):
        
        """Initialize the CustomKGPipeline with the necessary components. Mimics
        the functionality of SimpleKGPipeline but allows for more customization.
        
        Args:
            llm: LLM instance for entity and relation extraction
            driver: Neo4j driver instance
            embedder: Embedding model instance
            schema_config: Configuration for the schema (entities, relations, patterns). 
                          If provided and 'create_schema' is True, will use schema-based extraction.
                          If None or 'create_schema' is False, will extract without schema.
            prompt_template: Custom prompt template for entity extraction
            text_splitter_config: Optional, text splitter configuration dictionary. Defaults to FixedSizeSplitter with chunk size 100000 and overlap 1000.
            resolver: Optional, entity resolver instance for resolving entities in the graph. Can be an instance of EntityResolver (like SinglePropertyExactMatchResolver) or BasePropertySimilarityResolver (like SpaCySemanticMatchResolver). If None, no entity resolution will be performed.
            examples_config: Configuration for examples. If provided and 'pass_examples' is True, will use few-shot learning with provided examples.
                           If None or 'pass_examples' is False, no examples will be used.
            on_error: Error handling strategy for entity extraction ("IGNORE" or "RAISE")
            batch_size: Batch size for writing to Neo4j. Defaults to 1000.
            max_concurrency: Maximum number of concurrent LLM requests. Defaults to 5.
        """
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.schema_config = schema_config
        self.examples_config = examples_config

        # Set up default text splitter parameters if not provided
        if text_splitter_config is None:
            self.chunk_size = 10**5
            self.chunk_overlap = 10**3
        else:
            self.chunk_size = text_splitter_config.get('chunk_size', 10**5)
            self.chunk_overlap = text_splitter_config.get('chunk_overlap', 10**3)
        
        # Determine if we should use schema-based extraction
        self.from_schema = (
            schema_config is not None and 
            isinstance(schema_config, dict) and 
            schema_config.get("create_schema", False)
        )
        
        self.prompt_template = prompt_template
        self.resolver = resolver

        # Determine if we should use examples for few-shot learning
        self.pass_examples = (
            examples_config is not None and 
            isinstance(examples_config, dict) and 
            examples_config.get("pass_examples", False)
        )

        self.on_error = on_error
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        
        # Process schema configuration if using schema
        if self.from_schema:
            self.entities, self.relations, self.triplets = build_schema_from_config(self.schema_config)
            
            # Validate that schema was actually created
            if self.entities is None and self.relations is None:
                raise ValueError(
                    "Schema creation failed. Check that your schema_config contains valid 'nodes' and 'edges' definitions."
                )
        else:
            self.entities, self.relations, self.triplets = None, None, None
        
        # Set up default lexical graph config
        self.lexical_graph_config = LexicalGraphConfig()
        
        if self.from_schema:  # If the creation from schema is set to True 
            # Process schema enforcement mode from the configuration file for the schema
            enforce_schema_mapping = {
                "STRICT": SchemaEnforcementMode.STRICT,  # Enforce strict schema validation
                "NONE": SchemaEnforcementMode.NONE,  # Do not enforce schema validation
            }
            self.enforce_schema = enforce_schema_mapping.get(
                self.schema_config.get('enforce_schema', 'NONE'),  # Default to NONE if not specified
                SchemaEnforcementMode.NONE  # Fallback to NONE if invalid value
            )
        else:
            self.enforce_schema = SchemaEnforcementMode.NONE
    
        # Process examples configuration
        if self.pass_examples:  # If the creation of examples is set to True
            examples = self.examples_config.get("examples", None)
            if examples is None:
                raise ValueError("Few-shot learning was considered, but there was no key 'examples' with a list of examples.")
            else:
                try:
                    self.examples = str(examples)  # Convert examples to string format
                except Exception as e:
                    raise ValueError(f"Could not convert examples to string. Received: {type(examples)} with value {examples}. Error: {e}")
        else:  # If the creation of examples is set to False, return an empty string
            self.examples = ""

    async def run_async(
        self, 
        text: str, 
        document_base_field: str,
        document_metadata: Optional[Dict[str, Any]] = None, 
        document_id: Optional[Any] = None
    ) -> PipelineResult:
        """Process a single text through the pipeline.
        
        Args:
            text: The text to process
            document_base_field: Field from which to create the document node in the lexical graph
            document_metadata: Optional metadata for the document node
            document_id: Optional unique ID for the document
            
        Returns:
            PipelineResult: The result of running the pipeline
        """
        # Create pipeline and add components
        pipe = self._create_pipeline()
        
        # Define inputs for the pipeline that are not included when making the 
        # pipeline connections, but that are needed to execute the pipeline 
        # (.run() method of every component). Note that some steps of the pipeline 
        # (like TextChunkEmbedder) do not require any more inputs.
        pipe_inputs = {
            "splitter": {  # Define additional inputs for the splitter component
                "text": text  # Where to find the text to process
            },
            "lexical_graph_builder": {  # Define additional inputs for the lexical graph builder component. Document node is created from the method `create_document_node` of the LexicalGraphBuilder class.
                "document_info": {  # DocumentInfo(DataModel) can take 3 parameters: `path` (used to create the document), `metadata` (additional metadata that will be included as a property of the document node) and `uid` (unique identifier for the document, if not provided a unique UUID will be created).
                    "path": document_base_field,  # 'path' can be anything from which we want to create a document from
                    "metadata": document_metadata,  # Optional metadata associated with the document that we want to include as properties of the document node
                    "uid": document_id  # Optional unique identifier for the document, if not provided a unique UUID will be created automatically
                },
            }
        }
        
        # Only add schema inputs if we're using schema
        if self.from_schema:
            pipe_inputs["schema"] = {  # Define additional inputs for the schema builder component
                "entities": self.entities,  # List of entities from which to create the schema
                "relations": self.relations,  # List of relations from which to create the schema
                "potential_schema": self.triplets  # List of triplets to create the schema from
            }

        # Complete the pipe inputs with the extractor component inputs
        pipe_inputs["extractor"] = {  # Define additional inputs for the entity relation extractor component
                "examples": self.examples,  # Examples for few-shot learning in the prompt
                "lexical_graph_config": self.lexical_graph_config,  # Lexical graph configuration. Oddly enough, this must be included even if we are not creating a lexical graph in the extractor component, but this is needed to link the lexical graph created in the LexicalGraphBuilder component to the entity graph created in the LLMEntityRelationExtractor component.
            }

        # Run the pipeline asynchronously (.run() method of the Pipeline class)
        # already handles the execution of all components in the pipeline in
        # an asynchronous manner, so we don't need to worry about that.
        return await pipe.run(pipe_inputs)
    
    def _create_pipeline(self) -> Pipeline:
        """Create and configure the pipeline with all components.
        
        Returns:
            Pipeline: The configured pipeline
        """
        pipe = Pipeline()  # Create a new pipeline instance
        
        # Add all components to the pipeline
        pipe.add_component(
            FixedSizeSplitter(
                chunk_size=self.chunk_size,  # Increase chunk size for faster processing (at the cost of accuracy, remember that LLMs read mostly the first and last tokens). Consider LLM context window size.
                chunk_overlap=self.chunk_overlap,  # Overlap between chunks to ensure context is preserved (must be less than chunk_size)
                approximate=True  # Avoid splitting words in the middle of chunk boundaries
            ),
            "splitter"  # Name of the component in the pipeline
        )

        pipe.add_component(
            TextChunkEmbedder(embedder=self.embedder), 
            "chunk_embedder"
        )

        pipe.add_component(
            LexicalGraphBuilder(self.lexical_graph_config), 
            "lexical_graph_builder"
        )

        pipe.add_component(
            Neo4jWriter(
                self.driver,  # Neo4j driver instance
                batch_size=self.batch_size  # Batch size for writing nodes and relationships to Neo4j (default is 1000, can be adjusted based on performance needs)
            ), 
            "lg_writer"
        )

        if self.from_schema:  # Add schema builder only if `from_schema` is True
            pipe.add_component(
                SchemaBuilder(),  # To enable the creation of a schema based on the provided entities, relationships and patterns 
                "schema"
            )
        else:  # If `from_schema` is False, omit the schema component
            pass

        pipe.add_component(
            LLMEntityRelationExtractor(
                llm=self.llm,  # LLM instance to extract entities and relations
                prompt_template=self.prompt_template,  # Use a prompt template for the creation of the knowledge graph
                create_lexical_graph=False,  # Do not create a lexical graph from the text chunks here, we already have one (more controlled in the pipeline)
                enforce_schema=self.enforce_schema,  # Whether to enforce the schema ("STRICT") or not ("NONE")
                on_error=self.on_error,  # Whether to ignore errors ("IGNORE") or raise them ("RAISE") during entity extraction
                max_concurrency=self.max_concurrency  # Maximum number of concurrent requests to the LLM
            ),
            "extractor"
        )

        pipe.add_component(
            Neo4jWriter(
                self.driver, 
                batch_size=self.batch_size
            ), 
            "eg_writer"
        )
        
        # Add entity resolution if resolver has been provided
        if self.resolver is not None:
            if not isinstance(self.resolver, (EntityResolver, BasePropertySimilarityResolver)):
                # Ensure the resolver is an instance of EntityResolver or its subclasses
                raise TypeError("Resolver must be an instance of EntityResolver or BasePropertySimilarityResolver")
            pipe.add_component(
                self.resolver,  # Custom resolver instance
                "resolver"
            )

        # Configure connections between components
        self._configure_pipeline_connections(pipe)
        
        return pipe
    
    def _configure_pipeline_connections(self, pipe: Pipeline) -> None:
        """Configure the connections between components in the pipeline.
        
        Args:
            pipe: The pipeline to configure
        """
        # Connect text processing components
        pipe.connect(
            start_component_name="splitter",  # Name of the starting component
            end_component_name="chunk_embedder",  # Name of the next component to connect to
            input_config={"text_chunks": "splitter"}  # Pass "splitter" output as "text_chunks" input to "chunk_embedder". E.g., TextChunkEmbedder.run() requires a `text_chunks` parameter.
        )
        
        # Connect lexical graph components
        pipe.connect(
            start_component_name="chunk_embedder",  # Output of the "chunk_embedder" component (when executing .run() method): "The input text chunks with each one having an added embedding."
            end_component_name="lexical_graph_builder",
            input_config={"text_chunks": "chunk_embedder"}  # Pass the output of "chunk_embedder" as input to "lexical_graph_builder", through the parameter `text_chunks`
        )
        
        pipe.connect(
            start_component_name="lexical_graph_builder",  # Output of the LexicalGraphBuilder.run() method: "GraphResult" containing the created lexical graph, with the Graph object itself and its configuration.
            end_component_name="lg_writer",
            input_config={  # Neo4jWriter.run() requires a `graph` parameter, which is the output of LexicalGraphBuilder.run(), and a `lexical_graph_config` parameter, which is the configuration for the lexical graph.
                "graph": "lexical_graph_builder.graph",
                "lexical_graph_config": "lexical_graph_builder.config",
            }
        )
        
        # Connect entity extraction components
        pipe.connect(
            start_component_name="chunk_embedder",  # Output of the "chunk_embedder" component (when executing .run() method): "The input text chunks with each one having an added embedding."
            end_component_name="extractor", 
            input_config={"chunks": "chunk_embedder"}  # Pass the output of "chunk_embedder" as input to "extractor", through the parameter `chunks`
        )
        
        if self.from_schema:  # If `from_schema` is True, connect the schema builder with the extractor
            pipe.connect(
                start_component_name="schema",  # Output of the SchemaBuilder.run() method: "SchemaConfig" containing the created schema, with the list of node types, relationship types and patterns.
                end_component_name="extractor", 
                input_config={"schema": "schema"}  # Pass the output of "schema" as input to "extractor", through the parameter `schema`
            )
        else:  # If `from_schema` is False, do not connect the schema builder with the extractor
            pass
        
        # Connect entity graph writer
        pipe.connect(
            start_component_name="extractor",  # Output of the LLMEntityRelationExtractor.run() method: "Neo4jGraph" containing the created entity graph, with the Graph object itself.
            end_component_name="eg_writer",
            input_config={"graph": "extractor"}  # Pass the output of "extractor" as input to "eg_writer", through the parameter `graph`
        )
        
        # Ensure lexical graph is created before entity graph
        pipe.connect(
            start_component_name="lg_writer", 
            end_component_name="eg_writer", 
            input_config={}  # This was empty in the example code: there is no specific input configuration needed for the Neo4jWriter component (all inputs are provided by the previous components)
        )
        
        # Connect entity resolution if provided
        if self.resolver is not None:
            pipe.connect(
                start_component_name="eg_writer",
                end_component_name="resolver",  
                input_config={}  # .run() method of the SpaCySemanticMatchResolver component requires no specific input configuration, as it will use the Neo4j database to resolve entities based on the provided filter_query and resolve_properties.
            )