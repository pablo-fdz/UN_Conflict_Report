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
import re
from datetime import datetime
from dotenv import load_dotenv
import json
from neo4j_graphrag.retrievers.base import Retriever
from library.kg_indexer import KGIndexer
from library.kg_builder.utilities import GeminiLLM, get_rate_limit_checker
from neo4j_graphrag.generation import RagTemplate
from library.graphrag import CustomGraphRAG

# Neo4j and Neo4j GraphRAG imports
import neo4j

class GraphRAGConstructionPipeline:
    """Main coordinator for the GraphRAG pipeline."""
    
    def __init__(self):
    
        # Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
        # modules in parent directory)
        self.script_dir = Path(__file__).parent  # Get the directory where this script is located
        self.graphrag_pipeline_dir = self.script_dir.parent.parent  # Get the graphrag_pipeline directory
        self.config_files_path = os.path.join(self.graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
        self._load_configs()
        self._setup_credentials()
        self._set_llm_rate_limit()
        self.llm_usage = 0  # Track LLM usage for rate limiting

    def _load_configs(self):
        """Load all configuration files."""

        try:
            # Load environment variables from .env file
            load_dotenv(os.path.join(self.config_files_path, '.env'), override=True)
            
            # Load KG building configurations
            with open(os.path.join(self.config_files_path, 'kg_building_config.json'), 'r') as f:
                self.build_config = json.load(f)
            # Load GraphRAG configurations
            with open(os.path.join(self.config_files_path, 'graphrag_config.json'), 'r') as f:
                self.graphrag_config = json.load(f)
        
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
    
    def _set_llm_rate_limit(self):
        
        # Get rate limit from config, default to 20
        rpm = self.graphrag_config['llm_config'].get('max_requests_per_minute', 20)
        # Subtract 20% for safety
        safe_rpm = round(rpm - rpm * 0.2)
        print(f"Applying a global rate limit of LLM requests of {safe_rpm} requests per minute.")
        self.check_rate_limit = get_rate_limit_checker(safe_rpm)

    def _get_indexes(self, driver: neo4j.Driver):
        """Get the vector and fulltext indexes in the database."""

        # Initialize the KG indexer
        indexer = KGIndexer(driver=driver)

        try:
            existing_indexes = indexer.list_all_indexes()
            self.embeddings_index_name = [index['name'] for index in existing_indexes if index['type'] == 'VECTOR'][0]
            self.fulltext_index_name = [index['name'] for index in existing_indexes if index['type'] == 'FULLTEXT'][0]
        except IndexError:
            raise ValueError("No vector and/or fulltext indexes found in the database. Please create the necessary indexes before running the GraphRAG pipeline.")

        return self.embeddings_index_name, self.fulltext_index_name

    def _create_graphrag_pipeline(self, retriever: Retriever):
        """Create the main GraphRAG pipeline with all components."""

        # Initialize LLM with GraphRAG configurations
        llm = GeminiLLM(
            model_name=self.graphrag_config['llm_config']['model_name'],
            google_api_key=self.gemini_api_key,
            model_params=self.graphrag_config['llm_config']['model_params']
        )
        
        # Create RAGTemplate using configuration files
        rag_template = RagTemplate(
            template=self.graphrag_config['rag_template_config'].get('template', None),  # Use custom template if specified, otherwise use default
            expected_inputs=['query_text', 'context', 'examples'],  # Define expected inputs for the template
            system_instructions=self.graphrag_config['rag_template_config'].get('system_instructions', None),  # Use custom system instructions if specified, otherwise use default
        )

        graphrag = CustomGraphRAG(
            llm=llm,  # LLM for generating answers
            retriever=retriever,  # Retriever for fetching relevant context 
            prompt_template=rag_template  # RAG template for formatting the prompt
        )

        return graphrag
    
    async def run_async(
            self, 
            retriever: Retriever,
            retriever_search_params: dict[str, any] = None,
            country: str = None
        ):
        
        """
        Run the complete GraphRAG construction pipeline asynchronously.

        Args:
            retriever (Retriever): The retriever used to find relevant context to pass to the LLM.
            country (str): The country for which the report is generated. Defaults to None, which uses an empty string in the query text.
            retriever_search_params (dict[str, any]): Configuration for the search parameters of the input retriever. Defaults to None, which uses the default search parameters.
        
        Returns:
            str: The generated answer from the GraphRAG pipeline.
        """
        
        try:
                
            # Get the initialized GraphRAG pipeline
            graphrag = self._create_graphrag_pipeline(retriever)

            # Format the search text for the retriever (i.e., the text that will be used to search the knowledge graph)
            formatted_search_text = self.graphrag_config.get('search_text', '').format(country=country)  # Use the country in the search text if specified, otherwise use an empty string

            # Format the query text for generating the report with the input country
            formatted_query_text = self.graphrag_config.get('query_text', '').format(country=country)  # Use the country in the query text if specified, otherwise use an empty string
            
            # Check rate limit before LLM call
            self.check_rate_limit()

            # Generate the answer using the GraphRAG pipeline
            graphrag_results = graphrag.search(
                search_text=formatted_search_text,  # Search query for the retriever (i.e., the text that will be used to search the knowledge graph)
                query_text=formatted_query_text,  # User question that is used to search the knowledge graph (i.e., vector search and fulltext search is made based on this question); defaults to empty string if not provided
                message_history=None,  # Optional message history for conversational context (omitted for now)
                examples=self.graphrag_config.get('examples', ''),  # Optional examples to guide the LLM's response (defaults to empty string)
                retriever_config=retriever_search_params,  # Configuration for the search parameters of the input retriever
                return_context=self.graphrag_config.get('return_context', True),   # Whether to return the context used for generating the answer (defaults to True). Can be obtained with graphrag_results['retriever_result']
                structured_output=False  # Whether to return the output in a structured format (in this case, set to False since we want a markdown report - not a structured JSON output)
            )
            
            self.llm_usage += 1  # Increment the LLM usage counter

            # Get the generated answer from the GraphRAG results (the string)
            generated_answer = graphrag_results.answer

            if self.graphrag_config.get('return_context', True):  # If return_context is True, we can also access the context used for generating the answer
                retrieved_context = graphrag_results.retriever_result  # For now, not used, but can be useful for debugging or further processing

        except Exception as e:
            raise RuntimeError(f"Error during GraphRAG construction pipeline execution: {e}")
        
        print(f"\n=== LLM requests made in this run: {self.llm_usage} ===\n")

        return generated_answer  # Return the generated answer from the GraphRAG pipeline
    
    def save_report_to_markdown(
        self, 
        answer: str, 
        output_directory: str = None, 
        filename: str = None,
        country: str = None,
        retriever_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Save the GraphRAG answer to a markdown file.
        
        Args:
            answer (str): The generated answer from GraphRAG.
            output_directory (str): Optional directory where to save the markdown file. If none is provided, uses a default directory structure based on the country naming.
            filename (str): Optional custom filename. If None, auto-generates based on timestamp.
            country (str): Country name for the report (used in title and filename).
            retriever_type (str): Type of retriever used (e.g., "HybridCypher", "Vector").
            metadata (dict): Additional metadata to include in the report.
            
        Returns:
            str: Path to the saved markdown file
        """
        
        # Use default output directory if none provided
        if output_directory is None:
            output_directory = self._get_default_output_directory(country)

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Generate timestamp up to minute detail
            country_suffix = f"_{re.sub(r'[^\w\-]', '_', country)}" if country else ""  # Replace special characters with underscores for country name
            retriever_suffix = f"_{retriever_type}" if retriever_type else ""
            filename = f"security_report{country_suffix}{retriever_suffix}_{timestamp}.md"
        
        # Ensure filename has .md extension
        if not filename.endswith('.md'):
            filename += '.md'
            
        filepath = os.path.join(output_directory, filename)  # Full path to the markdown file
        
        # Prepare markdown content
        markdown_content = self._format_markdown_report(
            answer=answer,
            country=country,
            retriever_type=retriever_type,
            metadata=metadata
        )
        
        # Write to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return filepath
        except Exception as e:
            raise RuntimeError(f"Error saving markdown file to {filepath}: {e}")
    
    def _get_default_output_directory(self, country: str = None) -> str:
        """
        Generate the default output directory structure.
        
        Args:
            country (str): Country name for the report.
            
        Returns:
            str: Default output directory path.
        """

        # Get the parent directory of graphrag_pipeline_dir (outside the program
        # files)
        parent_dir = self.graphrag_pipeline_dir.parent  
        
        # Create the base reports directory
        reports_base = os.path.join(parent_dir, 'reports')
        
        # If country is specified, create a country-specific subdirectory
        if country:
            # Sanitize country name for filesystem
            safe_country = re.sub(r'[^\w\-]', '_', country)  # Replace any non-word characters with underscores
            country_path = os.path.join(reports_base, safe_country)
            return country_path
        else:
            # If no country specified, use a general directory
            general_path = os.path.join(reports_base, 'general')
            return general_path

    def _format_markdown_report(
        self, 
        answer: str, 
        country: str = None, 
        retriever_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Format the answer into a structured markdown report.
        
        Args:
            answer (str): The generated answer
            country (str): Country name
            retriever_type (str): Retriever type used
            metadata (dict): Additional metadata
            
        Returns:
            str: Formatted markdown content
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build header
        title = f"Security Report"
        if country:
            title += f" - {country}: Metadata"
            
        markdown_lines = [
            f"# {title}",
            "",
            f"**Generated on:** {timestamp}",
        ]
        
        if retriever_type:
            markdown_lines.append(f"**Retriever:** {retriever_type}")
            
        if metadata:
            markdown_lines.append("**Configuration:**")
            for key, value in metadata.items():
                markdown_lines.append(f"- {key}: {value}")
        
        markdown_lines.extend([
            "",
            "---",
            "",
            answer,
            "",
            "---",
            "",
            f"*Report generated using GraphRAG pipeline at {timestamp}*"
        ])
        
        return "\n".join(markdown_lines)