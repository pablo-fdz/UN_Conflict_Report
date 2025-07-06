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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Tuple, Union
import json
from neo4j_graphrag.retrievers.base import Retriever
from library.kg_indexer import KGIndexer
from library.kg_builder.utilities import GeminiLLM, get_rate_limit_checker
from neo4j_graphrag.generation import RagTemplate
from library.graphrag import CustomGraphRAG
from neo4j_graphrag.types import RetrieverResult

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
            country: str = None,
            output_directory: str = None
        ):
        
        """
        Run the complete GraphRAG construction pipeline asynchronously.

        Args:
            retriever (Retriever): The retriever used to find relevant context to pass to the LLM.
            country (str): The country for which the report is generated. Defaults to None, which uses an empty string in the query text.
            retriever_search_params (dict[str, any]): Configuration for the search parameters of the input retriever. Defaults to None, which uses the default search parameters.
            output_directory (str): Optional directory where the forecasts included in the report will be searched for, under the `assets` subdirectory. If None, uses a default directory structure based on the country naming.
        
        Returns:
            str: The generated answer from the GraphRAG pipeline and the retrieved context (if context return is enabled).
        """

        # Use default output directory if none provided
        if output_directory is None:
            output_directory = self._get_default_output_directory(country)

        try:
                
            # Get the initialized GraphRAG pipeline
            graphrag = self._create_graphrag_pipeline(retriever)

            # Get the latest forecast data for the country to pass ACLED hotspot
            # predictions into the report
            forecast_data, _, _, _ = self._get_latest_forecast_data(output_directory)

            # Initialize hotspot variables with default values
            total_hotspots = 0
            hotspot_regions = []

            if forecast_data and forecast_data.get('acled_cast_analysis'):  # If ACLED CAST analysis is available
                total_hotspots = forecast_data['acled_cast_analysis'].get('total_hotspots', 0)  # Get the total hotspots, default to 0 if not present
                hotspot_regions = forecast_data['acled_cast_analysis'].get('hotspot_regions', [])  # Get the hotspot regions, default to empty list if not present. This will be a list of dictionaries with 'name' (name of ADM1 region), 'avg1' (average number of violent events in the last 3 months), 'total_forecast' (forecasted number of violent events 2 months ahead, including current month), 'forecast_horizon_months' (forecast horizon in months, default to 2 if not present) and 'percent_increase' keys.
                hotspot_regions_list = []
                # For each of the hotspot regions, retrieve the name
                for region in hotspot_regions:
                    region_name = region.get('name', 'Unknown Region')
                    hotspot_regions_list.append(region_name)  # Append the region name to the list

            # Get current month and year, e.g., "July, 2025"
            current_month_year = datetime.now().strftime("%B, %Y")

            default_search_text = "Security events, conflicts, and political stability in {country}. Focus on the following conflict hotspots: {hotspot_regions_list}."

            # Format the search text for the retriever (i.e., the text that will be used to search the knowledge graph)
            formatted_search_text = self.graphrag_config.get('search_text', default_search_text).format(  # Use the country in the search text if specified, otherwise use an empty string
                country=country,
                hotspot_regions_list= ', '.join(hotspot_regions_list) if 'hotspot_regions_list' in locals() else ''  # Use the hotspot regions list if available, otherwise default to empty string
            )  

            # Format the query text for generating the report with the input country
            formatted_query_text = self.graphrag_config.get('query_text', '').format(  # Use the information in the query text if specified, otherwise use an empty string
                country=country,
                current_month_year=current_month_year,  # Current month and year for the report
                total_hotspots=total_hotspots if 'total_hotspots' in locals() else 0,  # Use the total hotspots if available, otherwise default to 0
                hotspot_regions=hotspot_regions if 'hotspot_regions' in locals() else []  # Use the hotspot regions if available, otherwise default to empty list
            )
            
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
                retrieved_context = graphrag_results.retriever_result  # This will be a RetrieverResult object, containing `items` and `metadata`

        except Exception as e:
            raise RuntimeError(f"Error during GraphRAG construction pipeline execution: {e}")
        
        print(f"\n=== LLM requests made in this run: {self.llm_usage} ===\n")

        return generated_answer, retrieved_context  # Return the generated answer and the retrieved context from the GraphRAG pipeline
    
    def save_report_to_markdown(
        self, 
        answer: str,
        context: Union[RetrieverResult, None] = None,
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
            context (Union[RetrieverResult, None]): Optional context used for generating the answer. 
                If provided, it will be saved in a separate "context" folder within the output directory.
            output_directory (str): Optional directory where to save the markdown file. If none is provided, uses a default directory structure based on the country naming.
            filename (str): Optional custom filename. If None, auto-generates based on timestamp.
            country (str): Country name for the report (used in title and filename).
            retriever_type (str): Type of retriever used (e.g., "HybridCypher", "Vector").
            metadata (dict): Additional metadata to include in the report.
            
        Returns:
            str: Path to the saved markdown file and to the JSON context file (if context is provided).
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
        report_content = self._format_markdown_report(
            answer=answer,
            country=country,
            output_directory=output_directory,  # Pass the output directory for locating assets
            retriever_type=retriever_type,
            metadata=metadata
        )
        
        # Write to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
        except Exception as e:
            raise RuntimeError(f"Error saving markdown file to {filepath}: {e}")
        
        context_filepath = None  # Initialize context filepath as None
        # Save context to a separate file if provided
        if context:
            context_dir = os.path.join(output_directory, 'context')
            os.makedirs(context_dir, exist_ok=True)
            context_filename = f"context_{os.path.splitext(filename)[0]}.json" # Use the report filename without extension and add .json
            context_filepath = os.path.join(context_dir, context_filename)
            try:
                context_items = context.items  # Get the items from the RetrieverResult (list of `content` and `metadata` for each retrieved result)
                retrieved_results = []
                for item in context_items:
                    retrieved_results.append({
                        'content': item.content,  # The content of the retrieved result
                        'metadata': item.metadata  # The metadata associated with the retrieved result
                    })
                # Save context as a JSON file
                context_json = {
                    'country': country,  # Country for which the report was generated
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp of when the context was saved
                    'retriever_type': retriever_type,  # Type of retriever used
                    'retrieved_results': retrieved_results  # List of retrieved results with content and metadata
                }
                with open(context_filepath, 'w', encoding='utf-8') as f:
                    json.dump(context_json, f, indent=4) # Use json.dump to write the dictionary as JSON
            except Exception as e:
                # Log or print a warning instead of raising an error to not stop the main report generation
                print(f"Warning: Could not save context file to {context_filepath}: {e}")
        
        return filepath, context_filepath
    
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

    def _get_latest_forecast_data(self, output_directory: str) -> Tuple[dict, str, str, str]:
        """
        Get the latest forecast data from ConflictForecast and ACLED for a given country.
        
        Args:
            output_directory (str): Directory where the report will be saved (as well as the
                forecast data in the `assets` subdirectory).
            country (str): Country name to get the forecast for.
            
        Returns:
            tuple: A tuple containing:
                - dict: Forecast data containing conflict forecast prediction (`conflict_forecast_prediction` with float value) and ACLED CAST analysis.
                - str: Path to the latest ConflictForecast time series plot.
                - str: Path to the latest ACLED CAST analysis hotspots plot.
                - str: Path to the latest forecast data JSON file.
        """
        
        # Initialize variables to hold forecast data and chart paths
        forecast_data = {}
        line_chart_path = None
        bar_chart_path = None
        forecast_data_path = None
        
        try:
            assets_dir = os.path.join(output_directory, 'assets')
            os.makedirs(assets_dir, exist_ok=True)  # Ensure assets directory exists

            # Find the most recent forecast JSON file
            forecast_files = [f for f in os.listdir(assets_dir) if f.startswith('forecast_') and f.endswith('.json')]

            if forecast_files:
                latest_file = max(forecast_files, key=lambda f: os.path.getmtime(os.path.join(assets_dir, f)))
                latest_filepath = os.path.join(assets_dir, latest_file)
                forecast_data_path = latest_filepath  # Store the path to the latest forecast data file

                # Load the JSON data from the file
                with open(latest_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Safely extract the required information
                forecast_data = {
                    "conflict_forecast_prediction": data.get("conflict_forecast_prediction", None),  # Defafults to None if not present
                    "acled_cast_analysis": data.get("acled_cast_analysis", None)  # Defaults to None if not present (e.g., if region in Europe or North America)
                }

            else:
                print(f"Warning: No forecast files found in {assets_dir}")

            # Find the most recent line chart
            line_chart_files = [f for f in os.listdir(assets_dir) if f.startswith(f'LineChart_') and f.endswith('.svg')]
            if line_chart_files:
                latest_line_chart = max(line_chart_files, key=lambda f: os.path.getmtime(os.path.join(assets_dir, f)))
                line_chart_path = os.path.join(assets_dir, latest_line_chart)
            else:
                print(f"Warning: No line chart found in {assets_dir}")

            # Find the most recent bar chart
            bar_chart_files = [f for f in os.listdir(assets_dir) if f.startswith(f'BarChart_') and f.endswith('.svg')]
            if bar_chart_files:
                latest_bar_chart = max(bar_chart_files, key=lambda f: os.path.getmtime(os.path.join(assets_dir, f)))
                bar_chart_path = os.path.join(assets_dir, latest_bar_chart)
            else:
                print(f"Warning: No bar chart found in {assets_dir}")

            return forecast_data, line_chart_path, bar_chart_path, forecast_data_path

        except FileNotFoundError:
            print(f"Warning: Asset file could not be found.")
            return {}, None, None
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from forecast file.")
            return {}, None, None
        except Exception as e:
            print(f"An unexpected error occurred in _get_latest_forecast_data: {e}")
            return {}, None, None

    def _format_markdown_report(
        self, 
        answer: str, 
        country: str = None,
        output_directory: str = None,
        retriever_type: str = None,
        metadata: dict = None
    ) -> str:
        """
        Format the answer into a structured markdown report.
        
        Args:
            answer (str): The generated answer
            country (str): Country name
            output_directory (str): Directory where the report will be saved (for 
                locating the assets).
            retriever_type (str): Retriever type used
            metadata (dict): Additional metadata
            
        Returns:
            str: Formatted markdown content
        """
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ===== 1. Get and format the latest forecast data for the report =====

        # Get the predictions data
        forecast_data, line_chart_path, bar_chart_path, forecast_data_path = self._get_latest_forecast_data(output_directory)

        processed_answer = answer # Initialize processed_answer with the original answer

        # --- Create and inject the "Conflict Forecast" section ---
        if line_chart_path and forecast_data.get('conflict_forecast_prediction'):
            conflict_forecast_prediction = forecast_data['conflict_forecast_prediction']
            
            # Create the structured text for this section
            conflict_forecast_text = [
                "",  # Initialize with a newline for spacing 
                "### Armed Conflict Probability Forecast (ConflictForecast)",
                ""  # Initialize with an empty line for spacing
            ]
            
            conflict_forecast_text.append(f"According to [ConflictForecast](https://conflictforecast.org/), there is a {conflict_forecast_prediction:.2%} estimated probability that {country} will experience an outbreak of armed conflict within the next three months.")
            conflict_forecast_text.append("")
            conflict_forecast_text.append(f"*This forecast reflects the likelihood that the country will exceed a threshold of 0.5 fatalities per one million inhabitants over the course of three months.*")
            conflict_forecast_text.append("")

            # Add more structured text here as needed.
            conflict_forecast_text.append(f"The trend in armed conflict risk, 2016-{datetime.now().year}.")

            # Make chart path relative for markdown file
            relative_line_chart_path = os.path.join('assets', os.path.basename(line_chart_path))
            conflict_forecast_text.extend([
                "",
                f"![Conflict Forecast Time Series]({relative_line_chart_path})",
                ""
            ])
            
            conflict_forecast_section = "\n".join(conflict_forecast_text)
            
            # Inject the section after "## 3. Forward Outlook" and the potential
            # introductory text to the section, but before the next H3 or H2 section
            # Check regex101 (https://regex101.com/v) in case of doubts
            processed_answer = re.sub(
                r"(## 3\. Forward Outlook[\s\S]*?)(?=\n###|\n##)",
                r"\1" + conflict_forecast_section,
                processed_answer,
                count=1,
                flags=re.DOTALL  # Use DOTALL to match across newlines
            )
        else:
            print("Warning: No Conflict Forecast time series plot and/or armed conflict risk predictions found. The section will not be included in the report.")

        # --- Create and inject the "ACLED" section ---
        if bar_chart_path and forecast_data.get('acled_cast_analysis'):
            acled_analysis = forecast_data['acled_cast_analysis']  
            forecast_horizon_months = acled_analysis.get('forecast_horizon_months', 2)  # Get the forecast horizon in months, default to 2 if not present
            total_hotspots = acled_analysis.get('total_hotspots', 0)  # Get the total hotspots, default to 0 if not present
            hotspot_regions = acled_analysis.get('hotspot_regions', [])  # Get the hotspot regions, default to empty list if not present. This will be a list of dictionaries with 'name' (name of ADM1 region), 'avg1' (average number of violent events in the last 3 months), 'total_forecast' (forecasted number of violent events 2 months ahead, including current month), 'forecast_horizon_months' (forecast horizon in months, default to 2 if not present) and 'percent_increase' keys.

            # Create the structured text for this section
            acled_forecast_text = [
                "",  # Initialize with a newline for spacing
                "#### Predicted Increase in Violent Events in the Short Term (ACLED)",
                ""
            ]

            # Get current month and year, e.g., "July, 2025"
            now = datetime.now()

            # Get next month and year, e.g., "August, 2025"
            next_month_year = (now.replace(day=1) + timedelta(days=32)).strftime("%B, %Y")  # Go to the first day of the current month, add 32 days to get into the next month, and then format it.

            if total_hotspots > 0:
                acled_forecast_text.append(f"[ACLED CAST](https://acleddata.com/conflict-alert-system/) predicts {total_hotspots} ADM1 regions in {country} to be hotspots for violent events in the next calendar month ({next_month_year}).")
                acled_forecast_text.append("")
                acled_forecast_text.append("*An ADM1 region is considered to be a hotspot if the predicted increase in the number of violent events in the next month compared to the 3-month average is at least of 25%.*")
                acled_forecast_text.append("")

            acled_forecast_text.append("The chart below shows regions with a predicted change in violent events.")

            # Make chart path relative for markdown file
            relative_bar_chart_path = os.path.join('assets', os.path.basename(bar_chart_path))
            acled_forecast_text.extend([
                "",
                f"![ACLED Hotspots Bar Chart]({relative_bar_chart_path})",
                ""
            ])

            if total_hotspots > 0:
                acled_forecast_text.append(f"Considering the hotspot criteria, the following regions are expected to have a significant increase in violent events in {next_month_year}:")
                acled_forecast_text.append("")  # Add a blank line for spacing before the table

                # Add Markdown table header
                acled_forecast_text.append("| Region | Avg. # Violent Events (3 months) | Forecasted # Violent Events | % Increase |")
                acled_forecast_text.append("|---|---|---|---|")
                for region in hotspot_regions:
                    name = region.get('name', 'Unknown Region')
                    avg1 = region.get('avg1', 0)  # Average number of violent events in the last 3 months
                    total_forecast = region.get('total_forecast', 0)  # Forecasted number of violent events 2 months ahead, including current month
                    percent_increase = region.get('percent_increase1', 0)  # Percent increase in violent events
                    acled_forecast_text.append(f"| {name} | {round(avg1)} | {round(total_forecast)} | {round(percent_increase, 1)}% |")

            acled_forecast_section = "\n".join(acled_forecast_text)
            
            # Inject the section after any intro text in "### Subnational Perspective", 
            # but before the next H4 or H3 heading
            processed_answer = re.sub(
                r"(### Subnational Perspective[\s\S]*?)(?=\n####|\n###)",
                r"\1" + acled_forecast_section,
                processed_answer,
                count=1,
                flags=re.DOTALL  # Use DOTALL to match across newlines
            )
        else:
            print("Warning: No ACLED CAST analysis and/or bar chart found. The section will not be included in the report.")

        # ===== 2. Format the markdown report =====

        markdown_lines = []  # Initialize an empty list to hold the markdown lines
        
        # ----- 2.1. Add the processed answer to the markdown lines -----

        markdown_lines.extend([
            processed_answer,  # Use the processed answer with injected sections
            "",
            "---",
            ""
        ])

        # ----- 2.2. Add a metadata section at the end of the report -----

        metadata_lines = [
            f"# Metadata",  # Title for the metadata section
            ""
        ]
        
        metadata_lines.append(f"**Generated on:** {timestamp}")
        metadata_lines.append("")

        if country:
            metadata_lines.append(f"**Country:** {country}")
            metadata_lines.append("")

        if retriever_type:
            metadata_lines.append(f"**Retriever used for report generation:** {retriever_type}")
            metadata_lines.append("")
            
        if forecast_data_path:
            metadata_lines.append(f"**Forecast data path:** {os.path.basename(forecast_data_path)}")
            metadata_lines.append("")

        if metadata:
            metadata_lines.append("**Configuration:**")
            for key, value in metadata.items():
                metadata_lines.append(f"- {key}: {value}")
        
        markdown_lines.extend(metadata_lines)
        
        return "\n".join(markdown_lines)