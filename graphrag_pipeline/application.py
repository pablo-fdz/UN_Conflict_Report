import logging
import importlib
import os
import sys
from datetime import datetime
import json
import runpy  # For running Python scripts dynamically
import re
from pathlib import Path
import glob

class Application:
    """
    Class which implements the application logic for the GraphRAG pipeline.
    """

    def __init__(
            self, 
            ingest_data: list[str] = [],
            build_kg: bool = False,
            resolve_ex_post: bool = False,
            graph_retrieval: list[str] = [],
            accuracy_eval: str = None,
            output_directory: str = None
        ):
        """
        Initializes the application with the provided parameters.
        
        Args:
            ingest_data (list[str]): List of countries for which to ingest 
                data from the sources set in the configuration files. Defaults to 
                an empty list (no ingestion).
            build_kg (str): Flag to indicate whether to build the knowledge
                graph out of the data sources set in the configuration files. 
                Defaults to False (knowledge graph is not built).
            resolve_ex_post (bool): Flag to indicate whether to resolve entities ex-post. 
                Defaults to False (ex-post resolution is not enabled).
            graph_retrieval (list[str]): List of countries for which to do GraphRAG. 
                Defaults to an empty list (GraphRAG is not performed, no report is generated).
            accuracy_eval (str): Flag or path for accuracy evaluation. Defaults to None.
            output_directory (str): Directory to save the generated reports. Defaults
                to None (reports saved in the default directory).
        """

        self.name = "GraphRAG Pipeline"  # Name of the application

        self.ingest_data = ingest_data
        self.build_kg = build_kg
        self.resolve_ex_post = resolve_ex_post
        self.graph_retrieval = graph_retrieval
        self.accuracy_eval = accuracy_eval
        self.output_directory = output_directory

        # Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
        # modules in parent directory)
        self.graphrag_pipeline_dir = Path(__file__).parent  # Get the directory where this script is located (graphrag_pipeline)

        # Ensure pipeline directory is in sys.path
        if str(self.graphrag_pipeline_dir) not in sys.path:
            sys.path.append(str(self.graphrag_pipeline_dir))

        self.__init_logging()  # Initialize logging for the application

        self.config_files_path = os.path.join(self.graphrag_pipeline_dir, 'config_files')  # Find path to config_files folder
        self._load_configs()  # Load all configuration files

        self.generated_reports: list[str] = []  # List to store paths of generated reports (for evaluation purposes)

    def run(self):
        """
        Runs the GraphRAG pipeline with the configured steps.
        """
        self.logger.info(f"Running {self.name}")
        
        try:
            # Step 1: Data Ingestion
            if self.ingest_data:
                self.logger.info(f"Running data ingestion for sources: {self.ingest_data}")
                self._run_data_ingestion()
            
            # Step 2: Knowledge Graph Building and Indexing
            if self.build_kg:
                self.logger.info(f"Building knowledge graph from sources: {self.build_kg}")
                self._run_kg_building()
            
            # Step 3: Ex-post entity Resolution
            if self.resolve_ex_post:
                self.logger.info("Running ex-post entity resolution")
                self._run_entity_resolution()
            
            # Step 4: Graph Retrieval and Report Generation
            if self.graph_retrieval:
                self.logger.info(f"Running graph retrieval: {self.graph_retrieval}")
                self._run_graph_retrieval()
            
            # Step 5: Accuracy Evaluation
            if self.accuracy_eval:
                self.logger.info(f"Running accuracy evaluation: {self.accuracy_eval}")
                self._run_accuracy_evaluation()

            self.logger.info(f"{self.name} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def _run_data_ingestion(self):
        """Run the data ingestion step for specified sources in the configuration file."""

        if self.ingest_data:  # Check if the list is not empty

            self.logger.info("Starting data ingestion process")

            try:
                
                for country in self.ingest_data:
                    self.logger.info(f"Ingesting data for country: {country}")
                    os.environ['GRAPHRAG_INGEST_COUNTRY'] = country
                    
                    for data_source, config in self.data_config.items():  # Iterate over all data sources in the config file
                        
                        if config['ingestion'] is True:  # If the ingestion is enabled for the data source, ingest data
                            
                            self.logger.info(f"Ingesting data from source: {data_source} for {country}")
                            
                            # Set the path to the appropriate script
                            script_path = f"pipeline.01_data_ingestion.{data_source}_ingestion"
                            
                            # Ensure the script path is valid
                            if not importlib.util.find_spec(script_path):
                                self.logger.error(f"Data ingestion module for {data_source} not found.")
                                continue  # Skip to the next data source if the module is not found
                        
                            # Execute the script directly as if it was run with python -m in the terminal
                            try:
                                self.logger.info(f"Executing script: {script_path}")
                                runpy.run_module(script_path, run_name="__main__")
                                self.logger.info(f"Successfully executed script for {data_source} for {country}")
                            except Exception as e:
                                self.logger.error(f"Error executing script for {data_source}: {str(e)}")
            
                # Clean up environment variable
                if 'GRAPHRAG_INGEST_COUNTRY' in os.environ:
                    del os.environ['GRAPHRAG_INGEST_COUNTRY']

            except ImportError as e:
                self.logger.error(f"Could not run data ingestion module: {str(e)}")
        
        else:
            pass  # If no ingestion is specified, skip this step
    
    def _run_kg_building(self):
        """Run the knowledge graph building step for specified sources."""

        if self.build_kg:

            self.logger.info("Starting knowledge graph building process")

            # ---------- 1. Build from data sources specified in the configuration file ----------

            for country in self.build_kg:
                
                print("Building knowledge graph for country:", country)

                self.logger.info(f"Building KG for country: {country}")
                
                os.environ['KG_BUILDING_COUNTRY'] = country

                for data_source, config in self.data_config.items():  # Iterate over all data sources in the config file
                    if config['include_in_kg'] is True:  # If the KG building is enabled for the data source, build from its data
                        
                        self.logger.info(f"Building KG from data source: {data_source}")
                        
                        # Set the path to the appropriate script
                        script_path = f"pipeline.02_kg_building.{data_source}_kg_building"
                        
                        # Ensure the script path is valid
                        if not importlib.util.find_spec(script_path):
                            self.logger.error(f"KG building module for {data_source} not found.")
                            continue  # Skip to the next data source if the module is not found
                        
                        # Execute the script directly as if it was run with python -m in the terminal
                        try:
                            self.logger.info(f"Executing script: {script_path}")
                            runpy.run_module(script_path, run_name="__main__")
                            self.logger.info(f"Successfully executed script for {data_source}")
                        except Exception as e:
                            self.logger.error(f"Error executing script for {data_source}: {str(e)}")
                    
                    else:
                        self.logger.debug(f"Skipping {data_source} - not enabled for KG building")
                        continue  # Skip to the next data source if KG building is not enabled
            
            # Clean up environment variables
            if 'KG_BUILDING_COUNTRY' in os.environ:
                del os.environ['KG_BUILDING_COUNTRY']
                
            # ---------- Index the knowledge graph ----------
        
            self._run_kg_indexing()  # Index the knowledge graph after building it

        # ---------- 2. Build from sample data ----------
        
        # Build KG with sample data if specified in the configuration, for 
        # development purposes. This is useful for testing and development without
        # needing to run the full data ingestion and KG building process.
        elif self.build_config['dev_settings']['build_with_sample_data'] == True:

            self.logger.info("Building KG with sample data for development purposes")

            try:
                # Set the path to the appropriate script
                script_path = f"pipeline.02_kg_building.sample_kg_building"
                
                # Ensure the script path is valid
                if not importlib.util.find_spec(script_path):
                    self.logger.error(f"KG building module for sample data not found.")
                
                # Execute the script directly as if it was run with python -m in the terminal
                try:
                    self.logger.info(f"Executing script: {script_path}")
                    runpy.run_module(script_path, run_name="__main__")
                    self.logger.info(f"Successfully executed script for sample data.")
                except Exception as e:
                    self.logger.error(f"Error executing script for sample data: {str(e)}")
            
            except ImportError as e:
                self.logger.error(f"Could not import data ingestion module: {str(e)}")
        
            # ---------- Index the knowledge graph ----------
            
            self._run_kg_indexing()  # Index the knowledge graph after building it

        else:
            pass  # If KG building is disabled, skip this step
    
    def _run_kg_indexing(self):
        """Run knowledge graph indexing."""
        
        self.logger.info(f"Indexing embeddings and full text of knowledge graph for retrieval.")

        try:
            # Set the path to the appropriate script
            script_path = f"pipeline.03_indexing.indexing"
            
            # Ensure the script path is valid
            if not importlib.util.find_spec(script_path):
                self.logger.error(f"Indexing script not found.")
            
            # Execute the script directly as if it was run with python -m in the terminal
            try:
                self.logger.info(f"Executing script: {script_path}")
                runpy.run_module(script_path, run_name="__main__")
                self.logger.info(f"Successfully executed script for indexing KG.")
            except Exception as e:
                self.logger.error(f"Error executing script for indexing KG: {str(e)}")
        
        except ImportError as e:
            self.logger.error(f"Could not import indexing script: {str(e)}")
    
    def _run_entity_resolution(self):
        """Run ex-post entity resolution."""

        if self.resolve_ex_post == True:

            self.logger.info(f"Running ex-post entity resolution with the configured resolver: {self.build_config['entity_resolution_config']['ex_post_resolver']}")

            try:
                # Set the path to the appropriate script
                script_path = f"pipeline.04_ex_post_resolution.ex_post_resolver"
                
                # Ensure the script path is valid
                if not importlib.util.find_spec(script_path):
                    self.logger.error(f"Ex-post resolving script not found.")
                
                # Execute the script directly as if it was run with python -m in the terminal
                try:
                    self.logger.info(f"Executing script: {script_path}")
                    runpy.run_module(script_path, run_name="__main__")
                    self.logger.info(f"Successfully executed script for running ex-post entity resolution.")
                except Exception as e:
                    self.logger.error(f"Error executing script for ex-post entity resolution: {str(e)}")
            
            except ImportError as e:
                self.logger.error(f"Could not import ex-post entity resolution script: {str(e)}")
        
        else:
            pass  # If ex-post entity resolution is disabled, skip this step
    
    def _run_graph_retrieval(self):
        """Run graph retrieval for the specified countries."""
        
        if self.graph_retrieval:  # If countries for graph retrieval are specified, run graph retrieval. Empty lists are falsy in Python.

            self.logger.info(f"Creating forward-looking security reports for the following countries: {', '.join(self.graph_retrieval)}")

            try:

                # Iterate over the countries specified for graph retrieval. For
                # each country, run the GraphRAG pipeline with the configured retrievers.
                for country in self.graph_retrieval:
                    
                    safe_country = re.sub(r'[^\w\-]', '_', country)  # Sanitize country name for use in environment variables

                    self.logger.info(f"Generating security report for {country}")

                    # Set environment variables for the script to access
                    os.environ['GRAPHRAG_COUNTRY'] = country
                    if self.output_directory:  # If output_directory is set, use it
                        os.environ['GRAPHRAG_OUTPUT_DIR'] = self.output_directory  # Set the output directory for GraphRAG

                    # Generate the forecast outputs that will be used in the report
                    cast_script_path = f"pipeline.05_graphrag.forecast_generation"  # Set the path to the appropriate script
                    if not importlib.util.find_spec(cast_script_path):  # Ensure the script path is valid
                        self.logger.error(f"Module for retrieving forecast predictions not found.")
                        break  # Break the loop if the module is not found
                    try:
                        self.logger.info(f"Executing script: {cast_script_path}")
                        runpy.run_module(cast_script_path, run_name="__main__")
                        self.logger.info(f"Successfully executed script for generating forecast outputs for {country}")
                    except Exception as e:
                        self.logger.error(f"Error executing script for generating forecast outputs for {country}: {str(e)}")
                        continue  # Skip to the next country if the script fails

                    for retriever, config in self.retrieval_config.items():  # Iterate over all retrievers in the config file
                        if config['enabled'] is True:  # If retrieval is enabled for the retriever, do GraphRAG using that retriever
                            
                            self.logger.info(f"Running GraphRAG for {country} with the retriever: {retriever}")
                            
                            # Set the path to the appropriate script
                            script_path = f"pipeline.05_graphrag.{retriever}_graphrag"
                            
                            # Ensure the script path is valid
                            if not importlib.util.find_spec(script_path):
                                self.logger.error(f"GraphRAG module for {retriever} not found.")
                                continue  # Skip to the next retriever if the module is not found
                            
                            try:
                                self.logger.info(f"Executing script: {script_path}")
                                runpy.run_module(script_path, run_name="__main__")
                                self.logger.info(f"Successfully executed script for {retriever}")

                                # --- record which file just got written ---
                                base_path = Path(self.output_directory or self.graphrag_pipeline_dir.parent / 'reports')
                                # look both in base/country (default) and base (in case scripts wrote directly there)
                                search_dirs = [base_path / country, base_path]
                                retr_short = retriever.replace('Retriever', '')
                                candidates = []
                                for d in search_dirs:
                                    if d.is_dir():
                                        pattern = f"security_report_{safe_country}_{retr_short}_*.md"
                                        candidates.extend(d.glob(pattern))
                                if candidates:
                                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                                    self.generated_reports.append(str(latest))
                                    self.logger.info(f"Recorded generated report: {latest}")
                                else:
                                    self.logger.warning(f"No report found for pattern '{retr_short}' in {search_dirs}")
                            except Exception as e:
                                self.logger.error(f"Error executing {script_path}: {e}")

                    else:
                        self.logger.debug(f"Skipping {retriever} - not enabled for GraphRAG")
                        continue  # Skip to the next retriever if GraphRAG for that retriever is not enabled
                
                # Clean up environment variables
                if 'GRAPHRAG_COUNTRY' in os.environ:
                    del os.environ['GRAPHRAG_COUNTRY']
                if 'GRAPHRAG_OUTPUT_DIR' in os.environ:
                    del os.environ['GRAPHRAG_OUTPUT_DIR']
            
            except ImportError as e:
                self.logger.error(f"Could not import GraphRAG module: {str(e)}")

        else:
            pass  # If no countries are passed for graph retrieval, skip this step
    
    def _run_accuracy_evaluation(self):
        """Run accuracy evaluation of the generated reports."""
        
        # Case 1: Default evaluation for countries from --retrieval
        if self.accuracy_eval == 'default_eval':
            if not self.generated_reports:
                raise ValueError("--accuracy-eval requires --retrieval (no reports found to eval).")
            self.logger.info("Running accuracy evaluation for generated reports")
            for report in self.generated_reports:
                self._execute_evaluation_script(report_path=report)

        # Case 2: Evaluation for a specific report file
        elif self.accuracy_eval and os.path.isfile(self.accuracy_eval):
            self.logger.info(f"Running accuracy evaluation for specific report: {self.accuracy_eval}")
            self._execute_evaluation_script(report_path=self.accuracy_eval)
        
        # Case 3: Invalid argument
        else:
            self.logger.error(f"Invalid path for --accuracy-eval: {self.accuracy_eval}. File not found.")
            raise FileNotFoundError(f"The report specified for --accuracy-eval does not exist: {self.accuracy_eval}")

    def _execute_evaluation_script(self, country: str = None, report_path: str = None):
        """Helper method to set environment and run the evaluation script."""
        
        script_path = "pipeline.06_evaluation.accuracy_evaluation"
        
        if not importlib.util.find_spec(script_path):
            self.logger.error(f"Accuracy evaluation module not found at {script_path}.")
            return

        try:
            # Set environment variables for the script
            if country:
                os.environ['GRAPHRAG_COUNTRY'] = country
            if report_path:
                os.environ['GRAPHRAG_EVAL_REPORT_PATH'] = report_path
            if hasattr(self, 'output_directory') and self.output_directory:
                os.environ['GRAPHRAG_OUTPUT_DIR'] = self.output_directory
            
            self.logger.info(f"Executing script: {script_path}")
            runpy.run_module(script_path, run_name="__main__")
            self.logger.info(f"Successfully executed accuracy evaluation for {country or report_path}.")

        except Exception as e:
            self.logger.error(f"Error executing accuracy evaluation script for {country or report_path}: {str(e)}")
        finally:
            # Clean up environment variables
            if 'GRAPHRAG_COUNTRY' in os.environ:
                del os.environ['GRAPHRAG_COUNTRY']
            if 'GRAPHRAG_EVAL_REPORT_PATH' in os.environ:
                del os.environ['GRAPHRAG_EVAL_REPORT_PATH']
            if 'GRAPHRAG_OUTPUT_DIR' in os.environ:
                del os.environ['GRAPHRAG_OUTPUT_DIR']

    def _load_configs(self):
        """Load all configuration files necessary to run the program."""

        try:
            
            # Load data configurations
            with open(os.path.join(self.config_files_path, 'data_ingestion_config.json'), 'r') as f:
                self.data_config = json.load(f)
            # Load KG building configurations
            with open(os.path.join(self.config_files_path, 'kg_building_config.json'), 'r') as f:
                self.build_config = json.load(f)
            # Load KG retrieval configurations
            with open(os.path.join(self.config_files_path, 'kg_retrieval_config.json'), 'r') as f:
                self.retrieval_config = json.load(f)
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {e.filename}. Please ensure the file exists in the config_files directory.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in configuration file: {e.msg}. Please check the file format.")

    def __init_logging(self):
        """
        Initializes logging for the application. Hybrid approach that logs to a 
        "current" log file and a timestamped log file in the logs directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Create a timestamp for the log file
        logs_dir = os.path.join(self.graphrag_pipeline_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)  # Ensure the logs directory exists
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, 'application.log'), mode='w'),  # Current log (replaced on each run)
                logging.FileHandler(os.path.join(logs_dir, f'application_{timestamp}.log'))  # Timestamped log
            ]
        )
        self.logger = logging.getLogger(self.name)