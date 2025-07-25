"""This script just performs argument parsing and calls the application. It acts
as the bridge between the command line interface and the application logic."""

import argparse
from application import Application

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the GraphRAG pipeline with configurable steps.'
    )
    
    # Add arguments for each pipeline step
    parser.add_argument(
        '--ingest',
        help='Space-separated strings of countries for which to download data from configured sources. This argument can accept multiple values. Data ingestion is not executed if no countries are provided or if the argument is not used. Example usage: --ingest "Sudan" "Mali"',
        nargs='*',  # Zero or more arguments can be provided
        default=[],  # Default is an empty list if no arguments are provided
        dest='ingest_data'
    )
    
    parser.add_argument(
        '--build-kg',
        help='Space-separated strings of countries for which to build the knowledge graph from ingested data. This argument can accept multiple values. KG building is not executed if no countries are provided, if the argument is not used, if data has not been ingested for those countries or if the "sample data" option in the configuration files is deactivated. Example usage: --build-kg "Sudan" "Mali"',
        nargs='*',  # Zero or more arguments can be provided
        default=[],  # Default is an empty list if no arguments are provided
        dest='build_kg'
    )
    
    parser.add_argument(
        '--resolve-ex-post',
        help='Perform ex-post entity resolution (if argument is provided). This will resolve entities in the knowledge graph after it has been built.',
        action='store_true',  # If the flag is present, resolve entities. By default, it is False.
        dest='resolve_ex_post'
    )
    
    parser.add_argument(
        '--retrieval',
        help='Space-separated strings of countries for which to retrieve the knowledge graph and generate security reports. This argument can accept multiple values. GraphRAG is not executed if no countries are provided or if the argument is not used. Example usage: --retrieval "Sudan"  "United States"',
        nargs='*',  # Zero or more arguments can be provided
        default=[],  # Default is an empty list if no arguments are provided
        dest='graph_retrieval'
    )

    parser.add_argument(
        '--accuracy-eval',
        help='Run accuracy evaluation. Can be used alone to evaluate reports from the --retrieval step, or with a path to a specific report. Example: --accuracy-eval /path/to/report.md',
        nargs='?', # 0 or 1 argument
        const='default_eval', # Value if flag is present without an argument
        default=None, # Value if flag is not present
        dest='accuracy_eval'
    )

    parser.add_argument(
        '--output-dir',
        help='String of the directory to save the generated reports. If not specified, uses default directory structure.',
        type=str,
        default=None,
        dest='output_directory'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Create application with parsed arguments
    app = Application(
        ingest_data=args.ingest_data,
        build_kg=args.build_kg,
        resolve_ex_post=args.resolve_ex_post,
        graph_retrieval=args.graph_retrieval,
        accuracy_eval=args.accuracy_eval,
        output_directory=args.output_directory
    )
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()