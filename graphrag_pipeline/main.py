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
        help='Download data from configured sources (if argument is provided).',
        action='store_true',  # If the flag is present, ingest data. By default, it is False.
        dest='ingest_data'
    )
    
    parser.add_argument(
        '--build-kg',
        help='Build the knowledge graph from the configured sources (if argument is provided).',
        action='store_true',  # If the flag is present, build the knowledge graph. By default, it is False.
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
        help='Space-separated names of countries for which to retrieve the knowledge graph and generate security reports. This argument can accept multiple values. GraphRAG is not executed if no countries are provided or if the argument is not used.',
        nargs='*',  # Zero or more arguments can be provided
        default=[],  # Default is an empty list if no arguments are provided
        dest='graph_retrieval'
    )

    parser.add_argument(
        '--output-dir',
        help='Directory to save the generated reports. If not specified, uses default directory structure.',
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
        output_directory=args.output_directory
    )
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()