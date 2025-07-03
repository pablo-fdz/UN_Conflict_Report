"""
ACLED Knowledge Graph Builder

This script builds knowledge graphs from ACLED conflict data.

Features:
- Automatic data file detection and loading
- Configurable sampling for testing
- Command-line interface for easy usage

Author: Generated for UN Conflict Report project
"""

import os
import sys
import asyncio
import json
import polars as pl
from pathlib import Path

# Setup paths for imports
script_dir = Path(__file__).parent
graphrag_pipeline_dir = script_dir.parent.parent
if str(graphrag_pipeline_dir) not in sys.path:
    sys.path.append(str(graphrag_pipeline_dir))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import project modules after path setup
# Note: These imports must come after path setup to ensure modules are found
try:
    from kg_construction_pipeline import KGConstructionPipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

async def main():
    """
    Main function to build knowledge graph from ACLED conflict data.
    
    1. Data Loading: Loads ACLED conflict data
    2. Knowledge Graph Construction: Creates entities, relationships, text chunks and document nodes with metadata
    """

    base_config_path = Path(__file__).parent.parent.parent / 'config_files'
    config_path = base_config_path / 'data_ingestion_config.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('acled', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")

    # Get the sample size parameter if defined
    sample_size_str = config.get('sample_size', 'all')
    sample_size = None
    if sample_size_str.lower() != 'all':
        try:
            sample_size = int(sample_size_str)
        except (ValueError, TypeError):
            print(f"Invalid sample size '{sample_size_str}'. Processing all data.")
            sample_size = None

    # Get the country for which to build the knowledge graph
    country = os.getenv('KG_BUILDING_COUNTRY')
    if not country:
        raise ValueError("Country not specified. Set GRAPHRAG_KG_COUNTRY environment variable.")

    # ==================== 1. Load data ====================

    try:
        # Find ACLED data files
        data_dir = os.path.join(graphrag_pipeline_dir, 'data', 'acled')
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"ACLED data directory not found: {data_dir}")
        
        # Get list of available files
        available_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        
        if not available_files:
            raise FileNotFoundError(f"No ACLED parquet files found in: {data_dir}")
        
        # Join the "country" with underscores if there is spacing
        data_file_pattern = country.replace(' ', '_')

        # Select file based on pattern (name of a country) or use first available
        if data_file_pattern:
            matching_files = [f for f in available_files if data_file_pattern.lower() in f.lower()]
            if not matching_files:
                print(f"No files matching country '{data_file_pattern}' found.")
                print(f"Available files: {available_files}")
                raise FileNotFoundError(f"No files matching country: {data_file_pattern}")
            selected_file = matching_files[0]
        else:
            selected_file = available_files[0]
        
        df_path = os.path.join(data_dir, selected_file)
        print(f"Loading data from: {selected_file}")
        
        df = pl.read_parquet(df_path)
        
        # Apply sampling if specified
        if sample_size:
            original_size = len(df)
            df = df.head(sample_size)
            print(f"Using sample of {len(df)} rows out of {original_size} total rows for testing")
        
        # Convert date column to string format for metadata
        if 'date' in df.columns:
            df = df.with_columns([
                pl.col('date').dt.strftime('%Y-%m-%d').alias('date')
            ])
    
    except Exception as e:
        print(f"Error loading ACLED data: {e}")
        return []

    # ==================== 2. Run KG pipeline ====================

    # Initialize the custom ACLED KG construction pipeline 
    # This uses enhanced SpaCy resolver with higher similarity threshold
    kg_pipeline = KGConstructionPipeline()

    # Define metadata mapping for ACLED data (document properties additional 
    # to base field to dataframe columns)
    metadata_mapping = {
        "date": "date",           # Event date
        "domain": "domain",
        "url": "url" # if available
    }
    
    results = await kg_pipeline.run_async(
        df=df,
        document_base_field='item_id',
        text_column='text',
        document_metadata_mapping=metadata_mapping,
        document_id_column='item_id'  # Use item_id as document ID
    )

    print(f"Processed {len(results)} documents successfully.")
    return results

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    # Run the main function with arguments
    results = asyncio.run(main())