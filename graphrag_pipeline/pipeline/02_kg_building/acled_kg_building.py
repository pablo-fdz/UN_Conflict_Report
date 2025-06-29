"""
ACLED Knowledge Graph Builder with Built-in Entity Resolution

This script builds knowledge graphs from ACLED conflict data for any country/region
using the standard KGConstructionPipeline with built-in entity resolution.

Features:
- Automatic data file detection and loading
- Built-in entity resolution using configuration files
- Configurable sampling for testing
- Command-line interface for easy usage

Usage Examples:
    # Process all data from Sudan with sample for testing
    python acled_kg_building.py --file-country "Sudan" --sample-size 100

    # List available data files
    python acled_kg_building.py --list-files
    
    # Process first available file with full data
    python acled_kg_building.py

RECENT CHANGES:
- Fully reverted to standard KGConstructionPipeline with built-in entity resolution
- Removed all custom resolver code and configuration
- Simplified interface for better stability and compatibility
- Uses configuration files for entity resolution settings

Author: Generated for UN Conflict Report project
"""

import os
import sys
import asyncio
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





async def main(data_file_pattern=None, sample_size=10, region=None):
    """
    Main function to build knowledge graph from ACLED conflict data.
    
    This script demonstrates the complete pipeline for building and refining a knowledge graph:
    1. Data Loading: Loads ACLED conflict data from any country/region
    2. Knowledge Graph Construction: Creates entities, relationships, and document nodes with metadata
    3. Entity Resolution: Uses the built-in SpaCy semantic matching resolver
    
    The resulting knowledge graph contains entities with proper relationships, ready for 
    downstream analysis and querying.
    
    Args:
        data_file_country (str, optional): Pattern to match ACLED data files. If None, uses first available file.
        sample_size (int, optional): Number of rows to process for testing. If None, processes all data.
        region (str, optional): Region code (currently not used - for future compatibility).
    """

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
        
        # Select file based on pattern or use first available
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
            df = df.tail(sample_size)
            print(f"Using sample of {len(df)} rows out of {original_size} total rows for testing")
        
        # Convert date column to string format for metadata
        if 'date' in df.columns:
            df = df.with_columns([
                pl.col('date').dt.strftime('%Y-%m-%d').alias('date')
            ])
        
        print(f"Loaded {len(df)} rows from ACLED data")
        print("Sample data columns:", df.columns[:5])  # Show first 5 columns
    
    except Exception as e:
        print(f"Error loading ACLED data: {e}")
        return []

    # ==================== 2. Run KG pipeline ====================

    # Initialize the standard KG construction pipeline
    kg_pipeline = KGConstructionPipeline()

    # Define metadata mapping for ACLED data (document properties additional 
    # to base field to dataframe columns)
    metadata_mapping = {
        "date": "date",           # Event date
        "domain": "domain",
        "url": "url" # if available
    }

    # Run the KG pipeline with the loaded data
    print("Starting Knowledge Graph construction with built-in entity resolution...")
    
    results = await kg_pipeline.run_async(
        df=df,
        document_base_field='item_id',
        text_column='text',
        document_metadata_mapping=metadata_mapping,
        document_id_column='item_id'  # Use item_id as document ID
    )

    print(f"Processed {len(results)} documents successfully.")
    print("Knowledge graph construction completed with built-in entity resolution.")
    return results

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build Knowledge Graph from ACLED conflict data with built-in entity resolution')
    parser.add_argument('--file-country', type=str, help='Pattern to match ACLED data files (e.g., "Sudan", "Mali", "2024")')
    parser.add_argument('--sample-size', type=int, help='Number of rows to process for testing (default: process all)')
    parser.add_argument('--list-files', action='store_true', help='List available ACLED data files and exit')
    
    args = parser.parse_args()
    
    # List files if requested
    if args.list_files:
        data_dir = os.path.join(Path(__file__).parent.parent.parent, 'data', 'acled')
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            print("Available ACLED data files:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"Data directory not found: {data_dir}")
        sys.exit(0)
    
    print("Starting ACLED Knowledge Graph Construction with Built-in Entity Resolution")
    print("=" * 80)
    
    # Run the main function with arguments
    results = asyncio.run(main(
        data_file_pattern=args.file_country,
        sample_size=args.sample_size,
        region=None  # No longer used
    ))
    
    print("=" * 80)
    if results:
        print(f"✅ SUCCESS: Processed {len(results)} documents.")
        print("Knowledge graph created with built-in entity resolution.")
    else:
        print("❌ FAILED: No documents were processed.")
    print("=" * 80)