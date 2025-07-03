"""
Google News Knowledge Graph Builder

This script builds knowledge graphs from Google News conflict data.

Features:
- Automatic data file detection and loading
- Configurable sampling for testing
- Command-line interface for easy usage

Author: Generated for UN Conflict Report project
"""

import os
import sys
import asyncio
import polars as pl
from pathlib import Path
import json

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
    Main function to build knowledge graph from Google News conflict data.
    
    1. Data Loading: Loads Google News conflict data
    2. Knowledge Graph Construction: Creates entities, relationships, text chunks and document nodes with metadata
    """

    base_config_path = Path(__file__).parent.parent.parent / 'config_files'
    config_path = base_config_path / 'data_ingestion_config.json'

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('google_news', {})
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

    google_news_data_dir = graphrag_pipeline_dir / 'data' / 'google_news'
    df = None  # Placeholder for actual DataFrame loading logic
    if df['date'].dtype == pl.Date:
        df = df.with_columns(pl.col('date').cast(pl.String))

    try:
        pass  # Placeholder for Google News data loading logic
    except Exception as e:
        print(f"Error loading Google News data: {e}")
        return []

    # ==================== 2. Run KG pipeline ====================

    # Initialize the custom Google News KG construction pipeline 
    # This uses enhanced SpaCy resolver with higher similarity threshold
    kg_pipeline = KGConstructionPipeline()

    # Define metadata mapping for Google News data (document properties additional 
    # to base field to dataframe columns)
    metadata_mapping = {
        'date': 'date',
        'url': 'decoded_url',
        'domain': 'source'
    }
    
    results = await kg_pipeline.run_async(
        df=df,
        document_base_field='id',
        text_column='full_text',
        document_metadata_mapping=metadata_mapping,
        document_id_column='item_id'  # Use item_id as document ID
    )

    print(f"Processed {len(results)} documents successfully.")
    return results

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    # Run the main function with arguments
    results = asyncio.run(main())