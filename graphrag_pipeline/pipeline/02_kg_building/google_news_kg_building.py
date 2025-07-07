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

os.environ['PYTHONIOENCODING'] = 'utf-8'

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
            config = config.get('google_news', {})
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
            print(f"Invalid sample size '{sample_size_str}'. The `sample_size` parameter should be a string containing an integer or `all` to process all data. Processing all data.")
            sample_size = None

    # Get the country for which to build the knowledge graph
    country = os.getenv('KG_BUILDING_COUNTRY')
    if not country:
        raise ValueError("Country not specified. Set KG_BUILDING_COUNTRY environment variable.")

    # ==================== 1. Load data ====================

    google_news_data_dir = graphrag_pipeline_dir / 'data' / 'google_news'
    all_results = []

    try:
        # Find Google News data files
        if not google_news_data_dir.exists():
            raise FileNotFoundError(f"Google News data directory not found: {google_news_data_dir}")

        # Get list of available files
        available_files = [f for f in os.listdir(google_news_data_dir) if f.endswith('.parquet')]
        
        if not available_files:
            raise FileNotFoundError(f"No Google News parquet files found in: {google_news_data_dir}")

        # Join the "country" with underscores if there is spacing
        data_file_pattern = country.replace(' ', '_')

        # Select file based on pattern (name of a country)
        matching_files = [f for f in available_files if data_file_pattern.lower() in f.lower()]
        
        if not matching_files:
            print(f"No files matching country '{data_file_pattern}' found.")
            print(f"Available files: {available_files}")
            raise FileNotFoundError(f"No files matching country: {data_file_pattern}")

        # Sort files by the end date in the filename, descending (most recent first)
        try:
            sorted_files = sorted(
                matching_files,
                key=lambda f: f.removesuffix('.parquet').split('_')[-1],
                reverse=True
            )
        except IndexError:
            print("Warning: Could not sort files by date due to unexpected filename format. Processing in default order.")
            sorted_files = matching_files

        # ==================== 2. Run KG pipeline ====================

        # Initialize the custom Google News KG construction pipeline 
        kg_pipeline = KGConstructionPipeline()

        # Define metadata mapping for Google News data (document properties additional 
        # to base field to dataframe columns)
        metadata_mapping = {
            'date': 'date',
            'url': 'decoded_url',
            'domain': 'source'
        }
        
        # Loop through all matching files
        for file_name in sorted_files:
            try:
                df_path = google_news_data_dir / file_name
                print(f"Loading and processing data from: {file_name}")
                
                df = pl.read_parquet(df_path)

                # Apply sampling if specified
                if sample_size:
                    original_size = len(df)
                    df = df.sample(sample_size, seed=42)
                    print(f"Using sample of {len(df)} rows out of {original_size} total rows for testing")

                # Debug: Check data types and ensure all columns are strings
                print(f"DataFrame columns: {df.columns}")
                print(f"DataFrame dtypes: {df.dtypes}")
                
                # Convert all metadata columns to strings to avoid WindowsPath issues
                if 'decoded_url' in df.columns:
                    df = df.with_columns(
                        pl.col('decoded_url').map_elements(
                            lambda x: str(x) if x is not None else None,
                            return_dtype=pl.Utf8
                        ).alias('decoded_url')
                    )
                
                if 'source' in df.columns:
                    df = df.with_columns(
                        pl.col('source').map_elements(
                            lambda x: str(x) if x is not None else None,
                            return_dtype=pl.Utf8
                        ).alias('source')
                    )

                # Convert date column to string format for metadata
                if 'date' in df.columns and df['date'].dtype == pl.Date:
                    df = df.with_columns(pl.col('date').cast(pl.String))

                # Ensure full_text column is UTF-8 string
                if 'full_text' in df.columns:
                    df = df.with_columns(pl.col('full_text').cast(pl.Utf8))

                # Debug: Print sample data
                print("Sample of metadata columns:")
                if len(df) > 0:
                    sample_row = df.head(1)
                    for col in ['date', 'decoded_url', 'source']:
                        if col in df.columns:
                            value = sample_row[col].to_list()[0]
                            print(f"  {col}: {value} (type: {type(value)})")


                # ==================== 2. Run KG pipeline for the current file ====================
                
                results = await kg_pipeline.run_async(
                    df=df,
                    document_base_field='id',
                    text_column='full_text',
                    document_metadata_mapping=metadata_mapping,
                    document_id_column=None
                )

                print(f"Processed {len(results)} documents from {file_name} successfully.")
                all_results.extend(results)

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue # Continue to the next file

    except Exception as e:
        print(f"Error during data loading phase: {e}")
        return []

    print(f"Total processed documents across all files: {len(all_results)}")
    return all_results

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    # Run the main function with arguments
    results = asyncio.run(main())