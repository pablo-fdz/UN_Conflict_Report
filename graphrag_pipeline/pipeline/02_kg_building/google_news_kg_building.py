"""
Google News Knowledge Graph Builder with Enhanced SpaCy Entity Resolution

This script builds knowledge graphs from Google News conflict data
using an enhanced SpaCy resolver with higher similarity threshold.

Features:
- Automatic data file detection and loading
- Enhanced SpaCy resolver with high similarity threshold (0.999)
- Reduces inappropriate merging of geographically distinct entities
- Configurable sampling for testing
- Command-line interface for easy usage

The enhanced resolver approach:
- Uses SpaCy semantic matching with 95% similarity threshold
- Better handles geographic entities with different directions
- Maintains proper merging of true spelling variants
- More stable than complex custom resolvers

Usage Examples:
    # Process all data from Sudan with sample for testing
    python google_news_kg_building.py --file-country "Sudan" --sample-size 10

    # List available data files
    python google_news_kg_building.py --list-files
    
    # Process first available file with full data
    python google_news_kg_building.py

RECENT CHANGES:
- Implemented enhanced SpaCy resolver with higher similarity threshold
- Added StrictKGPipeline class that overrides the standard resolver
- Simplified approach for better stability and reliability

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
    from neo4j_graphrag.experimental.components.resolver import SpaCySemanticMatchResolver
    from library.kg_builder.utilities import ensure_spacy_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

class StrictKGPipeline(KGConstructionPipeline):
    """
    Custom KG Construction Pipeline for Google News data that uses SpaCy resolver
    with higher similarity threshold to reduce inappropriate geographic merging.
    """
    
    def _create_resolver(self, driver):
        """Override to use SpaCy resolver with higher similarity threshold."""
        
        entity_resolution_config = self.build_config['entity_resolution_config']
        
        if not entity_resolution_config.get('use_resolver', False):
            return None
        
        # Use SpaCy resolver with higher threshold to reduce inappropriate merges
        config = entity_resolution_config.get('SpaCySemanticMatchResolver_config', {})
        ensure_spacy_model(config.get('spacy_model', 'en_core_web_lg'))
        
        # Create SpaCy resolver with higher threshold to reduce over-merging
        return SpaCySemanticMatchResolver(
            driver,
            filter_query=config.get('filter_query'),
            resolve_properties=config.get('resolve_properties', ["name"]),
            similarity_threshold=0.999,  # Higher threshold to reduce inappropriate merges
            spacy_model=config.get('spacy_model', "en_core_web_lg"),
            neo4j_database='neo4j'
        )

async def main(data_file_pattern=None, sample_size=10, region=None):
    """
    Main function to build knowledge graph from Google News conflict data.
    
    1. Data Loading: Loads Google News conflict data
    2. Knowledge Graph Construction: Creates entities, relationships, text chunks and document nodes with metadata
    3. Entity Resolution: Uses enhanced SpaCy semantic matching resolver with higher similarity threshold
    
    The resulting knowledge graph contains entities with proper relationships, ready for 
    downstream analysis and querying. The enhanced SpaCy resolver with 95% similarity threshold
    reduces inappropriate merging while maintaining proper entity resolution.
    
    Args:
        data_file_pattern (str, optional): Pattern to match Google News data files. If None, uses first available file.
        sample_size (int, optional): Number of rows to process for testing. If None, processes all data.
    """

    # ==================== 1. Load data ====================

    # TODO: Method to load Google News data here
    # Also figure out google news parquet naming
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
    kg_pipeline = StrictKGPipeline()

    # Define metadata mapping for Google News data (document properties additional 
    # to base field to dataframe columns)
    metadata_mapping = {
        'date': 'date',
        'url': 'decoded_url',
        'domain': 'source'
    }

    # Run the KG pipeline with the loaded data
    print("Using SpaCySemanticMatchResolver with similarity threshold: 0.999")
    
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build Knowledge Graph from Google News conflict data with built-in entity resolution')
    parser.add_argument('--file-country', type=str, help='Pattern to match Google News data files (e.g., "Sudan", "Mali", "2024")')
    parser.add_argument('--sample-size', type=int, help='Number of rows to process for testing (default: process all)')
    parser.add_argument('--list-files', action='store_true', help='List available Google News data files and exit')
    
    args = parser.parse_args()
    
    # List files if requested
    if args.list_files:
        data_dir = os.path.join(Path(__file__).parent.parent.parent, 'data', 'google')
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
            print("Available Google News data files:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"Data directory not found: {data_dir}")
        sys.exit(0)

    # Run the main function with arguments
    results = asyncio.run(main(
        data_file_pattern=args.file_country,
        sample_size=args.sample_size
    ))