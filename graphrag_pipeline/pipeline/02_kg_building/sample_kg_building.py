import os
import sys
import asyncio
import polars as pl
from .kg_construction_pipeline import KGConstructionPipeline

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory where this script is located
parent_dir = os.path.dirname(os.path.dirname(script_dir))  # Get the parent directory (graphrag_pipeline)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

async def main():

    # ==================== 1. Load data ====================

    try:
        # Load data
        df_path = os.path.join(parent_dir, 'example_notebooks', 'sample_data', 'factal_single_topic_report-2025-05-01-2025-06-05.csv')
        df = pl.read_csv(df_path)

        # Create an index for each row
        df = df.with_row_index(name="id", offset=1)

        # Convert the "id" to a string to ensure it is treated as a document ID
        df = df.with_columns(pl.col('id').cast(pl.String))
    
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return

    # ==================== 2. Run KG pipeline ====================

    # Initialize the KG construction pipeline
    kg_pipeline = KGConstructionPipeline()

    # Define metadata mapping for sample data(document properties additional 
    # to base field to dataframe columns)
    metadata_mapping = {
        "source": "Source URL",
        "published_date": "Published date"
    }

    # Run the KG pipeline with the loaded data
    results = await kg_pipeline.run_async(
        df=df,
        document_base_field='id',
        text_column='Published text',
        document_metadata_mapping=metadata_mapping,
        document_id_column=None  # Use default document ID generation
    )

    return results

# Asyncio event loop to run the main function in a script
if __name__ == "__main__":
    results = asyncio.run(main())
    print(f"Processed {len(results)} documents")