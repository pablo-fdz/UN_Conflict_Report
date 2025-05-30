# Utilities
from typing import Dict, List, Tuple, Any, Optional, Union
import time
import polars as pl
from .custom_kg_pipeline import CustomKGPipeline

# Neo4j and Neo4j GraphRAG imports
from neo4j_graphrag.experimental.pipeline.pipeline import PipelineResult

async def build_kg_from_df(
    kg_pipeline: CustomKGPipeline,
    df: Union[pl.DataFrame, pl.LazyFrame],
    document_base_field: str,
    text_column: str = 'text',
    document_metadata_mapping: Optional[Dict[str, str]] = None,
    document_id_column: Optional[Any] = None,
) -> List[PipelineResult]:
    """Process a dataframe through the KG pipeline row by row.
    
    Args:
        kg_pipeline: An initialized CustomKGPipeline instance
        df: The polars dataframe to process
        document_base_field: Field from which to create the document node in the lexical graph
        text_column: The column containing the text to process
        document_id_column: Optional column to use as document ID
        document_metadata_mapping: Optional mapping of document property names to dataframe columns
    
    Returns:
        List of pipeline results, one per row
    """
    
    results = []
    row_counter = 0
    start_time = time.time()
    
    # Iterate over rows in the dataframe
    for row in df.iter_rows(named=True):
        # Print progress
        row_counter += 1
        print(f"Processing row {row_counter} of {df.shape[0]}")

        # Get the text from the row
        text = row[text_column]
        
        # Only process if text is not None or empty
        if text and text.strip():

            # Process document metadata: iterate over the document_metadata dictionary 
            # to extract values from each row and populate the processed_metadata dictionary
            processed_metadata = {}  # Initialize metadata as empty dictionary
            if document_metadata_mapping:  # Check if metadata mapping is provided
                for prop_name, column_name in document_metadata_mapping.items():
                    if column_name in row:
                        processed_metadata[prop_name] = row[column_name]
                    else:
                        print(f"Warning: Column '{column_name}' not found in row data. Skipping metadata property '{prop_name}'.")
            
            # Get the base field of the document
            doc_base_field = row[document_base_field]

            # Get document ID if column specified
            doc_id = row.get(document_id_column) if document_id_column else None
            
            # Process the text with the pipeline
            result = await kg_pipeline.run_async(
                text=text,
                document_base_field=doc_base_field,
                document_metadata=processed_metadata, 
                document_id=doc_id
            )
            results.append(result)
            print(f"Result: {result}")
        else:
            print(f"Skipping row {row_counter} due to empty text")
            results.append(None)

        # Update time tracking
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {(elapsed_time / row_counter) * (df.shape[0] - row_counter):.2f} seconds\n")
    
    return results