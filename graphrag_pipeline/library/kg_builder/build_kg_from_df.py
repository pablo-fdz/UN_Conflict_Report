# Utilities
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import time
import polars as pl
import uuid
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
    rate_limit_checker: Optional[Callable] = None
) -> Tuple[List[PipelineResult], int]:
    """Process a dataframe through the KG pipeline row by row.
    
    Args:
        kg_pipeline: An initialized CustomKGPipeline instance
        df: The polars dataframe to process
        document_base_field: Field from which to create the document node in the lexical graph
        text_column: The column containing the text to process
        document_metadata_mapping: Optional mapping of document property names to dataframe columns
        document_id_column: Optional column to use as document ID. A random UUID will be generated if not provided.
        rate_limit_checker: Optional function to check and enforce rate limits before LLM calls.
    
    Returns:
        tuple: List of pipeline results, one per row; and the number of LLM calls made.
    """
    
    results = []
    llm_calls = 0
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
                        value = row[column_name]
                        if value is None:
                            processed_metadata[prop_name] = ""
                        elif not isinstance(value, str):
                            raise TypeError(f"Metadata fields must be converted to strings before passing into pipeline. Expected string value for column '{column_name}', got {type(value).__name__}")
                        else:
                            processed_metadata[prop_name] = value
                    else:
                        print(f"Warning: Column '{column_name}' not found in row data. Skipping metadata property '{prop_name}'.")
            
            # Get the base field of the document
            doc_base_field = row[document_base_field]

            # Get document ID if column specified
            doc_id = row.get(document_id_column) if document_id_column else str(uuid.uuid4())
            
            # Check rate limit before making an LLM call
            if rate_limit_checker:
                rate_limit_checker()

            # Process the text with the pipeline
            result = await kg_pipeline.run_async(
                text=text,
                document_base_field=doc_base_field,
                document_metadata=processed_metadata, 
                document_id=doc_id
            )
            results.append(result)
            llm_calls += 1  # Increment LLM call count
            print(f"Result: {result}")
        else:
            print(f"Skipping row {row_counter} due to empty text")
            results.append(None)

        # Update time tracking
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {(elapsed_time / row_counter) * (df.shape[0] - row_counter):.2f} seconds\n")
    
    return results, llm_calls