# Utilities
from typing import Any
import polars as pl
import datetime
import json

def to_string(value: Any) -> str:
    """Convert various data types to string format appropriately.
    
    Args:
        value: Any value to convert to string
        
    Returns:
        String representation of the value
    """
    if value is None:
        return ""
    
    # Handle date/time objects
    if isinstance(value, (pl.Date, datetime.date)):
        return value.isoformat()
    elif isinstance(value, datetime.datetime):
        return value.isoformat()
    elif isinstance(value, datetime.time):
        return value.isoformat()
    
    # Handle numeric types with potential formatting issues
    elif isinstance(value, float):
        # Avoid scientific notation for small/large numbers
        return f"{value:.10g}"
    
    # Handle collection types by converting to JSON
    elif isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)
    
    # Handle boolean values explicitly
    elif isinstance(value, bool):
        return str(value).lower()
    
    # Default: standard string conversion
    return str(value)
