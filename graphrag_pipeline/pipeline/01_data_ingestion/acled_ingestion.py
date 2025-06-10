"""
ACLED data ingestion script (example, modify as needed). However, def main() is necessary to run the script as a standalone program.
This script retrieves data from the ACLED API, processes it, and stores it in the appropriate location.
"""

import os
import sys
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

def load_config():
    """Load configuration settings for ACLED data ingestion."""
    
    # Get the directory path of the current script
    script_dir = Path(__file__).parent

    # Navigate to config_files directory
    config_path = Path(script_dir).parent.parent / 'config_files' / 'data_ingestion_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('acled', {})
    except FileNotFoundError:
        raise(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise(f"Error parsing configuration file at {config_path}")

def fetch_acled_data(api_key, start_date, end_date, countries=None):
    """
    Fetch data from ACLED API.
    
    Args:
        api_key: ACLED API key
        start_date: Start date for data retrieval (YYYY-MM-DD)
        end_date: End date for data retrieval (YYYY-MM-DD)
        countries: List of country codes to retrieve data for
        
    Returns:
        DataFrame containing ACLED data
    """
    
    # Example API request logic
    url = "https://api.acleddata.com/acled/read"
    params = {
        "key": api_key,
        "email": "your_registered_email@example.com",
        "start_date": start_date,
        "end_date": end_date,
        "event_date_format": "YYYY-MM-DD",
        "format": "json"
    }
    
    if countries:
        params["countries"] = ",".join(countries)
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        return df
    
    except requests.RequestException as e:
        raise(f"Error fetching data from ACLED API: {str(e)}")

def process_data(df):
    """
    Process and clean the ACLED data.
    
    Args:
        df: Raw DataFrame from ACLED API
        
    Returns:
        Processed DataFrame
    """
    
    if df.empty:
        raise("No data to process")
        return df
    
    try:
        # Example processing steps
        # 1. Drop duplicates
        df = df.drop_duplicates()
        
        # 2. Convert date columns
        date_cols = ['event_date', 'timestamp']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # 3. Other cleaning steps
        # ...
        
        print(f"Data processing complete. {len(df)} records after processing.")
        return df
    
    except Exception as e:
        raise(f"Error during data processing: {str(e)}")

def save_data(df, output_path=None):
    """
    Save the processed data.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the data
    """
    if df.empty:
        raise("No data to save")
    
    try:
        # Determine output path if not provided
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = Path(script_dir).parent.parent / 'data' / 'acled'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"acled_data_{timestamp}.csv"
        
        # Save the data
        df.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path}")
        
        # Also save a copy as processed_data.csv for KG building step
        processed_path = Path(output_path).parent / "processed_data.csv"
        df.to_csv(processed_path, index=False)
        print(f"Data also saved as {processed_path} for further processing")
        
    except Exception as e:
        raise(f"Error saving data: {str(e)}")

# Necessary defining main() for the program to run as a script
def main():
    """Main function to run the ACLED data ingestion pipeline."""
    print("Starting ACLED data ingestion")
    
    # Load configuration
    config = load_config()
    if not config:
        raise("Failed to load configuration. Aborting ingestion.")
    
    # Extract parameters from config
    api_key = config.get('api_key')
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    countries = config.get('countries')
    
    if not api_key:
        raise("API key not found in configuration. Aborting ingestion.")
    
    # Execute the ingestion pipeline
    raw_data = fetch_acled_data(api_key, start_date, end_date, countries)
    processed_data = process_data(raw_data)
    save_data(processed_data)
    
    print("ACLED data ingestion completed successfully")

# Necessary to run the main function when this script is executed
if __name__ == "__main__":
    main()