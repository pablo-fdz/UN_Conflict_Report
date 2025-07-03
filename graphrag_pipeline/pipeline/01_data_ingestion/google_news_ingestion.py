import json
import sys
from datetime import datetime
from pathlib import Path 

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if graphrag_pipeline_dir not in sys.path:
    sys.path.append(graphrag_pipeline_dir)

from library.data_ingestor.google_news_ingestor import GoogleNewsIngestor
from library.data_ingestor.utilities import date_range_converter

def load_config():
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir}")  # Debug print
    # Correct path: go up two levels to graphrag_pipeline, then into config_files
    config_path = script_dir.parent.parent / 'config_files' / 'data_ingestion_config.json'
    print(f"Looking for config at: {config_path}")  # Debug print
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            config = config.get('google_news', {})
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")

def main():

    config = load_config()
    country = config.get('country')  

    print(f"Fetching Google News data for {country}...")

    time_range = config.get('ingestion_date_range', '2 months')
    start_date, end_date = date_range_converter(time_range)

    query_language = config.get('query', {}).get('language', 'en')
    query_country = config.get('query', {}).get('country', 'US')

    print(f'Start date: {start_date}, End date: {end_date}, Country: {country}')
    print('This takes a while... (20 min aprox)')

    gn_ingestor = GoogleNewsIngestor(country, start_date, end_date, query_language, query_country)

    gn_ingestor.get_google_news_data()
    gn_ingestor.print_query_summary()
    gn_ingestor.process_data()
    gn_ingestor.print_urls_and_texts_summary()
    gn_ingestor.save_data()

    print("Google News data files saved.")

if __name__ == "__main__":
    main()