import json
import sys
from datetime import datetime
from pathlib import Path 

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "library"))

from data_ingestor.google import GoogleNewsIngestor

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

    start_date = config.get('start_date')
    end_date = config.get('end_date')

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