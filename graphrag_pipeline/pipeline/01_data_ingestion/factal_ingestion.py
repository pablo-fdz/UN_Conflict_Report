"""
Factal data ingestion script.
This script retrieves data via the Factal API, processes it, and stores it in "graphrag_pipeline/data" as "Factal_{name of the country}_{date}.parquet".
"""

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from dotenv import find_dotenv, load_dotenv

def load_config():
    script_dir = Path(__file__).parent
    config_path = script_dir.parent.parent / 'config_files' / 'data_ingestion_config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('factal', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")

print("Fetching Factal data...")

def get_factal_data(
    country,
    start_date=None,
    end_date=None,
    limit=None 
):
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv('FACTAL_API_KEY')
    country = country.capitalize()

    def get_id(name, kind="location", category="Country"):
        r = requests.get(
            "https://www.factal.com/api/v2/topic/",
            params={"name": name, "kind": kind, "category": category},
            headers={'Authorization': f'Token {api_key}'}
        )
        if r.ok:
            results = r.json().get('results', [])
            return results[0]['id'] if results else None
        return None

    topic_id = get_id(country, kind="location", category="Country")
    if not topic_id:
        print(f"Topic ID for {country} not found.")
        return pl.DataFrame()

    url = 'https://www.factal.com/api/v2/item/'
    params = {
        'topics': str(topic_id),
        'kind': "location",
        'category': "Country",
        'page_size': 100,
        'limit': None,
    }
    if start_date: params['date__gte'] = start_date
    if start_date and end_date: params['date__range'] = f"{start_date},{end_date}"

    headers = {'Authorization': f'Token {api_key}'}
    results, next_url, total = [], url, 0

    while next_url:
        r = requests.get(next_url, headers=headers, params=params)
        if not r.ok:
            break
        data = r.json()
        items = data.get('results', [])
        results.extend(items)
        total += len(items)
        if limit and total >= limit:
            results = results[:limit]
            break
        next_url = data.get('next')
        params = {}

    return pl.DataFrame(results)


def process_data(results, country):
    if results.is_empty():
        raise ValueError("No data to process")
    results = results.unique()
    for col in ['event_date', 'timestamp']:
        if col in results.columns:
            results = results.with_columns(pl.col(col).str.strptime(pl.Datetime, strict=False))
    
    def extract_topics_to_df(df, column):
        if column not in df.columns:
            return pl.DataFrame()
        all_topics = [
            {**dict(topic), 'item_id': row['id']}
            for row in df.iter_rows(named=True) if row[column]
            for topic in row[column]
        ]
        if not all_topics:
            return pl.DataFrame()
        topics_df = pl.DataFrame(all_topics)
        if 'topic' in topics_df.columns:
            topics_flat = pl.DataFrame(topics_df['topic'].to_list())
            topics_flat = topics_flat.with_columns(topics_df['item_id'])
            return topics_flat.select(['item_id'] + [c for c in topics_flat.columns if c != 'item_id'])
        return topics_df

    def fill_categories(df):
        return df.with_columns([
            pl.when(pl.col("category") == "Country").then(pl.col("topic")).otherwise(None).alias("country"),
            pl.when(pl.col("category") == "State").then(pl.col("topic")).otherwise(None).alias("state"),
            pl.when(pl.col("category") == "Town").then(pl.col("topic")).otherwise(None).alias("town"),
            pl.when(pl.col("category") == "POI").then(pl.col("topic")).otherwise(None).alias("location"),  # Changed from "poi" to "location"
            pl.when(pl.col("kind") == "arc").then(pl.col("topic")).otherwise(None)
                .str.replace(r" at \w+\+\w+, ", " at ")
                .str.replace(r"^\w+\+\w+\s", "").alias("topic2"),
            pl.when(pl.col("kind") == "vertical").then(pl.col("topic")).otherwise(None).alias("theme"),
            pl.when(pl.col("kind") == "tag").then(pl.col("topic")).otherwise(None).alias("tag"),
        ])

    def group_by_factal_id(df):
        def extract_country(state, country):
            if state and isinstance(state, str) and ',' in state:
                m = re.search(r',\s*([^,]+)$', state)
                return m.group(1).strip() if m else country
            return country

        result_rows = []
        for item_id in df['item_id'].unique():
            id_data = df.filter(pl.col("item_id") == item_id)
            first = id_data.row(0, named=True)
            row = {k: first[k] for k in ["item_id", "url", "text", "domain", "date", "severity"]}
            
            # Only check for columns that exist in the DataFrame
            for col in ["country", "state", "town", "location", "topic2", "theme", "tag"]:
                if col in df.columns:
                    vals = id_data.select(pl.col(col)).filter(pl.col(col).is_not_null()).unique().to_series().to_list()
                    row[col] = vals[0] if vals else None
                else:
                    row[col] = None
                    
            row["country"] = extract_country(row.get("state"), row.get("country"))
            row["topics"] = id_data.select(pl.col("topic")).unique().to_series().to_list()
            summaries = id_data.select(pl.col("topic_summary")).filter(pl.col("topic_summary").is_not_null()).to_series().to_list()
            row["topic_summary"] = summaries[0] if summaries else None
            result_rows.append(row)
        return pl.DataFrame(result_rows)

    topics_df = extract_topics_to_df(results, "topics")
    items_merged = results.join(topics_df, left_on='id', right_on='item_id', how='left')

    items_merged = items_merged.with_columns([
        pl.when((pl.col('url') == '') & (pl.col('url_domain') == 'x.com'))
          .then(pl.col('source') + pl.lit('/status/') + pl.col('tweet_id').cast(pl.Utf8))
          .otherwise(pl.col('url')).alias('url'),
        (pl.lit("factal_") + pl.col("id").cast(pl.Utf8)).alias("item_id")
    ])

    clean_df = items_merged.select([
        'item_id',
        'url',
        pl.col('content').alias('text'),
        pl.col('source').alias('domain'),
        pl.col('date').str.to_datetime().dt.date(),
        'severity',
        pl.col('name').alias('topic'),
        'kind',
        'category',
        pl.col('description').alias('topic_summary')
    ])

    clean_df = fill_categories(clean_df)
    processed_data = group_by_factal_id(clean_df)
    
    # Create conditional location prefix based on availability of state/town/country
    processed_data = processed_data.with_columns([
        pl.when(pl.col("state").is_not_null())
        .then(pl.concat_str([pl.lit("State, country: "), pl.col("state")]))
        .when(pl.col("town").is_not_null())
        .then(pl.concat_str([pl.lit("Town, country: "), pl.col("town")]))
        .when(pl.col("country").is_not_null())
        .then(pl.concat_str([pl.lit("Country: "), pl.col("country")]))
        .otherwise(pl.lit("Location: N/A"))
        .alias("location_prefix")
    ])
    
    # Concatenate the metadata to the text column
    processed_data = processed_data.with_columns([
        pl.concat_str([
            pl.col("location_prefix"),
            pl.lit(". Theme: "),
            pl.col("theme").fill_null("N/A"),
            pl.lit(". Tag: "),
            pl.col("tag").fill_null("N/A"),
            pl.lit(". Topics: "),
            pl.col("topics").list.join(", ").fill_null("N/A"),
            pl.lit(". Text: "),
            pl.col("text")
        ]).alias("text")
    ])
    
    # Remove the temporary location_prefix column
    processed_data = processed_data.drop("location_prefix")
    
    processed_data = processed_data.with_columns(pl.lit(country).alias("country_keyword"))
    processed_data = processed_data.rename({"topic2": "topic"})
    processed_data = processed_data.select(["country_keyword"] + [c for c in processed_data.columns if c != "country_keyword"])
    return processed_data


def save_data(processed_data, country, start_date, end_date):
    if processed_data.is_empty():
        raise ValueError("No data to save.")
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / 'data' / 'factal'
    output_dir.mkdir(parents=True, exist_ok=True)
    prev_day = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    date = f'{start_date}_{prev_day}' if start_date and end_date else f'generated_{datetime.now().strftime('%Y-%m-%d')}'
    output_path = output_dir / f"Factal_{country}_{date}.parquet"
    processed_data.write_parquet(output_path)

def main():
    config = load_config()
    if not config:
        raise ValueError("Failed to load configuration. Aborting ingestion.")
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    country = config.get('country')
    raw_data = get_factal_data(country, start_date=start_date, end_date=end_date)
    processed_data = process_data(raw_data, country)
    save_data(processed_data, country, start_date, end_date)
    print("Factal data file saved.")

if __name__ == "__main__":
    main()