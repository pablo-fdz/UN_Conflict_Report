"""
ACLED data ingestion script.
This script retrieves data via the ACLED API, processes it, and stores it in "graphrag_pipeline/data" as "acled_{name of the country}_{date}.parquet".
"""

import sys
from pathlib import Path

# Add the parent directory (graphrag_pipeline) to the Python path (needed for importing
# modules in parent directory)
script_dir = Path(__file__).parent  # Get the directory where this script is located
graphrag_pipeline_dir = script_dir.parent.parent  # Get the graphrag_pipeline directory
if str(graphrag_pipeline_dir) not in sys.path:
    sys.path.append(str(graphrag_pipeline_dir))

import io
import json
import os
from datetime import datetime
from typing import Optional
from library.data_ingestor.utilities import date_range_converter

import polars as pl
import requests
from dotenv import find_dotenv, load_dotenv

def load_config():
    base_config_path = Path(__file__).parent.parent.parent / 'config_files'
    config_path = base_config_path / 'data_ingestion_config.json'
    load_dotenv(os.path.join(base_config_path, '.env'), override=True)
    try:
        return json.loads(config_path.read_text()).get('acled', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")

def get_acled_data(
    country: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pl.DataFrame:

    # load_dotenv(find_dotenv(), override=True)
    country = country.capitalize()
    email = email or os.getenv("ACLED_EMAIL")
    api_key = api_key or os.getenv("ACLED_API_KEY")
    if not (email and api_key):
        raise ValueError("ACLED credentials missing: set email & api_key.")

    params = {
        "email": email,
        "key": api_key,
        "country": country,
        "format": "csv",
        "limit": 0
    }
    if start_date and end_date:
        params.update({"event_date": f"{start_date}|{end_date}", "event_date_where": "BETWEEN"})
    elif start_date:
        params.update({"event_date": start_date, "event_date_where": ">="})
    elif end_date:
        params.update({"event_date": end_date, "event_date_where": "<="})

    r = requests.get("https://api.acleddata.com/acled/read.csv", params=params, timeout=60)
    r.raise_for_status()
    results = pl.read_csv(io.BytesIO(r.content))
    return results.head(limit) if limit else results

def process_data(results, country):
    if results.is_empty():
        raise ValueError("No data to process")

    results = results.with_columns([
        pl.when(pl.col("tags") == "crowd size=no report").then(None).otherwise(pl.col("tags")).alias("tags"),
        (pl.lit("acled_") + pl.col("event_id_cnty").cast(pl.Utf8)).alias("item_id")
    ])

    processed_data = results.select([
        "item_id",
        pl.col("event_date").str.to_datetime().dt.date().alias("date"),
        pl.col("notes").alias("text"),
        # Modified domain column to append ACLED reporting information
        pl.concat_str([
            pl.col("source"),
            pl.lit(", recorded by ACLED on "),
            pl.col("event_date").str.to_datetime().dt.strftime("%d %B %Y")
        ]).alias("domain"),
        "event_type", "sub_event_type", "actor1", "assoc_actor_1", "actor2", "assoc_actor_2", "interaction",
        "country",
        pl.col("admin1").alias("state"),
        pl.col("admin2").alias("town"),
        "location", "fatalities", "tags"
    ])

    processed_data = processed_data.with_columns([
        pl.when(pl.col("country").is_not_null())
        .then(pl.concat_str([
            pl.lit("Country: "), pl.col("country"),
            pl.lit(". State: "), pl.col("state"),
            pl.lit(". Town: "), pl.col("town"), pl.lit(". ")
        ])).otherwise(pl.lit("Location: N/A. ")).alias("location_prefix")
    ])

    actor_expr = lambda actor, assoc: pl.when(pl.col(actor).is_not_null()).then(
        pl.col(actor) +
        pl.when(pl.col(assoc).is_not_null())
        .then(pl.concat_str([pl.lit(" (associated with: "), pl.col(assoc), pl.lit(")")]))
        .otherwise(pl.lit(""))
    ).otherwise(pl.lit(""))

    actors_prefix_expr = pl.concat_str([
        pl.when(pl.col("interaction").is_not_null())
        .then(pl.concat_str([pl.lit(" Opposing sides: "), pl.col("interaction"), pl.lit(".")]))
        .otherwise(pl.lit("")),
        pl.when(pl.col("actor1").is_not_null())
        .then(pl.concat_str([pl.lit(" Actor 1: "), actor_expr("actor1", "assoc_actor_1"), pl.lit(".")]))
        .otherwise(pl.lit("")),
        pl.when(pl.col("actor2").is_not_null())
        .then(pl.concat_str([pl.lit(" Actor 2: "), actor_expr("actor2", "assoc_actor_2"), pl.lit(".")]))
        .otherwise(pl.lit(""))
    ])

    processed_data = processed_data.with_columns(
        pl.when(
            pl.any_horizontal(
                pl.col("interaction").is_not_null(),
                pl.col("actor1").is_not_null(),
                pl.col("actor2").is_not_null(),
                pl.col("assoc_actor_1").is_not_null(),
                pl.col("assoc_actor_2").is_not_null(),
            )
        )
        .then(actors_prefix_expr)
        .otherwise(pl.lit(" Participants: N/A."))
        .alias("actors_prefix")
    )

    processed_data = processed_data.with_columns([
        pl.when(pl.col("event_type").is_not_null())
        .then(pl.concat_str([
            pl.lit(" Type of event: "), pl.col("event_type"),
            pl.lit(" ("), pl.col("sub_event_type"), pl.lit(").")
        ]))
        .otherwise(pl.lit(" Event type: N/A."))
        .alias("event_prefix")
    ])

    processed_data = processed_data.with_columns(
        pl.concat_str([
            pl.col("date").dt.strftime("On %d %B %Y. "),
            pl.col("location_prefix"),
            pl.lit("Text: "),
            pl.col("text"),
            pl.lit(" Number of fatalities: "),
            pl.col("fatalities").fill_null(0).cast(pl.Utf8 ).replace("", "0"),
            pl.lit(". "),
            pl.col("actors_prefix"),
            pl.col("event_prefix"),
            pl.when(pl.col("tags").is_not_null())
            .then(pl.concat_str([pl.lit(" Number of participants: "), pl.col("tags")]))
            .otherwise(pl.lit("")),
            pl.lit(".")
        ])
        .alias("text")
    )

    processed_data = processed_data.drop(["location_prefix", "actors_prefix", "event_prefix"])
    processed_data = processed_data.with_columns(pl.lit(country).alias("country_keyword"))
    return processed_data.select(["country_keyword"] + [c for c in processed_data.columns if c != "country_keyword"])

def save_data(processed_data, country, start_date, end_date):
    if processed_data.is_empty():
        raise ValueError("No data to save.")
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'acled'
    output_dir.mkdir(parents=True, exist_ok=True)
    date = f'{start_date}_{end_date}' if start_date and end_date else f'generated_{datetime.now():%Y-%m-%d}'
    output_path = output_dir / f"Acled_{country}_{date}.parquet"
    processed_data.write_parquet(output_path)

def main():
    config = load_config()
    country = os.getenv('GRAPHRAG_INGEST_COUNTRY')
    if not country:
        raise ValueError("Country not specified. Set GRAPHRAG_INGEST_COUNTRY environment variable.")
    print(f"Fetching ACLED data for {country}...")
    if not config:
        raise ValueError("Failed to load configuration. Aborting ingestion.")
    time_range = config.get('ingestion_date_range', '2 months')
    start_date, end_date = date_range_converter(time_range)
    raw_data = get_acled_data(country, start_date=start_date, end_date=end_date)
    processed_data = process_data(raw_data, country)
    save_data(processed_data, country, start_date, end_date)
    print("ACLED data file saved.")

if __name__ == "__main__":
    main()