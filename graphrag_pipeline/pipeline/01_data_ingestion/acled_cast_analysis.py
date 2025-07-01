"""
ACLED CAST (Conflict Alert and Surveillance Tool) analysis script.
This script retrieves ACLED CAST forecasting data, processes it to identify conflict hotspots,
creates visualizations, and saves outputs to "graphrag_pipeline/data/images".
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pycountry
import requests
import urllib.request
from dateutil.relativedelta import relativedelta
from dotenv import find_dotenv, load_dotenv
from shapely.geometry import Point


def load_config():
    """Load configuration from data_ingestion_config.json"""
    config_path = Path(__file__).parent.parent.parent / 'config_files' / 'data_ingestion_config.json'
    try:
        return json.loads(config_path.read_text()).get('acled_cast', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")


def get_acled_cast_data(country: str, email: Optional[str] = None, api_key: Optional[str] = None) -> pl.DataFrame:
    """
    Retrieve ACLED CAST forecasting data for a specific country.
    
    Args:
        country: Country name
        email: ACLED API email
        api_key: ACLED API key
    
    Returns:
        Polars DataFrame with CAST data
    """
    load_dotenv(find_dotenv(), override=True)
    
    email = email or os.getenv("ACLED_EMAIL")
    api_key = api_key or os.getenv("ACLED_API_KEY")
    if not (email and api_key):
        raise ValueError("ACLED credentials missing: set email & api_key.")
    
    # Set up API parameters
    parameters = {
        "email": email,
        "key": api_key,
        "country": country,
    }
    
    # GET request to ACLED CAST API
    response = requests.get("https://api.acleddata.com/cast/read.csv", params=parameters, timeout=60)
    response.raise_for_status()
    
    cast = pl.read_csv(io.BytesIO(response.content))
    
    # Filter for exact country match
    cast = cast.filter(pl.col("country") == country)
    
    # Process month names to numbers
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    
    cast = (
        cast.with_columns(pl.col("month").replace(month_map).alias("month_num"))
        .with_columns(
            (pl.col("year").cast(str) + "-" + pl.col("month_num").cast(str).str.zfill(2)).alias("year_month")
        )
        .drop("month_num")
    )
    
    return cast


def create_rolling_averages(cast: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate rolling averages for 1, 3, 6, and 12 month windows.
    
    Args:
        cast: Raw CAST data
    
    Returns:
        DataFrame with rolling averages added
    """
    cast = cast.with_columns(
        pl.col("year_month").str.strptime(pl.Date, "%Y-%m").alias("year_month_dt")
    ).sort(["admin1", "year_month_dt"])

    window_sizes = [1, 3, 6, 12]

    for window in window_sizes:
        avg_col = []
        rows = cast.to_dicts()

        data_dict = {}
        for row in rows:
            key = (row["admin1"], row["year_month_dt"])
            data_dict[key] = row["total_observed"]

        for row in rows:
            admin = row["admin1"]
            current_date = row["year_month_dt"]

            values = []
            for i in range(1, window + 1):
                check_date = current_date - relativedelta(months=i)
                key = (admin, check_date)
                if key in data_dict and data_dict[key] is not None:
                    values.append(data_dict[key])

            if values:
                avg_val = sum(values) / len(values)
            else:
                avg_val = None

            avg_col.append(avg_val)

        cast = cast.with_columns(pl.Series(name=f"avg{window}", values=avg_col))

    return cast.with_columns(
        pl.col("year_month_dt").dt.strftime("%Y-%m").alias("year_month")
    ).drop("year_month_dt")


def calculate_percent_increase(cast_clean: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate percentage increase and identify hotspots.
    
    Args:
        cast_clean: DataFrame with rolling averages
    
    Returns:
        DataFrame with percentage increases and hotspot flags
    """
    cast_clean = cast_clean.with_columns(
        pl.col(["avg1", "avg3", "avg6", "avg12"]).fill_null(strategy="forward")
    )
    cast_clean = cast_clean.with_columns(pl.col("total_observed").fill_null(0))

    windows = [1, 3, 6, 12]
    for w in windows:
        percent_col = f"percent_increase{w}"
        hot_col = f"hotspot{w}"
        cast_clean = cast_clean.with_columns(
            pl.when(
                (pl.col(f"avg{w}") == 0) & (pl.col("total_forecast") > 0)
            )  # Handle "inf" increase when average observed events = 0, and forecasted events > 0
            .then(pl.col("total_forecast") * 10)
            .otherwise(
                (pl.col("total_forecast") - pl.col(f"avg{w}")) / pl.col(f"avg{w}") * 100
            )
            .fill_null(0)
            .fill_nan(0)
            .alias(percent_col)
        ).with_columns((pl.col(percent_col) >= 25).cast(pl.Int8).alias(hot_col))
    
    return cast_clean


def identify_hotspots_and_regions(cast_clean: pl.DataFrame, window: int = 1, horizon: int = 2):
    """
    Identify hotspots and get all regions for the last month of the specified horizon.
    
    Args:
        cast_clean: Processed CAST data
        window: Number of past months to calculate average
        horizon: Number of months ahead to check for hotspots (will use the last month only)
    
    Returns:
        Tuple of (hotspots DataFrame, all_regions DataFrame, hotspots_list)
    """
    hot_col = f"hotspot{window}"
    
    # Get current month and calculate the target month (last month of horizon)
    current_date = datetime.now()
    target_month = current_date + relativedelta(months=horizon-1)
    target_month_str = target_month.strftime("%Y-%m")

    # Filter hotspots (hotspot1 == 1 and in the target month only)
    hotspots = cast_clean.filter(
        (pl.col("year_month") == target_month_str) & (pl.col(hot_col) == 1)
    )

    # Get all regions for the target month only
    all_regions = cast_clean.filter(
        pl.col("year_month") == target_month_str
    )

    hotspots_list = hotspots["admin1"].unique().to_list()
    
    return hotspots, all_regions, hotspots_list


def save_acled_cast(hotspots, country: str):
    """
    Save the processed data as parquet file.
    
    Args:
        hotspots: Processed hotspots DataFrame (Polars)
        country: Country name
    """
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'acled_cast'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Polars DataFrame to pandas for parquet saving
    if hasattr(hotspots, 'to_pandas'):
        # It's a Polars DataFrame
        df_to_save = hotspots.to_pandas()
    else:
        # It's already a pandas DataFrame
        df_to_save = hotspots
    
    date = datetime.now().strftime('%Y-%m-%d')
    output_path = output_dir / f"ACLED_CAST_{country}_{date}.parquet"
    df_to_save.to_parquet(output_path)


def create_tabular_chart(all_regions, country):
    """
    Create a horizontal bar chart showing forecasted events data with percent_increase1, avg1, and total_forecast values.
    """
    # Convert Polars DataFrame to pandas for visualization
    df_pandas = all_regions.to_pandas()
    
    # Filter only regions with data and sort by percent_increase1 descending
    df_filtered = df_pandas[
        (df_pandas['admin1'].notna()) & 
        (df_pandas['total_forecast'] != 0) |
        (df_pandas['avg1'] != 0) |
        (df_pandas['percent_increase1'] != 0)
    ].copy()
    
    if len(df_filtered) == 0:
        print("No data available for tabular chart")
        return go.Figure()
    
    # Sort by percent_increase1 in ascending order (for proper horizontal order)
    df_sorted = df_filtered.sort_values('percent_increase1', ascending=True)
    
    # Prepare data for the bar chart
    admin_names = df_sorted['admin1'].tolist()
    forecast_values = df_sorted['total_forecast'].round(0).astype(int).tolist()
    average_values = df_sorted['avg1'].round(0).astype(int).tolist()
    percent_changes = df_sorted['percent_increase1'].round(1).tolist()
    
    # Create multi-line text labels
    text_labels = []
    for i, pct in enumerate(percent_changes):
        avg_val = average_values[i]
        forecast_val = forecast_values[i]
        text_labels.append(f"<b>{pct}%</b>   (from {avg_val} to {forecast_val})")
    
    # Create color mapping for bars
    colors = []
    for pct in percent_changes:
        if pct >= 50:
            colors.append('#d73600')
        elif pct >= 25:
            colors.append('#ff6b35')
        elif pct >= 0:
            colors.append('#ffd700')
        else:
            colors.append('#5b9bd5')
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=admin_names,
        x=percent_changes,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=0.5)
        ),
        text=text_labels,
        textposition='outside',
        textfont=dict(size=10),  # Make bar labels bigger
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Percent Change: %{x:.1f}%<br>"
            "Average Events: %{customdata[0]}<br>"
            "Forecasted Events: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=list(zip(average_values, forecast_values)),
        name="Percent Change"
    ))
    
    # Calculate dynamic height based on number of regions
    num_regions = len(df_sorted)
    bar_height = 50
    calculated_height = max(50, num_regions * bar_height)  # Usual scaling for larger n
        
    
    # # Calculate x-axis range to provide more space for labels
    # max_pct = max(abs(min(percent_changes)), abs(max(percent_changes)))
    # x_range = [-max_pct * 1.3, max_pct * 1.8]  # Even more space on the right for multi-line labels
    
    x_range = [min(percent_changes)-350, max(percent_changes)+400]
    
    # Update axes with better formatting
    fig.update_xaxes(
        showticklabels=True,
        tickfont=dict(size=12),
        showgrid=False,
        gridcolor='lightgray',
        automargin=True
    )

    fig.update_yaxes(
        tickfont=dict(size=12),
        tickmode='linear',
        showgrid=False,
        side='left',
        categoryorder='array',
        categoryarray=admin_names,
        automargin=True# Ensure proper ordering
    )
    
    # Update layout with better spacing
    fig.update_layout(
        title=f"Predicted Violence Increase for {country} for the Next Month<br>"
              f"<sub>(Number of Forecasted Events Relative to Last Month)</sub>",
        xaxis_title="Percent Change (%)",
        yaxis_title="Regions",
        font=dict(size=11, family="Arial, sans-serif"),
        uniformtext_minsize=12,  # Ensures all bar labels are at least size 12
        uniformtext_mode='show', # Show even if they overflow
        margin=dict(l=20, r=0, t=120, b=80),  # Increased right margin for multi-line labels
        height=calculated_height,
        width=2000,  # Increased width to accommodate multi-line labels
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
        xaxis=dict(range=x_range),  # Set x-axis range for more space
        bargap=0.3
    )
    
    # Add vertical line at 0% (neutral line)
    fig.add_vline(
        x=0, 
        line_dash="solid", 
        line_color="gray", 
        line_width=1
    )
    
    return fig


def save_visualizations(fig: go.Figure, country: str, hotspots):
    """
    Save the interactive plots in multiple formats.
    
    Args:
        fig: Plotly bar chart figure to save
        country: Country name
        hotspots: Merged data for statistics (Polars DataFrame)
    """
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    # Save tabular chart
    table_base_filename = f"BarChart_{country.replace(' ', '_')}_{timestamp}"
    
    try:
        # Save as HTML (interactive)
        table_html_file = output_dir / f"{table_base_filename}.html"
        fig.write_html(table_html_file)

        # # Save as PDF (static, high quality)
        # table_pdf_file = output_dir / f"{table_base_filename}.pdf"
        # fig.write_image(table_pdf_file)

        # Save as SVG (vector format)
        table_svg_file = output_dir / f"{table_base_filename}.svg"
        fig.write_image(table_svg_file)

    except Exception as e:
        print(f"Error saving some visualization formats: {e}")
        # At least save the HTML version
        table_html_file = output_dir / f"{table_base_filename}.html"
        fig.write_html(table_html_file)
        print(f"Saved only HTML chart: {table_html_file}")


def save_hotspots_list(hotspots_list: list, country: str, horizon: int, all_regions: pl.DataFrame):
    """
    Save the hotspots list with detailed information as a JSON file.
    
    Args:
        hotspots_list: List of hotspot regions
        country: Country name
        horizon: Forecast horizon in months
        all_regions: DataFrame with all regions data including metrics
    """
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'acled_cast'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get detailed information for each hotspot region
    hotspots_details = []
    for region in hotspots_list:
        # Filter data for this specific region and get the latest entry
        region_data = all_regions.filter(pl.col("admin1") == region)
        if region_data.height > 0:
            # Get the row with max percent_increase1 for this region
            region_sorted = region_data.sort("percent_increase1", descending=True)
            region_row = region_sorted.head(1)
            
            hotspot_info = {
                "name": region,
                "avg1": float(region_row["avg1"][0]) if region_row["avg1"][0] is not None else 0.0,
                "total_forecast": float(region_row["total_forecast"][0]) if region_row["total_forecast"][0] is not None else 0.0,
                "percent_increase1": float(region_row["percent_increase1"][0]) if region_row["percent_increase1"][0] is not None else 0.0
            }
            hotspots_details.append(hotspot_info)
    
    # Create metadata for the JSON file
    hotspots_data = {
        "country": country,
        "forecast_horizon_months": horizon,
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_hotspots": len(hotspots_list),
        "hotspot_regions": hotspots_details
    }
    
    date = datetime.now().strftime('%Y-%m-%d')
    output_path = output_dir / f"hotspots_{country.replace(' ', '_')}_{date}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hotspots_data, f, indent=2, ensure_ascii=False)
    


def main():
    """Main execution function."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            raise ValueError("Failed to load configuration. Aborting analysis.")
        
        country = config.get('country')
        window = config.get('window', 1)
        horizon = config.get('horizon', 2)
        
        print(f"Fetching ACLED CAST data for {country}...")
        
        # Step 1: Get CAST data
        cast_data = get_acled_cast_data(country)
        
        # Step 2: Process data
        cast_with_averages = create_rolling_averages(cast_data)
        cast_processed = calculate_percent_increase(cast_with_averages)
        
        # Step 3: Identify hotspots and regions
        hotspots, all_regions, hotspots_list = identify_hotspots_and_regions(cast_processed, window, horizon)
        
        # # Step 6: Create visualizations
        fig = create_tabular_chart(all_regions, country)
        
        # Step 7: Save outputs
        save_visualizations(fig, country, hotspots)
        # save_data(hotspots, country)
        save_acled_cast(hotspots, country)
        save_hotspots_list(hotspots_list, country, horizon, all_regions)
        
        print("ACLED CAST JSON and visualizations saved.")
        
    except Exception as e:
        print(f"Error during ACLED CAST analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()