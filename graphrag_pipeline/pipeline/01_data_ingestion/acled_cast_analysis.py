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
    Identify hotspots and get all regions for the specified horizon.
    
    Args:
        cast_clean: Processed CAST data
        window: Number of past months to calculate average
        horizon: Number of months ahead to check for hotspots
    
    Returns:
        Tuple of (hotspots DataFrame, all_regions DataFrame)
    """
    hot_col = f"hotspot{window}"
    
    # Get current month and calculate target months
    current_date = datetime.now()
    months_to_check = []
    for i in range(horizon):
        check_date = current_date + relativedelta(months=i)
        months_to_check.append(check_date.strftime("%Y-%m"))

    # Filter hotspots (hotspot1 == 1 and in the next 2 months)
    hotspots = cast_clean.filter(
        (pl.col("year_month").is_in(months_to_check)) & (pl.col(hot_col) == 1)
    )

    # Get all regions for the same time period
    all_regions = cast_clean.filter(
        pl.col("year_month").is_in(months_to_check)
    )

    hotspots_list = hotspots["admin1"].unique().to_list()
    
    return hotspots, all_regions, hotspots_list


def get_coordinates(country: str, email: str, api_key: str) -> pl.DataFrame:
    """
    Get coordinates for admin1 regions from ACLED API.
    
    Args:
        country: Country name
        email: ACLED API email
        api_key: ACLED API key
    
    Returns:
        DataFrame with admin1 coordinates
    """
    parameters = {
        "email": email,
        "key": api_key,
        "country": country,
        "fields": "admin1|longitude|latitude",
        "limit": 0
    }

    url = "https://api.acleddata.com/acled/read.csv"
    response = requests.get(url, params=parameters, timeout=60)
    response.raise_for_status()

    coords = pl.read_csv(io.BytesIO(response.content))
    coords = coords.group_by("admin1").agg([
        pl.col("longitude").mean().alias("longitude"),
        pl.col("latitude").mean().alias("latitude"),
    ])
    return coords


def get_shapefile(country: str) -> gpd.GeoDataFrame:
    """
    Download and load country shapefile from GADM.
    
    Args:
        country: Country name
    
    Returns:
        GeoDataFrame with admin1 shapes
    """
    # Get ISO-3 code for the country
    iso3 = pycountry.countries.lookup(country).alpha_3

    # Download GADM level-1 shapefile if not cached
    zip_url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"
    zip_path = Path(f"gadm41_{iso3}.zip")
    if not zip_path.exists():
        with urllib.request.urlopen(zip_url) as resp:
            zip_path.write_bytes(resp.read())

    # Read the level-1 layer directly from the zip
    layer = f"gadm41_{iso3}_1"
    adm1_shapes = gpd.read_file(f"zip://{zip_path}!{layer}.shp")
    return adm1_shapes


def create_merged_mapping(coords: pl.DataFrame, adm1_shapes: gpd.GeoDataFrame, 
                         all_regions: pl.DataFrame) -> gpd.GeoDataFrame:
    """
    Streamlined pipeline to merge coordinates, admin shapes, and all regions data.
    
    Args:
        coords: Coordinates DataFrame
        adm1_shapes: Admin1 shapes GeoDataFrame
        all_regions: All regions DataFrame with conflict data
    
    Returns:
        Merged GeoDataFrame ready for visualization
    """
    # Convert Polars coords to pandas for easier merging
    coords_pd = coords.to_pandas()
    all_regions_pd = all_regions.to_pandas()
    
    # Start with the admin1 shapes as base and initialize empty columns
    final_df = adm1_shapes.copy()
    final_df['admin1'] = None
    final_df['longitude'] = None
    final_df['latitude'] = None
    final_df['match_method'] = None
    
    # Step 1: Direct name matching (NAME_1)
    
    name1_matches = coords_pd.merge(
        final_df[['NAME_1']], 
        left_on="admin1", 
        right_on="NAME_1", 
        how="inner"
    )
    
    # Update final_df with NAME_1 matches
    for _, coord_row in name1_matches.iterrows():
        mask = final_df['NAME_1'] == coord_row['NAME_1']
        final_df.loc[mask, 'admin1'] = coord_row['admin1']
        final_df.loc[mask, 'longitude'] = coord_row['longitude']
        final_df.loc[mask, 'latitude'] = coord_row['latitude']
        final_df.loc[mask, 'match_method'] = 'NAME_1'
    
    matched_admin1 = name1_matches['admin1'].tolist()
    
    # Step 2: VARNAME_1 matching for unmatched coords
    if 'VARNAME_1' in adm1_shapes.columns:
        unmatched_coords = coords_pd[~coords_pd['admin1'].isin(matched_admin1)]
        
        if len(unmatched_coords) > 0:
            varname_matches = unmatched_coords.merge(
                final_df[['NAME_1', 'VARNAME_1']], 
                left_on="admin1", 
                right_on="VARNAME_1", 
                how="inner"
            )
            
            # Update final_df with VARNAME_1 matches
            for _, coord_row in varname_matches.iterrows():
                mask = final_df['NAME_1'] == coord_row['NAME_1']
                final_df.loc[mask, 'admin1'] = coord_row['admin1']
                final_df.loc[mask, 'longitude'] = coord_row['longitude']
                final_df.loc[mask, 'latitude'] = coord_row['latitude']
                final_df.loc[mask, 'match_method'] = 'VARNAME_1'
            
            matched_admin1.extend(varname_matches['admin1'].tolist())
    
    # Step 3: Spatial matching for remaining unmatched coords
    still_unmatched = coords_pd[~coords_pd['admin1'].isin(matched_admin1)]
    
    if len(still_unmatched) > 0:
        
        # Create Point geometries
        still_unmatched = still_unmatched.copy()
        still_unmatched['point_geom'] = [
            Point(row['longitude'], row['latitude']) for _, row in still_unmatched.iterrows()
        ]
        
        # Convert to GeoDataFrame
        unmatched_gdf = gpd.GeoDataFrame(still_unmatched, geometry='point_geom', crs=adm1_shapes.crs)
        
        # Spatial join
        spatial_matches = gpd.sjoin(
            unmatched_gdf, 
            final_df[['NAME_1', 'geometry']], 
            how='inner', 
            predicate='within'
        )
        
        # Update final_df with spatial matches
        for _, coord_row in spatial_matches.iterrows():
            mask = final_df['NAME_1'] == coord_row['NAME_1']
            final_df.loc[mask, 'admin1'] = coord_row['admin1']
            final_df.loc[mask, 'longitude'] = coord_row['longitude']
            final_df.loc[mask, 'latitude'] = coord_row['latitude']
            final_df.loc[mask, 'match_method'] = 'spatial'
        
        matched_admin1.extend(spatial_matches['admin1'].tolist())
    
    # Step 4: Add all regions data including avg1 and total_forecast
    
    # Aggregate data for each admin1 (get max percentage and corresponding avg1/total_forecast)
    all_regions_agg = all_regions_pd.loc[
        all_regions_pd.groupby('admin1')['percent_increase1'].idxmax()
    ][['admin1', 'percent_increase1', 'avg1', 'total_forecast']].reset_index(drop=True)
    
    # Merge all regions data
    final_df = final_df.merge(
        all_regions_agg, 
        on='admin1', 
        how='left'
    )
    
    # Fill missing values
    final_df['percent_increase1'] = final_df['percent_increase1'].fillna(0)
    final_df['avg1'] = final_df['avg1'].fillna(0)
    final_df['total_forecast'] = final_df['total_forecast'].fillna(0)
    final_df['longitude'] = pd.to_numeric(final_df['longitude'], errors='coerce')
    final_df['latitude'] = pd.to_numeric(final_df['latitude'], errors='coerce')
    
    
    return final_df


def create_visualization(final_merged_df: gpd.GeoDataFrame, country: str) -> go.Figure:
    """
    Create interactive choropleth map visualization.
    
    Args:
        final_merged_df: Merged GeoDataFrame with all data
        country: Country name
    
    Returns:
        Plotly Figure object
    """
    merged = final_merged_df.copy()

    # Custom color scale: blue → white → red
    custom_scale = [
        (0.00, "#5b9bd5"),   # blue
        (0.50, "#ffffff"),   # white (center = 0%)
        (1.00, "#d73600"),   # dark red
    ]

    min_val = merged["percent_increase1"].min()
    max_val = merged["percent_increase1"].max()
    abs_max = max(abs(min_val), abs(max_val))

    # Create choropleth map
    geojson = json.loads(merged.to_json())

    fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="NAME_1",
        featureidkey="properties.NAME_1",
        color="percent_increase1",
        color_continuous_scale=custom_scale,
        range_color=(-abs_max, abs_max),  # Center the scale at 0
        hover_name="NAME_1",
        hover_data={
            "percent_increase1": ":.1f",
            "avg1": True,
            "total_forecast": True,
        },
        labels={
            "percent_increase1": "% Change in Conflict Risk ",
            "avg1": "Average Observed Events Last Month ",
            "total_forecast": "Total Forecasted Events Next Month "
        },
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "% Change in Conflict Risk: %{customdata[0]:.1f}%<br>"
            "Average Observed Events Last Month: %{customdata[1]}<br>"
            "Total Forecasted Events Next Month: %{customdata[2]}"
            "<extra></extra>"  # hides the trace name box
        )
    )

    # Add region labels
    centroids = merged.copy()
    centroids["pnt"] = centroids.representative_point()
    centroids["lon"] = centroids.pnt.x
    centroids["lat"] = centroids.pnt.y

    # Build label text: name + % only if non-zero
    centroids["label"] = np.where(
        centroids["percent_increase1"] != 0,
        centroids["NAME_1"] + "<br>" +
        centroids["percent_increase1"].round(1).astype(str) + "%",
        centroids["NAME_1"]  # just the name when value == 0
    )

    fig.add_trace(
        go.Scattergeo(
            lon=centroids["lon"],
            lat=centroids["lat"],
            text=centroids["label"],
            mode="text",
            textfont=dict(size=8, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Layout tweaks
    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        title=f"{country}: Conflict Risk Change by Region"
              "<br><sub>Red = Increase, Blue = Decrease, White = No Change</sub>",
        margin=dict(l=0, r=0, t=80, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="% Change in<br>Conflict Risk", side="right")
        )
    )

    return fig


def save_visualizations(fig: go.Figure, country: str, final_merged_df: gpd.GeoDataFrame):
    """
    Save the interactive plot in multiple formats.
    
    Args:
        fig: Plotly figure to save
        country: Country name
        final_merged_df: Merged data for statistics
    """
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y-%m-%d")
    base_filename = f"Map_{country.replace(' ', '_')}_{timestamp}"

    # Save as HTML (interactive)
    html_file = output_dir / f"{base_filename}.html"
    fig.write_html(html_file)

    # Save as PDF (static, high quality)
    pdf_file = output_dir / f"{base_filename}.pdf"
    fig.write_image(pdf_file, width=1200, height=800)

    # Save as SVG (vector format)
    svg_file = output_dir / f"{base_filename}.svg"
    fig.write_image(svg_file, width=1200, height=800)


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
    


def save_data(final_merged_df: gpd.GeoDataFrame, country: str):
    """
    Save the processed data as parquet file.
    
    Args:
        final_merged_df: Final merged data
        country: Country name
    """
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'acled_cast'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to regular DataFrame for parquet saving (remove geometry)
    df_to_save = pd.DataFrame(final_merged_df.drop(columns=['geometry']))
    
    date = datetime.now().strftime('%Y-%m-%d')
    output_path = output_dir / f"ACLED_CAST_{country}_{date}.parquet"
    df_to_save.to_parquet(output_path)


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
        
        # Step 4: Get geographical data
        load_dotenv(find_dotenv(), override=True)
        email = os.getenv("ACLED_EMAIL")
        api_key = os.getenv("ACLED_API_KEY")
        
        if not email or not api_key:
            raise ValueError("ACLED credentials missing: set ACLED_EMAIL & ACLED_API_KEY environment variables.")
        
        coords = get_coordinates(country, email, api_key)
        adm1_shapes = get_shapefile(country)
        
        # Step 5: Merge all data
        final_merged_df = create_merged_mapping(coords, adm1_shapes, all_regions)
        
        # Step 6: Create visualization
        fig = create_visualization(final_merged_df, country)
        
        # Step 7: Save outputs
        save_visualizations(fig, country, final_merged_df)
        save_data(final_merged_df, country)
        save_hotspots_list(hotspots_list, country, horizon, all_regions)
        
        print("ACLED CAST JSON and visualizations saved.")
        
    except Exception as e:
        print(f"Error during ACLED CAST analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
