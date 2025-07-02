"""
Forecast analysis script.
This script retrieves ACLED CAST and Conflict Forecast data, processes it to identify conflict hotspots,
creates visualizations, and saves outpus.
Workflow
1. ACLED CAST Processing:
   - Download data
   - Process and analyze
   - Create bar chart visualization
   - Save processed data

2. Conflict Forecast Processing:
   - Download data
   - Create line chart visualization

3. Comprehensive Analysis:
   - Generate combined JSON with both sources
   - Include conflict forecast prediction text
   - Mark data availability status

4. Summary Report:
   - Show what data was successfully processed
   - Display output locations
"""

import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
import polars as pl
import requests
from dateutil.relativedelta import relativedelta
from dotenv import find_dotenv, load_dotenv
from .graphrag_construction_pipeline import GraphRAGConstructionPipeline


def load_config():
    """Load configuration from graphrag_config.json"""
    config_path = Path(__file__).parent.parent.parent / 'config_files' / 'graphrag_config.json'
    try:
        return json.loads(config_path.read_text()).get('acled_cast', {})
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error parsing configuration file at {config_path}")

#------------- Get ACLED CAST Data -------------

def get_acled_cast_data(country: str, email: Optional[str] = None, api_key: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Retrieve ACLED CAST forecasting data for a specific country.
    
    Args:
        country: Country name
        email: ACLED API email
        api_key: ACLED API key
    
    Returns:
        Polars DataFrame with CAST data or None if no data found
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
    
    try:
        # GET request to ACLED CAST API
        response = requests.get("https://api.acleddata.com/cast/read.csv", params=parameters, timeout=60)
        response.raise_for_status()
        
        cast = pl.read_csv(io.BytesIO(response.content))
        
        # Check if the API returned "No data has been found" 
        if cast.columns == ["No data has been found"]:
            return None
            
        # Check if 'country' column exists
        if "country" not in cast.columns:
            print(f"Warning: 'country' column not found in ACLED CAST data for {country}")
            print(f"Available columns: {cast.columns}")
            return None
        
        # Filter for exact country match
        cast = cast.filter(pl.col("country") == country)
        
        # Check if any data remains after filtering
        if cast.height == 0:
            print(f"Warning: No ACLED CAST data found for {country} after filtering.")
            return None
        
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
    
    except Exception as e:
        print(f"Error retrieving ACLED CAST data for {country}: {str(e)}")
        return None


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

#------------- Identify Hotspots -------------

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


def save_acled_cast(hotspots, country: str, output_dir):
    """
    Save the processed data as parquet file.
    
    Args:
        hotspots: Processed hotspots DataFrame (Polars)
        country: Country name
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert Polars DataFrame to pandas for parquet saving
    if hasattr(hotspots, 'to_pandas'):
        # It's a Polars DataFrame
        df_to_save = hotspots.to_pandas()
    else:
        # It's already a pandas DataFrame
        df_to_save = hotspots
    
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = output_dir / f"ACLED_CAST_{country}_{date}.parquet"
    df_to_save.to_parquet(output_path)

#------------- Create Visuals -------------

def get_conflict_forecast_prediction(country: str) -> str:
    """Get the latest conflict forecast prediction for the country."""
    try:
        import polars as pl
        import requests
        import pycountry
        
        # ISO3 code lookup function (same as in plot_conflict_forecast)
        def iso3(name):
            name = name.lower().strip()
            
            # Handle common country name mappings first
            country_mappings = {
                'united states': 'USA',
                'usa': 'USA',
                'us': 'USA',
                'america': 'USA',
                'united states of america': 'USA',
                'russia': 'RUS',
                'russian federation': 'RUS',
                'iran': 'IRN',
                'south korea': 'KOR',
                'north korea': 'PRK',
                'uk': 'GBR',
                'britain': 'GBR',
                'great britain': 'GBR',
                'united kingdom': 'GBR',
            }
            
            if name in country_mappings:
                return country_mappings[name]
            
            # Try exact matches first
            try:
                for country in pycountry.countries:
                    try:
                        country_name_attr = getattr(country, 'name', '')
                        official_name_attr = getattr(country, 'official_name', '')
                        alpha_3_attr = getattr(country, 'alpha_3', '')
                        
                        names_to_check = [
                            country_name_attr.lower() if country_name_attr else '',
                            official_name_attr.lower() if official_name_attr else '',
                            alpha_3_attr.lower() if alpha_3_attr else ''
                        ]
                        if name in names_to_check:
                            return alpha_3_attr
                    except (AttributeError, TypeError):
                        continue
            except Exception:
                pass
            
            # Then try substring matching
            matches = []
            try:
                for country in pycountry.countries:
                    try:
                        country_name_attr = getattr(country, 'name', '')
                        official_name_attr = getattr(country, 'official_name', '')
                        alpha_3_attr = getattr(country, 'alpha_3', '')
                        
                        names_to_check = [
                            country_name_attr.lower() if country_name_attr else '',
                            official_name_attr.lower() if official_name_attr else ''
                        ]
                        for country_name in names_to_check:
                            if country_name and (name in country_name or country_name in name):
                                matches.append((alpha_3_attr, len(country_name), country_name))
                    except (AttributeError, TypeError):
                        continue
            except Exception:
                pass
                    
            if matches:
                matches.sort(key=lambda x: x[1])
                return matches[0][0]
                
            return None
        
        # Get latest file listing and find target file
        files = requests.get(
            "http://api.backendless.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/C5F2917E-C2F6-4F7D-9063-69555274134E/services/fileService/get-latest-file-listing"
        ).json()
        file_url = next((f["publicUrl"] for f in files if f["name"] == "conflictforecast_ons_armedconf_03.csv"), None)
        if not file_url:
            return "Data not available"

        # Read and filter data
        df = pl.read_csv(requests.get(file_url).content)
        
        iso = iso3(country)
        if not iso:
            return "Country not found"
        
        df = df.filter(pl.col("isocode") == iso)
        if df.height == 0:
            return "No data available for this country"

        # Parse dates and filter
        df = df.with_columns([
            pl.col("period").cast(str).str.slice(0, 4).alias("year"),
            pl.col("period").cast(str).str.slice(4, 6).alias("month"),
        ])
        df = df.with_columns([
            (pl.col("year") + "-" + pl.col("month")).str.to_datetime("%Y-%m").alias("date")
        ]).filter(pl.col("year").cast(int) >= 2016)
        if df.height == 0:
            return "No recent data available"

        pdf = df.select(["date", "ons_armedconf_03_all"]).to_pandas()
        
        # Get the last available value
        last_value = pdf["ons_armedconf_03_all"].iloc[-1]
        
        return last_value
        
    except Exception as e:
        print(f"Error getting conflict forecast prediction: {e}")
        return "Conflict forecast data not available"


def save_comprehensive_analysis(country: str, horizon: int, output_dir, 
                               hotspots_list: Optional[list] = None, all_regions: Optional[pl.DataFrame] = None,
                               acled_available: bool = False, conflict_forecast_available: bool = False):
    """
    Save comprehensive analysis combining ACLED CAST and Conflict Forecast data.
    
    Args:
        country: Country name
        horizon: Forecast horizon in months
        output_dir: Output directory
        hotspots_list: List of hotspot regions (if ACLED data available)
        all_regions: DataFrame with all regions data (if ACLED data available)
        acled_available: Whether ACLED data was successfully processed
        conflict_forecast_available: Whether Conflict Forecast data is available
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analysis data
    analysis_data = {
        "country": country,
        "analysis_date": datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        "data_sources": {
            "acled_cast": acled_available,
            "conflict_forecast": conflict_forecast_available
        }
    }
    
    # Add Conflict Forecast data if available (before ACLED data)
    if conflict_forecast_available:
        conflict_prediction = get_conflict_forecast_prediction(country)
        analysis_data["conflict_forecast_prediction"] = conflict_prediction
    else:
        analysis_data["conflict_forecast_prediction"] = "Data not available"
    
    # Add ACLED CAST data if available
    if acled_available and hotspots_list is not None and all_regions is not None:
        # Get detailed information for each hotspot region
        hotspots_details = []
        for region in hotspots_list:
            region_data = all_regions.filter(pl.col("admin1") == region)
            if region_data.height > 0:
                region_sorted = region_data.sort("percent_increase1", descending=True)
                region_row = region_sorted.head(1)
                
                hotspot_info = {
                    "name": region,
                    "avg1": float(region_row["avg1"][0]) if region_row["avg1"][0] is not None else 0.0,
                    "total_forecast": float(region_row["total_forecast"][0]) if region_row["total_forecast"][0] is not None else 0.0,
                    "percent_increase1": float(region_row["percent_increase1"][0]) if region_row["percent_increase1"][0] is not None else 0.0
                }
                hotspots_details.append(hotspot_info)
        
        analysis_data.update({
            "acled_cast_analysis": {
                "forecast_horizon_months": horizon,
                "total_hotspots": len(hotspots_list),
                "hotspot_regions": hotspots_details
            }
        })
    else:
        analysis_data["acled_cast_analysis"] = "Data not available"
    
    # Save the comprehensive analysis
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = output_dir / f"forecast_data_{country.replace(' ', '_')}_{date}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"Comprehensive analysis saved to: {output_path}")


def save_hotspots_list(hotspots_list: list, country: str, horizon: int, all_regions: pl.DataFrame,output_dir):
    """
    Save the hotspots list with detailed information as a JSON file.
    [DEPRECATED] - Use save_comprehensive_analysis instead for combined data sources.
    """
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
    
    # Get conflict forecast prediction
    conflict_forecast_prediction = get_conflict_forecast_prediction(country)
    
    # Create metadata for the JSON file
    hotspots_data = {
        "country": country,
        "forecast_horizon_months": horizon,
        "analysis_date": datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        "conflict_forecast_prediction": conflict_forecast_prediction,
        "total_hotspots": len(hotspots_list),
        "hotspot_regions": hotspots_details
    }
    
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = output_dir / f"forecast_data_{country.replace(' ', '_')}_{date}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hotspots_data, f, indent=2, ensure_ascii=False)
        

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
        # title=f"Predicted Violence Increase for {country} for the Next Month<br>"
        #       f"Number of Forecasted Events Relative to Last Month",
        xaxis_title="Percent Change (%)",
        yaxis_title="Regions",
        font=dict(size=11, family="Arial, sans-serif"),
        uniformtext_minsize=12,  # Ensures all bar labels are at least size 12
        uniformtext_mode='show', # Show even if they overflow
        margin=dict(l=50, r=50, t=30, b=80),  # Increased right margin for multi-line labels
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


def save_barchart(fig: go.Figure, country: str, output_dir):
    """
    Save the interactive plots in multiple formats.
    
    Args:
        fig: Plotly bar chart figure to save
        country: Country name
        hotspots: Merged data for statistics (Polars DataFrame)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Save tabular chart
    table_base_filename = f"BarChart_{country.replace(' ', '_')}_{timestamp}"
    
    try:
        # # Save as HTML (interactive)
        # table_html_file = output_dir / f"{table_base_filename}.html"
        # fig.write_html(table_html_file)

        # Save as SVG (vector format)
        table_svg_file = output_dir / f"{table_base_filename}.svg"
        fig.write_image(table_svg_file)

    except Exception as e:
        print(f"Error saving visualizations: {e}")
        # # At least save the HTML version
        # table_html_file = output_dir / f"{table_base_filename}.html"
        # fig.write_html(table_html_file)
        # print(f"Saved only HTML chart: {table_html_file}")
    

#------------- Create Conflict Forecast Visuals ---------

def plot_conflict_forecast(country: str):
    """
    Plot conflict probability forecast for a specific country with gradient colors.
    """
    import polars as pl
    import requests
    import pycountry
    import plotly.graph_objects as go

    # Improved ISO3 code lookup
    def iso3(name):
        name = name.lower().strip()
        
        # Handle common country name mappings first
        country_mappings = {
            'united states': 'USA',
            'usa': 'USA',
            'us': 'USA',
            'america': 'USA',
            'united states of america': 'USA',
            'russia': 'RUS',
            'russian federation': 'RUS',
            'iran': 'IRN',
            'south korea': 'KOR',
            'north korea': 'PRK',
            'uk': 'GBR',
            'britain': 'GBR',
            'great britain': 'GBR',
            'united kingdom': 'GBR',
        }
        
        if name in country_mappings:
            return country_mappings[name]
        
        # Try exact matches first
        try:
            for country in pycountry.countries:
                try:
                    country_name_attr = getattr(country, 'name', '')
                    official_name_attr = getattr(country, 'official_name', '')
                    alpha_3_attr = getattr(country, 'alpha_3', '')
                    
                    names_to_check = [
                        country_name_attr.lower() if country_name_attr else '',
                        official_name_attr.lower() if official_name_attr else '',
                        alpha_3_attr.lower() if alpha_3_attr else ''
                    ]
                    if name in names_to_check:
                        return alpha_3_attr
                except (AttributeError, TypeError):
                    continue
        except Exception:
            pass
        
        # Then try substring matching (but prioritize shorter matches)
        matches = []
        try:
            for country in pycountry.countries:
                try:
                    country_name_attr = getattr(country, 'name', '')
                    official_name_attr = getattr(country, 'official_name', '')
                    alpha_3_attr = getattr(country, 'alpha_3', '')
                    
                    names_to_check = [
                        country_name_attr.lower() if country_name_attr else '',
                        official_name_attr.lower() if official_name_attr else ''
                    ]
                    for country_name in names_to_check:
                        if country_name and (name in country_name or country_name in name):
                            matches.append((alpha_3_attr, len(country_name), country_name))
                except (AttributeError, TypeError):
                    continue
        except Exception:
            pass
                
        if matches:
            # Sort by length to prefer shorter, more precise matches
            matches.sort(key=lambda x: x[1])
            return matches[0][0]
            
        print(f"Country '{name}' not found in ISO lookup.")
        return None

    try:
        # Get latest file listing and find target file
        files = requests.get(
            "http://api.backendless.com/C177D0DC-B3D5-818C-FF1E-1CC11BC69600/C5F2917E-C2F6-4F7D-9063-69555274134E/services/fileService/get-latest-file-listing"
        ).json()
        file_url = next((f["publicUrl"] for f in files if f["name"] == "conflictforecast_ons_armedconf_03.csv"), None)
        if not file_url:
            raise ValueError("Target file not found.")

        # Read and filter data
        df = pl.read_csv(requests.get(file_url).content)
        
        iso = iso3(country)
        if not iso:
            raise ValueError(f"ISO code not found for {country}")
        
        df = df.filter(pl.col("isocode") == iso)
        if df.height == 0:
            raise ValueError(f"No data for {country}")

        # Parse dates and filter
        df = df.with_columns([
            pl.col("period").cast(str).str.slice(0, 4).alias("year"),
            pl.col("period").cast(str).str.slice(4, 6).alias("month"),
        ])
        df = df.with_columns([
            (pl.col("year") + "-" + pl.col("month")).str.to_datetime("%Y-%m").alias("date")
        ]).filter(pl.col("year").cast(int) >= 2016)
        if df.height == 0:
            raise ValueError(f"No data for {country} from 2020")

        pdf = df.select(["date", "ons_armedconf_03_all"]).to_pandas()
        x, y = pdf["date"], pdf["ons_armedconf_03_all"]
        norm = (y - y.min()) / (y.max() - y.min())

        fig = go.Figure()
        # Add colored line segments
        for i in range(len(x)-1):
            c = int(255 * ((norm.iloc[i] + norm.iloc[i+1])/2))
            fig.add_trace(go.Scatter(
                x=[x.iloc[i], x.iloc[i+1]], y=[y.iloc[i], y.iloc[i+1]],
                mode='lines', line=dict(color=f'rgb({c},0,{255-c})', width=2),
                showlegend=False, hoverinfo='skip'
            ))
        # Last value label
        fig.add_trace(go.Scatter(
            x=[x.iloc[-1]], y=[y.iloc[-1]], mode='markers+text',
            marker=dict(color='darkred', size=8),
            text=[f'{y.iloc[-1]:.2f}'], textposition='top right',
            textfont=dict(size=12, color='darkred'), showlegend=False
        ))

        fig.update_layout(
            xaxis_title='Date', yaxis_title='Conflict Probability',
            font=dict(size=12, family='Arial, sans-serif'), showlegend=False,
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=50, r=20, t=30, b=80)
        )
        fig.update_xaxes(
            showgrid=True, gridcolor='lightgray', gridwidth=1,
            dtick="M3", tickformat="%b", tickangle=45, automargin=True
        )
        fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5, automargin=True)
        
        # Add custom year annotations
        years = pdf["date"].dt.year.unique()
        for year in sorted(years):
            # Find the first date of each year in the data
            year_data = pdf[pdf["date"].dt.year == year]
            if not year_data.empty:
                first_date = year_data["date"].min()
                fig.add_annotation(
                    x=first_date,
                    y=pdf["ons_armedconf_03_all"].min()-(((pdf["ons_armedconf_03_all"].max())-(pdf["ons_armedconf_03_all"].min()))*0.1),  # Position below the plot
                    text=str(year),
                    showarrow=False,
                    font=dict(size=12),
                    xanchor="left"
                )
        
        return fig

    except Exception as e:
        print(f"Error in plot_conflict_forecast: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    
def save_linechart(fig: go.Figure, country: str,output_dir):
    """
    Save the interactive plots in multiple formats.
    
    Args:
        fig: Plotly line chart figure to save
        country: Country name
    """
    # Check if figure is None
    if fig is None:
        print(f"Warning: No conflict forecast figure to save for {country}. Skipping line chart save.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Save tabular chart
    linechart_filename = f"LineChart_{country.replace(' ', '_')}_{timestamp}"
    
    try:
        # # Save as HTML (interactive)
        # line_html_file = output_dir / f"{linechart_filename}.html"
        # fig.write_html(line_html_file)

        # Save as SVG (vector format)
        table_svg_file = output_dir / f"{linechart_filename}.svg"
        fig.write_image(table_svg_file)

    except Exception as e:
        print(f"Error saving visualizations: {e}")
        # # At least save the HTML version
        # line_html_file = output_dir / f"{linechart_filename}.html"
        # fig.write_html(line_html_file)
        # print("Saved only HTML chart for ConfForecast")

#------------- Main Execution Function -------------
def main(country: Optional[str] = None, custom_output_directory: Optional[str] = None):
    """Main execution function with streamlined workflow for both data sources."""
    
    # Ensure we have a country
    if country is None:
        country = os.getenv('GRAPHRAG_COUNTRY')
    
    if not country:
        raise ValueError("Country must be specified either as parameter or GRAPHRAG_COUNTRY environment variable")
    
    graphrag_construction_pipeline = GraphRAGConstructionPipeline()
    default_output_directory = graphrag_construction_pipeline._get_default_output_directory(country=country)
    if custom_output_directory is None:
        custom_output_directory = os.getenv('GRAPHRAG_OUTPUT_DIR')
    
    if custom_output_directory:
        output_dir = Path(custom_output_directory) / 'assets'
    else:
        output_dir = Path(default_output_directory) / 'assets'
    
    # Initialize tracking variables
    acled_available = False
    conflict_forecast_available = False
    hotspots_list = []
    all_regions = None
    
    try:
        # Load configuration
        config = load_config()
        window = config.get('window', 1) if config else 1
        horizon = config.get('horizon', 2) if config else 2
        
        # === ACLED CAST DATA PROCESSING ===
        print(f"Processing ACLED CAST data for {country}...")
        try:
            cast_data = get_acled_cast_data(country)
            
            if cast_data is not None:
                print("ACLED CAST data retrieved successfully")
                
                # Process ACLED data
                cast_with_averages = create_rolling_averages(cast_data)
                cast_processed = calculate_percent_increase(cast_with_averages)
                hotspots, all_regions, hotspots_list = identify_hotspots_and_regions(cast_processed, window, horizon)
                
                # Create and save ACLED visualization
                fig_cast = create_tabular_chart(all_regions, country)
                if fig_cast is not None:
                    save_barchart(fig_cast, country, output_dir)
                
                # Save ACLED data
                save_acled_cast(hotspots, country, output_dir)
                
                acled_available = True
                
            else:
                print("No ACLED CAST data available")
                
        except Exception as e:
            print(f"Error processing ACLED CAST data: {e}")
        
        # === CONFLICT FORECAST DATA PROCESSING ===
        print(f"Processing Conflict Forecast data for {country}...")
        try:
            fig_cf = plot_conflict_forecast(country)
            
            if fig_cf is not None:
                save_linechart(fig_cf, country, output_dir)
                conflict_forecast_available = True
            else:
                print("No Conflict Forecast data available")
                
        except Exception as e:
            print(f"Error processing Conflict Forecast data: {e}")
        
        # === GENERATE COMPREHENSIVE ANALYSIS ===
        print("Generating comprehensive analysis...")
        save_comprehensive_analysis(
            country=country,
            horizon=horizon,
            output_dir=output_dir,
            hotspots_list=hotspots_list,
            all_regions=all_regions,
            acled_available=acled_available,
            conflict_forecast_available=conflict_forecast_available
        )
        
        
    except Exception as e:
        print(f"Critical error during analysis: {str(e)}")
        # Still try to save whatever data we have
        if acled_available or conflict_forecast_available:
            try:
                save_comprehensive_analysis(
                    country=country,
                    horizon=2,  # default value
                    output_dir=output_dir,
                    hotspots_list=hotspots_list,
                    all_regions=all_regions,
                    acled_available=acled_available,
                    conflict_forecast_available=conflict_forecast_available
                )
                print("Partial analysis saved despite errors.")
            except Exception as save_error:
                print(f"Failed to save partial analysis: {save_error}")
        raise


if __name__ == "__main__":
    main()