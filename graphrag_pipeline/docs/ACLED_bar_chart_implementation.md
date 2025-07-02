# ACLED CAST Analysis Pipeline - Vertical Bar Chart Implementation

## Overview
The `acled_cast_analysis.py` script has been successfully updated to include a **vertical bar chart** that visualizes conflict risk changes by region, replacing the previous tabular display.

## Key Features Implemented

### 1. Vertical Bar Chart (`create_tabular_chart` function)
- **Purpose**: Displays percent change in conflict risk for each Admin1 region
- **Chart Type**: Vertical bar chart with color-coded bars
- **Sorting**: Regions sorted by percent_increase1 (highest risk first)
- **Colors**: 
  - 🔴 **Dark Red** (#d73600): ≥50% increase (High Risk)
  - 🟠 **Orange-Red** (#ff6b35): 25-49% increase (Moderate Risk)  
  - 🟡 **Yellow** (#ffd700): 0-24% increase (Low Risk)
  - 🔵 **Blue** (#5b9bd5): Negative values (Decreased Risk)

### 2. Interactive Features
- **Hover Information**: Shows region name, percent change, forecast value, and average
- **Text Labels**: Percent change displayed on top of each bar
- **Responsive Height**: Chart height adjusts based on number of regions
- **Rotated Labels**: X-axis labels rotated 45° for better readability

### 3. Visualization Elements
- **Title**: "Forecasted Events (All Event Types) for {Country}"
- **Subtitle**: "Relative to 1-Month Average - Sorted by Risk Level"
- **X-axis**: Admin1 Regions
- **Y-axis**: Percent Change (%)
- **Reference Lines**: 
  - Horizontal line at 25% (hotspot threshold)
  - Horizontal line at 0% (neutral baseline)

### 4. File Outputs
The script now generates and saves:

#### Choropleth Map Files:
- `Map_{Country}_{Date}.html` - Interactive map
- `Map_{Country}_{Date}.pdf` - Static high-quality map  
- `Map_{Country}_{Date}.svg` - Vector format map

#### Bar Chart Files:
- `BarChart_{Country}_{Date}.html` - Interactive bar chart
- `BarChart_{Country}_{Date}.pdf` - Static high-quality chart
- `BarChart_{Country}_{Date}.svg` - Vector format chart

#### Data Files:
- `{Country}_merged_data_{Date}.parquet` - Complete processed dataset
- `hotspots_list_{Country}_{Date}.json` - Detailed hotspots information

## Technical Implementation

### Function Structure
```python
def create_tabular_chart(final_merged_df: gpd.GeoDataFrame, country: str) -> go.Figure:
    """Create a vertical bar chart showing percent increase in conflict risk by region."""
```

### Data Processing
1. **Filtering**: Only includes regions with valid admin1 names and non-zero percent_increase1
2. **Sorting**: Descending order by percent_increase1 (highest risk first)
3. **Color Mapping**: Dynamic color assignment based on risk thresholds
4. **Data Preparation**: Rounds values and formats for display

### Integration with Main Pipeline
- Called from `main()` function after data processing
- Integrated with `save_visualizations()` for multi-format output
- Works alongside existing choropleth map generation

## Usage Example

```python
# Run the complete pipeline
python acled_cast_analysis.py

# Or test just the bar chart functionality
python test_bar_chart.py
```

## Configuration
The bar chart respects all existing configuration parameters from `data_ingestion_config.json`:
- `country`: Target country for analysis
- `window`: Time window for averages (affects percent_increase1 calculation)
- `horizon`: Forecast horizon (affects data filtering)

## Quality Assurance
✅ **Syntax Validation**: No lint errors or syntax issues
✅ **Import Testing**: Script imports successfully
✅ **Function Testing**: Bar chart generation tested with sample data
✅ **Color Coding**: Risk-based color scheme implemented correctly
✅ **File Output**: Multi-format saving (HTML, PDF, SVG) working
✅ **Data Integration**: Seamlessly integrated with existing pipeline

## Output Files Location
All generated files are saved to:
```
graphrag_pipeline/data/images/
```

The vertical bar chart implementation provides an intuitive, visually appealing way to quickly identify high-risk regions and understand the relative conflict risk changes across all Admin1 areas in the target country.
