# banxico-adapted-app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import folium_static, st_folium
import re # Keep for potential city/state cleaning, review if needed later
import sys

# Define comprehensive state data for enhanced analysis
STATE_DATA = {
    'Aguascalientes': {
        'population': 1500412,
        'adult_population_pct': 68.5,
        'area_km2': 5616,
        'urban_pct': 81.2,
        'gdp_per_capita': 17852
    },
    'Baja California': {
        'population': 3786109,
        'adult_population_pct': 73.1,
        'area_km2': 71450,
        'urban_pct': 92.3,
        'gdp_per_capita': 19728
    },
    'Baja California Sur': {
        'population': 883649,
        'adult_population_pct': 72.7,
        'area_km2': 73909,
        'urban_pct': 86.4,
        'gdp_per_capita': 20851
    },
    'Campeche': {
        'population': 941096,
        'adult_population_pct': 67.3,
        'area_km2': 57507,
        'urban_pct': 75.2,
        'gdp_per_capita': 42163
    },
    'Coahuila de Zaragoza': {
        'population': 3354804,
        'adult_population_pct': 71.2,
        'area_km2': 151571,
        'urban_pct': 90.1,
        'gdp_per_capita': 20125
    },
    'Colima': {
        'population': 733701,
        'adult_population_pct': 70.1,
        'area_km2': 5625,
        'urban_pct': 89.5,
        'gdp_per_capita': 16219
    },
    'Chiapas': {
        'population': 5803144,
        'adult_population_pct': 62.1,
        'area_km2': 73311,
        'urban_pct': 50.5,
        'gdp_per_capita': 6970
    },
    'Chihuahua': {
        'population': 3870533,
        'adult_population_pct': 71.5,
        'area_km2': 247412,
        'urban_pct': 85.4,
        'gdp_per_capita': 18088
    },
    'Ciudad de México': {
        'population': 9338373,
        'adult_population_pct': 77.8,
        'area_km2': 1495,
        'urban_pct': 99.5,
        'gdp_per_capita': 33665
    },
    'Durango': {
        'population': 1903497,
        'adult_population_pct': 66.9,
        'area_km2': 123317,
        'urban_pct': 69.7,
        'gdp_per_capita': 14078
    },
    'Guanajuato': {
        'population': 6330734,
        'adult_population_pct': 66.8,
        'area_km2': 30607,
        'urban_pct': 70.1,
        'gdp_per_capita': 14706
    },
    'Guerrero': {
        'population': 3607623,
        'adult_population_pct': 64.7,
        'area_km2': 63596,
        'urban_pct': 58.2,
        'gdp_per_capita': 8238
    },
    'Hidalgo': {
        'population': 3240790,
        'adult_population_pct': 66.9,
        'area_km2': 20846,
        'urban_pct': 52.7,
        'gdp_per_capita': 10615
    },
    'Jalisco': {
        'population': 8726308,
        'adult_population_pct': 70.2,
        'area_km2': 78588,
        'urban_pct': 87.3,
        'gdp_per_capita': 18175
    },
    'Estado de México': {
        'population': 17727868,
        'adult_population_pct': 69.7,
        'area_km2': 22357,
        'urban_pct': 87.0,
        'gdp_per_capita': 11644
    },
    'Michoacán': {
        'population': 4941831,
        'adult_population_pct': 66.3,
        'area_km2': 58643,
        'urban_pct': 69.1,
        'gdp_per_capita': 10634
    },
    'Morelos': {
        'population': 1979715,
        'adult_population_pct': 69.8,
        'area_km2': 4893,
        'urban_pct': 84.3,
        'gdp_per_capita': 12602
    },
    'Nayarit': {
        'population': 1248922,
        'adult_population_pct': 68.9,
        'area_km2': 27815,
        'urban_pct': 69.2,
        'gdp_per_capita': 11347
    },
    'Nuevo León': {
        'population': 6128074,
        'adult_population_pct': 73.4,
        'area_km2': 64156,
        'urban_pct': 94.8,
        'gdp_per_capita': 27248
    },
    'Oaxaca': {
        'population': 4297758,
        'adult_population_pct': 64.5,
        'area_km2': 93757,
        'urban_pct': 47.9,
        'gdp_per_capita': 7872
    },
    'Puebla': {
        'population': 6603151,
        'adult_population_pct': 66.8,
        'area_km2': 34290,
        'urban_pct': 72.1,
        'gdp_per_capita': 11897
    },
    'Querétaro': {
        'population': 2541803,
        'adult_population_pct': 69.3,
        'area_km2': 11684,
        'urban_pct': 70.2,
        'gdp_per_capita': 20810
    },
    'Quintana Roo': {
        'population': 1912993,
        'adult_population_pct': 71.5,
        'area_km2': 44705,
        'urban_pct': 88.9,
        'gdp_per_capita': 20355
    },
    'San Luis Potosí': {
        'population': 2880503,
        'adult_population_pct': 67.3,
        'area_km2': 60983,
        'urban_pct': 65.1,
        'gdp_per_capita': 15059
    },
    'Sinaloa': {
        'population': 3100860,
        'adult_population_pct': 68.8,
        'area_km2': 57377,
        'urban_pct': 73.2,
        'gdp_per_capita': 13952
    },
    'Sonora': {
        'population': 3076858,
        'adult_population_pct': 71.3,
        'area_km2': 179355,
        'urban_pct': 86.0,
        'gdp_per_capita': 20289
    },
    'Tabasco': {
        'population': 2527412,
        'adult_population_pct': 67.9,
        'area_km2': 24731,
        'urban_pct': 57.4,
        'gdp_per_capita': 17428
    },
    'Tamaulipas': {
        'population': 3574192,
        'adult_population_pct': 71.2,
        'area_km2': 80175,
        'urban_pct': 88.0,
        'gdp_per_capita': 16784
    },
    'Tlaxcala': {
        'population': 1464291,
        'adult_population_pct': 67.8,
        'area_km2': 3991,
        'urban_pct': 80.4,
        'gdp_per_capita': 8722
    },
    'Veracruz': {
        'population': 8094410,
        'adult_population_pct': 68.4,
        'area_km2': 71820,
        'urban_pct': 61.2,
        'gdp_per_capita': 10675
    },
    'Yucatán': {
        'population': 2381597,
        'adult_population_pct': 69.7,
        'area_km2': 39612,
        'urban_pct': 84.0,
        'gdp_per_capita': 14289
    },
    'Zacatecas': {
        'population': 1651236,
        'adult_population_pct': 66.8,
        'area_km2': 75539,
        'urban_pct': 59.1,
        'gdp_per_capita': 11750
    },
}

# For backward compatibility
STATE_POPULATIONS_UPDATED = {state: data['population'] for state, data in STATE_DATA.items()}


# Set page configuration
st.set_page_config(
    page_title="Banxico Branch Analysis",
    page_icon="🏛️", # Changed icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (simplified, can be adjusted)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #600000; /* Banxico Red */
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #8C1919; /* Darker Banxico Red */
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #AE2525; /* Lighter Banxico Red */
    }
    .info-box {
        background-color: #fef0f0; /* Light red background */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #AE2525;
        margin-bottom: 20px;
    }
    .stat-box {
        text-align: center;
    }
    /* Style for expanders */
    .stExpander > details {
        border-left: 3px solid #AE2525;
        border-radius: 5px;
        background-color: #fafafa;
    }
    .stExpander > details > summary {
        font-weight: bold;
        color: #8C1919;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    """Load and preprocess data from banxico_branches-complete.csv"""
    data_path = "banxico_branches-complete.csv"
    try:
        # Print original row count for debugging
        raw_count = sum(1 for line in open(data_path, 'r')) - 1  # Subtract header
        
        # Specify encoding if needed, e.g., encoding='latin1' or 'utf-8'
        try:
            df = pd.read_csv(data_path)
            initial_count = len(df)
        except UnicodeDecodeError:
            st.warning("UTF-8 decoding failed, trying latin-1...")
            df = pd.read_csv(data_path, encoding='latin1')
            initial_count = len(df)
            
        # Log initial data count    
        print(f"INFO: Initial data loaded with {initial_count} rows (raw file has {raw_count} lines including header)")
            
        # --- Basic Cleaning ---
        # Rename columns for consistency if they exist
        rename_map = {
             # 'OLD_LAT_COL': 'latitud', # Not needed, already matches
             # 'OLD_LON_COL': 'longitud', # Not needed, already matches
             'estado': 'estado',       # Already matches
             'municipio': 'ciudad',    # Map from new CSV column
             'direccion': 'nombre',    # Use direccion as the branch name (as per clarification)
             'horario': 'direccion'    # horario actually contains the address (as per clarification)
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # For this dataset, the sucursal field contains the bank name
        # and banco is a numeric ID
        if 'banco' in df.columns and 'sucursal' in df.columns:
            # Store the original ID
            df['banco_id'] = df['banco']
            # Use sucursal as the bank name
            df['banco'] = df['sucursal']

        # Check for essential columns
        required_cols = ['latitud', 'longitud', 'nombre', 'banco', 'estado', 'ciudad']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing essential columns in CSV: {', '.join(missing_cols)}. Please ensure the CSV has these columns.")
            return create_sample_data() # Return sample data or empty df

        # Count before cleaning coordinates
        before_coord_clean = len(df)
        
        # Clean coordinates
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        df = df.dropna(subset=['latitud', 'longitud'])
        
        # Count after cleaning coordinates
        after_coord_clean = len(df)
        if before_coord_clean != after_coord_clean:
            print(f"INFO: Removed {before_coord_clean - after_coord_clean} rows with invalid coordinates")

        # Clean geo data
        for col in ['estado', 'ciudad', 'banco', 'nombre']:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.strip().fillna('Sin especificar')
                df.loc[df[col] == '', col] = 'Sin especificar' # Handle empty strings

        # Count before removing outliers
        before_outlier_clean = len(df)
        
        # Remove outlier coordinates (adjust bounds as needed for Mexico)
        df = df[(df['latitud'] > 14) & (df['latitud'] < 33)]
        df = df[(df['longitud'] > -120) & (df['longitud'] < -85)]
        
        # Count after removing outliers
        after_outlier_clean = len(df)
        if before_outlier_clean != after_outlier_clean:
            print(f"INFO: Removed {before_outlier_clean - after_outlier_clean} outlier coordinates")

        # Add a placeholder column indicating it's a branch location
        df['is_branch'] = 1 
        
        # Final count
        final_count = len(df)
        print(f"INFO: Final dataset has {final_count} rows. Rows removed during processing: {initial_count - final_count}")
        
        # Add data quality metrics to the dataframe for display in the app
        df.attrs['data_quality'] = {
            'initial_count': initial_count,
            'missing_coords': before_coord_clean - after_coord_clean,
            'outliers': before_outlier_clean - after_outlier_clean,
            'final_count': final_count
        }

        return df

    except FileNotFoundError:
        st.warning(f"Data file not found at {data_path}. Using sample data.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

@st.cache_data
def create_sample_data():
    """Create sample data mimicking banxico_branches.csv structure"""
    states = ['Ciudad de México', 'Jalisco', 'Nuevo León', 'Estado de México',
              'Baja California', 'Veracruz']
    banks = ['Banco Principal', 'Banco Secundario', 'Banco Regional', 'Otro Banco']
    np.random.seed(42)
    n_samples = 150
    data = {
        'nombre': [f"Sucursal Demo {i+1}" for i in range(n_samples)],
        'banco': np.random.choice(banks, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'estado': np.random.choice(states, n_samples),
        'ciudad': [f"Ciudad Demo {i%15 + 1}" for i in range(n_samples)],
        'latitud': np.random.uniform(19, 26, n_samples),
        'longitud': np.random.uniform(-105, -97, n_samples),
    }
    df = pd.DataFrame(data)
    df['is_branch'] = 1
    return df

# --- Color Mapping for Banks ---
def get_bank_colors(bank_names):
    """Assigns colors to bank names"""
    unique_banks = sorted(bank_names.unique())
    colors = px.colors.qualitative.Vivid + px.colors.qualitative.Pastel + px.colors.qualitative.Set3
    color_map = {bank: colors[i % len(colors)] for i, bank in enumerate(unique_banks)}
    # Assign a default color for 'Sin especificar' or other common unknowns
    color_map['Sin especificar'] = '#808080' # Grey
    return color_map

# --- Geographic Analysis Functions ---
def create_folium_map(df, center, zoom, map_type='clusters', bank_color_map=None, filtered_banks=None, selected_state="All States", base_map_tile='CartoDB positron'):
    """Create a Folium map visualization of branch locations.
    
    Args:
        df (pd.DataFrame): Dataframe with branch locations ('latitud', 'longitud', 'nombre', 'banco', 'estado', 'ciudad').
        center (list): Latitude and Longitude for map center.
        zoom (int): Initial zoom level for the map.
        map_type (str): 'clusters' or 'heatmap'.
        bank_color_map (dict): Mapping from bank name to hex color.
        filtered_banks (list): Optional list of banks to display. If None or empty, show all.
        selected_state (str): State used for filtering (for determining if map is empty).
        base_map_tile (str): Folium tile layer identifier. Defaults to 'CartoDB positron'.
    """
    map_df = df.copy() # Work with a copy of the potentially pre-filtered df

    # Note: Filtering by state and bank is now done *before* calling this function.
    # The selected_state arg is primarily used here to determine if the df is empty for display purposes.

    if map_df.empty:
        st.warning(f"No data points to display on the map for the selected filters (State: {selected_state}, Banks: {filtered_banks if filtered_banks else 'All'}).")
        # Provide a default map centered on Mexico using the passed center/zoom
        empty_map = folium.Map(location=center, zoom_start=zoom, tiles=base_map_tile)
        
        # Add a title indicating no banks to display
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; 
                    background-color: rgba(255, 255, 255, 0.8); padding: 10px 15px; 
                    border-radius: 5px; font-size: 16px; font-weight: bold; text-align: center;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            No banks to display for the selected filters
        </div>
        '''
        empty_map.get_root().html.add_child(folium.Element(title_html))
        
        return empty_map

    # Create map using provided center and zoom
    m = folium.Map(location=center, zoom_start=zoom, tiles=base_map_tile)
    
    # Add a title showing which banks are being displayed
    banks_displayed = map_df['banco'].unique()
    banks_list = ", ".join(sorted(banks_displayed))
    state_info = f" in {selected_state}" if selected_state != "All States" else ""
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; 
                background-color: rgba(255, 255, 255, 0.8); padding: 10px 15px; 
                border-radius: 5px; font-size: 16px; font-weight: bold; text-align: center;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        Displaying: {banks_list}{state_info}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    if bank_color_map is None:
        # If no color map provided, generate one (might happen for overview map)
        # We need the original unfiltered df banco column for consistent colors
        # This is a fallback, ideally the color map is generated once and passed in.
        try:
            # Attempt to load original data again just for colors if needed
            original_df_for_colors = load_data()
            bank_color_map = get_bank_colors(original_df_for_colors['banco'])
        except Exception:
            # Fallback if loading fails or running in a context where load_data isn't easily available
             bank_color_map = get_bank_colors(map_df['banco']) # Use current df as last resort

    # --- Create Legend HTML ---
    # Only include banks that are actually present in the filtered map_df
    banks_on_map = sorted(map_df['banco'].unique())
    if 'Sin especificar' in banks_on_map: # Move 'Sin especificar' to the end if present
        banks_on_map.remove('Sin especificar')
        banks_on_map.append('Sin especificar')

    legend_html = '''
         <div style="
             position: fixed;
             bottom: 50px;
             left: 50px;
             width: 180px; /* Adjust width as needed */
             height: auto; /* Adjust height based on content */
             max-height: 300px; /* Prevent legend from becoming too tall */
             z-index:9999;
             font-size:14px;
             background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
             border:1px solid grey;
             border-radius: 8px;
             padding: 10px;
             overflow-y: auto; /* Add scroll if too many items */
             ">
         <h4 style="margin-top:0; margin-bottom: 5px; text-align: center;">Bank Legend</h4>
         <ul style="list-style-type: none; padding-left: 0;">
     '''
    for bank in banks_on_map:
        color = bank_color_map.get(bank, '#808080') # Default grey
        legend_html += f'<li><span style="background-color:{color}; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border-radius: 50%;"></span>{bank}</li>' # Using circles
    legend_html += '</ul></div>'
    # Add Legend to map
    m.get_root().html.add_child(folium.Element(legend_html))
    # --- End Legend ---

    if map_type == 'clusters':
        marker_cluster = MarkerCluster(
            name="Branch Clusters",
            disable_clustering_at_zoom=9
        ).add_to(m)
        for idx, row in map_df.iterrows():
            popup_text = f"<b>{row['nombre']}</b><br>" \
                         f"<b>Bank:</b> {row['banco']}<br>" \
                         f"<b>State:</b> {row['estado']}<br>"
            
            icon_color_hex = bank_color_map.get(row['banco'], '#808080')

            folium.CircleMarker(
                location=[row['latitud'], row['longitud']],
                radius=5,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{row['banco']} - {row['nombre']}", # Updated tooltip
                color=icon_color_hex,
                fill=True,
                fill_color=icon_color_hex,
                fill_opacity=0.7
            ).add_to(marker_cluster)

    elif map_type == 'heatmap':
        locations = map_df[['latitud', 'longitud']].values.tolist()
        HeatMap(locations, radius=15).add_to(m)

    # LayerControl only useful if multiple layers exist (e.g., heatmap + clusters)
    # folium.LayerControl().add_to(m) # Removed for now as only one layer type is active at a time
    return m

# --- Distribution Analysis Functions ---
def analyze_bank_distribution(df, top_n=20):
    """Analyze branch distribution by bank"""
    bank_counts = df['banco'].value_counts().reset_index()
    bank_counts.columns = ['Bank', 'Number of Branches']
    total_branches = bank_counts['Number of Branches'].sum()
    bank_counts['Percentage'] = bank_counts['Number of Branches'].apply(lambda x: round((x / total_branches * 100), 1))
    return bank_counts.head(top_n)

def analyze_state_distribution(df):
    """Analyze branch distribution by state"""
    state_counts = df['estado'].value_counts().reset_index()
    state_counts.columns = ['State', 'Number of Branches']
    total_branches = state_counts['Number of Branches'].sum()
    state_counts['Percentage'] = state_counts['Number of Branches'].apply(lambda x: round((x / total_branches * 100), 1))
    return state_counts

def analyze_city_distribution(df, state=None, top_n=15):
    """Analyze branch distribution by city within a state, separating 'Sin especificar'."""
    target_df = df[df['estado'] == state] if state and state != "All States" else df
    city_counts_all = target_df['ciudad'].value_counts().reset_index()
    city_counts_all.columns = ['City', 'Number of Branches']

    # Separate 'Sin especificar'
    sin_especificar_count = 0
    if 'Sin especificar' in city_counts_all['City'].values:
        sin_especificar_row = city_counts_all[city_counts_all['City'] == 'Sin especificar']
        if not sin_especificar_row.empty:
             sin_especificar_count = sin_especificar_row['Number of Branches'].iloc[0]
        city_counts = city_counts_all[city_counts_all['City'] != 'Sin especificar'].copy()
    else:
        city_counts = city_counts_all.copy()

    # Calculate percentage based on total including 'Sin especificar' for accuracy
    total_branches_in_filter = target_df.shape[0]
    if total_branches_in_filter > 0:
        city_counts['Percentage'] = city_counts['Number of Branches'].apply(lambda x: round((x / total_branches_in_filter * 100), 1))
    else:
        city_counts['Percentage'] = 0

    return city_counts.head(top_n), sin_especificar_count

def calculate_branch_density(df):
    """Calculate branch density by state using population data"""
    state_branches = df.groupby('estado').size().reset_index(name='location_count')
    state_branches['population'] = state_branches['estado'].map(STATE_POPULATIONS_UPDATED)
    state_branches = state_branches.dropna(subset=['population']) # Only include states with population data

    if state_branches.empty or 'population' not in state_branches.columns or state_branches['population'].eq(0).any():
        st.warning("Could not calculate density for some states due to missing or zero population data.")
        state_branches['branches_per_100k'] = np.nan # Avoid division by zero
    else:
        state_branches['branches_per_100k'] = state_branches.apply(
            lambda row: round((row['location_count'] / row['population'] * 100000), 2) if row['population'] > 0 else 0, 
            axis=1
        )

    return state_branches.sort_values('branches_per_100k', ascending=False).dropna(subset=['branches_per_100k'])

def find_underserved_areas_by_density(df):
    """Identify potentially underserved areas based on branch density"""
    density_data = calculate_branch_density(df)
    if density_data.empty or 'branches_per_100k' not in density_data.columns:
        return pd.DataFrame()

    # Define 'underserved' based on quantile (e.g., bottom 25%)
    # Ensure there are enough data points to calculate quantile
    if len(density_data) >= 4:
        underserved_threshold = density_data['branches_per_100k'].quantile(0.25)
        underserved_states = density_data[density_data['branches_per_100k'] <= underserved_threshold]
    else:
        # If too few states with density data, return the lowest ones or empty
        underserved_states = density_data.sort_values('branches_per_100k').head(max(1, len(density_data) // 4)) # Return lowest if few states

    return underserved_states.sort_values('branches_per_100k')


# --- Main Application Layout ---
def run_streamlit_ui():
    # Load data
    raw_df = load_data() # Load the original, unfiltered data first
    if raw_df.empty:
         st.error("Failed to load or process data. Cannot display dashboard.")
         return # Stop execution if data loading failed

    # --- Key Metrics (Calculated on Raw Data *before* filtering Bienestar) ---
    # Display these first, potentially unaffected by the filter
    st.markdown("<h1 class='main-header'>Mexican Banking Branch Network Analysis</h1>", unsafe_allow_html=True)
    st.markdown("--- ")
    st.markdown("<h2 class='sub-header'>Key Metrics (complete network)</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4) # Changed to 4 columns
    # Calculate metrics based on the *original* raw data
    metrics = {
        "Total Branches": len(raw_df),
        "Unique Banks": raw_df['banco'].nunique(),
        "States Covered": raw_df['estado'].nunique(),
        "Municipalities Covered": raw_df['ciudad'].nunique(),
    }
    cols = [col1, col2, col3, col4] # Changed to 4 columns
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.metric(label, f"{value:,}")
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Apply Filter and Create the main 'df' for analysis ---
    df = raw_df.copy() # Start with a copy of the raw data

    # --- Now proceed with analysis using the potentially filtered 'df' ---

    # Assign bank colors using the potentially filtered df
    # Make sure this happens *after* df is potentially filtered
    bank_color_map = get_bank_colors(df['banco'])
    all_banks = sorted(df['banco'].unique())

    # --- Page Content (Sequential Layout) ---

    # Overview Map (Uses the potentially filtered df)
    st.markdown("<h2 class='sub-header'>Branch Distribution Map</h2>", unsafe_allow_html=True)
    map_type_ov = st.selectbox("Map Type", ["Cluster Map", "Heat Map"], index=1, key="overview_map_type") # Default to Heat Map (index=1)
    map_view_ov = "clusters" if map_type_ov == "Cluster Map" else "heatmap"

    # --- Initialize Session State for Map View (Needed for Overview Map now) ---
    if 'map_center' not in st.session_state:
        st.session_state.map_center = [23.6345, -102.5528] # Default Mexico Center
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 5 # Default Zoom
    if 'current_state_filter' not in st.session_state: # Keep this initialization too
        st.session_state.current_state_filter = "All States"

    with st.spinner("Generating overview map..."):
        # Calculate center/zoom based on the potentially filtered df
        # Handle case where df might be empty after filtering
        if df.empty:
            st.warning("No branches remain after applying filters.")
            # Display a default empty map or message
            # Use session state center/zoom even for empty map
            m_ov = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles='CartoDB positron') # Use default tile
            folium_static(m_ov, width=1400, height=700) # Use consistent size
        else:
            # Use session state center/zoom instead of calculating mean
            # overview_center = [df['latitud'].mean(), df['longitud'].mean()] # Removed
            # overview_zoom = 5 # Removed

            m_ov = create_folium_map(
                df, # Pass the filtered df
                center=st.session_state.map_center, # Use session state center
                zoom=st.session_state.map_zoom,     # Use session state zoom
                map_type=map_view_ov,
                bank_color_map=bank_color_map,
                filtered_banks=None # Show all banks *within the filtered df*
            )
            folium_static(m_ov, width=1400, height=700) # Use consistent width/height

    # Filterable Map Section (Uses the potentially filtered df)
    st.markdown("<h2 class='sub-header'>Interactive Map</h2>", unsafe_allow_html=True)

    # --- Session State Initialization moved above Overview map ---
    # if 'map_center' not in st.session_state:
    #     st.session_state.map_center = [23.6345, -102.5528] # Default Mexico Center
    # if 'map_zoom' not in st.session_state:
    #     st.session_state.map_zoom = 5 # Default Zoom
    # if 'current_state_filter' not in st.session_state:
    #     st.session_state.current_state_filter = "All States" # Track state filter changes


    # Define base map options
    base_map_options = {
        "Default (Positron)": "CartoDB positron",
        "Street Map": "OpenStreetMap",
        "Satellite": "Esri.WorldImagery", # Using Esri satellite view
        "Terrain": "OpenTopoMap", # Replaced Stamen Terrain
        "Dark Mode": "CartoDB dark_matter"
    }

    # Filters in columns
    map_filter_col1, map_filter_col2, map_filter_col3 = st.columns(3)

    with map_filter_col1:
        selected_base_map_name = st.selectbox(
            "Select Base Map:",
            options=list(base_map_options.keys()),
            index=0, # Default to Positron
            key="base_map_select"
        )
        selected_base_map_tile = base_map_options[selected_base_map_name]

    with map_filter_col2:
         # Prepare state list including "All States"
        all_states_list = ["All States"] + sorted([s for s in df['estado'].unique() if s != 'Sin especificar'])
        selected_state_map = st.selectbox(
            "Filter by State:",
            options=all_states_list,
            index=0, # Default to "All States"
            key="state_filter_map"
        )

    with map_filter_col3:
        # Define default banks and add "All Banks" option
        default_banks_list = ['Banamex', 'BBVA Bancomer', 'Santander', 'Banorte', 'Azteca']
        available_default_banks = [bank for bank in default_banks_list if bank in all_banks]
        all_banks_option = "-- All Banks --" # Special option text
        map_options = [all_banks_option] + all_banks # Prepend "All Banks" to the list

        # Select banks to display
        selected_map_banks = st.multiselect(
            "Filter by Bank(s):",
            options=map_options,
            default=available_default_banks if available_default_banks else [all_banks_option],
            key="bank_filter_map"
        )

    # --- Filter Data for Interactive Map ---
    interactive_map_df = df.copy()
    # Apply state filter
    if selected_state_map != "All States":
        interactive_map_df = interactive_map_df[interactive_map_df['estado'] == selected_state_map]

    # Apply bank filter
    banks_to_display_on_map = None # None means show all
    if selected_map_banks:
        if all_banks_option in selected_map_banks:
            banks_to_display_on_map = None # Show all if "All Banks" is selected
        else:
            banks_to_display_on_map = selected_map_banks
            interactive_map_df = interactive_map_df[interactive_map_df['banco'].isin(banks_to_display_on_map)]

    # --- Dynamic Label ---
    label_state = selected_state_map
    label_banks = "All Banks" if banks_to_display_on_map is None else f"{len(banks_to_display_on_map)} selected bank(s)"
    st.markdown(f"**Displaying:** `{len(interactive_map_df):,}` branches | **State:** `{label_state}` | **Banks:** `{label_banks}`")


    # --- Recalculate Center/Zoom ONLY if state changes or first load ---
    # Check if the state filter has changed
    if selected_state_map != st.session_state.current_state_filter:
        if not interactive_map_df.empty:
            st.session_state.map_center = [interactive_map_df['latitud'].mean(), interactive_map_df['longitud'].mean()]
            st.session_state.map_zoom = 6 if selected_state_map != "All States" else 5
        else:
            # If filtered df is empty, reset to default Mexico view
            st.session_state.map_center = [23.6345, -102.5528]
            st.session_state.map_zoom = 5
        # Update the tracked state filter
        st.session_state.current_state_filter = selected_state_map
        print(f"INFO: State filter changed to {selected_state_map}. Recalculating map center and zoom.") # Debug log


    # Map display logic
    if not selected_map_banks:
        st.warning("Please select at least one bank or '-- All Banks --' to display on the map.")
        # Display an empty map or a placeholder message
        m_filtered = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles=selected_base_map_tile)
        folium_static(m_filtered, width=1400, height=700)
    elif interactive_map_df.empty:
         # If filters result in no data, show warning but still display map at last known center/zoom
        st.warning(f"No branches match the current filter criteria (State: {selected_state_map}, Banks: {label_banks}).")
        m_filtered = folium.Map(location=st.session_state.map_center, zoom_start=st.session_state.map_zoom, tiles=selected_base_map_tile)
        folium_static(m_filtered, width=1400, height=700)
    else:
        with st.spinner("Generating interactive map..."):
             # Pass the potentially filtered df and the session state center/zoom
            m_filtered = create_folium_map(
                interactive_map_df, # Pass the already filtered data
                center=st.session_state.map_center, # Use stateful center
                zoom=st.session_state.map_zoom, # Use stateful zoom
                map_type='clusters', # Always use clusters for colored markers/legend
                bank_color_map=bank_color_map,
                filtered_banks=banks_to_display_on_map, # Pass for info in create_folium_map warning
                selected_state=selected_state_map, # Pass for info in create_folium_map warning
                base_map_tile=selected_base_map_tile
            )
            # Use folium_static instead of st_folium for better performance
            folium_static(m_filtered, width=1400, height=700)


    # ----- REDESIGNED BANK DISTRIBUTION ANALYSIS SECTION -----
    st.markdown("--- ")
    st.markdown("<h1 class='main-header'>🏦 Bank Distribution Analysis</h1>", unsafe_allow_html=True)
    
    # Enhanced Bank Analysis - Overview Card
    total_banks = df['banco'].nunique()
    top_bank = df['banco'].value_counts().idxmax()
    top_bank_count = df['banco'].value_counts().max()
    top_bank_pct = round((top_bank_count / len(df) * 100), 1)
    
    bank_metrics_col1, bank_metrics_col2, bank_metrics_col3 = st.columns(3)
    with bank_metrics_col1:
        st.metric("Total Banks", f"{total_banks:,}")
    with bank_metrics_col2:
        st.metric("Leading Bank", top_bank)
    with bank_metrics_col3:
        st.metric("Market Share", f"{top_bank_pct}%", help=f"{top_bank} has {top_bank_count:,} branches")
    
    # Interactive Bank Distribution
    st.markdown("<h2 class='sub-header'>Bank Market Distribution</h2>", unsafe_allow_html=True)
    
    # Set up tabs for different views of bank data
    bank_tab1, bank_tab2, bank_tab3, bank_tab4 = st.tabs(["📊 Bank Ranking", "🔍 Top Banks Analysis", "🔄 Compare Banks", "🏆 Competitive Analysis"])
    
    # -- Tab 1: Bank Ranking --
    with bank_tab1:
        # Allow user to control how many banks to show
        bank_count_options = [10, 20, 30, 50, "All"]
        selected_bank_count = st.select_slider(
            "Number of banks to display:", 
            options=bank_count_options,
            value=20
        )
        
        display_count = df['banco'].nunique() if selected_bank_count == "All" else selected_bank_count
        
        # Allow user to choose between horizontal and vertical layout
        layout_type = st.radio(
            "Chart layout:",
            options=["Horizontal (better for many banks)", "Vertical (better for comparison)"],
            horizontal=True
        )
        
        bank_counts = analyze_bank_distribution(df, top_n=display_count)
        
        # Create a custom color scale based on bank size
        custom_colors = px.colors.sequential.BuPu[::-1]  # Reverse the color scale
        
        if layout_type == "Horizontal":
            # Horizontal bar chart (better for many banks)
            fig_bank_bar = px.bar(
                bank_counts.sort_values('Number of Branches'), 
                y='Bank', 
                x='Number of Branches',
                title=f'Top {display_count} Banks by Branch Count',
                text='Number of Branches',
                color='Number of Branches',
                color_continuous_scale=custom_colors,
                hover_data=['Percentage']
            )
            
            fig_bank_bar.update_traces(
                texttemplate='%{x:,.0f}', 
                textposition='outside',
                hovertemplate='<b>Bank:</b> %{y}<br><b>Branches:</b> %{x:,}<br><b>Market Share:</b> %{customdata[0]:.1f}%<extra></extra>'
            )
            
            fig_bank_bar.update_layout(
                height=max(500, display_count * 25),  # Dynamically adjust height
                yaxis_title="",
                xaxis_title="Number of Branches",
                margin=dict(l=220, r=50, t=50, b=50),
                coloraxis_showscale=False
            )
            
        else:
            # Vertical bar chart (better for comparing values)
            fig_bank_bar = px.bar(
                bank_counts, 
                x='Bank', 
                y='Number of Branches',
                title=f'Top {display_count} Banks by Branch Count',
                text='Number of Branches',
                color='Number of Branches',
                color_continuous_scale=custom_colors,
                hover_data=['Percentage']
            )
            
            fig_bank_bar.update_traces(
                texttemplate='%{y:,.0f}', 
                textposition='outside',
                hovertemplate='<b>Bank:</b> %{x}<br><b>Branches:</b> %{y:,}<br><b>Market Share:</b> %{customdata[0]:.1f}%<extra></extra>'
            )
            
            fig_bank_bar.update_layout(
                height=600,
                xaxis_title="",
                yaxis_title="Number of Branches",
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=50, b=120),
                coloraxis_showscale=False
            )
        
        st.plotly_chart(fig_bank_bar, use_container_width=True)
        
    # -- Tab 2: Top Banks Analysis -- 
    with bank_tab2:
        # More detailed analysis of top banks
        st.markdown("<h3>Leading Banks Detailed Analysis</h3>", unsafe_allow_html=True)
        
        # Allow user to select number of top banks to analyze
        top_n_banks = st.slider("Number of top banks to analyze:", min_value=3, max_value=10, value=5)
        
        top_banks = df['banco'].value_counts().nlargest(top_n_banks).index.tolist()
        top_banks_df = df[df['banco'].isin(top_banks)]
        
        # Geographic spread analysis of top banks
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Geographic Spread</h4>", unsafe_allow_html=True)
            # Calculate state coverage for each bank
            bank_state_coverage = {}
            for bank in top_banks:
                states_covered = df[df['banco'] == bank]['estado'].nunique()
                total_states = df['estado'].nunique()
                coverage_pct = round((states_covered / total_states * 100), 1)
                bank_state_coverage[bank] = {
                    'States Covered': states_covered,
                    'Total States': total_states,
                    'Coverage %': coverage_pct
                }
            
            coverage_df = pd.DataFrame.from_dict(bank_state_coverage, orient='index')
            
            fig_coverage = px.bar(
                coverage_df.reset_index(), 
                x='index', 
                y='Coverage %',
                title='State Coverage by Top Banks (%)',
                labels={'index': 'Bank'},
                color='Coverage %',
                color_continuous_scale='Reds',
                text='Coverage %'
            )
            
            fig_coverage.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='outside'
            )
            
            fig_coverage.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="% of States Covered",
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
        
        with col2:
            st.markdown("<h4>Branch Concentration</h4>", unsafe_allow_html=True)
            
            # Calculate concentration in top 3 states for each bank
            bank_concentration = []
            
            for bank in top_banks:
                bank_df = df[df['banco'] == bank]
                total_branches = len(bank_df)
                state_counts = bank_df['estado'].value_counts()
                top_3_states = state_counts.nlargest(3)
                top_3_pct = round((top_3_states.sum() / total_branches * 100), 1)
                
                top_state = state_counts.index[0] if not state_counts.empty else "N/A"
                top_state_pct = round((state_counts.iloc[0] / total_branches * 100), 1) if not state_counts.empty else 0
                
                bank_concentration.append({
                    'Bank': bank,
                    'Top 3 States %': top_3_pct,
                    'Leading State': top_state,
                    'Leading State %': top_state_pct
                })
            
            concentration_df = pd.DataFrame(bank_concentration)
            
            fig_concentration = px.bar(
                concentration_df,
                x='Bank',
                y='Top 3 States %',
                title='Branch Concentration in Top 3 States (%)',
                color='Leading State %',
                color_continuous_scale='Blues',
                text='Top 3 States %',
                hover_data=['Leading State', 'Leading State %']
            )
            
            fig_concentration.update_traces(
                texttemplate='%{y:.1f}%',
                textposition='outside',
                hovertemplate='<b>Bank:</b> %{x}<br>' +
                              '<b>Top 3 States:</b> %{y:.1f}%<br>' +
                              '<b>Leading State:</b> %{customdata[0]}<br>' +
                              '<b>Leading State Share:</b> %{customdata[1]:.1f}%<extra></extra>'
            )
            
            fig_concentration.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="% of Branches in Top 3 States",
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig_concentration, use_container_width=True)
        
        # Branch network comparison for selected banks
        st.markdown("<h4>Branch Network Comparison</h4>", unsafe_allow_html=True)
        
        # Allow selecting which top banks to compare
        selected_compare_banks = st.multiselect(
            "Select banks to compare:",
            options=top_banks,
            default=top_banks[:min(5, len(top_banks))]
        )
        
        if selected_compare_banks:
            compare_df = df[df['banco'].isin(selected_compare_banks)]
            
            # Create a radar chart for comparison across key metrics
            bank_metrics = []
            
            for bank in selected_compare_banks:
                bank_df = df[df['banco'] == bank]
                total_branches = len(bank_df)
                
                # Calculate metrics
                states_covered = bank_df['estado'].nunique()
                cities_covered = bank_df['ciudad'].nunique()
                
                # Normalize metrics for radar chart
                max_states = df['estado'].nunique()
                states_norm = round((states_covered / max_states * 100), 1)
                
                max_cities = df['ciudad'].nunique()
                cities_norm = round((cities_covered / max_cities * 100), 1)
                
                # Calculate branch density (branches per state)
                branch_per_state = round((total_branches / states_covered), 1)
                max_branch_per_state = 50  # Adjust based on data
                branch_density_norm = min(round((branch_per_state / max_branch_per_state * 100), 1), 100)
                
                # Market share
                market_share = round((total_branches / len(df) * 100), 1)
                
                bank_metrics.append({
                    'Bank': bank,
                    'Geographic Coverage': states_norm,
                    'Urban Presence': cities_norm,
                    'Branch Density': branch_density_norm,
                    'Market Share': market_share,
                    'Absolute Branches': total_branches
                })
            
            metrics_df = pd.DataFrame(bank_metrics)
            
            # Create a spider/radar chart
            categories = ['Geographic Coverage', 'Urban Presence', 'Branch Density', 'Market Share']
            
            fig_radar = go.Figure()
            
            for i, bank in enumerate(metrics_df['Bank']):
                bank_data = metrics_df[metrics_df['Bank'] == bank]
                values = bank_data[categories].values.flatten().tolist()
                values.append(values[0])  # Close the loop
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],  # Close the loop
                    fill='toself',
                    name=f"{bank} ({bank_data['Absolute Branches'].values[0]:,} branches)"
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Bank Network Comparison (normalized metrics)",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Show the raw data in a table
            with st.expander("View detailed metrics table"):
                display_metrics = metrics_df.copy()
                # Add the original values for context
                for bank in selected_compare_banks:
                    bank_df = df[df['banco'] == bank]
                    idx = display_metrics[display_metrics['Bank'] == bank].index
                    display_metrics.loc[idx, 'States Covered'] = bank_df['estado'].nunique()
                    display_metrics.loc[idx, 'Cities Covered'] = bank_df['ciudad'].nunique()
                    
                # Reorder columns for display
                display_cols = ['Bank', 'Absolute Branches', 'States Covered', 'Cities Covered', 
                                'Market Share', 'Geographic Coverage', 'Urban Presence', 'Branch Density']
                display_metrics = display_metrics[display_cols]
                st.dataframe(display_metrics.set_index('Bank'), use_container_width=True)
        
        else:
            st.info("Please select at least one bank to display the comparison.")
    
    # -- Tab 3: Compare Banks --
    with bank_tab3:
        st.markdown("<h3>Custom Bank Comparison</h3>", unsafe_allow_html=True)
        
        # Allow selecting any banks to compare
        all_banks = sorted(df['banco'].unique())
        custom_compare_banks = st.multiselect(
            "Select banks to compare:",
            options=all_banks,
            default=all_banks[:min(3, len(all_banks))] if all_banks else []
        )
        
        if len(custom_compare_banks) > 1:
            compare_banks_df = df[df['banco'].isin(custom_compare_banks)]
            
            # Branch count comparison
            bank_branch_counts = compare_banks_df['banco'].value_counts().reset_index()
            bank_branch_counts.columns = ['Bank', 'Branch Count']
            bank_branch_counts['Market Share (%)'] = bank_branch_counts['Branch Count'].apply(lambda x: round((x / len(df) * 100), 2))
            
            # Sort by branch count
            bank_branch_counts = bank_branch_counts.sort_values('Branch Count', ascending=False)
            
            # Create a horizontal bar chart
            fig_bank_compare = px.bar(
                bank_branch_counts,
                y='Bank',
                x='Branch Count',
                text='Branch Count',
                color='Market Share (%)',
                color_continuous_scale='Viridis',
                title='Branch Network Size Comparison',
                hover_data=['Market Share (%)']
            )
            
            fig_bank_compare.update_traces(
                texttemplate='%{x:,}',
                textposition='outside',
                hovertemplate='<b>Bank:</b> %{y}<br><b>Branches:</b> %{x:,}<br><b>Market Share:</b> %{customdata[0]:.2f}%<extra></extra>'
            )
            
            fig_bank_compare.update_layout(
                height=max(400, len(custom_compare_banks) * 40),
                yaxis_title="",
                xaxis_title="Number of Branches"
            )
            
            st.plotly_chart(fig_bank_compare, use_container_width=True)
            
            # Geographic presence comparison
            st.markdown("<h4>Geographic Presence Comparison</h4>", unsafe_allow_html=True)
            
            # State coverage comparison
            bank_state_data = []
            
            for bank in custom_compare_banks:
                bank_df = df[df['banco'] == bank]
                states_present = bank_df['estado'].value_counts().reset_index()
                states_present.columns = ['State', 'Branch Count']
                states_present['Bank'] = bank
                states_present['% of Bank Branches'] = states_present['Branch Count'].apply(lambda x: round((x / len(bank_df) * 100), 1))
                bank_state_data.append(states_present)
            
            all_bank_state_data = pd.concat(bank_state_data)
            
            # Create a grouped bar chart by state and bank
            fig_state_presence = px.bar(
                all_bank_state_data,
                x='State',
                y='Branch Count',
                color='Bank',
                title='State Presence Comparison',
                barmode='group',
                hover_data=['% of Bank Branches']
            )
            
            fig_state_presence.update_layout(
                height=500,
                xaxis_title="",
                yaxis_title="Branch Count",
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=50, b=120)
            )
            
            # Add a dropdown to filter states
            all_states = sorted(all_bank_state_data['State'].unique())
            
            # Determine which states to preselect (those with most banks present)
            state_bank_counts = all_bank_state_data.groupby('State')['Bank'].nunique().sort_values(ascending=False)
            preselected_states = state_bank_counts.head(10).index.tolist()
            
            selected_states = st.multiselect(
                "Select states to compare:",
                options=all_states,
                default=preselected_states if preselected_states else all_states[:min(10, len(all_states))]
            )
            
            if selected_states:
                filtered_data = all_bank_state_data[all_bank_state_data['State'].isin(selected_states)]
                
                fig_filtered_presence = px.bar(
                    filtered_data,
                    x='State',
                    y='Branch Count',
                    color='Bank',
                    title=f'State Presence Comparison ({len(selected_states)} states)',
                    barmode='group',
                    hover_data=['% of Bank Branches']
                )
                
                fig_filtered_presence.update_layout(
                    height=500,
                    xaxis_title="",
                    yaxis_title="Branch Count",
                    xaxis_tickangle=-45,
                    margin=dict(l=50, r=50, t=50, b=120)
                )
                
                st.plotly_chart(fig_filtered_presence, use_container_width=True)
            else:
                st.info("Please select at least one state to display the comparison.")
            
            # Display a heatmap of bank presence across states
            st.markdown("<h4>Bank Presence Heatmap</h4>", unsafe_allow_html=True)
            
            # Create a pivot table for the heatmap
            presence_pivot = all_bank_state_data.pivot_table(
                values='Branch Count',
                index='Bank',
                columns='State',
                fill_value=0
            )
            
            # Filter to selected states if any
            if selected_states:
                presence_pivot = presence_pivot[selected_states]
            
            # Normalize by row (bank) to show distribution pattern
            normalized_pivot = presence_pivot.div(presence_pivot.sum(axis=1), axis=0) * 100
            
            fig_heatmap = px.imshow(
                normalized_pivot,
                title="Bank Branch Distribution Across States (%)",
                labels=dict(x="State", y="Bank", color="% of Branches"),
                color_continuous_scale="Viridis",
                aspect="auto"
            )
            
            fig_heatmap.update_layout(
                height=max(400, len(custom_compare_banks) * 30),
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=50, b=120)
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        else:
            st.info("Please select at least two banks to display the comparison.")
    
    # -- Tab 4: Competitive Analysis --
    with bank_tab4:
        st.markdown("<h3>Competitive Bank Analysis</h3>", unsafe_allow_html=True)
        
        # Allow selecting a central bank for comparison (default to Banamex if available)
        all_banks = sorted(df['banco'].unique())
        default_central_bank = "BANAMEX" if "BANAMEX" in all_banks else all_banks[0] if all_banks else None
        
        central_bank = st.selectbox(
            "Select central bank for analysis:",
            options=all_banks,
            index=all_banks.index(default_central_bank) if default_central_bank in all_banks else 0
        )
        
        # Create filter controls in columns
        filter_col1, filter_col2 = st.columns([1, 1])
        
        with filter_col1:
            # Create a list of all potential competitors (all banks except the central one)
            all_competitors = [bank for bank in all_banks if bank != central_bank]
            
            # Option to select all competitors
            use_all_competitors = st.checkbox("Compare against all competitors", value=True,
                                             help="Check to compare against all competitors, uncheck to select specific competitors")
            
            if use_all_competitors:
                competitor_banks = all_competitors
                st.info(f"Comparing {central_bank} against all {len(all_competitors)} competitors")
            else:
                # Allow selecting specific competitor banks to compare against
                competitor_banks = st.multiselect(
                    "Select specific competitors to compare against:",
                    options=all_competitors,
                    default=all_competitors[:min(3, len(all_competitors))]
                )
        
        with filter_col2:
            # Add state filter
            all_states = sorted(df['estado'].unique())
            
            # Option to filter by state
            filter_by_state = st.checkbox("Filter by specific states", value=False,
                                         help="Check to analyze only specific states")
            
            if filter_by_state:
                selected_states = st.multiselect(
                    "Select states to include in analysis:",
                    options=all_states,
                    default=all_states[:min(5, len(all_states))]
                )
            else:
                selected_states = all_states
        
        if central_bank and competitor_banks:
            # First apply state filtering to the entire dataset if needed
            if filter_by_state and selected_states:
                filtered_df = df[df['estado'].isin(selected_states)]
                
                # Show how many states are being filtered
                if len(selected_states) < len(all_states):
                    st.info(f"Analysis filtered to {len(selected_states)} states: {', '.join(selected_states[:3])}{' and others' if len(selected_states) > 3 else ''}")
            else:
                filtered_df = df
            
            # Then filter by banks
            central_df = filtered_df[filtered_df['banco'] == central_bank]
            competitors_df = filtered_df[filtered_df['banco'].isin(competitor_banks)]
            comparison_df = pd.concat([central_df, competitors_df])
            
            # Calculate key metrics
            total_branches = len(comparison_df)
            central_branches = len(central_df)
            total_states = comparison_df['estado'].nunique()
            central_states = central_df['estado'].nunique()
            
            # Create metrics for overview
            st.subheader(f"{central_bank} vs. Competitors: Key Metrics")
            col1, col2, col3 = st.columns(3)
            
            competitors_text = "all competitors" if use_all_competitors else f"{len(competitor_banks)} selected competitors"
            
            with col1:
                st.metric(
                    f"{central_bank} Market Share", 
                    f"{round(central_branches / total_branches * 100, 1)}%",
                    help=f"Percentage of branches belonging to {central_bank} compared to {competitors_text}"
                )
            
            with col2:
                competitors_branches = len(competitors_df)
                branch_delta = central_branches - (competitors_branches / len(competitor_banks))
                delta_text = f"{branch_delta:+,.0f} vs. avg competitor"
                
                st.metric(
                    f"Branch Count", 
                    f"{central_branches:,}",
                    delta=delta_text,
                    help=f"Total number of {central_bank} branches compared to competitor average"
                )
            
            with col3:
                competitors_avg_states = sum(df[df['banco'] == bank]['estado'].nunique() for bank in competitor_banks) / len(competitor_banks)
                state_delta = central_states - competitors_avg_states
                delta_text = f"{state_delta:+,.1f} states vs. avg"
                
                st.metric(
                    f"State Coverage", 
                    f"{central_states}/{total_states} states",
                    delta=delta_text,
                    help=f"Number of states where {central_bank} has presence vs. the average competitor coverage"
                )
            
            # --- Create comprehensive comparison table ---
            st.subheader(f"{central_bank} vs. {competitors_text}: Detailed Comparison")
            
            # Calculate per-bank metrics
            bank_metrics = []
            
            # Add central bank first
            central_cities = central_df['ciudad'].nunique()
            bank_metrics.append({
                'Bank': central_bank,
                'Branch Count': central_branches,
                'Market Share (%)': round(central_branches / total_branches * 100, 1),
                'States Covered': central_states,
                'Cities Covered': central_cities,
                'Branches per State': round(central_branches / central_states, 1) if central_states > 0 else 0,
                'Branches per City': round(central_branches / central_cities, 1) if central_cities > 0 else 0,
                'Type': 'Central Bank'
            })
            
            # Add competitor banks
            for bank in competitor_banks:
                bank_df = df[df['banco'] == bank]
                branch_count = len(bank_df)
                state_count = bank_df['estado'].nunique()
                city_count = bank_df['ciudad'].nunique()
                
                bank_metrics.append({
                    'Bank': bank,
                    'Branch Count': branch_count,
                    'Market Share (%)': round(branch_count / total_branches * 100, 1),
                    'States Covered': state_count,
                    'Cities Covered': city_count,
                    'Branches per State': round(branch_count / state_count, 1) if state_count > 0 else 0,
                    'Branches per City': round(branch_count / city_count, 1) if city_count > 0 else 0,
                    'Type': 'Competitor'
                })
            
            # Convert to DataFrame and display
            metrics_df = pd.DataFrame(bank_metrics)
            st.dataframe(metrics_df.set_index('Bank'), use_container_width=True)
            
            # --- Geographic Distribution Comparison ---
            st.subheader(f"{central_bank} vs. {competitors_text}: Geographic Comparison")
            
            # State-level comparison
            # Calculate state-level branch counts for each bank using size() for robustness
            state_pivot = comparison_df.pivot_table(
                index='estado',
                columns='banco',
                aggfunc='size', # Use size instead of count('sucursal')
                fill_value=0
            ).reset_index()

            # Add a total column - Check if state_pivot has non-'estado' columns before summing
            numeric_cols = state_pivot.select_dtypes(include=np.number).columns
            if not numeric_cols.empty:
                state_pivot['Total'] = state_pivot[numeric_cols].sum(axis=1)
            else:
                state_pivot['Total'] = 0 # Handle case where pivot is empty or has no banks

            # Calculate market share percentages for each bank by state
            banks_in_pivot = [col for col in state_pivot.columns if col not in ['estado', 'Total']]
            for bank in banks_in_pivot:
                # Avoid division by zero if Total is 0
                if state_pivot['Total'].sum() > 0: # Check if there's any total count > 0
                    state_pivot[f'{bank} (%)'] = state_pivot.apply(lambda row: (row[bank] / row['Total'] * 100) if row['Total'] > 0 else 0, axis=1)
                else:
                    state_pivot[f'{bank} (%)'] = 0 # Set percentage to 0 if total is 0


            # Sort by central bank presence (if central bank column exists)
            if central_bank in state_pivot.columns:
                state_pivot = state_pivot.sort_values(by=central_bank, ascending=False)
            else:
                # If central bank has no branches in selection, sort by Total or state name
                 if 'Total' in state_pivot.columns:
                      state_pivot = state_pivot.sort_values(by='Total', ascending=False)
                 else: # Fallback if 'Total' somehow isn't there
                     state_pivot = state_pivot.sort_values(by='estado')


            # Create a heatmap
            # Determine which banks and % columns actually exist in the pivot table
            banks_to_show = [bank for bank in ([central_bank] + competitor_banks) if bank in state_pivot.columns]
            percent_cols_to_show = [f'{bank} (%)' for bank in banks_to_show if f'{bank} (%)' in state_pivot.columns]

            # Construct the list of columns for the heatmap data based on existing columns
            # Select only the existing columns to avoid KeyError
            heatmap_data_cols = ['estado'] + banks_to_show # Only need bank counts and state for heatmap source data
            heatmap_data = state_pivot[[col for col in heatmap_data_cols if col in state_pivot.columns]].copy()


            # Rename columns for better display - No longer needed as we only use bank columns for heatmap
            # heatmap_data.columns = [ ... ]

            # Create the heatmap using only the bank count columns
            # Ensure 'estado' exists before setting index
            if 'estado' in heatmap_data.columns:
                 heatmap_pivot_data = heatmap_data.set_index('estado')
                 # Ensure banks_to_show only contains columns actually present after filtering
                 valid_banks_to_show = [bank for bank in banks_to_show if bank in heatmap_pivot_data.columns]

                 if not heatmap_pivot_data.empty and valid_banks_to_show:
                     fig_heatmap = px.imshow(
                         heatmap_pivot_data[valid_banks_to_show], # Use the pivot data with only existing bank counts
                         labels=dict(x="Bank", y="State", color="Branch Count"),
                         x=valid_banks_to_show,
                         color_continuous_scale="Viridis",
                         title=f"Branch Distribution by State: {central_bank} vs {competitors_text}",
                         aspect="auto"
                     )
                     fig_heatmap.update_layout(height=600)
                     st.plotly_chart(fig_heatmap, use_container_width=True)
                 else:
                    st.warning("No data available to display the state distribution heatmap for the selected banks and states.")
            else:
                st.warning("Could not generate state distribution heatmap due to missing 'estado' column.")
            
            # --- Accessibility Analysis ---
            st.subheader(f"{central_bank} vs. {competitors_text}: Accessibility Analysis")
            
            # Calculate overlap between central bank and competitors
            central_cities_set = set(central_df['ciudad'].unique())
            
            overlap_data = []
            for bank in competitor_banks:
                bank_df = df[df['banco'] == bank]
                bank_cities = set(bank_df['ciudad'].unique())
                
                # Calculate overlap
                common_cities = central_cities_set.intersection(bank_cities)
                only_central = central_cities_set - bank_cities
                only_competitor = bank_cities - central_cities_set
                
                overlap_data.append({
                    'Competitor': bank,
                    'Cities with Both Banks': len(common_cities),
                    f'Cities with only {central_bank}': len(only_central),
                    f'Cities with only {bank}': len(only_competitor),
                    'Overlap Percentage': round(len(common_cities) / len(central_cities_set.union(bank_cities)) * 100, 1)
                })
            
            # Display overlap data
            overlap_df = pd.DataFrame(overlap_data)
            st.dataframe(overlap_df.set_index('Competitor'), use_container_width=True)
            
            # Visualize the top 10 states where central bank has advantage/disadvantage
            st.subheader(f"{central_bank} Competitive Position by State")
            
            # Calculate advantage/disadvantage for central bank
            advantage_data = []
            
            for state in state_pivot['estado'].unique():
                state_row = state_pivot[state_pivot['estado'] == state].iloc[0]
                central_share = state_row[f'{central_bank} (%)'] if f'{central_bank} (%)' in state_row else 0
                
                # Calculate the average competitor share
                competitor_shares = [state_row[f'{bank} (%)'] for bank in competitor_banks 
                                    if f'{bank} (%)' in state_row]
                avg_competitor_share = sum(competitor_shares) / len(competitor_shares) if competitor_shares else 0
                
                # Calculate the advantage
                advantage = central_share - avg_competitor_share
                
                advantage_data.append({
                    'State': state,
                    f'{central_bank} Share (%)': round(central_share, 1),
                    'Avg Competitor Share (%)': round(avg_competitor_share, 1),
                    'Advantage (pp)': round(advantage, 1)
                })
            
            # Convert to DataFrame
            advantage_df = pd.DataFrame(advantage_data)
            
            # Create two views - states with advantage and disadvantage
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"#### Top States with {central_bank} Advantage")
                top_advantage = advantage_df.sort_values(by='Advantage (pp)', ascending=False).head(10)
                fig = px.bar(
                    top_advantage,
                    x='State',
                    y='Advantage (pp)',
                    color='Advantage (pp)',
                    color_continuous_scale=[(0, "green"), (1, "darkgreen")],
                    title=f"States where {central_bank} has the largest advantage",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown(f"#### Top States with {central_bank} Disadvantage")
                top_disadvantage = advantage_df.sort_values(by='Advantage (pp)', ascending=True).head(10)
                fig = px.bar(
                    top_disadvantage,
                    x='State',
                    y='Advantage (pp)',
                    color='Advantage (pp)',
                    color_continuous_scale=[(0, "red"), (1, "darkred")],
                    title=f"States where {central_bank} has the largest disadvantage",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif not central_bank:
            st.info("Please select a central bank to display the analysis.")
        elif not competitor_banks:
            st.info("Please select at least one competitor bank to display the analysis.")
        elif central_df.empty:
            if filter_by_state:
                st.warning(f"{central_bank} has no branches in the selected states. Please adjust your state selection.")
            else:
                st.warning(f"No branch data found for {central_bank}. Please select a different bank.")
        elif competitors_df.empty:
            if filter_by_state:
                st.warning(f"The selected competitors have no branches in the selected states. Please adjust your filters.")
            else:
                st.warning("No branch data found for the selected competitors. Please select different banks.")
            
    # Add informative expander about the analysis
    with st.expander("About Bank Distribution Analysis"):
        st.markdown("""
        The Bank Distribution Analysis examines the distribution and market share of banks across Mexico:
        
        - **Bank Ranking**: Shows the relative size of bank branch networks
        - **Top Banks Analysis**: Provides deeper insights into the geographic spread and concentration of leading banks
        - **Bank Comparison**: Allows custom comparison of selected banks' branch networks and state presence
        - **Competitive Analysis**: Provides detailed insights comparing a central bank (e.g., Banamex) against selected competitors, including market share, geographic distribution, and competitive positioning. Supports filtering by specific states and competitor banks.
        
        The analysis uses data from banxico_branches-complete.csv with proper data cleaning and normalization applied.
        """)
    
    # ----- REDESIGNED GEOGRAPHIC DISTRIBUTION SECTION -----
    st.markdown("--- ")
    st.markdown("<h1 class='main-header'>🌎 Geographic Distribution Analysis</h1>", unsafe_allow_html=True)
    
    # Interactive state selector with multiple visualization options
    geo_tab1, geo_tab2, geo_tab3 = st.tabs(["📊 State Rankings", "🏙️ Bank Presence by State", "📈 Regional Analysis"])
    
    # -- Tab 1: State Rankings --
    with geo_tab1:
        st.markdown("<h3>Banking Infrastructure by State</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # State filtering and ranking options
            st.markdown("<h4>Ranking Options</h4>", unsafe_allow_html=True)
            
            # Choose ranking metric
            ranking_metric = st.radio(
                "Rank states by:",
                options=["Total Branches", "Branch Density (per 100k people)", "Bank Variety"],
                index=0
            )
            
            # Choose number of states to display
            display_n_states = st.slider(
                "Number of states to display:",
                min_value=5,
                max_value=min(32, df['estado'].nunique()),
                value=15,
                step=1
            )
            
            # Choose sorting order
            sort_order = st.radio(
                "Sorting order:",
                options=["Highest to Lowest", "Lowest to Highest"],
                index=0,
                horizontal=True
            )
            
            # Display options
            display_type = st.radio(
                "Display as:",
                options=["Bar Chart", "Table"],
                index=0,
                horizontal=True
            )
            
        with col2:
            # Prepare data based on selected metric
            state_distribution = analyze_state_distribution(df)
            
            if ranking_metric == "Total Branches":
                # Already have this from analyze_state_distribution
                state_data = state_distribution.rename(columns={
                    'State': 'Estado', 
                    'Number of Branches': 'Value'
                })
                state_data['Metric'] = 'Total Branches'
                
            elif ranking_metric == "Branch Density (per 100k people)":
                # Calculate branch density
                density_data = calculate_branch_density(df)
                if not density_data.empty:
                    state_data = density_data[['estado', 'branches_per_100k']].copy()
                    state_data.columns = ['Estado', 'Value']
                    state_data['Metric'] = 'Branches per 100k people'
                else:
                    st.warning("Branch density data could not be calculated for all states.")
                    state_data = pd.DataFrame(columns=['Estado', 'Value', 'Metric'])
                    
            elif ranking_metric == "Bank Variety":
                # Calculate number of unique banks per state
                bank_variety = df.groupby('estado')['banco'].nunique().reset_index()
                bank_variety.columns = ['Estado', 'Value']
                bank_variety['Metric'] = 'Unique Banks'
                
                # Calculate percentage of all banks present
                total_banks = df['banco'].nunique()
                bank_variety['Percentage'] = bank_variety['Value'].apply(lambda x: round((x / total_banks * 100), 1))
                state_data = bank_variety
            
            # Sort the data based on user selection
            ascending = sort_order == "Lowest to Highest"
            state_data = state_data.sort_values('Value', ascending=ascending)
            
            # Limit to the selected number of states
            state_data = state_data.head(display_n_states)
            
            # Display based on user selection
            if display_type == "Bar Chart":
                # Create the chart
                color_scale = 'Reds' if ranking_metric != "Bank Variety" else 'Greens'
                
                fig_state_ranking = px.bar(
                    state_data,
                    x='Estado',
                    y='Value',
                    title=f'States by {ranking_metric} ({sort_order})',
                    text='Value',
                    color='Value',
                    color_continuous_scale=color_scale,
                    hover_data=['Percentage'] if 'Percentage' in state_data.columns else None
                )
                
                # Format number display based on metric
                if ranking_metric == "Branch Density (per 100k people)":
                    texttemplate = '%{y:.2f}'
                else:
                    texttemplate = '%{y:,}'
                    
                fig_state_ranking.update_traces(
                    texttemplate=texttemplate,
                    textposition='outside'
                )
                
                # Custom hover template based on metric
                if ranking_metric == "Bank Variety":
                    hovertemplate = '<b>State:</b> %{x}<br><b>Unique Banks:</b> %{y:,}<br><b>% of All Banks:</b> %{customdata[0]:.1f}%<extra></extra>'
                elif ranking_metric == "Branch Density (per 100k people)":
                    hovertemplate = '<b>State:</b> %{x}<br><b>Branches per 100k:</b> %{y:.2f}<extra></extra>'
                else:
                    hovertemplate = '<b>State:</b> %{x}<br><b>Total Branches:</b> %{y:,}<extra></extra>'
                    
                fig_state_ranking.update_traces(
                    hovertemplate=hovertemplate
                )
                
                fig_state_ranking.update_layout(
                    height=500,
                    xaxis_title="",
                    yaxis_title=state_data['Metric'].iloc[0],
                    xaxis_tickangle=-45,
                    margin=dict(l=50, r=50, t=50, b=120)
                )
                
                st.plotly_chart(fig_state_ranking, use_container_width=True)
                
            else:  # Display as table
                # Format the data for display
                display_df = state_data.copy()
                if 'Percentage' in display_df.columns:
                    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
                
                if ranking_metric == "Branch Density (per 100k people)":
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:.2f}")
                else:
                    display_df['Value'] = display_df['Value'].apply(lambda x: f"{x:,}")
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    'Value': ranking_metric
                })
                
                # Drop the Metric column as it's redundant in the table
                if 'Metric' in display_df.columns:
                    display_df = display_df.drop('Metric', axis=1)
                
                # Display the table
                st.dataframe(
                    display_df.set_index('Estado'),
                    use_container_width=True,
                    height=min(35 * (len(display_df) + 1), 500)
                )
    
    # -- Tab 2: Bank Presence by State --
    with geo_tab2:
        st.markdown("<h3>Bank Market Share by State (100% Stacked Bars)</h3>", unsafe_allow_html=True)
        
        # State selection for market share analysis
        all_states = sorted(df['estado'].unique())
        
        # Find states with most branches for default selection
        top_branch_states = df['estado'].value_counts().nlargest(10).index.tolist()
        
        selected_market_states = st.multiselect(
            "Select states to analyze:",
            options=all_states,
            default=top_branch_states[:5] if top_branch_states else all_states[:min(5, len(all_states))]
        )
        
        if selected_market_states:
            # Filter data for selected states
            market_df = df[df['estado'].isin(selected_market_states)]
            
            # Bank selection for market share analysis
            all_banks_in_states = sorted(market_df['banco'].unique())
            top_banks_in_states = market_df['banco'].value_counts().nlargest(10).index.tolist()
            
            selected_market_banks = st.multiselect(
                "Select banks to include (or leave empty for all banks):",
                options=all_banks_in_states,
                default=[]
            )
            
            # Filter by selected banks if any
            if selected_market_banks:
                market_df = market_df[market_df['banco'].isin(selected_market_banks)]
            
            # Calculate market share for each bank in each state
            bank_state_counts = market_df.groupby(['estado', 'banco']).size().reset_index(name='Count')
            
            # Calculate percentage within each state
            state_totals = bank_state_counts.groupby('estado')['Count'].transform('sum')
            bank_state_counts['Percentage'] = bank_state_counts.apply(
                lambda row: round((row['Count'] / state_totals[row.name] * 100), 1), 
                axis=1
            )
            
            # Sort states by total branches
            state_order = bank_state_counts.groupby('estado')['Count'].sum().sort_values(ascending=False).index.tolist()
            
            # Filter to top banks for better visualization if there are many banks and user didn't select specific ones
            if not selected_market_banks and len(all_banks_in_states) > 15:
                # Get the top banks across all selected states
                top_banks_overall = bank_state_counts.groupby('banco')['Count'].sum().nlargest(15).index.tolist()
                
                # Filter to top banks
                filtered_bank_state = bank_state_counts[bank_state_counts['banco'].isin(top_banks_overall)]
                
                # Add an "Others" category for the rest
                others_df = []
                
                for state in selected_market_states:
                    state_data = bank_state_counts[bank_state_counts['estado'] == state]
                    top_state_data = state_data[state_data['banco'].isin(top_banks_overall)]
                    other_banks_count = state_data['Count'].sum() - top_state_data['Count'].sum()
                    
                    if other_banks_count > 0:
                        other_pct = 100 - top_state_data['Percentage'].sum()
                        others_df.append({
                            'estado': state,
                            'banco': 'Otros Bancos',
                            'Count': other_banks_count,
                            'Percentage': other_pct
                        })
                
                if others_df:
                    others_df = pd.DataFrame(others_df)
                    filtered_bank_state = pd.concat([filtered_bank_state, others_df])
                
                plot_data = filtered_bank_state
                st.caption(f"Showing top 15 banks plus 'Otros Bancos' category. Select specific banks to see detailed breakdown.")
            else:
                plot_data = bank_state_counts
            
            # Create the 100% stacked bar chart
            fig_market_share = px.bar(
                plot_data,
                x='estado',
                y='Percentage',
                color='banco',
                title=f'Bank Market Share by State ({len(selected_market_states)} states selected)',
                category_orders={"estado": state_order},
                barmode='stack',
                text='Percentage',
                hover_data=['Count']
            )
            
            # Customize text display - only show percentage for segments > 5%
            fig_market_share.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='inside',
                textfont=dict(size=10, color="white"),
                insidetextanchor='middle',
                hovertemplate='<b>State:</b> %{x}<br><b>Bank:</b> %{fullData.name}<br><b>Share:</b> %{y:.1f}%<br><b>Branches:</b> %{customdata[0]}<extra></extra>'
            )
            
            # Apply a fixed height based on number of states
            fig_market_share.update_layout(
                height=max(400, len(selected_market_states) * 50),
                xaxis_title="State",
                yaxis_title="Market Share (%)",
                yaxis_range=[0, 100],
                yaxis_ticksuffix="%",
                legend_title_text='Bank',
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_market_share, use_container_width=True)
            
            # Add a data table with the exact figures
            with st.expander("View detailed market share data"):
                detail_data = plot_data.pivot_table(
                    values=['Count', 'Percentage'],
                    index='banco',
                    columns='estado',
                    fill_value=0
                )
                
                # Flatten the hierarchical columns
                detail_data.columns = [f"{col[1]} ({col[0]})" for col in detail_data.columns]
                
                # Add a total column
                if 'Count' in bank_state_counts.columns:
                    bank_totals = bank_state_counts.groupby('banco')['Count'].sum()
                    detail_data['Total Branches'] = bank_totals
                
                # Sort by total branches
                detail_data = detail_data.sort_values('Total Branches', ascending=False)
                
                st.dataframe(detail_data, use_container_width=True)
        
        else:
            st.info("Please select at least one state to analyze market share.")
        
    # -- Tab 3: Regional Analysis --
    with geo_tab3:
        st.markdown("<h3>Regional Banking Analysis</h3>", unsafe_allow_html=True)
        
        # Define regions (adjusted for state names in the dataset)
        regions = {
            'Norte': ['Baja California', 'Baja California Sur', 'Chihuahua', 'Coahuila de Zaragoza'],
            'Centro-Norte': ['Aguascalientes'],
            'Centro': ['Ciudad de México', 'Colima'],
            'Sur-Sureste': ['Chiapas', 'Campeche']
        }
        
        # Map states to regions
        df['region'] = df['estado'].map({state: region for region, states in regions.items() for state in states})
        
        # Calculate regional statistics
        region_stats = []
        
        for region in regions:
            region_df = df[df['region'] == region]
            
            if len(region_df) > 0:
                states_count = region_df['estado'].nunique()
                branches_count = len(region_df)
                banks_count = region_df['banco'].nunique()
                
                # Population data for the region
                region_states = regions[region]
                region_population = sum([STATE_POPULATIONS_UPDATED.get(state, 0) for state in region_states])
                
                # Branch density
                if region_population > 0:
                    branches_per_100k = round((branches_count / region_population * 100000), 2)
                else:
                    branches_per_100k = 0
                
                # Top banks
                top_banks = region_df['banco'].value_counts().nlargest(3)
                top_bank_names = ", ".join(top_banks.index.tolist())
                
                region_stats.append({
                    'Region': region,
                    'States': states_count,
                    'Branches': branches_count,
                    'Banks': banks_count,
                    'Population': region_population,
                    'Branches per 100k': branches_per_100k,
                    'Top Banks': top_bank_names,
                    'Branch %': round((branches_count / len(df) * 100), 1)
                })
        
        region_stats_df = pd.DataFrame(region_stats)
        
        # Create regional comparison
        col1, col2 = st.columns([2, 3])
        
        with col1:
            # Regional statistics table
            st.markdown("<h4>Regional Summary</h4>", unsafe_allow_html=True)
            
            # Format the population nicely
            region_stats_df['Population'] = region_stats_df['Population'].apply(lambda x: f"{x:,}")
            
            # Format branches nicely
            region_stats_df['Branches'] = region_stats_df['Branches'].apply(lambda x: f"{x:,}")
            
            # Add % sign to Branch %
            region_stats_df['Branch %'] = region_stats_df['Branch %'].apply(lambda x: f"{x}%")
            
            # Display statistics
            st.dataframe(
                region_stats_df.set_index('Region'),
                use_container_width=True
            )
        
        with col2:
            # Regional visualization
            st.markdown("<h4>Regional Branch Distribution</h4>", unsafe_allow_html=True)
            
            # Choose visualization type
            viz_type = st.radio(
                "Visualization type:",
                options=["Branch Count", "Branch Density", "Bank Variety"],
                horizontal=True
            )
            
            if viz_type == "Branch Count":
                metric = "Branches"
                plot_df = pd.DataFrame({
                    'Region': region_stats_df['Region'],
                    'Value': region_stats_df['Branches'].apply(lambda x: int(x.replace(',', ''))),
                    'Percentage': region_stats_df['Branch %'].apply(lambda x: float(x.replace('%', '')))
                })
                color_scale = 'Reds'
                
            elif viz_type == "Branch Density":
                metric = "Branches per 100k"
                plot_df = pd.DataFrame({
                    'Region': region_stats_df['Region'],
                    'Value': region_stats_df['Branches per 100k'],
                    'Population': region_stats_df['Population']
                })
                color_scale = 'Oranges'
                
            else:  # Bank Variety
                metric = "Banks"
                plot_df = pd.DataFrame({
                    'Region': region_stats_df['Region'],
                    'Value': region_stats_df['Banks'],
                    'Top Banks': region_stats_df['Top Banks']
                })
                color_scale = 'Greens'
            
            # Create the visualization
            fig_region = px.bar(
                plot_df.sort_values('Value', ascending=False),
                x='Region',
                y='Value',
                title=f'Regional Comparison: {metric}',
                text='Value',
                color='Value',
                color_continuous_scale=color_scale
            )
            
            # Customize hover info based on metric
            if viz_type == "Branch Count":
                hovertemplate = '<b>Region:</b> %{x}<br><b>Branches:</b> %{y:,}<br><b>% of Total:</b> %{customdata:.1f}%<extra></extra>'
                customdata = plot_df['Percentage']
            elif viz_type == "Branch Density":
                hovertemplate = '<b>Region:</b> %{x}<br><b>Branches per 100k:</b> %{y:.2f}<br><b>Population:</b> %{customdata}<extra></extra>'
                customdata = plot_df['Population']
            else:  # Bank Variety
                hovertemplate = '<b>Region:</b> %{x}<br><b>Banks Present:</b> %{y:,}<br><b>Top Banks:</b> %{customdata}<extra></extra>'
                customdata = plot_df['Top Banks']
            
            fig_region.update_traces(
                texttemplate='%{y:,}' if viz_type != "Branch Density" else '%{y:.2f}',
                textposition='outside',
                hovertemplate=hovertemplate,
                customdata=customdata
            )
            
            fig_region.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title=metric,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig_region, use_container_width=True)
        
        # Regional bank presence analysis
        st.markdown("<h4>Bank Presence by Region</h4>", unsafe_allow_html=True)
        
        # Select top N banks to analyze
        top_n_regional = st.slider(
            "Number of top banks to analyze:",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        # Get top banks overall
        top_banks_overall = df['banco'].value_counts().nlargest(top_n_regional).index.tolist()
        
        # Calculate bank presence by region
        region_bank_data = []
        
        for bank in top_banks_overall:
            bank_df = df[df['banco'] == bank]
            total_branches = len(bank_df)
            
            for region in regions:
                region_branches = len(bank_df[bank_df['region'] == region])
                if region_branches > 0:
                    region_bank_data.append({
                        'Bank': bank,
                        'Region': region,
                        'Branches': region_branches,
                        'Bank Total': total_branches,
                        'Region Share': round((region_branches / total_branches * 100), 1)
                    })
        
        region_bank_df = pd.DataFrame(region_bank_data)
        
        # Create a heatmap of bank presence across regions
        bank_region_pivot = region_bank_df.pivot_table(
            values='Branches',
            index='Bank',
            columns='Region',
            fill_value=0
        )
        
        # Add a total column
        bank_region_pivot['Total'] = bank_region_pivot.sum(axis=1)
        
        # Sort by total
        bank_region_pivot = bank_region_pivot.sort_values('Total', ascending=False)
        
        # Drop the total column for visualization
        plot_pivot = bank_region_pivot.drop('Total', axis=1)
        
        # Create two different visualizations
        viz_option = st.radio(
            "Visualization option:",
            options=["Absolute Branch Count", "Regional Distribution (%)"],
            horizontal=True
        )
        
        if viz_option == "Regional Distribution (%)":
            # Normalize to show distribution pattern
            plot_data = plot_pivot.div(plot_pivot.sum(axis=1), axis=0) * 100
            colorscale = "Blues"
            title = "Regional Distribution of Bank Branches (%)"
            colorbar_title = "% of Bank Branches"
            hovertemplate = '<b>Bank:</b> %{y}<br><b>Region:</b> %{x}<br><b>Share:</b> %{z:.1f}%<extra></extra>'
        else:
            # Use absolute numbers
            plot_data = plot_pivot
            colorscale = "Viridis"
            title = "Branch Count by Bank and Region"
            colorbar_title = "Branch Count"
            hovertemplate = '<b>Bank:</b> %{y}<br><b>Region:</b> %{x}<br><b>Branches:</b> %{z:,}<extra></extra>'
        
        fig_region_heatmap = px.imshow(
            plot_data,
            title=title,
            labels=dict(x="Region", y="Bank", color=colorbar_title),
            color_continuous_scale=colorscale,
            aspect="auto",
            text_auto=".1f" if viz_option == "Regional Distribution (%)" else True
        )
        
        fig_region_heatmap.update_traces(
            hovertemplate=hovertemplate,
            texttemplate="%{z:.1f}%" if viz_option == "Regional Distribution (%)" else "%{z:,}"
        )
        
        fig_region_heatmap.update_layout(
            height=max(400, len(top_banks_overall) * 30),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig_region_heatmap, use_container_width=True)


    # ----- ENHANCED DEMOGRAPHIC AND ACCESSIBILITY ANALYSIS -----
    st.markdown("--- ")
    st.markdown("<h1 class='main-header'>📊 Demographic & Accessibility Analysis</h1>", unsafe_allow_html=True)
    
    # --- DEBUG: Print unique states in df before demographic analysis ---
    st.write(f"DEBUG: States in df for demographic analysis: {sorted(df['estado'].unique())}")
    # ---------------------------------------------------------------------

    demo_tab1, demo_tab2 = st.tabs(["📈 Population Demographics", "🚶‍♂️ Branch Accessibility"])
    
    # -- Tab 1: Population Demographics --
    with demo_tab1:
        st.markdown("<h3>State Demographics Analysis</h3>", unsafe_allow_html=True)
        
        # Create a DataFrame with state demographic data
        state_demo_data = []
        for estado, data in STATE_DATA.items():
            # Check if the state name from STATE_DATA or a common variation exists in the dataframe
            state_in_df = estado in df['estado'].unique()
            # Explicitly check for 'México' variation for 'Estado de México'
            if not state_in_df and estado == 'Estado de México':
                state_in_df = 'México' in df['estado'].unique()
                if state_in_df:
                    print(f"INFO: Matched 'Estado de México' (STATE_DATA) with 'México' (DataFrame).") # Log successful match

            # Only include states that are present in our dataset
            # if estado in df['estado'].unique():
            if state_in_df:
                 # Find the actual name used in the DataFrame for this state
                actual_state_name_in_df = estado if estado in df['estado'].unique() else 'México'

                state_branch_count = len(df[df['estado'] == actual_state_name_in_df])
                state_pop = data['population']
                branches_per_100k = round((state_branch_count / state_pop * 100000), 2) if state_pop > 0 else 0

                state_demo_data.append({
                    'Estado': estado, # Use the name from STATE_DATA for consistency in the chart hover
                    'Población': data['population'],
                    'Población_adulta_porcentaje': data['adult_population_pct'],
                    'Superficie_km2': data['area_km2'],
                    'Urbanización_porcentaje': data['urban_pct'],
                    'PIB_per_cápita': data['gdp_per_capita'],
                    'Número_de_sucursales': state_branch_count,
                    'Sucursales_por_100k': branches_per_100k
                })
        
        state_df = pd.DataFrame(state_demo_data)
        
        # Calculate adult population in absolute numbers
        state_df['Población_adulta'] = (state_df['Población'] * state_df['Población_adulta_porcentaje'] / 100).round(0)
        
        # Bubble chart for demographic analysis
        st.markdown("<h4>Population Demographics vs Banking Access</h4>", unsafe_allow_html=True)
        
        # Let user select the metrics for the chart axes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_metric = st.selectbox(
                "X-axis:",
                options=["Población", "Población_adulta", "Superficie_km2", "Urbanización_porcentaje", "PIB_per_cápita"],
                index=0,  # Default to "Población" (Total Population)
                format_func=lambda x: {
                    "Población": "Total Population", 
                    "Población_adulta": "Adult Population",
                    "Superficie_km2": "Area (km²)", 
                    "Urbanización_porcentaje": "Urbanization (%)",
                    "PIB_per_cápita": "GDP per Capita"
                }[x]
            )
        
        with col2:
            y_metric = st.selectbox(
                "Y-axis:",
                options=["Sucursales_por_100k", "Urbanización_porcentaje", "Población_adulta_porcentaje", "PIB_per_cápita"],
                index=2,  # Default to "Población_adulta_porcentaje" (Adult Population %)
                format_func=lambda x: {
                    "Sucursales_por_100k": "Branches per 100k People",
                    "Urbanización_porcentaje": "Urbanization (%)",
                    "Población_adulta_porcentaje": "Adult Population (%)",
                    "PIB_per_cápita": "GDP per Capita"
                }[x]
            )
        
        with col3:
            size_metric = st.selectbox(
                "Bubble size:",
                options=["Superficie_km2", "Población", "Número_de_sucursales"],
                index=0,  # Default to "Superficie_km2" (Area in km²)
                format_func=lambda x: {
                    "Superficie_km2": "Area (km²)",
                    "Población": "Total Population",
                    "Número_de_sucursales": "Number of Branches"
                }[x]
            )
        
        # Mapping of metric names to display names for the chart
        metric_display_names = {
            "Población": "Total Population",
            "Población_adulta": "Adult Population",
            "Población_adulta_porcentaje": "Adult Population (%)",
            "Superficie_km2": "Area (km²)",
            "Urbanización_porcentaje": "Urbanization (%)",
            "PIB_per_cápita": "GDP per Capita (USD)",
            "Número_de_sucursales": "Number of Branches",
            "Sucursales_por_100k": "Branches per 100k People"
        }
        
        # Create the bubble chart
        fig_bubble = px.scatter(
            state_df,
            x=x_metric,
            y=y_metric,
            size=size_metric,
            color="Número_de_sucursales",
            hover_name="Estado",
            size_max=50,
            color_continuous_scale="Viridis",
            labels={
                x_metric: metric_display_names[x_metric],
                y_metric: metric_display_names[y_metric],
                size_metric: metric_display_names[size_metric],
                "Número_de_sucursales": "Number of Branches"
            },
            title=f"{metric_display_names[x_metric]} vs {metric_display_names[y_metric]} by State"
        )
        
        # Add text annotations for state names
        fig_bubble.update_traces(
            textposition="top center",
            hovertemplate="<b>%{hovertext}</b><br>" +
                         f"{metric_display_names[x_metric]}: %{{x:,.0f}}<br>" +
                         f"{metric_display_names[y_metric]}: %{{y:.1f}}<br>" +
                         f"{metric_display_names[size_metric]}: %{{marker.size:,.0f}}<br>" +
                         "Branches: %{marker.color:,.0f}"
        )
        
        fig_bubble.update_layout(
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),
            coloraxis_colorbar=dict(title="Number of Branches")
        )
        
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Correlation analysis between demographics and banking presence
        st.markdown("<h4>Correlation Analysis</h4>", unsafe_allow_html=True)
        
        # Calculate correlations between demographic factors and branch density
        correlation_metrics = [
            "Población", "Población_adulta_porcentaje", "Superficie_km2", 
            "Urbanización_porcentaje", "PIB_per_cápita", "Sucursales_por_100k"
        ]
        
        corr_matrix = state_df[correlation_metrics].corr()
        
        # Create a heatmap of the correlations
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Between Demographic Factors and Banking Presence",
            labels=dict(x="Factors", y="Factors", color="Correlation")
        )
        
        # Rename axis labels with display names
        fig_corr.update_xaxes(ticktext=[metric_display_names[m] for m in correlation_metrics], 
                            tickvals=list(range(len(correlation_metrics))))
        fig_corr.update_yaxes(ticktext=[metric_display_names[m] for m in correlation_metrics], 
                            tickvals=list(range(len(correlation_metrics))))
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        with st.expander("View Detailed State Demographic Data"):
            # Display the data in a table
            display_df = state_df.copy()
            # Rename columns for display
            display_df.columns = [metric_display_names.get(col, col) for col in display_df.columns]
            st.dataframe(display_df.set_index("Estado"), use_container_width=True)
    
    # -- Tab 2: Branch Accessibility --
    with demo_tab2:
        st.markdown("<h3>Branch Accessibility Analysis</h3>", unsafe_allow_html=True)
        
        # Explanation of the analysis
        st.info("""
        This section analyzes branch accessibility based on travel time. 
        The visualization shows areas covered by branches within different travel times,
        helping identify underserved areas.
        """)
        
        # Create filter controls in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Let user select a state to analyze
            selected_access_state = st.selectbox(
                "Select state to analyze:",
                options=sorted(df['estado'].unique()),
                index=0
            )
        
        # Filter data for the selected state
        state_df = df[df['estado'] == selected_access_state]
        
        with filter_col2:
            # Let user select banks to include
            state_banks = sorted(state_df['banco'].unique())
            selected_banks = st.multiselect(
                "Filter by bank(s):",
                options=state_banks,
                default=[]
            )
            
            if selected_banks:
                state_df = state_df[state_df['banco'].isin(selected_banks)]
        
        with filter_col3:
            # Let user select the number of banks to show in the metrics
            top_n_banks = st.slider(
                "Top banks to show in metrics:", 
                min_value=3, 
                max_value=min(10, len(state_banks)), 
                value=5
            )
        
        # Display some key metrics
        st.markdown("<h4>Accessibility Metrics</h4>", unsafe_allow_html=True)
        
        # Create columns for basic metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Branches", len(state_df))
        
        with col2:
            # Calculate population per branch
            state_pop = STATE_DATA[selected_access_state]['population']
            pop_per_branch = round(state_pop / len(state_df)) if len(state_df) > 0 else "N/A"
            st.metric("Population per Branch", f"{pop_per_branch:,}" if isinstance(pop_per_branch, int) else pop_per_branch)
        
        with col3:
            # Calculate area per branch
            state_area = STATE_DATA[selected_access_state]['area_km2']
            area_per_branch = round(state_area / len(state_df)) if len(state_df) > 0 else "N/A"
            st.metric("Area per Branch (km²)", f"{area_per_branch:,}" if isinstance(area_per_branch, int) else area_per_branch)
        
        # Bank-specific metrics
        # Display branch count by bank in the selected state
        original_state_df = df[df['estado'] == selected_access_state]
        bank_counts = original_state_df['banco'].value_counts().reset_index()
        bank_counts.columns = ['Bank', 'Number of Branches']
        
        if not bank_counts.empty:
            st.markdown("<h4>Branch Distribution by Bank</h4>", unsafe_allow_html=True)
            
            # Create a bar chart for bank branch counts
            fig_bank_counts = px.bar(
                bank_counts.head(top_n_banks),
                y='Bank',
                x='Number of Branches',
                title=f"Top {top_n_banks} Banks by Branch Count in {selected_access_state}",
                orientation='h',
                color='Number of Branches',
                color_continuous_scale='Viridis'
            )
            
            fig_bank_counts.update_layout(
                xaxis_title="Number of Branches",
                yaxis_title="",
                height=400
            )
            
            st.plotly_chart(fig_bank_counts, use_container_width=True)
        
        # Accessibility map with isochrones
        st.markdown("<h4>Accessibility Map</h4>", unsafe_allow_html=True)
        
        # Transportation and distance controls in columns
        map_control_col1, map_control_col2 = st.columns(2)
        
        with map_control_col1:
            # Let user select transportation mode
            transport_mode = st.radio(
                "Transportation mode:",
                options=["Walking", "Driving"],
                horizontal=True
            )
        
        with map_control_col2:
            # Let user select the time radius to display
            time_radius = st.select_slider(
                "Time radius (minutes):",
                options=[5, 10, 15, 20, 30],
                value=10
            )
        
        # More realistic travel distance estimation
        # Average walking speed is about 5 km/h = 83.33 meters per minute
        # Average driving speed in urban areas is about 30 km/h = 500 meters per minute
        # Rural driving could be faster at 60 km/h = 1000 meters per minute
        # These are rough estimates and would vary by city/terrain/traffic
        
        if transport_mode == "Walking":
            # Walking: ~5 km/h or 83.33 meters per minute
            # For urban areas with stops at intersections, we'll use a conservative estimate
            meters_per_minute = 80
        else:  # Driving
            # Driving: 
            # - Urban/congested areas: ~20-30 km/h = 333-500 meters per minute
            # - Suburban: ~40-50 km/h = 667-833 meters per minute
            # We'll use a balanced estimate considering traffic congestion in cities
            meters_per_minute = 500
        
        # Calculate radius in meters
        radius_meters = time_radius * meters_per_minute
        
        # Cache the map generation to improve performance
        @st.cache_data(ttl=3600, max_entries=10)
        def generate_accessibility_map(state_name, banks_filter, transport_mode, time_radius_min, radius_m):
            """Generate map with accessibility circles - cached for performance"""
            # Filter data
            map_df = original_state_df if not banks_filter else original_state_df[original_state_df['banco'].isin(banks_filter)]
            
            if map_df.empty:
                # Return empty map centered on state
                state_center = [
                    original_state_df['latitud'].mean() if not original_state_df.empty else 23.6345, 
                    original_state_df['longitud'].mean() if not original_state_df.empty else -102.5528
                ]
                empty_map = folium.Map(location=state_center, zoom_start=7)
                
                # Add a title indicating no banks to display
                title_html = f'''
                <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; 
                            background-color: rgba(255, 255, 255, 0.8); padding: 10px 15px; 
                            border-radius: 5px; font-size: 16px; font-weight: bold; text-align: center;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    No banks to display for the selected filters
                </div>
                '''
                empty_map.get_root().html.add_child(folium.Element(title_html))
                
                return empty_map, True
            
            # Create a map centered on the state with the branches
            state_center = [map_df['latitud'].mean(), map_df['longitud'].mean()]
            
            # Create a folium map with performance optimizations
            m = folium.Map(
                location=state_center, 
                zoom_start=9,
                prefer_canvas=True  # Use canvas renderer for better performance
            )
            
            # Add a title showing which banks are being displayed
            banks_displayed = map_df['banco'].unique()
            banks_list = ", ".join(sorted(banks_displayed)) if len(banks_displayed) <= 8 else f"{len(banks_displayed)} banks"
            state_info = state_name if state_name != "All States" else "Mexico"
            title_html = f'''
            <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index:9999; 
                        background-color: rgba(255, 255, 255, 0.8); padding: 10px 15px; 
                        border-radius: 5px; font-size: 16px; font-weight: bold; text-align: center;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                Displaying: {banks_list} in {state_info}
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Create a feature group for each bank for easy filtering
            all_bank_groups = {}
            
            # Get consistent colors for banks
            bank_color_map = get_bank_colors(map_df['banco'])
            
            # Reduce the number of bank groups if there are too many
            unique_banks = map_df['banco'].unique()
            
            # If we have too many banks, group the small ones into "Others"
            if len(unique_banks) > 8:
                # Get bank counts
                bank_counts = map_df['banco'].value_counts()
                major_banks = bank_counts.nlargest(7).index
                
                # Keep major banks in their own groups
                for bank in major_banks:
                    bank_group = folium.FeatureGroup(name=f"{bank}")
                    all_bank_groups[bank] = bank_group
                    m.add_child(bank_group)
                
                # Group minor banks
                others_group = folium.FeatureGroup(name="Other Banks")
                all_bank_groups["Others"] = others_group
                m.add_child(others_group)
            else:
                # Use all banks as is
                for bank in unique_banks:
                    bank_group = folium.FeatureGroup(name=f"{bank}")
                    all_bank_groups[bank] = bank_group
                    m.add_child(bank_group)
            
            # Create a separate circle layer that will be added first (at the bottom layer)
            circles_group = folium.FeatureGroup(name="Accessibility Circles")
            
            # Add all the circles to this base layer first
            batch_size = 100
            for i in range(0, len(map_df), batch_size):
                batch = map_df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    # Create a circle to represent the isochrone
                    folium.Circle(
                        location=[row['latitud'], row['longitud']],
                        radius=radius_m,  # Use calculated radius in meters
                        color='green' if transport_mode == "Walking" else 'orange',
                        fill=True,
                        fill_color='green' if transport_mode == "Walking" else 'orange',
                        fill_opacity=0.1,
                        weight=1,  # Thinner border for performance
                        popup=f"{time_radius_min} min {transport_mode.lower()} distance",
                        z_index_offset=-1000  # Ensure circles are at the bottom
                    ).add_to(circles_group)
            
            # Add the circles layer FIRST to the map (will be at the bottom)
            m.add_child(circles_group)
            
            # Create a marker layer that will ALWAYS be on top
            marker_group = folium.FeatureGroup(name="Branch Markers")
            
            # Add all the markers directly to the top layer
            for i in range(0, len(map_df), batch_size):
                batch = map_df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    bank = row['banco']
                    
                    # Get color for this bank
                    bank_color = bank_color_map.get(bank, '#808080')  # Default to gray if not found
                    
                    # Create marker for this branch (with VERY HIGH z-index)
                    folium.CircleMarker(
                        location=[row['latitud'], row['longitud']],
                        radius=5,
                        color=bank_color,
                        fill=True,
                        fill_color=bank_color,
                        fill_opacity=0.9,  # More opaque to be more visible
                        popup=f"<b>{bank}</b><br>{row['nombre']}",
                        tooltip=bank,
                        z_index_offset=1000000  # Very high z-index to ensure it's on top
                    ).add_to(marker_group)
                    
                    # Also add to bank group for toggle functionality (invisible version)
                    if bank in all_bank_groups:
                        bank_group = all_bank_groups[bank]
                    else:
                        bank_group = all_bank_groups.get("Others", all_bank_groups[list(all_bank_groups.keys())[0]])
                    
                    # Add invisible marker to bank group just for toggling
                    folium.CircleMarker(
                        location=[row['latitud'], row['longitud']],
                        radius=0,  # Invisible
                        opacity=0,
                        fill_opacity=0
                    ).add_to(bank_group)
            
            # Add the marker layer LAST (will be at the top)
            m.add_child(marker_group)
            
            # Add layer control to toggle banks
            folium.LayerControl().add_to(m)
            
            # Add a legend
            legend_html = f'''
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
                    <p><b>Legend:</b></p>
                    <div style="max-height: 150px; overflow-y: auto;">
            '''
            
            # Add bank colors to legend
            banks_on_map = sorted(map_df['banco'].unique())
            for bank in banks_on_map:
                color = bank_color_map.get(bank, '#808080')  # Default to gray if not found
                legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {bank}</p>'
                
            # Add the isochrone legend
            legend_html += f'''
                    </div>
                    <p><i class="fa fa-circle" style="color:{'green' if transport_mode == "Walking" else 'orange'}"></i> {time_radius_min} min {transport_mode} range ({radius_m/1000:.1f} km)</p>
                    <p><small>Use layer control (top right) to toggle banks</small></p>
                </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m, False
            
        # Generate the map
        with st.spinner("Generating accessibility map..."):
            map_obj, is_empty = generate_accessibility_map(
                selected_access_state,
                selected_banks if selected_banks else None,
                transport_mode,
                time_radius,
                radius_meters
            )
            
            if is_empty:
                st.warning("No branches match the selected filters. Please adjust your selection.")
            
            # Render with appropriate settings
            folium_static(
                map_obj, 
                width=1100, 
                height=600
            )
            
            # Add performance note
            if not is_empty and len(state_df) > 50:
                st.info("""
                **Performance Note**: For smoother map interaction with large numbers of branches:
                - Use the bank filter to select specific banks
                - Wait for the map to fully load before zooming/panning
                - Try different transportation modes or time ranges
                """)
                
            if not is_empty:
                # Add a "Regenerate Map" button for when the cache needs clearing
                if st.button("Regenerate Map (if markers appear incorrect)"):
                    st.experimental_rerun()
            
            # Underserved Areas Analysis
            st.markdown("<h4>Underserved Areas Analysis</h4>", unsafe_allow_html=True)
            
            # More realistic coverage calculation based on:
            # 1. Number of branches
            # 2. State area and urban/rural distribution
            # 3. Selected transport mode and time
            
            # Get state data
            urban_pct = STATE_DATA[selected_access_state]['urban_pct'] / 100
            state_area = STATE_DATA[selected_access_state]['area_km2']
            state_pop = STATE_DATA[selected_access_state]['population']
            
            # Calculate total coverage area (km²)
            branch_coverage_radius_km = radius_meters / 1000  # convert to km
            single_branch_coverage_km2 = 3.14159 * (branch_coverage_radius_km ** 2)
            
            # Account for overlap in urban areas (branches tend to cluster)
            # Higher overlap in urban areas, less in rural
            overlap_factor = 0.6 if urban_pct > 0.7 else 0.4
            
            # Total theoretical coverage without overlap
            max_theoretical_coverage_km2 = single_branch_coverage_km2 * len(state_df)
            
            # Adjusted coverage accounting for overlap
            effective_coverage_km2 = max_theoretical_coverage_km2 * (1 - (overlap_factor * urban_pct))
            
            # Cap at state area
            effective_coverage_km2 = min(effective_coverage_km2, state_area)
            
            # Calculate coverage percentage of state area
            area_coverage_pct = round((effective_coverage_km2 / state_area) * 100)
            
            # Population coverage is higher than area coverage because:
            # 1. Branches are concentrated in populated areas
            # 2. Population is not evenly distributed
            rural_weight = 1 - urban_pct
            
            # Population coverage calculation
            # - Higher for walking in urban areas (where density is high)
            # - Much higher for driving (larger radius)
            # - Lower in rural areas (where population is sparse)
            
            # Base coverage from area
            pop_coverage_pct = area_coverage_pct
            
            # Adjust for urban concentration
            urban_boost = urban_pct * 2 # Urban areas get double coverage weight
            
            # Apply transport mode multiplier (driving reaches more people)
            transport_multiplier = 1.0 if transport_mode == "Walking" else 2.5
            
            # Calculate final population coverage percentage
            effective_pop_coverage_pct = min(95, round(pop_coverage_pct * (1 + urban_boost) * transport_multiplier))
            
            # Ensure sensible limits
            if transport_mode == "Walking" and time_radius <= 10:
                effective_pop_coverage_pct = min(effective_pop_coverage_pct, 70)  # Walking has limited range
            
            # Ensure we don't exceed 95% even with many branches
            effective_pop_coverage_pct = min(effective_pop_coverage_pct, 95)
            
            # Final coverage percentages
            covered_pct = effective_pop_coverage_pct
            uncovered_pct = 100 - covered_pct
            
            # Create coverage analysis explanation
            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:15px; border-radius:10px;'>
                <h5>Coverage Analysis Methodology:</h5>
                <ul>
                    <li>State Area: {state_area:,} km²</li>
                    <li>Urban Population: {urban_pct*100:.1f}%</li>
                    <li>Total Branches Analyzed: {len(state_df)}</li>
                    <li>Coverage Radius per Branch: {branch_coverage_radius_km:.2f} km ({time_radius} min {transport_mode.lower()})</li>
                    <li>Estimated Geographic Coverage: {area_coverage_pct}% of state area</li>
                    <li>Estimated Population Coverage: {effective_pop_coverage_pct}% of state population</li>
                </ul>
                <p><small>Note: Coverage estimates account for branch distribution, population density, and transportation mode.</small></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a DataFrame for the coverage data
            coverage_data = pd.DataFrame({
                'Category': ['Covered', 'Underserved'],
                'Population_Percentage': [covered_pct, uncovered_pct],
                'Population': [round(state_pop * covered_pct / 100), round(state_pop * uncovered_pct / 100)]
            })
            
            # Create a pie chart for the coverage
            fig_coverage = px.pie(
                coverage_data,
                values='Population_Percentage',
                names='Category',
                title=f"Estimated Population Coverage ({time_radius} min {transport_mode.lower()} distance)",
                color='Category',
                color_discrete_map={'Covered': 'green', 'Underserved': 'red'},
                hover_data=['Population']
            )
            
            fig_coverage.update_traces(
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Percentage: %{value}%<br>Population: %{customdata[0]:,}'
            )
            
            st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Recommendations for underserved areas
            if uncovered_pct > 30:
                st.warning(f"""
                ### Significant Service Gaps Identified
                
                Based on the analysis, approximately {uncovered_pct}% of the population in {selected_access_state} 
                lacks convenient access to banking services within a {time_radius}-minute {transport_mode.lower()} distance.
                
                **Recommendations:**
                - Consider opening new branches in underserved areas, particularly in medium-sized cities
                - Explore mobile banking solutions for remote regions
                - Partner with local businesses to offer basic banking services in rural areas
                """)
            elif uncovered_pct > 15:
                st.warning(f"""
                ### Moderate Service Gaps Identified
                
                Based on the analysis, approximately {uncovered_pct}% of the population in {selected_access_state} 
                may lack convenient access to banking services.
                
                **Recommendations:**
                - Focus on improving coverage in specific underserved municipalities
                - Consider extending hours in existing branches to improve accessibility
                """)
            else:
                st.success(f"""
                ### Good Coverage
                
                The analysis shows that approximately {covered_pct}% of the population in {selected_access_state} 
                has access to banking services within a {time_radius}-minute {transport_mode.lower()} distance.
                
                The current branch network provides good coverage for this state.
                """)

    st.markdown("--- ")

    # --- Add Disclaimer at Bottom ---
    st.info(
        "Analysis based on data from banxico_branches-complete.csv, which contains a partial dataset covering 9 states in Mexico. Bank names are taken from the 'sucursal' field, branch names from 'direccion', and addresses from 'horario' field in the dataset. Population data used for density calculations is estimated for 2024."
    )
    
    # Add data quality report as collapsible box at the bottom
    if hasattr(df, 'attrs') and 'data_quality' in df.attrs:
        with st.expander("📊 Data Processing Report - Why different branch counts?"):
            quality = df.attrs['data_quality']
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown(f"""
                ### Data Processing Pipeline

                The raw data file contains **{quality['initial_count']:,}** rows, but the analysis shows **{quality['final_count']:,}** branches.
                
                During data processing, some rows were removed due to data quality issues:
                
                | Processing Step | Records Removed | Reason |
                | --------------- | -------------- | ------ |
                | Invalid coordinates | {quality['missing_coords']:,} | Missing latitude/longitude values |
                | Outlier coordinates | {quality['outliers']:,} | Coordinates outside Mexico's borders |
                | **Total removed** | **{quality['initial_count'] - quality['final_count']:,}** | |
                
                This explains the discrepancy between the raw file count and the branch count shown in the analysis.
                """)
            
            with col2:
                # Create a simple flowchart using markdown
                st.markdown("""
                ### Data Cleaning Workflow
                
                ```
                ┌───────────────────┐
                │ Load Raw CSV Data │
                └─────────┬─────────┘
                          ▼
                ┌───────────────────┐
                │  Convert to Nums  │
                │  Remove NA Coords │
                └─────────┬─────────┘
                          ▼
                ┌───────────────────┐
                │   Filter Outlier  │
                │    Coordinates    │
                └─────────┬─────────┘
                          ▼
                ┌───────────────────┐
                │  Clean Text Data  │
                │ (States, Cities)  │
                └─────────┬─────────┘
                          ▼
                ┌───────────────────┐
                │  Final Clean Data │
                └───────────────────┘
                ```
                """)
                
            st.markdown("The cleaning process ensures that all branches have valid geographic coordinates that can be accurately mapped and analyzed.")

# --- Function to run analysis as a script ---
def run_script_analysis():
    print("Starting Banxico Branch Analysis Script...")
    df = load_data()
    if df is None or df.empty:
        print("Failed to load data. Exiting.")
        return

    print(f"\nLoaded {len(df)} branch locations.")
    print(f"Found {df['banco'].nunique()} unique banks across {df['estado'].nunique()} states.")

    print("\n--- Top 20 Banks by Branch Count ---")
    bank_distribution = analyze_bank_distribution(df, top_n=20)
    print(bank_distribution.to_string())

    print("\n--- Top 15 States by Branch Count ---")
    state_distribution = analyze_state_distribution(df)
    print(state_distribution.head(15).to_string())

    print("\n--- Branch Density by State (Top 15) ---")
    density_data = calculate_branch_density(df)
    if not density_data.empty:
        print(density_data.head(15).to_string())
    else:
        print("Could not calculate branch density (check population data mapping).")

    print("\n--- Underserved Areas (based on branch density) ---")
    underserved = find_underserved_areas_by_density(df)
    if not underserved.empty:
        print(underserved.to_string())
    else:
        print("Could not determine underserved areas based on density.")

    print("\nAnalysis Script Finished.")


# --- Entry Point ---
if __name__ == "__main__":
    # Check if Streamlit is running the script
    if 'streamlit' in sys.modules:
        run_streamlit_ui() # Call the UI function when run via Streamlit
    else:
        # Execute the console-based analysis if run directly
        run_script_analysis()