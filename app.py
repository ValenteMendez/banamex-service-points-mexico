import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster, HeatMap, BeautifyIcon
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from PIL import Image
import io
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re # Add import for regex

# Set page configuration
st.set_page_config(
    page_title="Banamex Service Point Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (inherited and potentially adjusted)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #002244; /* Dark Blue */
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #005599; /* Medium Blue */
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0077CC; /* Bright Blue */
    }
    .info-box {
        background-color: #f0f8ff; /* AliceBlue */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0077CC;
        margin-bottom: 20px;
    }
    .stat-box {
        /* background-color: #E6F3FF; /* Light Blue */ */
        /* padding: 15px; */
        /* border-radius: 5px; */
        text-align: center;
        /* box-shadow: 2px 2px 5px rgba(0,0,0,0.1); */
    }
    .highlight {
        color: #FF5733; /* Orange Red */
        font-weight: bold;
    }
    /* Style for expanders */
    .stExpander > details {
        border-left: 3px solid #0077CC;
        border-radius: 5px;
        background-color: #fafafa; /* Slightly off-white */
    }
    .stExpander > details > summary {
        font-weight: bold;
        color: #005599;
    }
    /* Hide GitHub fork button and related elements */
    .css-1fjgr1s, /* Fork button */
    .css-1fjgr1s + div, /* GitHub icon */
    .css-1fjgr1s + div + div /* Three dots menu */
    {
        display: none !important;
    }
    /* Hide the main Streamlit menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Data Loading and Processing ---
@st.cache_data
def load_data():
    """Load and preprocess the data from CSV file"""
    data_path = "all_branches_detailed.csv"
    try:
        df = pd.read_csv(data_path)
        
        # Clean coordinates
        df['latitud'] = pd.to_numeric(df['latitud'], errors='coerce')
        df['longitud'] = pd.to_numeric(df['longitud'], errors='coerce')
        df = df.dropna(subset=['latitud', 'longitud'])
        
        # Extract service types
        service_types = {
            '100': 'has_regular_branch', '110': 'has_premium_branch', '300': 'has_atm',
            '400': 'has_automatic_branch', '600': 'has_afore', '900': 'has_banamex_1',
            '950': 'has_special_service'
        }
        for code, col_name in service_types.items():
            df[col_name] = df['tipos_servicio'].astype(str).str.contains(code, na=False).astype(int)
            
        # --- Correction: Identify ATMs based on 'numero' if 'tipos_servicio' is missing --- 
        # Find rows where numero starts with 'ATM-' but has_atm is currently 0 
        # (likely because tipos_servicio was NaN or didn't contain '300')
        missing_atm_mask = (df['numero'].astype(str).str.startswith('ATM-', na=False)) & (df['has_atm'] == 0)
        df.loc[missing_atm_mask, 'has_atm'] = 1
        # Optionally, update tipos_servicio as well for consistency, though not strictly necessary for counts
        # df.loc[missing_atm_mask, 'tipos_servicio'] = df.loc[missing_atm_mask, 'tipos_servicio'].fillna('').astype(str) + ',300'
        # Note: The above line might add duplicate ',300' if '300' was present but not caught initially.
        # A safer update might be to just ensure 'has_atm' is correct for counting.

        # --- Recalculate total_services AFTER potential ATM correction --- 
        df['total_services'] = df[list(service_types.values())].sum(axis=1)
        
        # Extract accessibility features
        df['has_ramp'] = df['rampas'].fillna(0).astype(int)
        df['has_parking'] = df['estacionamiento'].fillna(0).astype(int)
        df['exchanges_dollars'] = df['recibeDolar'].fillna(0).astype(int)
        
        # Clean geo data initially
        df['estado'] = df['estado'].str.strip()
        df['ciudad'] = df['ciudad'].str.strip()

        # --- Fill missing 'ciudad' using 'dirComplemento' ---
        # Function to extract city from dirComplemento
        def extract_city_from_complemento(complemento, estado):
            if pd.isna(complemento) or pd.isna(estado):
                return None
            
            complemento_str = str(complemento)
            estado_str = str(estado)
            
            # Attempt 1: Find text between last comma and C.P. or state
            # Regex explanation:
            # ,            # Match the last comma
            # \\s*         # Match any whitespace
            # ([^,]+?)    # Capture group 1: One or more characters that are not commas (non-greedy)
            # \\s*,?       # Match any whitespace and an optional comma
            # (?:C\\.P\\.|C.P.) # Match "C.P." or "C.P" (non-capturing group)
            # .*           # Match rest of the string (up to state)
            # |            # OR (if C.P. pattern fails)
            # ,\\s*        # Match the last comma and whitespace
            # ([^,]+)     # Capture group 2: One or more non-comma characters (greedy, likely the city)
            # \\s*,\\s*    # Match whitespace, comma, whitespace
            # {re.escape(estado_str)} # Match the escaped state name
            # \\s*$        # Match optional whitespace at the end of the string
            match_cp = re.search(r",\s*([^,]+?)\s*,?\s*(?:C\.P\.|C\.P\.)", complemento_str)
            match_state_end = re.search(rf",\s*([^,]+)\s*,?\s*{re.escape(estado_str)}\s*$", complemento_str, re.IGNORECASE)

            if match_cp:
                return match_cp.group(1).strip()
            elif match_state_end:
                 # Check if the captured group is the state itself (sometimes state appears twice)
                 potential_city = match_state_end.group(1).strip()
                 if potential_city.lower() != estado_str.lower():
                     return potential_city
            # Fallback: Try extracting the last comma-separated value before the state, excluding C.P. parts
            parts = complemento_str.split(',')
            if len(parts) > 1:
                # Find the state part
                state_index = -1
                for i in range(len(parts) - 1, -1, -1):
                     if estado_str.lower() in parts[i].lower():
                         state_index = i
                         break
                
                # Look for the part before the state or before a C.P. part
                potential_city_index = -1
                if state_index > 0:
                    potential_city_index = state_index - 1
                    # Skip if it looks like a postal code part
                    if "C.P." in parts[potential_city_index] or "C.P" in parts[potential_city_index]:
                        if potential_city_index > 0:
                             potential_city_index -= 1 # Try one more step back
                        else: potential_city_index = -1 # Give up if C.P. is right before state

                # Take the last non-state, non-CP part if no better match
                elif len(parts) > 1 and estado_str.lower() not in parts[-1].lower() and "C.P" not in parts[-1].upper():
                    potential_city_index = -1 # Indicate using the last part
                    potential_city = parts[-1].strip()
                    if potential_city.lower() != estado_str.lower():
                         return potential_city


                if potential_city_index != -1 and potential_city_index < len(parts):
                     potential_city = parts[potential_city_index].strip()
                     # Final check to avoid capturing the state name again
                     if potential_city.lower() != estado_str.lower():
                        return potential_city
                         
            return None # Return None if no logic could extract a city

        # Identify rows where 'ciudad' is NaN or empty string
        missing_city_mask = df['ciudad'].isna() | (df['ciudad'] == '') | (df['ciudad'].str.upper() == 'SIN ESPECIFICAR')

        # Apply the function only to rows where city is missing
        extracted_cities = df.loc[missing_city_mask].apply(
            lambda row: extract_city_from_complemento(row['dirComplemento'], row['estado']), axis=1
        )

        # Fill the missing 'ciudad' values
        df.loc[missing_city_mask, 'ciudad'] = extracted_cities

        # Final cleanup: Fill any remaining NaNs/empty strings with 'Sin especificar'
        df['estado'] = df['estado'].fillna('Sin especificar')
        df['ciudad'] = df['ciudad'].fillna('Sin especificar')
        df.loc[df['ciudad'] == '', 'ciudad'] = 'Sin especificar' # Ensure empty strings are also replaced
        
        # Remove outlier coordinates (optional, adjust bounds if necessary)
        df = df[(df['latitud'] > 14) & (df['latitud'] < 33)]
        df = df[(df['longitud'] > -120) & (df['longitud'] < -85)]
        
        return df
    except FileNotFoundError:
        st.warning(f"Data file not found at {data_path}. Using sample data.")
        return create_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

@st.cache_data
def create_sample_data():
    """Create sample data if real data isn't available"""
    states = ['Ciudad de México', 'Jalisco', 'Nuevo León', 'Estado de México', 
              'Baja California', 'Quintana Roo', 'Yucatán', 'Sinaloa', 'Chihuahua', 'Veracruz']
    np.random.seed(42)
    n_samples = 200
    data = {
        'numero': [f"{i+1000}" for i in range(n_samples)],
        'nombre': [f"Sucursal Demo {i+1}" for i in range(n_samples)],
        'direccion': [f"Calle Ficticia #{i*10}" for i in range(n_samples)],
        'estado': np.random.choice(states, n_samples),
        'ciudad': [f"Ciudad Demo {i%20 + 1}" for i in range(n_samples)],
        'latitud': np.random.uniform(19, 26, n_samples), # More focused coordinates
        'longitud': np.random.uniform(-105, -97, n_samples),
        'estacionamiento': np.random.randint(0, 2, n_samples),
        'rampas': np.random.randint(0, 2, n_samples),
        'recibeDolar': np.random.randint(0, 2, n_samples),
        'tipos_servicio': [','.join(np.random.choice(['100', '110', '300', '400', '600', '900'], 
                                    size=np.random.randint(1, 4), replace=False)) for _ in range(n_samples)]
    }
    df = pd.DataFrame(data)
    # Process sample data similar to real data
    service_types = {
        '100': 'has_regular_branch', '110': 'has_premium_branch', '300': 'has_atm',
        '400': 'has_automatic_branch', '600': 'has_afore', '900': 'has_banamex_1',
        '950': 'has_special_service'
    }
    for code, col_name in service_types.items():
        df[col_name] = df['tipos_servicio'].astype(str).str.contains(code, na=False).astype(int)
    df['total_services'] = df[list(service_types.values())].sum(axis=1)
    df['has_ramp'] = df['rampas']
    df['has_parking'] = df['estacionamiento']
    df['exchanges_dollars'] = df['recibeDolar']
    return df

@st.cache_data
def get_service_type_names():
    """Return mapping of service type codes to names"""
    return {
        'has_regular_branch': 'Regular Branch', 'has_premium_branch': 'Premium Branch',
        'has_atm': 'ATM', 'has_automatic_branch': 'Automatic Branch',
        'has_afore': 'AFORE Services', 'has_banamex_1': 'Banamex 1',
        'has_special_service': 'Special Services'
    }

# --- Geographic Analysis Functions ---
@st.cache_data
def create_folium_map(df, map_type='clusters', service_filter=None, cluster_labels=None):
    """Create a Folium map visualization of branch locations"""
    map_df = df.copy()
    
    # Filter by service
    if service_filter:
        col_name = f'has_{service_filter.lower().replace(" ", "_")}'
        if col_name in map_df.columns:
            map_df = map_df[map_df[col_name] == 1]
        
    # Add cluster labels if provided
    if cluster_labels is not None and 'cluster' not in map_df.columns:
         # Ensure index alignment if df was filtered before clustering
        map_df = map_df.join(cluster_labels.rename('cluster'), how='inner')

    if map_df.empty:
        st.warning("No data points match the current filters for the map.")
        # Use the standard default center even if empty
        return folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='CartoDB positron')

    # Use fixed default center and zoom, matching banxico-adapted-app.py
    # mexico_center = [map_df['latitud'].mean(), map_df['longitud'].mean()] # Removed dynamic center calculation
    m = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='CartoDB positron')

    service_names_map = get_service_type_names()
    
    # Define colors for clusters if applicable
    cluster_colors = px.colors.qualitative.Vivid
    
    if map_type == 'clusters':
        # Use disable_clustering_at_zoom to control when clusters break apart
        marker_cluster = MarkerCluster(
            name="Branch Clusters", 
            disable_clustering_at_zoom=9  # Clusters will disappear at zoom level 9+
        ).add_to(m)
        for idx, row in map_df.iterrows():
            services = [name for col, name in service_names_map.items() if row.get(col) == 1]
            popup_text = f"<b>{row['nombre']} (#{row['numero']})</b><br>" \
                         f"<i>{row['direccion']}</i><br>" \
                         f"<b>State:</b> {row['estado']}<br>" \
                         f"<b>City:</b> {row['ciudad']}<br>" \
                         f"<b>Services:</b> {', '.join(services) or 'N/A'}<br>"
            
            accessibility = []
            if row.get('has_parking') == 1: accessibility.append("Parking")
            if row.get('has_ramp') == 1: accessibility.append("Ramp")
            if row.get('exchanges_dollars') == 1: accessibility.append("USD Exchange")
            if accessibility: popup_text += f"<b>Features:</b> {', '.join(accessibility)}<br>"

            icon_color = 'darkblue' # Default
            icon_symbol = 'university' # Default: bank/university

            if cluster_labels is not None and 'cluster' in row:
                cluster_id = int(row['cluster'])
                popup_text += f"<b>Cluster:</b> {cluster_id}<br>"
                icon_color = cluster_colors[cluster_id % len(cluster_colors)]
                icon_symbol = str(cluster_id) # Use cluster number as icon symbol
            else: # Fallback color logic if no clustering
                 if row.get('has_premium_branch') == 1: icon_color = 'purple'
                 elif row.get('has_regular_branch') == 1: icon_color = 'blue'
                 elif row.get('has_automatic_branch') == 1: icon_color = 'green'
                 elif row.get('has_atm') == 1: icon_color = 'gray'
                 else: icon_color = 'orange'

            folium.Marker(
                location=[row['latitud'], row['longitud']],
                popup=folium.Popup(popup_text, max_width=350),
                # Use BeautifyIcon for more icon options if needed, or standard Icon
                icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix='fa' if len(icon_symbol)>1 else 'glyphicon')
                # Example using BeautifyIcon for number icons:
                # icon=BeautifyIcon(
                #     icon="number", number=int(row['cluster']) if cluster_labels is not None else 0,
                #     border_color=icon_color, text_color=icon_color, background_color="#FFF",
                #     inner_icon_style="font-size:12px;padding-top:2px;"
                # )
            ).add_to(marker_cluster)
            
    elif map_type == 'heatmap':
        locations = map_df[['latitud', 'longitud']].values.tolist()
        HeatMap(locations, radius=15).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# --- Service and Feature Analysis Functions ---
@st.cache_data
def analyze_services(df):
    """Analyze service distribution across branches"""
    service_names_map = get_service_type_names()
    service_cols = list(service_names_map.keys())
    
    service_counts = df[service_cols].sum().reset_index()
    service_counts.columns = ['Service Code', 'Count']
    service_counts['Service Name'] = service_counts['Service Code'].map(service_names_map)
    service_counts = service_counts[['Service Name', 'Count']].sort_values('Count', ascending=False)
    return service_counts

@st.cache_data
def analyze_accessibility(df):
    """Analyze accessibility features across branches"""
    feature_map = {'has_ramp': 'Ramp Access', 'has_parking': 'Parking Available', 'exchanges_dollars': 'Dollar Exchange'}
    feature_cols = list(feature_map.keys())
    
    feature_counts = df[feature_cols].sum().reset_index()
    feature_counts.columns = ['Feature Code', 'Count']
    feature_counts['Feature Name'] = feature_counts['Feature Code'].map(feature_map)
    feature_counts['Percentage'] = (feature_counts['Count'] / len(df) * 100).round(1)
    return feature_counts[['Feature Name', 'Count', 'Percentage']].sort_values('Percentage', ascending=False)

@st.cache_data
def analyze_service_combinations(df, top_n=10):
    """Analyze common service combinations"""
    service_names_map = get_service_type_names()
    # Exclude special services from combinations for clarity?
    service_cols = [k for k,v in service_names_map.items() if v not in ['Special Services']] 
    
    combinations = df.groupby(service_cols).size().reset_index(name='Count')
    combinations = combinations.sort_values('Count', ascending=False).head(top_n)
    
    combination_names = []
    for _, row in combinations.iterrows():
        services = [service_names_map[col] for col in service_cols if row[col] == 1]
        combo_name = ' + '.join(sorted(services)) if services else 'No Standard Services'
        combination_names.append(combo_name)
    
    combinations['Combination'] = combination_names
    combinations['Percentage'] = (combinations['Count'] / len(df) * 100).round(1)
    return combinations[['Combination', 'Count', 'Percentage']]

# --- Geographic Distribution Functions ---
@st.cache_data
def analyze_state_distribution(df):
    """Analyze branch distribution by state"""
    state_counts = df['estado'].value_counts().reset_index()
    state_counts.columns = ['State', 'Number of Branches']
    state_counts['Percentage'] = (state_counts['Number of Branches'] / state_counts['Number of Branches'].sum() * 100).round(1)
    return state_counts

@st.cache_data
def analyze_city_distribution(df, state=None, top_n=15):
    """Analyze branch distribution by city within a state, separating 'Sin especificar'."""
    target_df = df[df['estado'] == state] if state else df
    city_counts_all = target_df['ciudad'].value_counts().reset_index()
    city_counts_all.columns = ['City', 'Number of Branches']
    
    # Separate 'Sin especificar'
    sin_especificar_count = 0
    if 'Sin especificar' in city_counts_all['City'].values:
        sin_especificar_count = city_counts_all.loc[city_counts_all['City'] == 'Sin especificar', 'Number of Branches'].iloc[0]
        city_counts = city_counts_all[city_counts_all['City'] != 'Sin especificar'].copy()
    else:
        city_counts = city_counts_all.copy()

    # Calculate percentage based on total including 'Sin especificar' for accuracy
    total_branches_in_filter = target_df.shape[0] 
    city_counts['Percentage'] = (city_counts['Number of Branches'] / total_branches_in_filter * 100).round(1)
    
    return city_counts.head(top_n), sin_especificar_count

@st.cache_data
def analyze_services_by_state(df):
    """Analyze service availability across states"""
    service_names_map = get_service_type_names()
    service_cols = list(service_names_map.keys())
    
    # Consider analyzing all states or just top N? Analyze all for comprehensiveness.
    services_by_state = df.groupby('estado')[service_cols].sum().reset_index()
    
    melted_df = pd.melt(
        services_by_state, id_vars=['estado'], value_vars=service_cols,
        var_name='Service Code', value_name='Count'
    )
    melted_df['Service Name'] = melted_df['Service Code'].map(service_names_map)
    return melted_df[['estado', 'Service Name', 'Count']]

# --- Advanced Analysis / Clustering ---
# Updated 2024 State Populations
STATE_POPULATIONS_UPDATED = {
    'Aguascalientes': 1500412,
    'Baja California': 3786109,
    'Baja California Sur': 883649,
    'Campeche': 941096,
    'Coahuila': 3354804, # Cleaned from Coahuila de Zaragoza
    'Colima': 733701,
    'Chiapas': 5803144,
    'Chihuahua': 3870533,
    'Ciudad de México': 9338373,
    'Durango': 1903497,
    'Guanajuato': 6330734,
    'Guerrero': 3607623,
    'Hidalgo': 3240790,
    'Jalisco': 8726308,
    'Estado de México': 17727868, # Mapped from 'México'
    'Michoacán': 4941831, # Cleaned from Michoacán de Ocampo
    'Morelos': 1979715,
    'Nayarit': 1248922,
    'Nuevo León': 6128074,
    'Oaxaca': 4297758,
    'Puebla': 6603151,
    'Querétaro': 2541803,
    'Quintana Roo': 1912993,
    'San Luis Potosí': 2880503,
    'Sinaloa': 3100860,
    'Sonora': 3076858,
    'Tabasco': 2527412,
    'Tamaulipas': 3574192,
    'Tlaxcala': 1464291,
    'Veracruz': 8094410, # Cleaned from Veracruz de Ignacio de la Llave
    'Yucatán': 2381597,
    'Zacatecas': 1651236,
}

@st.cache_data
def calculate_service_density(df):
    """Calculate branch and service density by state using updated population data"""
    state_branches = df.groupby('estado').size().reset_index(name='location_count') # Renamed for clarity
    # Use the updated population data dictionary (Confirmed: Uses STATE_POPULATIONS_UPDATED with 2024 data)
    state_branches['population'] = state_branches['estado'].map(STATE_POPULATIONS_UPDATED)
    state_branches = state_branches.dropna(subset=['population'])
    
    if state_branches.empty: return pd.DataFrame() # Handle case where no population data matches

    # Use consistent naming 'locations' or 'service points'
    state_branches['locations_per_100k'] = (state_branches['location_count'] / state_branches['population'] * 100000).round(2)
    
    atm_by_state = df[df['has_atm'] == 1].groupby('estado').size().reset_index(name='atm_count')
    state_branches = state_branches.merge(atm_by_state, on='estado', how='left').fillna({'atm_count': 0})
    state_branches['atms_per_100k'] = (state_branches['atm_count'] / state_branches['population'] * 100000).round(2)
    
    return state_branches.sort_values('locations_per_100k', ascending=False) # Sort by location density

@st.cache_data
def analyze_accessibility_by_region(df):
    """Analyze accessibility features by region"""
    # Simplified regions
    regions = {
        'North': ['Baja California', 'Baja California Sur', 'Sonora', 'Chihuahua', 'Coahuila', 'Nuevo León', 'Tamaulipas'],
        'West': ['Sinaloa', 'Durango', 'Zacatecas', 'Nayarit', 'Jalisco', 'Aguascalientes', 'Colima', 'Michoacán'],
        'Central': ['San Luis Potosí', 'Guanajuato', 'Querétaro', 'Hidalgo', 'Estado de México', 'Ciudad de México', 'Morelos', 'Tlaxcala', 'Puebla'],
        'South/Southeast': ['Guerrero', 'Oaxaca', 'Chiapas', 'Veracruz', 'Tabasco', 'Campeche', 'Yucatán', 'Quintana Roo']
    }
    
    def get_region(state):
        for region, states in regions.items():
            if state in states: return region
        return 'Unknown'
        
    df_copy = df.copy() # Avoid modifying cached df
    df_copy['region'] = df_copy['estado'].apply(get_region)
    
    feature_map = {'has_ramp': 'Ramp Access', 'has_parking': 'Parking Available', 'exchanges_dollars': 'Dollar Exchange'}
    feature_cols = list(feature_map.keys())

    region_features = df_copy.groupby('region')[feature_cols].mean().reset_index()
    
    # Convert means to percentages and rename columns
    for col, name in feature_map.items():
        region_features[f'{name} (%)'] = (region_features[col] * 100).round(1)
        
    return region_features[['region'] + [f'{name} (%)' for name in feature_map.values()]]

@st.cache_data
def find_underserved_areas(df):
    """Identify potentially underserved areas based on branch density"""
    density_data = calculate_service_density(df)
    if density_data.empty: return pd.DataFrame()

    # Define 'underserved' based on quantile (e.g., bottom 25%)
    underserved_threshold = density_data['locations_per_100k'].quantile(0.25)
    underserved_states = density_data[density_data['locations_per_100k'] <= underserved_threshold]
    
    # Add ATM density for context
    return underserved_states.sort_values('locations_per_100k')

@st.cache_data
def perform_branch_clustering(df, n_clusters=5):
    """Performs K-Means clustering on branch features"""
    service_cols = list(get_service_type_names().keys())
    feature_cols = ['has_ramp', 'has_parking', 'exchanges_dollars']
    cluster_features = service_cols + feature_cols + ['total_services']
    
    # Select only numeric features present in the dataframe
    valid_features = [f for f in cluster_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if not valid_features:
        st.error("No valid numeric features found for clustering.")
        return None, None
        
    data_for_clustering = df[valid_features].copy()
    
    # Handle potential NaNs just in case (should be handled in load_data)
    data_for_clustering = data_for_clustering.fillna(0) 
    
    if len(data_for_clustering) < n_clusters:
        st.warning(f"Not enough data points ({len(data_for_clustering)}) for {n_clusters} clusters. Skipping clustering.")
        return None, None

    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    try:
        cluster_labels = kmeans.fit_predict(scaled_data)
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        return None, None

    # Analyze cluster centers (inverse transform for interpretability)
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers = scaler.inverse_transform(cluster_centers_scaled)
    
    cluster_summary = pd.DataFrame(cluster_centers, columns=valid_features)
    cluster_summary['Cluster Size'] = pd.Series(cluster_labels).value_counts().sort_index()
    
    # Add cluster labels back to the original index
    cluster_labels_series = pd.Series(cluster_labels, index=data_for_clustering.index, name='cluster')

    return cluster_labels_series, cluster_summary.round(2)

# --- Main Application Layout ---
def run_streamlit_ui():
    # Load data
    df = load_data()
    
    # --- Remove Sidebar Navigation and add Logo at Top ---
    # st.sidebar.image("https://www.banamex.com/assets/img/home/citibanamexLogo.jpg", width=200)
    # st.sidebar.title("Navigation")
    # page_options = [
    #     "Overview", "Geographic Distribution", "Service Analysis", 
    #     "Accessibility Analysis", "Branch Clustering", "Strategic Insights"
    # ]
    # page = st.sidebar.radio("Select Section", page_options)
    
    # # --- Remove JS for scrolling to top on page change ---
    # st.markdown('''
    #     <script>
    #         // Use setTimeout to delay the scroll slightly, allowing Streamlit to render
    #         setTimeout(function() {
    #             window.scrollTo(0, 0);
    #         }, 100); // Delay of 100 milliseconds
    #     </script>
    # ''', unsafe_allow_html=True)
    # # --- End JS removal ---

    # st.sidebar.markdown("---")
    # st.sidebar.info(
    #     "Uses data from April 1st, 2025, retrieved from https://www.banamex.com/es/localizador-sucursales.html."
    # )
    
    # --- Add Logo to Main Page --- 
    #col1, col2 = st.columns([10, 1])
    #with col2:
    #    st.image("https://www.banamex.com/assets/img/home/citibanamexLogo.jpg", width=100)
    
    # --- Page Content (Sequential Layout) --- 
    
    # --- Overview Section ---
    st.markdown("<h1 class='main-header'>Banamex Service Point Network Analysis</h1>", unsafe_allow_html=True)
    #st.markdown("<div class='info-box'>Welcome! This dashboard analyzes Banamex's branch network using location, service, and accessibility data.</div>", unsafe_allow_html=True)
    st.markdown("--- ") # Add separator
    
    # Key metrics
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    metrics = {
        "Total Service Points": len(df),
        "Regular Branches": df['has_regular_branch'].sum(),
        "ATMs": df['has_atm'].sum(),
        "States Covered": df['estado'].nunique()
    }
    cols = [col1, col2, col3, col4]
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.metric(label, f"{value:,}")
            st.markdown("</div>", unsafe_allow_html=True)

    # Map
    st.markdown("<h2 class='sub-header'>Service Point Distribution Map</h2>", unsafe_allow_html=True)
    map_type_ov = st.selectbox("Map Type", ["Cluster Map", "Heat Map"], key="overview_map_type")
    service_filter_ov = st.selectbox("Filter by Service", 
                                  ["All Services"] + sorted(get_service_type_names().values()), 
                                  key="overview_service_filter")
    
    map_service_filter_ov = None if service_filter_ov == "All Services" else service_filter_ov
    map_view_ov = "clusters" if map_type_ov == "Cluster Map" else "heatmap"
    
    with st.spinner("Generating overview map..."):
        m_ov = create_folium_map(df, map_type=map_view_ov, service_filter=map_service_filter_ov)
        # Update width and height to match banxico-adapted-app.py
        folium_static(m_ov, width=1400, height=700)

    # Service Distribution Bar Chart
    st.markdown("<h2 class='sub-header'>Service Distribution</h2>", unsafe_allow_html=True)
    service_counts = analyze_services(df)
    # Create bar chart for overview service distribution
    fig_service_bar_ov = px.bar(
        service_counts, x='Service Name', y='Count',
        title='Service Types Across All Locations', text='Count'
    )
    fig_service_bar_ov.update_traces(
        marker_color='#0077CC', # Single Blue Color
        texttemplate='%{text:,.0f}', textposition='outside',
        hovertemplate='<b>Service:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>'
    )
    fig_service_bar_ov.update_layout(
        xaxis_title='Service Name', yaxis_title='Count', margin=dict(t=50)
    )
    st.plotly_chart(fig_service_bar_ov, use_container_width=True)

    # About Data Expander
    with st.expander("About the Data and Service Types"):
        st.markdown(
            "Analysis based on Banamex branch locator data (extracted March 2025). Includes location, services, and accessibility. "
        )
        st.markdown(
            "***Note:** Some entries with IDs starting `ATM-` may lack explicit service codes in the source data. These are assumed to be ATMs and are included in the ATM counts during processing.*"
        )
        st.markdown("**Service Types:**")
        service_names_map = get_service_type_names()
        # Restore original keys with parenthetical names
        service_descriptions = {
            'Regular Branch (Sucursales)': "Standard bank branches with teller services.",
            'Premium Branch (Priority)': "Premium banking locations with enhanced services.",
            'ATM (Cajeros)': "Automated Teller Machines.",
            'Automatic Branch (Cajeros Smart)': "Self-service banking locations.",
            'AFORE Services (Afore)': "Retirement fund administration services.",
            'Banamex 1 (Banca Privada)': "High-end private banking services.",
            'Special Services (Banca Patrimonial)': "Wealth management banking services."
        }
        # Modify the loop to find the correct key and display it
        for simple_name in service_names_map.values():
            found_key = None
            found_desc = 'Description not available.'
            # Find the corresponding entry in service_descriptions
            for key, desc in service_descriptions.items():
                # Match if the dictionary key starts with the simple name from service_names_map
                if key.startswith(simple_name):
                    found_key = key
                    found_desc = desc
                    break # Found the match

            # Use the full key (with parenthesis) for display if found
            display_name = found_key if found_key else simple_name
            st.markdown(f"- **{display_name}**: {found_desc}")
    
    st.markdown("--- ") # Add separator

    # --- Geographic Distribution Section ---
    st.markdown("<h1 class='main-header'>Geographic Distribution Analysis</h1>", unsafe_allow_html=True)

    # State Distribution
    state_distribution = analyze_state_distribution(df)
    
    # --- Create Top 15 States Bar Chart with Percentage in Hover ---
    top_15_states_data = state_distribution.nlargest(15, 'Number of Branches')
    fig_state_bar = px.bar(
        top_15_states_data,
        x='State',
        y='Number of Branches', 
        title='Top 15 States by Number of Service Locations', # Updated Title
        text='Number of Branches', 
        hover_data=['Percentage'] 
    )
    fig_state_bar.update_traces(
        marker_color='#0077CC', 
        texttemplate='%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>State:</b> %{x}<br><b>Number of Locations:</b> %{y:,}<br><b>Percentage:</b> %{customdata[0]:.1f}%<extra></extra>' # Updated Hover Text
    )
    fig_state_bar.update_layout(
        xaxis_title='State',
        yaxis_title='Number of Service Locations' # Updated Y-axis label
    )
    st.plotly_chart(fig_state_bar, use_container_width=True)
    st.caption("*Note: This chart counts the total number of unique service point locations per state.*") # Already reflects locations

    # --- Service Point Density Chart ---
    density_data = calculate_service_density(df) # Confirmed uses 2024 population data via STATE_POPULATIONS_UPDATED
    if not density_data.empty:
        density_data_top_15 = density_data.nlargest(15, 'locations_per_100k') # Use top 15 data
        fig_density = px.bar(
            density_data_top_15, # Use top 15 data
            x='estado',
            y='locations_per_100k', # Use updated column name
            title='Top 15 States by Service Location Density (Locations per 100k People)', # Updated Title from 10 to 15
            text='locations_per_100k' # Use updated column name
        )
        fig_density.update_traces(
            marker_color='#0077CC', # Single Blue Color
            texttemplate='%{text:.2f}',
            textposition='outside',
            hovertemplate='<b>State:</b> %{x}<br><b>Density:</b> %{y:.2f}<extra></extra>' # Hover text matches y-axis
        )
        fig_density.update_layout(
            xaxis_title='State', yaxis_title='Service Locations per 100k People', margin=dict(t=50) # Updated Y-axis label
        )
        st.plotly_chart(fig_density, use_container_width=True)
    else:
        st.info("Population data not available or doesn't match states for density calculation.")

    # --- Distribution by City (Commented Out) ---
    # st.markdown("<h2 class='sub-header'>Distribution by City</h2>", unsafe_allow_html=True)
    # all_states = sorted([s for s in df['estado'].unique() if s != 'Sin especificar'])
    # selected_state_gd = st.selectbox("Select State to Analyze Cities", ["All States"] + all_states, key="geo_dist_state_select")
    # 
    # state_filter_gd = None if selected_state_gd == "All States" else selected_state_gd
    # # Get both city data and the count for 'Sin especificar'
    # city_data, sin_especificar_count = analyze_city_distribution(df, state=state_filter_gd, top_n=15)
    # 
    # title_gd = f'Top 15 Cities by Service Point Count' + (f' in {selected_state_gd}' if state_filter_gd else '')
    # if not city_data.empty:
    #     # Create bar chart for city distribution
    #     fig_city_bar = px.bar(
    #         city_data, x='City', y='Number of Branches', title=title_gd, text='Number of Branches'
    #     )
    #     fig_city_bar.update_traces(
    #         marker_color='#0077CC', # Single Blue Color
    #         texttemplate='%{text:,.0f}', textposition='outside',
    #         hovertemplate='<b>City:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>'
    #     )
    #     fig_city_bar.update_layout(
    #          xaxis_title='City', yaxis_title='Number of Service Points', margin=dict(t=50)
    #     )
    #     st.plotly_chart(fig_city_bar, use_container_width=True)
    # else:
    #      st.info(f"No specified cities found for {selected_state_gd if state_filter_gd else 'All States'}.")
    #
    # # Display the count for 'Sin especificar' below the chart
    # if sin_especificar_count > 0:
    #     st.caption(f"* Additionally, {sin_especificar_count} location(s) have 'Sin especificar' value for 'city'.")

    # Service Distribution by State
    with st.expander("Service Distribution Across Top States"):
            #st.markdown("<h3 class='section-header'>Service Counts in Top 10 States</h3>", unsafe_allow_html=True)
            service_by_state = analyze_services_by_state(df)
            top_15_states = state_distribution.head(15)['State'].tolist()
            service_by_top_state = service_by_state[service_by_state['estado'].isin(top_15_states)]

            fig_service_state = px.bar(
                service_by_top_state, x='estado', y='Count', color='Service Name',
                title='Service Distribution in Top 15 States', barmode='stack',
                category_orders={"estado": top_15_states}
            )
            fig_service_state.update_layout(xaxis_title="State", yaxis_title="Service Count")
            st.plotly_chart(fig_service_state, use_container_width=True)
            st.caption("*Note: This chart sums each type of service provided across all locations within a state. A single location can offer multiple services.*")
            
    st.markdown("--- ") # Add separator

    # --- Service Analysis Section ---
    st.markdown("<h1 class='main-header'>Service Analysis</h1>", unsafe_allow_html=True)
    #st.markdown("<div class='info-box'>Analyzing the types of services offered, common combinations, and correlations between services.</div>", unsafe_allow_html=True)

    # Service Distribution Pie & Bar
    st.markdown("<h2 class='sub-header'>Service Type Distribution</h2>", unsafe_allow_html=True)
    # service_counts reused from Overview section
    # col1_sa, col2_sa = st.columns(2) # Commented out column layout as pie chart is removed
    # with col1_sa:
    #     # Create Pie chart for service distribution
    #     fig_service_pie_sa = px.pie(
    #         service_counts, values='Count', names='Service Name', 
    #         title='Service Type Distribution (%)', hole=0.3,
    #         color_discrete_sequence=px.colors.sequential.Blues_r # Use shades of blue
    #     )
    #     fig_service_pie_sa.update_traces(
    #         textposition='inside', textinfo='percent+label',
    #         hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>'
    #     )
    #     st.plotly_chart(fig_service_pie_sa, use_container_width=True)
        
    # Display only the bar chart now
    # with col2_sa: # No longer needed
    service_percentages = service_counts.copy()
    service_percentages['Percentage'] = (service_percentages['Count'] / len(df) * 100).round(1)
    # Create Bar chart for service availability %
    fig_service_perc_sa = px.bar(
        service_percentages, x='Service Name', y='Percentage',
        title='Service Availability (% of Locations)', text='Percentage'
    )
    fig_service_perc_sa.update_traces(
        marker_color='#0077CC', # Single Blue Color
        texttemplate='%{text:.1f}%', textposition='outside',
        hovertemplate='<b>Service:</b> %{x}<br><b>Percentage:</b> %{y:.1f}%<extra></extra>'
    )
    fig_service_perc_sa.update_layout(
        xaxis_title='Service Name', yaxis_title='Percentage of Locations', margin=dict(t=50)
    )
    st.plotly_chart(fig_service_perc_sa, use_container_width=True)

    # Service Combinations
    st.markdown("<h2 class='sub-header'>Common Service Combinations</h2>", unsafe_allow_html=True)
    combinations_data = analyze_service_combinations(df, top_n=10)
    
    # Create the bar chart for combinations
    fig_combo_bar = px.bar(
        combinations_data,
        x='Combination',
        y='Count',
        title='Top 10 Service Combinations',
        text='Count'
    )
    
    # Update traces for single color, text position, and hover format
    fig_combo_bar.update_traces(
        marker_color='#0077CC', # Set a single color (Bright Blue)
        texttemplate='%{text:,.0f}', 
        textposition='outside',
        hovertemplate='<b>Combination:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>' # Format hover text
    )
    
    # Update layout for titles and increased top margin
    fig_combo_bar.update_layout(
        xaxis_title='Service Combination',
        yaxis_title='Number of Locations',
        margin=dict(t=50) # Add top margin to prevent text clipping
    )
    
    st.plotly_chart(fig_combo_bar, use_container_width=True)

    # Expanders for details
    with st.expander("Number of Services per Location & Correlation"):
        # Number of services
            st.markdown("<h3 class='section-header'>Number of Services per Location</h3>", unsafe_allow_html=True)
            service_count_dist = df['total_services'].value_counts().reset_index()
            service_count_dist.columns = ['Number of Services', 'Count']
            service_count_dist = service_count_dist.sort_values('Number of Services')

            # Create bar chart for number of services
            fig_total_serv = px.bar(
                service_count_dist,
                x='Number of Services',
                y='Count',
                title='Distribution of Total Services per Location',
                text='Count'
            )
            fig_total_serv.update_traces(
                marker_color='#0077CC', # Set single color (Medium Blue)
                texttemplate='%{text:,.0f}',
                textposition='outside',
                hovertemplate='<b>Services:</b> %{x}<br><b>Count:</b> %{y:,}<extra></extra>'
            )
            fig_total_serv.update_layout(
                xaxis_title='Number of Different Services Offered',
                yaxis_title='Number of Locations',
                margin=dict(t=50) # Add top margin
            )
            st.plotly_chart(fig_total_serv, use_container_width=True)

            # Display locations with 0 services
            #zero_service_locations = df[df['total_services'] == 0]
            #if not zero_service_locations.empty:
            #    st.markdown("**Locations Reporting 0 Standard Services (Sample):**")
            #    st.dataframe(zero_service_locations[['numero', 'nombre', 'direccion', 'ciudad', 'estado']].head(5))
            #else:
            #    st.write("No locations found reporting 0 standard services.")

        # Correlation heatmap
            st.markdown("<h3 class='section-header'>Service Correlation Heatmap</h3>", unsafe_allow_html=True)
            st.markdown("Shows how often different services appear together. Closer to 1 means they often co-occur, closer to -1 means they rarely do, 0 means no correlation.")
            service_names_map = get_service_type_names()
            service_cols = list(service_names_map.keys())
            correlation_matrix = df[service_cols].corr()
            correlation_matrix.index = correlation_matrix.index.map(service_names_map)
            correlation_matrix.columns = correlation_matrix.columns.map(service_names_map)
            
            fig_corr = px.imshow(
                correlation_matrix, text_auto='.2f', color_continuous_scale='RdBu_r', 
                title='Correlation Between Service Types', aspect="auto", range_color=[-1,1]
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)

    # Service Insights (ATM / Premium)
    st.markdown("<h2 class='sub-header'>Specific Service Insights</h2>", unsafe_allow_html=True)
    col1_si, col2_si = st.columns(2)
    with col1_si:
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-header'>ATM Insights</h3>", unsafe_allow_html=True)
            atm_percentage = (df['has_atm'].sum() / len(df) * 100).round(1)
            standalone_atms = ((df['has_atm'] == 1) & (df['has_regular_branch'] == 0) & (df['has_premium_branch'] == 0)).sum()
            standalone_percentage = (standalone_atms / df['has_atm'].sum() * 100 if df['has_atm'].sum() > 0 else 0).round(1)
            st.markdown(f"- **{atm_percentage}%** of locations have ATMs.")
            st.markdown(f"- **{standalone_atms:,}** are Standalone ATMs ({standalone_percentage}% of all ATMs).")
            st.markdown(f"- **{(df['has_atm'] & (df['has_regular_branch'] | df['has_premium_branch'])).sum():,}** ATMs are co-located with a Branch.")
            st.markdown("</div>", unsafe_allow_html=True)
    with col2_si:
            st.markdown("<div class='stat-box'>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-header'>Premium Service Insights</h3>", unsafe_allow_html=True)
            premium_percentage = (df['has_premium_branch'].sum() / len(df) * 100).round(1)
            banamex1_percentage = (df['has_banamex_1'].sum() / len(df) * 100).round(1)
            st.markdown(f"- **{premium_percentage}%** are Premium Branches.")
            st.markdown(f"- **{banamex1_percentage}%** offer 'Banamex 1' service.")
            st.markdown(f"- **{(df['has_premium_branch'] & df['has_banamex_1']).sum():,}** locations offer both.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("--- ") # Add separator

    # --- Branch Clustering Section ---
    st.markdown("<h1 class='main-header'>Branch Clustering Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Using K-Means clustering to group branches based on their services and features to identify distinct operational profiles.</div>", unsafe_allow_html=True)

    # Select number of clusters
    n_clusters = st.slider("Select Number of Clusters (K)", min_value=2, max_value=10, value=5)

    # Perform clustering
    cluster_labels, cluster_summary = perform_branch_clustering(df, n_clusters)

    if cluster_labels is not None and cluster_summary is not None:
        st.markdown(f"<h2 class='sub-header'>Cluster Profiles (K={n_clusters})</h2>", unsafe_allow_html=True)
        st.markdown("The table below shows the average characteristics for each cluster. Higher values indicate that feature/service is more common in that cluster.")
        
        # Display cluster summary table
        st.dataframe(cluster_summary)

        # Display cluster sizes
        st.markdown("<h3 class='section-header'>Cluster Sizes</h3>", unsafe_allow_html=True)
        cluster_sizes = cluster_summary[['Cluster Size']].reset_index().rename(columns={'index':'Cluster'})
        # Create bar chart for cluster sizes
        fig_cluster_size = px.bar(
            cluster_sizes, x='Cluster', y='Cluster Size',
            title='Number of Branches per Cluster', text='Cluster Size'
        )
        fig_cluster_size.update_traces(
            marker_color='#0077CC', # Single Blue Color
            texttemplate='%{text:,.0f}', textposition='outside',
            hovertemplate='<b>Cluster:</b> %{x}<br><b>Size:</b> %{y:,}<extra></extra>'
        )
        fig_cluster_size.update_layout(
            xaxis_title='Cluster ID', yaxis_title='Number of Branches', margin=dict(t=50)
        )
        st.plotly_chart(fig_cluster_size, use_container_width=True)

        # Optional: PCA Visualization (in expander)
        with st.expander("Cluster Visualization using PCA (2D & 3D)"):
            st.markdown("Visualizing clusters using the first 2 or 3 Principal Components (PCA) for dimensionality reduction. This helps see separation, but is an approximation.")
            
            service_cols = list(get_service_type_names().keys())
            feature_cols = ['has_ramp', 'has_parking', 'exchanges_dollars', 'total_services']
            valid_features = [f for f in service_cols + feature_cols if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
            
            if len(valid_features) >= 2:
                data_for_pca = df[valid_features].fillna(0)
                scaled_data = StandardScaler().fit_transform(data_for_pca)
                
                # --- 2D PCA Plot ---
                st.markdown("**2D PCA Plot**")
                pca_2d = PCA(n_components=2)
                principal_components_2d = pca_2d.fit_transform(scaled_data)
                pca_df_2d = pd.DataFrame(data = principal_components_2d, columns = ['PC1', 'PC2'], index=data_for_pca.index)
                
                # Join cluster labels using index alignment
                pca_df_2d = pca_df_2d.join(cluster_labels) # Assumes cluster_labels has same index
                pca_df_2d['cluster'] = pca_df_2d['cluster'].astype(str) # For discrete color mapping
                
                fig_pca_2d = px.scatter(
                    pca_df_2d, x='PC1', y='PC2', color='cluster',
                    title='Branch Clusters (2D PCA Visualization)',
                    hover_data={ 'PC1':False, 'PC2':False, 'cluster':True, 'Branch Name': df.loc[pca_df_2d.index, 'nombre'] },
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                st.plotly_chart(fig_pca_2d, use_container_width=True)
                explained_variance_2d = pca_2d.explained_variance_ratio_.sum() * 100
                st.write(f"Top 2 PCs explain {explained_variance_2d:.1f}% of the variance.")

                # --- 3D PCA Plot ---
                if len(valid_features) >= 3:
                    st.markdown("**3D PCA Plot**")
                    pca_3d = PCA(n_components=3)
                    principal_components_3d = pca_3d.fit_transform(scaled_data)
                    pca_df_3d = pd.DataFrame(data = principal_components_3d, columns = ['PC1', 'PC2', 'PC3'], index=data_for_pca.index)
                    
                    # Join cluster labels using index alignment
                    pca_df_3d = pca_df_3d.join(cluster_labels)
                    pca_df_3d['cluster'] = pca_df_3d['cluster'].astype(str)
                    
                    fig_pca_3d = px.scatter_3d(
                        pca_df_3d, x='PC1', y='PC2', z='PC3', color='cluster',
                        title='Branch Clusters (3D PCA Visualization)',
                        hover_data={ 'PC1':False, 'PC2':False, 'PC3':False, 'cluster':True, 'Branch Name': df.loc[pca_df_3d.index, 'nombre'] },
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    fig_pca_3d.update_traces(marker_size=3)
                    st.plotly_chart(fig_pca_3d, use_container_width=True)
                    explained_variance_3d = pca_3d.explained_variance_ratio_.sum() * 100
                    st.write(f"Top 3 PCs explain {explained_variance_3d:.1f}% of the variance.")
                else:
                    st.warning("Not enough features (at least 3 required) for 3D PCA visualization.")
            else:
                st.warning("Not enough features (at least 2 required) for PCA visualization.")

    else:
        st.warning("Clustering could not be performed with the current settings or data.")

    st.markdown("--- ") # Add separator

    # --- Strategic Insights Section ---
    st.markdown("<h1 class='main-header'>Strategic Insights</h1>", unsafe_allow_html=True)
    #st.markdown("<div class='info-box'>Synthesizing analysis findings into potential strategic actions regarding network expansion, service offerings, and accessibility improvements.</div>", unsafe_allow_html=True)

    # Underserved Areas Analysis
    st.markdown("<h2 class='sub-header'>Potentially Underserved Areas (based on location density)</h2>", unsafe_allow_html=True) # Updated sub-header
    underserved_areas = find_underserved_areas(df) # Uses calculate_service_density which is updated
    if not underserved_areas.empty:
        # Create bar chart for underserved areas
        fig_underserved = px.bar(
            underserved_areas, x='estado', y='locations_per_100k', # Use updated column name
            title='States with Lowest Service Location Density (Bottom 25%)', text='locations_per_100k' # Updated Title & text source
        )
        fig_underserved.update_traces(
            marker_color='#0077CC', # Single Blue Color
            texttemplate='%{text:.2f}',
            textposition='outside',
            hovertemplate='<b>State:</b> %{x}<br><b>Density:</b> %{y:.2f}<extra></extra>' # Hover text matches y-axis
        )
        fig_underserved.update_layout(
            xaxis_title='State', yaxis_title='Service Locations per 100k People', margin=dict(t=50) # Updated Y-axis label
        )
        st.plotly_chart(fig_underserved, use_container_width=True)
        
        lowest_density_states = underserved_areas['estado'].tolist()
        # Use consistent terminology in the finding text
        st.markdown(f"**Finding:** States like **{', '.join(lowest_density_states[:3])}** show the lowest service location density relative to population. This suggests potential for targeted expansion or alternative service models (digital, partnerships) in these areas.")
        
        with st.expander("View Underserved States Data"):
            st.dataframe(
                # Use updated column names for display
                underserved_areas[['estado', 'location_count', 'population', 'locations_per_100k', 'atms_per_100k']],
                column_config={
                    # Update column config labels
                    "location_count": st.column_config.NumberColumn("Location Count"),
                    "population": st.column_config.NumberColumn("Population (2024)", format="localized"),
                    "locations_per_100k": st.column_config.NumberColumn("Locations per 100k", format="%.2f"),
                    "atms_per_100k": st.column_config.NumberColumn("ATMs per 100k", format="%.2f"),
                },
                use_container_width=True
            )
    else:
        st.info("Could not identify underserved areas due to lack of population data or insufficient state variation.")

    # Service Gap & Opportunity Analysis (Improved Logic)
    st.markdown("<h2 class='sub-header'>Service Gap & Opportunity Analysis</h2>", unsafe_allow_html=True)
    
    # Calculate service prevalence (%) per state
    service_names_map = get_service_type_names()
    service_cols = list(service_names_map.keys())
    state_service_perc = df.groupby('estado')[service_cols].mean().reset_index()
    
    # Calculate national average prevalence for key services
    key_services = ['has_regular_branch', 'has_atm', 'has_premium_branch', 'has_automatic_branch', 'has_afore']
    national_avg_perc = df[key_services].mean()
    
    # Identify states significantly below national average (e.g., < 75% of national avg)
    service_gaps_detailed = {}
    for service_col in key_services:
            avg = national_avg_perc[service_col]
            threshold = avg * 0.75 
            below_avg_states = state_service_perc[state_service_perc[service_col] < threshold]['estado'].tolist()
            if below_avg_states:
                service_gaps_detailed[service_names_map[service_col]] = below_avg_states

    st.markdown("<h3 class='section-header'>States with Potential Service Gaps</h3>", unsafe_allow_html=True)
    if service_gaps_detailed:
        for service, states in service_gaps_detailed.items():
            st.markdown(f"**{service}**: Significantly below nat. average in **{len(states)} states** (e.g., {', '.join(states[:3])}{', ...' if len(states) > 3 else ''})")
    else:
            st.write("No significant service gaps identified based on the 75% threshold.")

    # Code for Service Opportunity Score now follows sequentially
    st.markdown("<h3 class='section-header'>Service Opportunity Score</h3>", unsafe_allow_html=True)
    st.markdown("*Score based on how many key services are below the national average threshold in each state.*")
    opportunity_score = {state: 0 for state in state_service_perc['estado']}
    for service, states in service_gaps_detailed.items():
        for state in states:
            if state in opportunity_score: opportunity_score[state] += 1
    
    opportunity_df = pd.DataFrame(list(opportunity_score.items()), columns=['State', 'Opportunity Score'])
    opportunity_df = opportunity_df.sort_values('Opportunity Score', ascending=False)
    
    # Create bar chart for opportunity score
    top_10_opp_states = opportunity_df.nlargest(10, 'Opportunity Score')
    fig_opp_score = px.bar(
        top_10_opp_states, x='State', y='Opportunity Score',
        title='Top 10 States by Service Opportunity Score', text='Opportunity Score'
    )
    fig_opp_score.update_traces(
        marker_color='#0077CC', # Single Blue Color
        texttemplate='%{text:,.0f}', textposition='outside',
        hovertemplate='<b>State:</b> %{x}<br><b>Score:</b> %{y}<extra></extra>'
    )
    fig_opp_score.update_layout(
        xaxis_title='State', yaxis_title='Opportunity Score', margin=dict(t=50)
    )
    st.plotly_chart(fig_opp_score, use_container_width=True)
    top_opportunity_states = opportunity_df.head(3)['State'].tolist()
    if top_opportunity_states:
            st.markdown(f"**Finding:** States like **{', '.join(top_opportunity_states)}** show the most potential service gaps across multiple categories, suggesting focused review for service enhancement or introduction.")

    # # Accessibility Improvement Opportunities
    # st.markdown("<h2 class='sub-header'>Accessibility Improvement Opportunities</h2>", unsafe_allow_html=True)
    # accessibility_by_state = df.groupby('estado')[['has_ramp', 'has_parking', 'exchanges_dollars']].mean().reset_index()
    # accessibility_by_state['Overall Accessibility (%)'] = accessibility_by_state[['has_ramp', 'has_parking', 'exchanges_dollars']].mean(axis=1) * 100
    # accessibility_by_state = accessibility_by_state.sort_values('Overall Accessibility (%)')

    # # Create bar chart for accessibility score
    # top_10_access_states = accessibility_by_state.nsmallest(10, 'Overall Accessibility (%)') # Lowest scores
    # fig_access_score = px.bar(
    #     top_10_access_states, x='estado', y='Overall Accessibility (%)',
    #     title='Top 10 States with Lowest Overall Accessibility Scores', text='Overall Accessibility (%)'
    # )
    # fig_access_score.update_traces(
    #     marker_color='#0077CC', # Single Blue Color
    #     texttemplate='%{text:.1f}%',
    #     textposition='outside',
    #     hovertemplate='<b>State:</b> %{x}<br><b>Avg. Accessibility:</b> %{y:.1f}%<extra></extra>'
    # )
    # fig_access_score.update_layout(
    #     xaxis_title='State', yaxis_title='Overall Accessibility (%)', margin=dict(t=50)
    # )
    # st.plotly_chart(fig_access_score, use_container_width=True)
    # lowest_access_states = accessibility_by_state.head(3)['estado'].tolist()
    # if lowest_access_states:
    #     st.markdown(f"**Key Finding:** States like **{', '.join(lowest_access_states)}** have the lowest average accessibility scores across ramps, parking, and dollar exchange. These could be priority areas for accessibility upgrades.")
    
    # Refined Strategic Recommendations (using expanders)
    st.markdown("<h2 class='sub-header'>Strategic Recommendations Summary</h2>", unsafe_allow_html=True)
    
    with st.expander("1. Targeted Network Expansion", expanded=True):
            st.markdown(f"- **Focus Expansion in Underserved States:** Prioritize states identified with the lowest location density (e.g., **{', '.join(lowest_density_states[:3])}**) for new location potential analysis or deployment of alternative solutions (mobile units, enhanced ATMs).")
            st.markdown(f"- **Address Specific Service Gaps:** In states with high 'Opportunity Scores' (e.g., **{', '.join(top_opportunity_states)}**), investigate the feasibility of adding missing key services like ATMs, AFORE, or specific branch types based on local demand analysis.")
            st.markdown("- **Optimize ATM Placement:** Beyond just state-level density, analyze intra-state distribution to place new ATMs in high-traffic areas currently lacking easy access, potentially leveraging standalone ATM insights.")

    with st.expander("2. Enhance Service Offerings & Accessibility", expanded=True):
            st.markdown(f"- **Prioritize Accessibility Upgrades:** Target states with the lowest overall accessibility scores with improvements regarding ramps, parking, and potentially dollar exchange where relevant.")
            st.markdown("- **Leverage Branch Clustering Insights:** Tailor services or marketing based on identified branch profiles (from the Clustering section). E.g., Promote specialized services at 'Full-Service Hub' clusters, or ensure basic transaction efficiency at 'ATM-Heavy' clusters.")
            st.markdown("- **Improve Digital Integration:** Promote digital channel adoption strongly in areas with lower physical branch density to maintain service levels. Ensure seamless online-to-offline experience.")
    
    with st.expander("3. Operational Efficiency", expanded=True):
        st.markdown("- **Review Service Combinations:** Analyze the most common and least common service combinations. Are the popular ones operationally efficient? Could less common but valuable combinations be promoted or standardized where appropriate?")
        st.markdown("- **Standardize Accessibility:** While prioritizing low-scoring states, aim for consistent accessibility standards across the network where feasible to enhance brand image and inclusivity.")
        st.markdown("- **Data-Driven Decisions:** Continuously monitor these metrics and update analysis with newer data (including population figures) to refine strategic planning.")
        
    st.markdown("--- ") # Add separator
    
    # --- Add Disclaimer at Bottom ---
    st.info(
        "Uses data from April 1st, 2025, retrieved from https://www.banamex.com/es/localizador-sucursales.html"
    )

# --- Function to run analysis as a script ---
def run_script_analysis():
    print("Starting Banamex Analysis Script...")
    df = load_data()
    if df is None or df.empty:
        print("Failed to load data. Exiting.")
        return

    print(f"\nLoaded {len(df)} service points.")

    print("\n--- Service Analysis ---")
    service_counts = analyze_services(df)
    print(service_counts.to_string())

    print("\n--- Accessibility Analysis ---")
    accessibility_counts = analyze_accessibility(df)
    print(accessibility_counts.to_string())

    print("\n--- Top 10 Service Combinations ---")
    service_combinations = analyze_service_combinations(df, top_n=10)
    print(service_combinations.to_string())

    print("\n--- State Distribution ---")
    state_distribution = analyze_state_distribution(df)
    print(state_distribution.head(10).to_string()) # Print top 10

    print("\n--- Service Density (Top 10) ---")
    density_data = calculate_service_density(df)
    if not density_data.empty:
        print(density_data.head(10).to_string())
    else:
        print("Could not calculate service density (check population data mapping).")
        
    print("\nAnalysis Script Finished.")

# --- Entry Point --- 
if __name__ == "__main__":
    # If run as a script (e.g., python enhanced_banamex_analysis_app.py), 
    # execute the console-based analysis.
    # If run with streamlit (e.g., streamlit run enhanced_banamex_analysis_app.py),
    # Streamlit will handle execution and call the UI functions.
    
    # Check if Streamlit is running the script. A simple check is often enough.
    # Streamlit sets specific environment variables or module properties.
    # A common way is to check if 'streamlit' is in sys.modules AFTER potential import.
    # However, for simplicity here, we assume direct execution means script mode.
    import sys
    if 'streamlit' not in sys.modules:
        run_script_analysis()
    else:
        # When run with `streamlit run`, Streamlit executes the whole script,
        # finds Streamlit commands (like st.write, st....) and renders them.
        # We need to explicitly call our UI function in this case.
        run_streamlit_ui() # Call the UI function when run via Streamlit




# --- End of File --- 