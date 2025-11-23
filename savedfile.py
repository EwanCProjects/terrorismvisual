import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import folium
from folium.plugins import FastMarkerCluster
from streamlit_folium import st_folium


@st.cache_data
def load_data():
    """Load only essential columns and optimize dtypes"""
    # Only load columns we actually use
    columns = ['longitude', 'latitude', 'iyear', 'imonth', 'iday', 'country_txt', 'summary']
    
    df = pd.read_parquet('globalterrorism.parquet', columns=columns)
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['longitude', 'latitude', 'iyear', 'imonth', 'iday', 'country_txt'])
    
    # Optimize data types to reduce memory
    df['iyear'] = df['iyear'].astype('int16')
    df['imonth'] = df['imonth'].astype('int8')
    df['iday'] = df['iday'].astype('int8')
    df['country_txt'] = df['country_txt'].astype('category')  # Major memory saver!
    df['longitude'] = df['longitude'].astype('float32')
    df['latitude'] = df['latitude'].astype('float32')
    
    return df



def haversine_vectorized(lat1, lon1, lats, lons):
    """Calculate distances using Haversine formula"""
    lat1, lon1, lats, lons = map(np.radians, [lat1, lon1, lats, lons])
    dlat, dlon = lats - lat1, lons - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a))

def apply_filters(df, filters):
    """Apply all active filters to dataframe"""
    result = df.copy()
    
    # Location filter (radius OR country)
    if filters['use_radius']:
        distances = haversine_vectorized(
            filters['lat'], filters['lon'], 
            result['latitude'].values, result['longitude'].values
        )
        result = result[distances <= filters['radius']]
    elif filters['use_country']:
        result = result[result['country_txt'] == filters['country']]
    
    # Date filter
    if filters['use_date']:
        result['date'] = pd.to_datetime(
            result[['iyear', 'imonth', 'iday']].rename(
                columns={'iyear': 'year', 'imonth': 'month', 'iday': 'day'}
            ), 
            errors='coerce'
        )
        result = result[
            (result['date'] >= pd.Timestamp(filters['start_date'])) & 
            (result['date'] <= pd.Timestamp(filters['end_date']))
        ]
    
    return result

def extract_bounds(st_data):
    """Extract map bounds from streamlit-folium data"""
    bounds = st_data.get("bounds")
    if not bounds:
        return -90, -180, 90, 180
    
    if isinstance(bounds, list) and len(bounds) == 2:
        return bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
    
    if isinstance(bounds, dict):
        sw = (bounds.get('_southWest') or bounds.get('southWest') or 
              bounds.get('southwest') or {})
        ne = (bounds.get('_northEast') or bounds.get('northEast') or 
              bounds.get('northeast') or {})
        return sw.get('lat', -90), sw.get('lng', -180), ne.get('lat', 90), ne.get('lng', 180)
    
    return -90, -180, 90, 180


df_agg = load_data()
st.title('Terrorist attacks since 1970')

# Display memory usage info
st.caption(f"📊 Dataset: {len(df_agg):,} events loaded")

# Initialize session state
if 'map_data' not in st.session_state:
    st.session_state['map_data'] = df_agg


with st.form("filters_form"):
    st.subheader("🔍 Filter Options")


    st.markdown("**🌍 Location Filter** — choose which input to use")

    location_mode = st.radio(
        "Use:",
        ["Neither", "Country", "Radius"],
        horizontal=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_lat = st.number_input("Latitude:", -90.0, 90.0, 30.0)
    with col2:
        filter_lon = st.number_input("Longitude:", -180.0, 180.0, 70.0)
    with col3:
        filter_radius = st.slider("Radius (km):", 1, 500, 50)

    filter_country = st.selectbox("Country:", sorted(df_agg['country_txt'].unique()))

    st.divider()


    use_date_filter = st.checkbox("📅 Filter by date range", value=False)

    filter_start, filter_end = st.date_input(
        "Select date range:",
        (date(2000, 1, 1), date(2019, 12, 31)),
        format="MM.DD.YYYY"
    )

    st.divider()


    col1, col2 = st.columns([1, 1])
    with col1:
        submit_filters = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)
    with col2:
        reset_filters = st.form_submit_button("Reset All", use_container_width=True)


if submit_filters:
    filters = {
        'use_radius': location_mode == "Radius",
        'use_country': location_mode == "Country",
        'use_date': use_date_filter,
    }

    if filters['use_radius']:
        filters.update({
            'lat': filter_lat,
            'lon': filter_lon,
            'radius': filter_radius
        })
    if filters['use_country']:
        filters['country'] = filter_country
    if filters['use_date']:
        filters.update({
            'start_date': filter_start,
            'end_date': filter_end
        })

    filtered = apply_filters(df_agg, filters)
    st.session_state['map_data'] = filtered

    active_filters = []
    if filters['use_radius']:
        active_filters.append(f"Radius: {filters['radius']} km from ({filters['lat']}, {filters['lon']})")
    if filters['use_country']:
        active_filters.append(f"Country: {filters['country']}")
    if filters['use_date']:
        active_filters.append(f"Date: {filters['start_date']} → {filters['end_date']}")

    if active_filters:
        st.success(f"✓ Found **{len(filtered):,}** events\n\n" + " • ".join(active_filters))
    else:
        st.info(f"No filters applied. Showing all {len(filtered):,} events.")

if reset_filters:
    st.session_state['map_data'] = df_agg
    st.success(f"✓ Filters reset. Showing all {len(df_agg):,} events.")


st.divider()
map_data = st.session_state['map_data']

if not map_data.empty:
    # Limit points shown on map for performance
    MAX_MAP_POINTS = 10000
    
    if len(map_data) > MAX_MAP_POINTS:
        st.warning(f"Displaying {MAX_MAP_POINTS:,} sampled points out of {len(map_data):,} for performance. Please reach out to casandjianewan@gmail.com for the full version")
        map_display_data = map_data.sample(n=MAX_MAP_POINTS, random_state=42)
    else:
        map_display_data = map_data
    
    # Determine map center
    if location_mode == "Radius":
        center_lat, center_lon = filter_lat, filter_lon
    else:
        center_lat = float(map_display_data['latitude'].median())
        center_lon = float(map_display_data['longitude'].median())
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")
    points = map_display_data[['latitude', 'longitude']].values.tolist()
    FastMarkerCluster(points).add_to(m)
    
    # Display map
    st_data = st_folium(m, width=900, height=600, returned_objects=["bounds"])
    
    if st_data and st_data.get("bounds"):
        st.session_state['last_bounds'] = st_data["bounds"]
    
    # Button to show events in current view
    if st.button("📌 Show Events in Current View"):
        bounds = st.session_state.get('last_bounds')
        if bounds:
            south, west, north, east = extract_bounds({"bounds": bounds})
            visible = map_data[
                (map_data['latitude'].between(south, north)) &
                (map_data['longitude'].between(west, east))
            ]

            with st.expander(f"📊 {len(visible):,} events visible in current map view", expanded=True):
                max_rows_to_display = 500  # Reduced from 1000
                
                if len(visible) > max_rows_to_display:
                    st.warning(
                        f"Showing first {max_rows_to_display:,} rows out of {len(visible):,}. "
                        "Download the full dataset of events in view below. Most data fields are missing, please reach out to me for the full version"
                    )
                    # Only convert what we display to avoid memory spike
                    display_df = visible.head(max_rows_to_display)
                else:
                    display_df = visible

                st.dataframe(display_df, height=400, use_container_width=True)

                # Download button - convert to CSV on demand
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv_data = convert_df(visible)
                st.download_button(
                    label="⬇️ Download full visible dataset as CSV",
                    data=csv_data,
                    file_name="visible_events.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("👆 Pan or zoom the map first, then click this button")
else:
    st.warning("⚠️ No data matches your filters")
