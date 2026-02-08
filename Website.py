import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(page_title="Birth Weight & Elevation Analysis", layout="wide")

# Title and description
st.title("Birth Weight & Elevation Analysis Dashboard")
st.markdown("### Interactive exploration of 2023 low birth weight rates by county elevation")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    # Use relative path - make sure CSV is in same folder as this script
    df = pd.read_csv('QAMO 5090 Website Data.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert elevation and birth weight to numeric, handling empty values
    df['elevation'] = pd.to_numeric(df['elevation'], errors='coerce')
    df['low birth weight 2023'] = pd.to_numeric(df['low birth weight 2023'], errors='coerce')
    
    # Remove rows where both elevation and birth weight are missing
    df_clean = df.dropna(subset=['elevation', 'low birth weight 2023'], how='all')
    
    # Create a location label
    df_clean['location'] = df_clean['county'] + ', ' + df_clean['state']
    
    return df_clean

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# State filter
states = ['All States'] + sorted(df['state'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State(s)",
    options=states,
    default=['All States']
)

# Elevation range slider
if df['elevation'].notna().any():
    min_elev = int(df['elevation'].min())
    max_elev = int(df['elevation'].max())
    
    elevation_range = st.sidebar.slider(
        "Elevation Range (feet)",
        min_value=min_elev,
        max_value=max_elev,
        value=(min_elev, max_elev),
        step=100
    )
else:
    elevation_range = (0, 10000)

# Birth weight range slider
if df['low birth weight 2023'].notna().any():
    min_bw = int(df['low birth weight 2023'].min())
    max_bw = int(df['low birth weight 2023'].max())
    
    birth_weight_range = st.sidebar.slider(
        "Low Birth Weight Count Range",
        min_value=min_bw,
        max_value=max_bw,
        value=(min_bw, max_bw)
    )
else:
    birth_weight_range = (0, 1000)

# Filter data based on selections
filtered_df = df.copy()

# Apply state filter
if 'All States' not in selected_states and len(selected_states) > 0:
    filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

# Apply elevation filter
filtered_df = filtered_df[
    (filtered_df['elevation'].between(elevation_range[0], elevation_range[1], inclusive='both') | 
     filtered_df['elevation'].isna())
]

# Apply birth weight filter
filtered_df = filtered_df[
    (filtered_df['low birth weight 2023'].between(birth_weight_range[0], birth_weight_range[1], inclusive='both') |
     filtered_df['low birth weight 2023'].isna())
]

# Remove rows with no valid data for the main metrics
filtered_df_plot = filtered_df.dropna(subset=['elevation', 'low birth weight 2023'])

# Key metrics
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Summary")
st.sidebar.metric("Counties Displayed", len(filtered_df_plot))
if len(filtered_df_plot) > 0:
    st.sidebar.metric("Avg Elevation", f"{filtered_df_plot['elevation'].mean():.0f} ft")
    st.sidebar.metric("Avg Low Birth Weight", f"{filtered_df_plot['low birth weight 2023'].mean():.1f}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Birth Weight vs Elevation Scatter Plot")
    
    if len(filtered_df_plot) > 0:
        # Create scatter plot
        fig_scatter = px.scatter(
            filtered_df_plot,
            x='elevation',
            y='low birth weight 2023',
            color='state',
            hover_data=['location', 'county', 'state'],
            title='Low Birth Weight Count by Elevation',
            labels={
                'elevation': 'Elevation (feet)',
                'low birth weight 2023': 'Low Birth Weight Count (2023)',
                'state': 'State'
            },
            opacity=0.7
        )
        
        # Add trendline if enough data points
        if len(filtered_df_plot) > 2:
            # Calculate correlation
            correlation = filtered_df_plot[['elevation', 'low birth weight 2023']].corr().iloc[0, 1]
            
            # Add trendline
            z = np.polyfit(filtered_df_plot['elevation'], filtered_df_plot['low birth weight 2023'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(filtered_df_plot['elevation'].min(), 
                                 filtered_df_plot['elevation'].max(), 100)
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name=f'Trend (r={correlation:.3f})',
                    line=dict(color='red', dash='dash')
                )
            )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("No data available for the selected filters.")

with col2:
    st.subheader("Distribution")
    
    if len(filtered_df_plot) > 0:
        # Elevation distribution
        fig_elev = px.histogram(
            filtered_df_plot,
            x='elevation',
            nbins=30,
            title='Elevation Distribution',
            labels={'elevation': 'Elevation (feet)', 'count': 'Number of Counties'}
        )
        fig_elev.update_layout(height=235, showlegend=False)
        st.plotly_chart(fig_elev, use_container_width=True)
        
        # Birth weight distribution
        fig_bw = px.histogram(
            filtered_df_plot,
            x='low birth weight 2023',
            nbins=30,
            title='Low Birth Weight Distribution',
            labels={'low birth weight 2023': 'Low Birth Weight Count', 'count': 'Number of Counties'}
        )
        fig_bw.update_layout(height=235, showlegend=False)
        st.plotly_chart(fig_bw, use_container_width=True)

# Second row - Additional visualizations
st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 10 States by Average Elevation")
    
    if len(filtered_df_plot) > 0:
        state_elevation = filtered_df_plot.groupby('state').agg({
            'elevation': 'mean',
            'county': 'count'
        }).reset_index()
        state_elevation.columns = ['state', 'avg_elevation', 'county_count']
        state_elevation = state_elevation.sort_values('avg_elevation', ascending=False).head(10)
        
        fig_state_elev = px.bar(
            state_elevation,
            x='state',
            y='avg_elevation',
            title='Top 10 States by Average Elevation',
            labels={'avg_elevation': 'Average Elevation (feet)', 'state': 'State'},
            color='avg_elevation',
            color_continuous_scale='Viridis'
        )
        fig_state_elev.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_state_elev, use_container_width=True)

with col4:
    st.subheader("Top 10 Counties by Low Birth Weight")
    
    if len(filtered_df_plot) > 0:
        top_counties = filtered_df_plot.nlargest(10, 'low birth weight 2023')[
            ['location', 'low birth weight 2023', 'elevation']
        ].copy()
        
        fig_top_counties = px.bar(
            top_counties,
            x='low birth weight 2023',
            y='location',
            orientation='h',
            title='Top 10 Counties by Low Birth Weight Count',
            labels={'low birth weight 2023': 'Low Birth Weight Count', 'location': 'County, State'},
            color='elevation',
            color_continuous_scale='RdYlGn_r'
        )
        fig_top_counties.update_layout(height=400)
        st.plotly_chart(fig_top_counties, use_container_width=True)

# Elevation categories analysis
st.markdown("---")
st.subheader("Analysis by Elevation Category")

if len(filtered_df_plot) > 0:
    # Create elevation categories
    filtered_df_plot['elevation_category'] = pd.cut(
        filtered_df_plot['elevation'],
        bins=[0, 1000, 3000, 5000, 10000],
        labels=['Low (0-1000 ft)', 'Medium (1000-3000 ft)', 'High (3000-5000 ft)', 'Very High (5000+ ft)']
    )
    
    category_stats = filtered_df_plot.groupby('elevation_category').agg({
        'low birth weight 2023': ['mean', 'median', 'sum', 'count']
    }).reset_index()
    
    category_stats.columns = ['Elevation Category', 'Mean', 'Median', 'Total', 'Counties']
    
    col5, col6 = st.columns(2)
    
    with col5:
        fig_cat_mean = px.bar(
            category_stats,
            x='Elevation Category',
            y='Mean',
            title='Average Low Birth Weight by Elevation Category',
            labels={'Mean': 'Average Low Birth Weight Count'},
            color='Mean',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cat_mean, use_container_width=True)
    
    with col6:
        fig_cat_count = px.bar(
            category_stats,
            x='Elevation Category',
            y='Counties',
            title='Number of Counties by Elevation Category',
            labels={'Counties': 'Number of Counties'},
            color='Counties',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_cat_count, use_container_width=True)

# Data table
st.markdown("---")
st.subheader("Detailed Data Table")

if st.checkbox("Show detailed county data"):
    display_columns = ['location', 'state', 'county', 'elevation', 'low birth weight 2023']
    st.dataframe(
        filtered_df_plot[display_columns].sort_values('low birth weight 2023', ascending=False),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df_plot[display_columns].to_csv(index=False)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name="birth_weight_elevation_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("*Data Source: QAMO 5090 Website Data | Built with Streamlit & Plotly*")
