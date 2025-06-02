from vega_datasets import data
import os
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

# Load data
df = pd.read_csv("data/nashville_accidents.csv")

# DateTime Conversions
df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y %I:%M:%S %p')
df['year'] = df['Date and Time'].dt.year
df['month'] = df['Date and Time'].dt.month
df['day'] = df['Date and Time'].dt.day
df['hour'] = df['Date and Time'].dt.hour
df['day_of_week'] = df['Date and Time'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['is_night'] = (df['hour'] >= 20) | (df['hour'] < 6) # Night is between 8PM and 5:59AM inclusive

# NaN
cols_drop_na = ['x', 'y', 'Long', 'Lat', 'Number of Fatalities', 'Number of Injuries',
           'City', 'State', 'Number of Motor Vehicles', 'Hit and Run',
           'Precinct', 'Zip Code', 'Street Address']
df.dropna(subset=cols_drop_na, inplace=True)

cols_na_unknwon = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions', 'Illumination Description', 'Collision Type Description']
df[cols_na_unknwon] = df[cols_na_unknwon].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

#-------------------------------------------------------------------------------------------------#
# Under which weather conditions are accidents disproportionately more severe than average?
#-------------------------------------------------------------------------------------------------#

# Create binary columns
df['has_injury'] = df['Number of Injuries'] > 0
df['has_fatality'] = df['Number of Fatalities'] > 0

# Group by weather condition
severity_df = df.groupby('Weather Description').agg(
    total_accidents=('Weather Description', 'count'),
    injury_accidents=('has_injury', 'sum'),
    fatal_accidents=('has_fatality', 'sum')
).reset_index()

# Calculate percentages
severity_df['% with Injury'] = severity_df['injury_accidents'] / severity_df['total_accidents'] * 100
severity_df['% with Fatality'] = severity_df['fatal_accidents'] / severity_df['total_accidents'] * 100

# Melt for plotting
melted = severity_df.melt(
    id_vars='Weather Description',
    value_vars=['% with Injury', '% with Fatality'],
    var_name='Severity Type',
    value_name='Percentage'
)

# Filter out weather types with very few accidents (e.g., < 5)
melted = melted[melted['Weather Description'].isin(
    severity_df[severity_df['total_accidents'] >= 5]['Weather Description']
)]

# Plot
chart = alt.Chart(melted).mark_bar().encode(
    x=alt.X('Weather Description:N', sort='-y', title='Weather Condition'),
    y=alt.Y('Percentage:Q', title='Percentage of Accidents'),
    color=alt.Color('Severity Type:N', title='Severity Type',
        scale=alt.Scale(
            domain=['% with Injury', '% with Fatality'],
            range=['orange', 'crimson']
        )
    ),
    tooltip=['Weather Description', 'Severity Type', 'Percentage']
).properties(
    title='Proportion of Accidents with Injuries or Fatalities by Weather Condition',
    width=700,
    height=400
).configure_axisX(labelAngle=-40)

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)

#-------------------------------------------------------------------------------------------------#
# Are there clusters of accidents in specific latitude/longitude regions?
#-------------------------------------------------------------------------------------------------#
# Filter copy of your data for Nashville bounds
df_map = df.copy()
df_map = df_map[(df_map['Lat'] >= 36.0) & (df_map['Lat'] <= 36.4) &
                (df_map['Long'] >= -87.0) & (df_map['Long'] <= -86.5)]

st.subheader("Nashville Accidents on Interactive Map (Zoomable, Pannable, With Roads)")

# Each dot = accident, with tooltip and interactivity
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[Long, Lat]',
    get_radius=40,
    get_fill_color='[255, 0, 0, 150]',
    pickable=True,
    auto_highlight=True,
)

# Map view configuration
view_state = pdk.ViewState(
    latitude=36.16,
    longitude=-86.78,
    zoom=10,
    pitch=0,
)

# Deck map setup with street basemap
deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Lat: {Lat}\nLong: {Long}"},
    map_style="mapbox://styles/mapbox/streets-v12",
)

# Show in Streamlit
st.pydeck_chart(deck)