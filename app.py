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
# How does weather impact the frequency and severity of accidents?
#-------------------------------------------------------------------------------------------------#

weather_emoji_map = {
    "Clear": "☀️ Clear",
    "Rain": "🌧️ Rain",
    "Snow": "❄️ Snow",
    "Cloudy": "☁️ Cloudy",
    "Fog": "🌫️ Fog",
    "Thunderstorm": "⛈️ Thunderstorm",
    "Drizzle": "🌦️ Drizzle",
    "Unknown": "❓ Unknown"
}
emoji_to_weather = {v: k for k, v in weather_emoji_map.items()}

# Clean and prep data
df['Time of Day'] = df['is_night'].map({True: 'Night', False: 'Day'})

# Sidebar weather selector
top_weather = df['Weather Description'].fillna("Unknown").value_counts().nlargest(8).index.tolist()
emoji_options = [weather_emoji_map.get(w, f"🔘 {w}") for w in top_weather]
weather_lookup = dict(zip(emoji_options, top_weather))
selected_emoji = st.sidebar.selectbox("Choose a weather condition to highlight", emoji_options)
selected_weather = weather_lookup[selected_emoji]

# CHART 1: Avg Injuries by Weather + Illumination
grouped1 = (
    df.groupby(['Weather Description', 'Illumination Description'])
    .agg(total_injuries=('Number of Injuries', 'sum'),
         accident_count=('Weather Description', 'count'))
    .reset_index()
)
grouped1['avg_injuries'] = grouped1['total_injuries'] / grouped1['accident_count']

top_illum = df['Illumination Description'].value_counts().nlargest(6).index
filtered1 = grouped1[
    grouped1['Weather Description'].isin(top_weather) &
    grouped1['Illumination Description'].isin(top_illum)
]

chart1 = alt.Chart(filtered1).mark_rect().encode(
    x=alt.X('Illumination Description:N', title='Lighting Condition'),
    y=alt.Y('Weather Description:N', title='Weather Condition'),
    color=alt.Color('avg_injuries:Q', scale=alt.Scale(scheme='reds'), title='Avg Injuries per Accident'),
    tooltip=['Weather Description', 'Illumination Description', 'avg_injuries'],
    opacity=alt.condition(
        f"datum['Weather Description'] == '{selected_weather}'", alt.value(1.0), alt.value(0.3)
    )
).properties(
    title='Average Injuries by Weather and Lighting Condition',
    width=400,
    height=300
)

# CHART 2: Avg Fatalities by Weather + Illumination
grouped2 = (
    df.groupby(['Weather Description', 'Illumination Description'])
    .agg(total_fatalities=('Number of Fatalities', 'sum'),
         accident_count=('Weather Description', 'count'))
    .reset_index()
)
grouped2['avg_fatalities'] = grouped2['total_fatalities'] / grouped2['accident_count']
filtered2 = grouped2[
    grouped2['Weather Description'].isin(top_weather) &
    grouped2['Illumination Description'].isin(top_illum)
]

chart2 = alt.Chart(filtered2).mark_rect().encode(
    x=alt.X('Illumination Description:N', title='Lighting Condition'),
    y=alt.Y('Weather Description:N', title='Weather Condition'),
    color=alt.Color('avg_fatalities:Q', scale=alt.Scale(scheme='reds'), title='Avg Fatalities per Accident'),
    tooltip=['Weather Description', 'Illumination Description', 'avg_fatalities'],
    opacity=alt.condition(
        f"datum['Weather Description'] == '{selected_weather}'", alt.value(1.0), alt.value(0.3)
    )
).properties(
    title='Average Fatalities by Weather and Lighting Condition',
    width=400,
    height=300
)

# CHART 3: Proportion of Accidents by Weather + Time 
top_weather_10 = df['Weather Description'].value_counts().nlargest(10).index
df_filtered3 = df[df['Weather Description'].isin(top_weather_10)]
counts = df_filtered3.groupby(['Weather Description', 'Time of Day']).size().reset_index(name='Accident Count')
weather_totals = df_filtered3.groupby('Weather Description').size().reset_index(name='Total')
normalized = counts.merge(weather_totals, on='Weather Description')
normalized['Proportion'] = normalized['Accident Count'] / normalized['Total']

chart3 = alt.Chart(normalized).mark_rect().encode(
    x=alt.X('Time of Day:N', title='Time of Day'),
    y=alt.Y('Weather Description:N', sort='-x', title='Weather Condition'),
    color=alt.Color('Proportion:Q', scale=alt.Scale(scheme='reds'), title='Proportion of Accidents'),
    tooltip=['Weather Description', 'Time of Day', 'Accident Count', 'Proportion'],
    opacity=alt.condition(
        f"datum['Weather Description'] == '{selected_weather}'", alt.value(1.0), alt.value(0.3)
    )
).properties(
    title='Proportion of Accidents by Weather Condition and Time of Day',
    width=300,
    height=400
)

# Display charts in Streamlit layout 
st.altair_chart(chart1, use_container_width=False)
st.altair_chart(chart2, use_container_width=False)
st.altair_chart(chart3, use_container_width=False)