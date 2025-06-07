from vega_datasets import data
import os
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load data
#-------------------------------------------------------------------------------------------------#
DATA_PATH = "data/nashville_accidents.csv"
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    uploaded = st.file_uploader("Upload nashville_accidents.csv", type="csv")
    if uploaded is not None:
        return pd.read_csv(uploaded)
    st.stop()

df = load_data(DATA_PATH)

#-------------------------------------------------------------------------------------------------#
# DateTime Conversions
#-------------------------------------------------------------------------------------------------#
df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y %I:%M:%S %p')
df['year'] = df['Date and Time'].dt.year
df['month'] = df['Date and Time'].dt.month
df['day'] = df['Date and Time'].dt.day
df['hour'] = df['Date and Time'].dt.hour
df['day_of_week'] = df['Date and Time'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['is_night'] = (df['hour'] >= 20) | (df['hour'] < 6) # Night is between 8PM and 5:59AM inclusive

#-------------------------------------------------------------------------------------------------#
# NaN Handling
#-------------------------------------------------------------------------------------------------#
cols_drop_na = ['x', 'y', 'Long', 'Lat', 'Number of Fatalities', 'Number of Injuries',
           'City', 'State', 'Number of Motor Vehicles', 'Hit and Run',
           'Precinct', 'Zip Code', 'Street Address']
df.dropna(subset=cols_drop_na, inplace=True)

cols_na_unknwon = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions',
                   'Illumination Description', 'Collision Type Description']
df[cols_na_unknwon] = df[cols_na_unknwon].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

#-------------------------------------------------------------------------------------------------#
# ‘Proportion of Accidents with Injuries or Fatalities’ (unfiltered)
#-------------------------------------------------------------------------------------------------#
df['has_injury']   = df['Number of Injuries']   > 0
df['has_fatality'] = df['Number of Fatalities'] > 0

severity_df = df.groupby('Weather Description').agg(
    total_accidents  = ('Weather Description', 'count'),
    injury_accidents = ('has_injury',   'sum'),
    fatal_accidents  = ('has_fatality', 'sum')
).reset_index()

severity_df['% with Injury']   = severity_df['injury_accidents'] / severity_df['total_accidents'] * 100
severity_df['% with Fatality'] = severity_df['fatal_accidents']  / severity_df['total_accidents'] * 100

melted = severity_df.melt(
    id_vars   = 'Weather Description',
    value_vars=['% with Injury', '% with Fatality'],
    var_name  = 'Severity Type',
    value_name= 'Percentage'
)

melted = melted[melted['Weather Description'].isin(
    severity_df[severity_df['total_accidents'] >= 5]['Weather Description']
)]

bar_chart = (
    alt.Chart(melted)
    .mark_bar()
    .encode(
        x=alt.X('Weather Description:N', sort='-y', title='Weather Condition'),
        y=alt.Y('Percentage:Q',          title='Percentage of Accidents'),
        color=alt.Color('Severity Type:N',
                        scale=alt.Scale(domain=['% with Injury', '% with Fatality'],
                                        range=['orange', 'crimson']),
                        title='Severity Type'),
        tooltip=['Weather Description', 'Severity Type',
                 alt.Tooltip('Percentage:Q', format='.1f')]
    )
    .properties(title='Proportion of Accidents with Injuries or Fatalities',
                height=400)
    .configure_axisX(labelAngle=-35)
)

st.altair_chart(bar_chart, use_container_width=True)

#-------------------------------------------------------------------------------------------------#
# Filters for Heat-maps only (Weather + Lighting)
#-------------------------------------------------------------------------------------------------#
weather_opts = sorted(df['Weather Description'].unique())
illum_opts   = sorted(df['Illumination Description'].unique())

st.markdown("### 🔎 Filter Heat-maps")
fcol1, fcol2 = st.columns(2, gap="medium")
with fcol1:
    sel_weather = st.multiselect("Weather Condition(s) ☁️🌧️",
                                 weather_opts,
                                 default=weather_opts)
with fcol2:
    sel_illum = st.multiselect("Lighting Condition(s) 💡🌙",
                               illum_opts,
                               default=illum_opts)

filtered_df = df[
    df['Weather Description'].isin(sel_weather) &
    df['Illumination Description'].isin(sel_illum)
]

#-------------------------------------------------------------------------------------------------#
# Heat-map 1: Average Injuries / Accident
#-------------------------------------------------------------------------------------------------#
inj_grp = (
    filtered_df.groupby(['Weather Description', 'Illumination Description'])
    .agg(total_injuries=('Number of Injuries', 'sum'),
         accident_count =('Weather Description', 'count'))
    .reset_index()
)
inj_grp['avg_injuries'] = inj_grp['total_injuries'] / inj_grp['accident_count']

inj_heat = (
    alt.Chart(inj_grp)
    .mark_rect()
    .encode(
        x=alt.X('Illumination Description:N', title='Lighting Condition'),
        y=alt.Y('Weather Description:N',      title='Weather Condition'),
        color=alt.Color('avg_injuries:Q',
                        scale=alt.Scale(scheme='reds'),
                        title='Avg Injuries / Accident'),
        tooltip=['Weather Description', 'Illumination Description',
                 alt.Tooltip('avg_injuries:Q', format='.2f')]
    )
    .properties(title='Average Injuries per Accident', height=350)
)

#-------------------------------------------------------------------------------------------------#
# Heat-map 2: Average Fatalities / Accident
#-------------------------------------------------------------------------------------------------#
fat_grp = (
    filtered_df.groupby(['Weather Description', 'Illumination Description'])
    .agg(total_fatalities=('Number of Fatalities', 'sum'),
         accident_count   =('Weather Description', 'count'))
    .reset_index()
)
fat_grp['avg_fatalities'] = fat_grp['total_fatalities'] / fat_grp['accident_count']

fat_heat = (
    alt.Chart(fat_grp)
    .mark_rect()
    .encode(
        x=alt.X('Illumination Description:N', title='Lighting Condition'),
        y=alt.Y('Weather Description:N',      title='Weather Condition'),
        color=alt.Color('avg_fatalities:Q',
                        scale=alt.Scale(scheme='reds'),
                        title='Avg Fatalities / Accident'),
        tooltip=['Weather Description', 'Illumination Description',
                 alt.Tooltip('avg_fatalities:Q', format='.3f')]
    )
    .properties(title='Average Fatalities per Accident', height=350)
)

#-------------------------------------------------------------------------------------------------#
# Display Heat-maps side-by-side (full width)
#-------------------------------------------------------------------------------------------------#
hcol1, hcol2 = st.columns(2, gap="large")
with hcol1:
    st.altair_chart(inj_heat, use_container_width=True)
with hcol2:
    st.altair_chart(fat_heat, use_container_width=True)
