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
    else:
        st.warning(f"File not found at `{path}` – please upload the CSV.")
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

cols_na_unknwon = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions', 'Illumination Description', 'Collision Type Description']
df[cols_na_unknwon] = df[cols_na_unknwon].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

#-------------------------------------------------------------------------------------------------#
# Filters
#-------------------------------------------------------------------------------------------------#
with st.sidebar:
    st.header("Filters")

    weather_opts = sorted(df['Weather Description'].unique())
    illum_opts   = sorted(df['Illumination Description'].unique())

    sel_weather = st.multiselect("Weather Condition(s)",
                                 weather_opts,
                                 default=weather_opts)          # All by default
    sel_illum   = st.multiselect("Lighting Condition(s)",
                                 illum_opts,
                                 default=illum_opts)

# Apply filters -----------------------------------------------------------
filtered_df = df[
    df['Weather Description'].isin(sel_weather) &
    df['Illumination Description'].isin(sel_illum)
]

#-------------------------------------------------------------------------------------------------#
# Under which weather conditions are accidents disproportionately more severe than average?
#-------------------------------------------------------------------------------------------------#

# Create binary columns
filtered_df['has_injury'] = filtered_df['Number of Injuries'] > 0
filtered_df['has_fatality'] = filtered_df['Number of Fatalities'] > 0

# Group by weather condition
severity_df = filtered_df.groupby('Weather Description').agg(
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
bar_chart = alt.Chart(melted).mark_bar().encode(
    x=alt.X('Weather Description:N', sort='-y', title='Weather Condition'),
    y=alt.Y('Percentage:Q', title='Percentage of Accidents'),
    color=alt.Color('Severity Type:N', title='Severity Type',
        scale=alt.Scale(
            domain=['% with Injury', '% with Fatality'],
            range=['orange', 'crimson']
        )
    ),
    tooltip=['Weather Description', 'Severity Type', alt.Tooltip('Percentage:Q', format='.1f')]
).properties(
    title='Proportion of Accidents with Injuries or Fatalities by Weather Condition',
    width=700,
    height=400
).configure_axisX(labelAngle=-40)

# Display in Streamlit
st.altair_chart(bar_chart, use_container_width=True)

#-------------------------------------------------------------------------------------------------#
# How does weather impact the frequency and severity of accidents?
#-------------------------------------------------------------------------------------------------#

# chart 1: injuries
inj_grp = (
    filtered_df.groupby(['Weather Description', 'Illumination Description'])
    .agg(total_injuries=('Number of Injuries', 'sum'),
         accident_count=('Weather Description', 'count'))
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
    .properties(title='Average Injuries per Accident',
                width=400, height=300)
)

# chart 2: fatalities
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
    .properties(title='Average Fatalities per Accident',
                width=400, height=300)
)

# display injuries/fatalities together
st.altair_chart(bar_chart, use_container_width=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.altair_chart(inj_heat, use_container_width=True)
with col2:
    st.altair_chart(fat_heat, use_container_width=True)
