from vega_datasets import data
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load data (no widgets inside cached functions)
#-------------------------------------------------------------------------------------------------#
DATA_PATH = "data/nashville_accidents.csv"

@st.cache_data
def read_csv_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def read_uploaded_file(uploaded) -> pd.DataFrame:
    return pd.read_csv(uploaded)

if os.path.exists(DATA_PATH):
    df = read_csv_file(DATA_PATH)
else:
    st.warning("`nashville_accidents.csv` not found – please upload it.")
    uploaded_file = st.file_uploader("**Upload CSV**", type="csv")
    if uploaded_file is None:
        st.stop()
    df = read_uploaded_file(uploaded_file)

#-------------------------------------------------------------------------------------------------#
# DateTime Conversions
#-------------------------------------------------------------------------------------------------#
df['Date and Time'] = pd.to_datetime(df['Date and Time'],
                                     format='%m/%d/%Y %I:%M:%S %p')
df['year']         = df['Date and Time'].dt.year
df['month']        = df['Date and Time'].dt.month
df['day']          = df['Date and Time'].dt.day
df['hour']         = df['Date and Time'].dt.hour
df['day_of_week']  = df['Date and Time'].dt.dayofweek            # Monday=0 … Sunday=6
df['is_weekend']   = df['day_of_week'].isin([5, 6])
df['is_night']     = (df['hour'] >= 20) | (df['hour'] < 6)       # 8 PM–5:59 AM

#-------------------------------------------------------------------------------------------------#
# NaN Handling
#-------------------------------------------------------------------------------------------------#
cols_drop_na = ['x', 'y', 'Long', 'Lat', 'Number of Fatalities', 'Number of Injuries',
                'City', 'State', 'Number of Motor Vehicles', 'Hit and Run',
                'Precinct', 'Zip Code', 'Street Address']
df.dropna(subset=cols_drop_na, inplace=True)

cols_na_unknown = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions',
                   'Illumination Description', 'Collision Type Description']
df[cols_na_unknown] = df[cols_na_unknown].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

#-------------------------------------------------------------------------------------------------#
# REMOVE "OTHER" & "UNKNOWN" FROM BOTH DIMENSIONS
#-------------------------------------------------------------------------------------------------#
exclude = ['OTHER', 'UNKNOWN']
df = df[
    ~df['Weather Description'].isin(exclude) &
    ~df['Illumination Description'].isin(exclude)
]

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
                height=420)
    .configure_axisX(labelAngle=-35)
)

st.altair_chart(bar_chart, use_container_width=True)

#-------------------------------------------------------------------------------------------------#
# Filters for Heat-map & Metric selector (lists already exclude OTHER/UNKNOWN)
#-------------------------------------------------------------------------------------------------#
weather_opts = sorted(df['Weather Description'].unique())
illum_opts   = sorted(df['Illumination Description'].unique())

st.markdown("###Select Heat-map Parameters")
fcol1, fcol2, fcol3 = st.columns([3,3,2], gap="medium")
with fcol1:
    sel_weather = st.multiselect("Weather Condition(s)",
                                 weather_opts,
                                 default=weather_opts)
with fcol2:
    sel_illum = st.multiselect("Lighting Condition(s)",
                               illum_opts,
                               default=illum_opts)
with fcol3:
    metric_choice = st.selectbox("Metric",
                                 ["Injuries", "Fatalities"], index=0)

filtered_df = df[
    df['Weather Description'].isin(sel_weather) &
    df['Illumination Description'].isin(sel_illum)
]

#-------------------------------------------------------------------------------------------------#
# Heat-map builder (fills missing combos with zeros)
#-------------------------------------------------------------------------------------------------#
def build_heatmap(df_in: pd.DataFrame,
                  metric: str,
                  weather_list: list,
                  illum_list: list) -> alt.Chart:
    if metric == "Injuries":
        grp = (df_in.groupby(['Weather Description', 'Illumination Description'])
               .agg(total=('Number of Injuries', 'sum'),
                    accident_cnt=('Weather Description', 'count'))
               .reset_index())
        title = 'Average Injuries per Accident'
        color_title = 'Avg Injuries / Accident'
        fmt = '.2f'
    else:
        grp = (df_in.groupby(['Weather Description', 'Illumination Description'])
               .agg(total=('Number of Fatalities', 'sum'),
                    accident_cnt=('Weather Description', 'count'))
               .reset_index())
        title = 'Average Fatalities per Accident'
        color_title = 'Avg Fatalities / Accident'
        fmt = '.3f'

    # complete grid to avoid blanks (OTHER/UNKNOWN already gone)
    full_idx = pd.MultiIndex.from_product(
        [weather_list, illum_list],
        names=['Weather Description', 'Illumination Description']
    )
    grp = (grp.set_index(['Weather Description', 'Illumination Description'])
              .reindex(full_idx, fill_value=0)
              .reset_index())

    grp['avg_val'] = np.where(grp['accident_cnt'] > 0,
                              grp['total'] / grp['accident_cnt'],
                              0)

    return (
        alt.Chart(grp)
        .mark_rect()
        .encode(
            x=alt.X('Illumination Description:N', title='Lighting Condition'),
            y=alt.Y('Weather Description:N',      title='Weather Condition'),
            color=alt.Color('avg_val:Q',
                            scale=alt.Scale(scheme='reds'),
                            title=color_title),
            tooltip=['Weather Description', 'Illumination Description',
                     alt.Tooltip('avg_val:Q', format=fmt)]
        )
        .properties(title=title,
                    width=800, height=420)
    )

heatmap_chart = build_heatmap(filtered_df,
                              metric_choice,
                              sel_weather,
                              sel_illum)

#-------------------------------------------------------------------------------------------------#
# Display single Heat-map
#-------------------------------------------------------------------------------------------------#
st.altair_chart(heatmap_chart, use_container_width=True)
