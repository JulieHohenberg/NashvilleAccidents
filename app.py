from vega_datasets import data
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load CSV  (widget-free inside @st.cache_data)
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
    up = st.file_uploader("📂 **Upload CSV**", type="csv")
    if up is None:
        st.stop()
    df = read_uploaded_file(up)

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
# NaN Handling & Remove “OTHER/UNKNOWN”
#-------------------------------------------------------------------------------------------------#
cols_drop_na = ['x', 'y', 'Long', 'Lat', 'Number of Fatalities', 'Number of Injuries',
                'City', 'State', 'Number of Motor Vehicles', 'Hit and Run',
                'Precinct', 'Zip Code', 'Street Address']
df.dropna(subset=cols_drop_na, inplace=True)

cols_na_unknown = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions',
                   'Illumination Description', 'Collision Type Description']
df[cols_na_unknown] = df[cols_na_unknown].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

exclude = ['OTHER', 'UNKNOWN']
df = df[
    ~df['Weather Description'].isin(exclude) &
    ~df['Illumination Description'].isin(exclude)
]

#-------------------------------------------------------------------------------------------------#
# “Proportion of Accidents with Injuries or Fatalities” (all categories)
#-------------------------------------------------------------------------------------------------#
df['has_injury']   = df['Number of Injuries']   > 0
df['has_fatality'] = df['Number of Fatalities'] > 0

sev_df = df.groupby('Weather Description').agg(
    total_accidents  = ('Weather Description', 'count'),
    injury_accidents = ('has_injury',   'sum'),
    fatal_accidents  = ('has_fatality', 'sum')
).reset_index()

sev_df['% with Injury']   = sev_df['injury_accidents'] / sev_df['total_accidents'] * 100
sev_df['% with Fatality'] = sev_df['fatal_accidents']  / sev_df['total_accidents'] * 100

sev_melt = sev_df.melt(
    id_vars   = 'Weather Description',
    value_vars=['% with Injury', '% with Fatality'],
    var_name  = 'Severity Type',
    value_name= 'Percentage'
)

bar_chart = (
    alt.Chart(sev_melt)
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
# Metric selector (single dropdown) — Injuries or Fatalities
#-------------------------------------------------------------------------------------------------#
metric_choice = st.selectbox("Heat-map Metric  📊",
                             ["Injuries", "Fatalities"], index=0)

#-------------------------------------------------------------------------------------------------#
# Build Heat-map with TOP-8 Weather × TOP-6 Lighting and **full grid** (no blank tiles)
#-------------------------------------------------------------------------------------------------#
def build_top_heatmap(df_in: pd.DataFrame, metric: str) -> alt.Chart:
    # Top categories ------------------------------------------------------
    top_weather = df_in['Weather Description'].value_counts().nlargest(8).index
    top_illum   = df_in['Illumination Description'].value_counts().nlargest(6).index
    df_sub = df_in[
        df_in['Weather Description'].isin(top_weather) &
        df_in['Illumination Description'].isin(top_illum)
    ]

    # Aggregate -----------------------------------------------------------
    if metric == "Injuries":
        grp = (df_sub.groupby(['Weather Description', 'Illumination Description'])
               .agg(total=('Number of Injuries', 'sum'),
                    accident_cnt=('Weather Description', 'count'))
               .reset_index())
        title_txt   = 'Average Injuries per Accident'
        color_title = 'Avg Injuries / Accident'
        fmt = '.2f'
    else:
        grp = (df_sub.groupby(['Weather Description', 'Illumination Description'])
               .agg(total=('Number of Fatalities', 'sum'),
                    accident_cnt=('Weather Description', 'count'))
               .reset_index())
        title_txt   = 'Average Fatalities per Accident'
        color_title = 'Avg Fatalities / Accident'
        fmt = '.3f'

    # Complete grid so *every* combo has a tile --------------------------
    full_idx = pd.MultiIndex.from_product(
        [top_weather, top_illum],
        names=['Weather Description', 'Illumination Description']
    )
    grp = (grp.set_index(['Weather Description', 'Illumination Description'])
              .reindex(full_idx, fill_value=0)
              .reset_index())

    grp['avg_val'] = np.where(grp['accident_cnt'] > 0,
                              grp['total'] / grp['accident_cnt'],
                              0)

    # Consistent colour scale --------------------------------------------
    max_val = grp['avg_val'].max()
    color_scale = alt.Scale(scheme='reds', domain=[0, max_val])

    # Plot ---------------------------------------------------------------
    return (
        alt.Chart(grp)
        .mark_rect()
        .encode(
            x=alt.X('Illumination Description:N', title='Lighting Condition'),
            y=alt.Y('Weather Description:N',      title='Weather Condition'),
            color=alt.Color('avg_val:Q',
                            scale=color_scale,
                            title=color_title),
            tooltip=['Weather Description',
                     'Illumination Description',
                     alt.Tooltip('avg_val:Q', format=fmt)]
        )
        .properties(title=title_txt,
                    width=800, height=420)
    )

heatmap_chart = build_top_heatmap(df, metric_choice)

#-------------------------------------------------------------------------------------------------#
# Display Heat-map
#-------------------------------------------------------------------------------------------------#
st.altair_chart(heatmap_chart, use_container_width=True)
