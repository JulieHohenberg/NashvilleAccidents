from vega_datasets import data
import os, streamlit as st, pandas as pd, numpy as np, altair as alt, pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load CSV (cached) ------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------#
DATA_PATH = "data/nashville_accidents.csv"

@st.cache_data
def load_csv(path_or_buf) -> pd.DataFrame:
    return pd.read_csv(path_or_buf)

if os.path.exists(DATA_PATH):
    df = load_csv(DATA_PATH)
else:
    st.warning("`nashville_accidents.csv` not found – upload it below.")
    up = st.file_uploader("📂 Upload CSV", type="csv")
    if up is None: st.stop()
    df = load_csv(up)

#-------------------------------------------------------------------------------------------------#
# Basic preprocessing ----------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------#

# Date/Time preprocessing 
df['Date and Time'] = pd.to_datetime(df['Date and Time'],
                                     format='%m/%d/%Y %I:%M:%S %p')

df['day_of_week'] = df['Date and Time'].dt.dayofweek   # Monday=0 … Sunday=6
df['hour']        = df['Date and Time'].dt.hour        # 0–23



df['has_injury']   = df['Number of Injuries']   > 0
df['has_fatality'] = df['Number of Fatalities'] > 0

cols_na_unknown = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions',
                   'Illumination Description', 'Collision Type Description']
df[cols_na_unknown] = df[cols_na_unknown].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

exclude = ['OTHER', 'UNKNOWN']
df = df[~df['Weather Description'].isin(exclude) &
        ~df['Illumination Description'].isin(exclude)]
#-------------------------------------------------------------------------------------------------#
# Title
#-------------------------------------------------------------------------------------------------#
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-family: 'Segoe UI', sans-serif; font-size: 48px; margin-bottom: 10px;">
            Weathering the Road
        </h1>
        <h3 style="font-family: 'Segoe UI', sans-serif; font-weight: normal; color: #444;">
            How Conditions and Location Impact Crash Severity and<br>
            Frequency in Nashville, Tennessee
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)

#-------------------------------------------------------------------------------------------------#
# Weather filter  (applies to *both* charts) 
#-------------------------------------------------------------------------------------------------#
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>⛅</div>
        <div style='font-size: 24px;'> <b>
            How does the weather impact the frequency and severity of accidents?
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

#top_weather = df['Weather Description'].value_counts().nlargest(8).index
#weather_sel = st.multiselect("Weather Condition(s)",
                             #list(top_weather),
                             #default=list(top_weather))

#df_weather = df[df['Weather Description'].isin(weather_sel)]

#-------------------------------------------------------------------------------------------------#
# Bar chart  (filtered **only** by Weather)
#-------------------------------------------------------------------------------------------------#
#======================== 1️⃣ WEATHER-ONLY ANALYSIS (BAR CHART) ================================#
with st.expander("Click to explore weather-based accident analysis", expanded=False):

    # Weather filter (affects this bar chart only)
    top_weather = df['Weather Description'].value_counts().nlargest(8).index
    weather_sel_bar = st.multiselect(
        "Weather Condition(s)",
        list(top_weather),
        default=list(top_weather),
        key="weather_sel_bar",
    )

    df_weather_bar = df[df['Weather Description'].isin(weather_sel_bar)]

    # Build bar-chart data
    sev_df = (
        df_weather_bar.groupby('Weather Description')
        .agg(
            total_acc=('Weather Description', 'count'),
            inj=('has_injury', 'sum'),
            fat=('has_fatality', 'sum')
        )
        .reset_index()
    )

    sev_df['% with Injury']   = sev_df['inj'] / sev_df['total_acc'] * 100
    sev_df['% with Fatality'] = sev_df['fat'] / sev_df['total_acc'] * 100
    sev_melt = sev_df.melt(
        'Weather Description',
        ['% with Injury', '% with Fatality'],
        var_name='Severity Type',
        value_name='Percentage'
    )

    bar_chart = (
        alt.Chart(sev_melt)
        .mark_bar()
        .encode(
            x=alt.X(
                'Weather Description:N',
                sort='-y',
                axis=alt.Axis(labelAngle=-35, labelOverlap=False)
            ),
            y='Percentage:Q',
            color=alt.Color(
                'Severity Type:N',
                scale=alt.Scale(
                    domain=['% with Injury', '% with Fatality'],
                    range=['orange', 'crimson']
                )
            ),
            tooltip=[
                'Weather Description',
                'Severity Type',
                alt.Tooltip('Percentage:Q', format='.1f')
            ]
        )
        .properties(
            title='Proportion of Accidents with Injuries or Fatalities',
            height=420,
            width=800
        )
    )

    st.altair_chart(bar_chart, use_container_width=True)

#-------------------------------------------------------------------------------------------------#
# Illumination filter (below bar chart, affects heat-map only) 
#-------------------------------------------------------------------------------------------------#

st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>𓍙</div>
        <div style='font-size: 24px;'> <b>
            How does lighting, in combination with weather, affect accident severity?
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)
#top_illum = df['Illumination Description'].value_counts().nlargest(6).index
#illum_sel = st.multiselect("Lighting Condition(s)",
                           #list(top_illum),
                           #default=list(top_illum))

#-------------------------------------------------------------------------------------------------#
# Metric selector 
#-------------------------------------------------------------------------------------------------#
#metric_choice = st.selectbox("Metric", ["Injuries", "Fatalities"], index=0)

#-------------------------------------------------------------------------------------------------#
# Heat-map  (filtered by Weather + Lighting + Metric)
#-------------------------------------------------------------------------------------------------#
with st.expander("Click to explore temporal & spatial patterns", expanded=False):

    # 1 ── Create labels once ────────────────────────────────────────────────────────────────
    day_mapping  = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',
                    4:'Friday',5:'Saturday',6:'Sunday'}
    hour_mapping = {h: pd.to_datetime(f"{h}:00").strftime("%-I %p") for h in range(24)}

    df['day_name']   = df['day_of_week'].map(day_mapping)
    df['hour_label'] = df['hour'].map(hour_mapping)
    df['day_sort']   = df['day_of_week']
    df['hour_sort']  = df['hour']

    # 2 ── Frequency table ──────────────────────────────────────────────────────────────────
    freq_grouped = (
        df.groupby(['day_name','hour_label','day_sort','hour_sort'])
          .size()
          .reset_index(name='accident_count')
    )

    # 3 ── One shared selection (default = all) ─────────────────────────────────────────────
    sel_time = alt.selection_multi(
        fields=['day_name', 'hour_label'],
        empty='all'
    )

    # 4 ── Day-hour heat-table (top chart) ─────────────────────────────────────────────────
    freq_chart = (
        alt.Chart(freq_grouped)
        .mark_rect()
        .encode(
            x=alt.X('hour_label:N',
                    sort=freq_grouped.sort_values('hour_sort')['hour_label'].unique(),
                    title='Hour of Day'),
            y=alt.Y('day_name:N',
                    sort=freq_grouped.sort_values('day_sort')['day_name'].unique(),
                    title='Day of Week'),
            color=alt.Color('accident_count:Q',
                            scale=alt.Scale(scheme='reds'),
                            title='Accident Frequency'),
            tooltip=['day_name', 'hour_label', 'accident_count']
        )
        .add_selection(sel_time)
        .properties(height=300)
    )

    # 5 ── Geo heat-map filtered by the same selection (bottom chart) ───────────────────────
    df_geo = df[(df['Lat'].between(36.0, 36.4)) & (df['Long'].between(-87.0, -86.5))]

    if df_geo.empty:
        st.warning("No latitude/longitude points fall inside the filtering window.")
    else:
        heatmap_geo = (
            alt.Chart(df_geo)
            .transform_filter(sel_time)
            .mark_rect()
            .encode(
                x=alt.X('Long:Q', bin=alt.Bin(maxbins=60), title='Longitude'),
                y=alt.Y('Lat:Q',  bin=alt.Bin(maxbins=60), title='Latitude'),
                color=alt.Color('count():Q',
                                scale=alt.Scale(scheme='reds'),
                                title='Accident Count'),
                tooltip=[
                    alt.Tooltip('count():Q',   title='Accidents'),
                    alt.Tooltip('day_name:N',  title='Day'),
                    alt.Tooltip('hour_label:N',title='Hour')
                ]
            )
            .configure_axisX(labelAngle=0)
            .properties(height=350)
        )

        # 6 ── Vertically concatenate and display ───────────────────────────────────────────
        stacked = (
            alt.vconcat(freq_chart, heatmap_geo)
               .resolve_scale(color='independent')   # separate colour legends
        )

        st.altair_chart(stacked, use_container_width=True)








