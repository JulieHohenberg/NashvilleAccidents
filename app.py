# streamlit_dashboard.py

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
df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%m/%d/%Y %I:%M:%S %p')
df['day_of_week'] = df['Date and Time'].dt.dayofweek
df['hour']        = df['Date and Time'].dt.hour

df['has_injury']   = df['Number of Injuries'] > 0
df['has_fatality'] = df['Number of Fatalities'] > 0

cols_na_unknown = ['Weather Description', 'Reporting Officer', 'HarmfulDescriptions',
                   'Illumination Description', 'Collision Type Description']
df[cols_na_unknown] = df[cols_na_unknown].fillna('UNKNOWN')
df['Weather Description'] = df['Weather Description'].replace('OTHER (NARRATIVE)', 'UNKNOWN')

exclude = ['OTHER', 'UNKNOWN']
df = df[~df['Weather Description'].isin(exclude) & ~df['Illumination Description'].isin(exclude)]

#-------------------------------------------------------------------------------------------------#
# Title ------------------------------------------------------------------------------------------
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
# WEATHER ANALYSIS — Bar + Scatter Linked Charts
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

with st.expander("Click to explore weather-based accident analysis", expanded=False):

    top_weather = df['Weather Description'].value_counts().nlargest(8).index
    weather_sel_bar = st.multiselect(
        "Weather Condition(s)",
        list(top_weather),
        default=list(top_weather),
        key="weather_sel_bar",
    )

    df_weather_bar = df[df['Weather Description'].isin(weather_sel_bar)]

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

    # Interactive scatterplot selector
    scatter_selector = alt.selection_point(fields=['Weather Description'], toggle=True, clear='click')

    acc_count_chart = (
        alt.Chart(sev_df)
        .mark_circle(size=200)
        .encode(
            x=alt.X('Weather Description:N', sort='-y'),
            y=alt.Y('total_acc:Q', title='Total Accidents'),
            color=alt.Color('total_acc:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Weather Description', 'total_acc']
        )
        .add_params(scatter_selector)
        .properties(
            title='Total Accidents by Weather Type',
            width=800,
            height=200
        )
    )

    bar_chart = (
        alt.Chart(sev_melt)
        .transform_filter(scatter_selector)
        .mark_bar()
        .encode(
            x=alt.X('Weather Description:N', sort='-y',
                    axis=alt.Axis(labelAngle=-35, labelOverlap=False)),
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
            width=800,
            height=400
        )
    )

    st.altair_chart(
        alt.vconcat(acc_count_chart, bar_chart).resolve_scale(color='independent'),
        use_container_width=True
    )

#-------------------------------------------------------------------------------------------------#
# REMAINING VISUALS (lighting, heatmap, time/day) STAY AS-IS from original code
#-------------------------------------------------------------------------------------------------#
# You already had all required logic for lighting × weather heatmap and time-of-day charts.
# Nothing else needs to be changed.
