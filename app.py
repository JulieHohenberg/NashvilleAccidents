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
df['Date and Time'] = pd.to_datetime(df['Date and Time'],
                                     format='%m/%d/%Y %I:%M:%S %p')
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
            ⛅ Weathering the Road
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

top_weather = df['Weather Description'].value_counts().nlargest(8).index
weather_sel = st.multiselect("Weather Condition(s)",
                             list(top_weather),
                             default=list(top_weather))

df_weather = df[df['Weather Description'].isin(weather_sel)]

#-------------------------------------------------------------------------------------------------#
# Bar chart  (filtered **only** by Weather)
#-------------------------------------------------------------------------------------------------#
sev_df = (df_weather.groupby('Weather Description')
          .agg(total_acc=('Weather Description', 'count'),
               inj=('has_injury', 'sum'),
               fat=('has_fatality', 'sum'))
          .reset_index())

sev_df['% with Injury']   = sev_df['inj'] / sev_df['total_acc'] * 100
sev_df['% with Fatality'] = sev_df['fat'] / sev_df['total_acc'] * 100
sev_melt = sev_df.melt('Weather Description',
                       ['% with Injury', '% with Fatality'],
                       var_name='Severity Type', value_name='Percentage')

bar_chart = (alt.Chart(sev_melt)
    .mark_bar()
    .encode(
        x=alt.X('Weather Description:N', sort='-y',
                axis=alt.Axis(labelAngle=-35, labelOverlap=False)),
        y='Percentage:Q',
        color=alt.Color('Severity Type:N',
                        scale=alt.Scale(domain=['% with Injury','% with Fatality'],
                                        range=['orange','crimson'])),
        tooltip=['Weather Description','Severity Type',
                 alt.Tooltip('Percentage:Q',format='.1f')])
    .properties(title='Proportion of Accidents with Injuries or Fatalities',
                height=420, width=800))

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
top_illum = df['Illumination Description'].value_counts().nlargest(6).index
illum_sel = st.multiselect("Lighting Condition(s)",
                           list(top_illum),
                           default=list(top_illum))

#-------------------------------------------------------------------------------------------------#
# Metric selector 
#-------------------------------------------------------------------------------------------------#
metric_choice = st.selectbox("Metric", ["Injuries", "Fatalities"], index=0)

#-------------------------------------------------------------------------------------------------#
# Heat-map  (filtered by Weather + Lighting + Metric)
#-------------------------------------------------------------------------------------------------#
def build_heat(df_in, metric, w_list, i_list):
    agg_col = 'Number of Injuries' if metric=="Injuries" else 'Number of Fatalities'
    title   = f"Average {metric} per Accident"
    fmt     = '.2f' if metric=="Injuries" else '.3f'

    df_sub = df_in[df_in['Illumination Description'].isin(i_list)]

    grp = (df_sub.groupby(['Weather Description', 'Illumination Description'])
              .agg(total=(agg_col,'sum'),
                   acc_cnt=('Weather Description','count'))
              .reset_index())

    # full grid so every tile shows
    full = pd.MultiIndex.from_product([w_list,i_list],
            names=['Weather Description','Illumination Description'])
    grp = (grp.set_index(['Weather Description','Illumination Description'])
              .reindex(full, fill_value=0).reset_index())
    grp['avg_val'] = np.where(grp['acc_cnt']>0, grp['total']/grp['acc_cnt'], 0)

    color_scale = alt.Scale(scheme='reds', domain=[0, grp['avg_val'].max()])

    return (alt.Chart(grp)
        .mark_rect()
        .encode(
            x='Illumination Description:N',
            y='Weather Description:N',
            color=alt.Color('avg_val:Q', scale=color_scale,
                            title=f'Avg {metric} / Accident'),
            tooltip=['Weather Description','Illumination Description',
                     alt.Tooltip('avg_val:Q',format=fmt)])
        .properties(title=title, width=800, height=420))

heatmap = build_heat(df_weather, metric_choice, weather_sel, illum_sel)
st.altair_chart(heatmap, use_container_width=True)
