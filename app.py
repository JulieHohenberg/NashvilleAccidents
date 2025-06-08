from vega_datasets import data
import os, streamlit as st, pandas as pd, numpy as np, altair as alt, pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load CSV  (widget-free inside @st.cache_data)
#-------------------------------------------------------------------------------------------------#
DATA_PATH = "data/nashville_accidents.csv"

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if os.path.exists(DATA_PATH):
    df = load_csv(DATA_PATH)
else:
    st.warning("`nashville_accidents.csv` not found – upload it below.")
    up = st.file_uploader("📂 Upload CSV", type="csv")
    if up is None: st.stop()
    df = load_csv(up)

#-------------------------------------------------------------------------------------------------#
# Quick preprocessing  (same as before)
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
# Global filters  (apply to both charts)
#-------------------------------------------------------------------------------------------------#
st.markdown("### Filters (weather & lighting)")

top_weather = df['Weather Description'].value_counts().nlargest(8).index
top_illum   = df['Illumination Description'].value_counts().nlargest(6).index

c1, c2 = st.columns(2, gap="medium")
with c1:
    weather_sel = st.multiselect("Weather Condition(s)",
                                 list(top_weather), default=list(top_weather))
with c2:
    illum_sel = st.multiselect("Lighting Condition(s)",
                               list(top_illum),   default=list(top_illum))

df_filt = df[(df['Weather Description'].isin(weather_sel)) &
             (df['Illumination Description'].isin(illum_sel))]

#-------------------------------------------------------------------------------------------------#
# Bar chart  (filtered)
#-------------------------------------------------------------------------------------------------#
sev_df = df_filt.groupby('Weather Description').agg(
            total_acc=('Weather Description', 'count'),
            inj=('has_injury', 'sum'),
            fat=('has_fatality', 'sum')
         ).reset_index()
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
# Metric selector (only affects heat-map)
#-------------------------------------------------------------------------------------------------#
st.markdown("### Metric (applies to heat-map)")
metric_choice = st.selectbox("Select metric", ["Injuries", "Fatalities"], index=0)

#-------------------------------------------------------------------------------------------------#
# Heat-map (filtered)
#-------------------------------------------------------------------------------------------------#
def build_heat(df_in, metric, w_list, i_list):
    agg_col = 'Number of Injuries' if metric=="Injuries" else 'Number of Fatalities'
    title   = f"Average {metric} per Accident"
    fmt     = '.2f' if metric=="Injuries" else '.3f'

    grp = (df_in.groupby(['Weather Description', 'Illumination Description'])
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

heatmap = build_heat(df_filt, metric_choice, weather_sel, illum_sel)
st.altair_chart(heatmap, use_container_width=True)
