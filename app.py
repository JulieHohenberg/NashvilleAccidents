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

# Add binary time-of-day indicators
df["hour"] = df["Date and Time"].dt.hour
df["is_night"] = df["hour"].apply(lambda h: h < 6 or h >= 20)
df["is_day"] = ~df["is_night"]

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

# -------------------------------------------------------------------------
# 2️⃣  HEAT-MAP DATA  – frequency *proportion* by weather × time-of-day
# -------------------------------------------------------------------------
hm_counts = (
    df_top.groupby(["Weather Description", "Time of Day"])
          .size()
          .reset_index(name="Accident Count")
)
hm_totals = (
    df_top.groupby("Weather Description")
          .size()
          .reset_index(name="Total")
)
hm = hm_counts.merge(hm_totals, on="Weather Description")
hm["Proportion"] = hm["Accident Count"] / hm["Total"]

# -------------------------------------------------------------------------
# 3️⃣  BAR-CHART DATA – severity by the **same** weather × time-of-day
# -------------------------------------------------------------------------
sev = (
    df_top.groupby(["Weather Description", "Time of Day"])
          .agg(
              total = ("Weather Description", "count"),
              inj   = ("has_injury",   "sum"),
              fat   = ("has_fatality", "sum"),
          )
          .reset_index()
)
sev["% with Injury"]   = sev["inj"] / sev["total"] * 100
sev["% with Fatality"] = sev["fat"] / sev["total"] * 100

sev_melt = sev.melt(
    id_vars    = ["Weather Description", "Time of Day"],
    value_vars = ["% with Injury", "% with Fatality"],
    var_name   = "Severity Type",
    value_name = "Percentage",
)

# -------------------------------------------------------------------------
# 4️⃣  SHARED POINT SELECTION – click a cell to update the bars
# -------------------------------------------------------------------------
cell_sel = alt.selection_point(
    fields = ["Weather Description", "Time of Day"],
    empty  = "none",      # no bars until a cell is clicked
    clear  = "dblclick",  # double-click background to reset
)

# -------------------------------- HEAT-MAP -------------------------------- #
heatmap = (
    alt.Chart(hm)
        .mark_rect()
        .encode(
            x      = alt.X("Time of Day:N",  title="Time of Day"),
            y      = alt.Y("Weather Description:N",
                           sort="-x",
                           title="Weather Condition"),
            color  = alt.Color("Proportion:Q",
                               scale=alt.Scale(scheme="reds"),
                               title="Accident Share"),
            stroke      = alt.condition(cell_sel, alt.value("black"), alt.value(None)),
            strokeWidth = alt.condition(cell_sel, alt.value(2),       alt.value(0)),
            tooltip = [
                "Weather Description",
                "Time of Day",
                alt.Tooltip("Accident Count:Q", format=","),
                alt.Tooltip("Proportion:Q",     format=".2%"),
            ],
        )
        .add_params(cell_sel)
        .properties(
            width = 700,
            height= 360,
            title = "Accident share by Weather & Time of Day  (click a cell)",
        )
)

# -------------------------------- BAR-CHART -------------------------------- #
bars = (
    alt.Chart(sev_melt)
        .transform_filter(cell_sel)   # show only the clicked cell’s data
        .mark_bar()
        .encode(
            x     = alt.X("Severity Type:N",
                          sort=["% with Injury", "% with Fatality"],
                          title=None),
            y     = alt.Y("Percentage:Q", title="Percentage of Accidents"),
            color = alt.Color("Severity Type:N",
                              scale=alt.Scale(
                                  domain=["% with Injury", "% with Fatality"],
                                  range=["orange", "crimson"]),
                              legend=None),
            tooltip=[
                "Severity Type",
                alt.Tooltip("Percentage:Q", format=".1f"),
            ],
        )
        .properties(
            width = 700,
            height= 240,
            title = "Severity distribution for selected cell  (dbl-click to reset)",
        )
)

# -------------------------------------------------------------------------
# 5️⃣  STACK VERTICALLY & DISPLAY
# -------------------------------------------------------------------------
linked_viz = alt.vconcat(heatmap, bars).resolve_scale(color="independent")

with st.expander("Weather impact on accident frequency & severity", expanded=True):
    st.altair_chart(linked_viz, use_container_width=True)


#-------------------------------------------------------------------------------------------------#
# Illumination filter (below bar chart, affects heat-map only) 
#-------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------#
# Lighting-and-Weather section ─ big emoji header (kept outside the dropdown)
#-------------------------------------------------------------------------------------------------#
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>𓍙</div>
        <div style='font-size: 24px;'><b>
            How does lighting, in combination with weather, affect accident severity?
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

#======================== 2️⃣ WEATHER × LIGHTING ANALYSIS (HEATMAP) ============================#
with st.expander("Click to explore lighting & weather interaction", expanded=False):

    # Re-select weather (needed for this analysis)
    top_weather2 = df['Weather Description'].value_counts().nlargest(8).index
    weather_sel_heat = st.multiselect(
        "Weather Condition(s)",
        list(top_weather2),
        default=list(top_weather2),
        key="weather_sel_heat",
    )

    # Lighting filter
    top_illum = df['Illumination Description'].value_counts().nlargest(6).index
    illum_sel = st.multiselect(
        "Lighting Condition(s)",
        list(top_illum),
        default=list(top_illum),
        key="illum_multiselect",
    )

    # Metric selector
    metric_choice = st.selectbox(
        "Metric",
        ["Injuries", "Fatalities"],
        index=0,
        key="metric_choice_select",
    )

    # ----------------- Heat-map (filtered by Weather + Lighting + Metric) ------------------ #
    def build_heat(df_in, metric, w_list, i_list):
        agg_col = 'Number of Injuries' if metric == "Injuries" else 'Number of Fatalities'
        title   = f"Average {metric} per Accident"
        fmt     = '.2f' if metric == "Injuries" else '.3f'

        df_sub = df_in[df_in['Illumination Description'].isin(i_list)]

        grp = (
            df_sub.groupby(['Weather Description', 'Illumination Description'])
            .agg(
                total=(agg_col, 'sum'),
                acc_cnt=('Weather Description', 'count')
            )
            .reset_index()
        )

        # full grid so every tile shows
        full = pd.MultiIndex.from_product(
            [w_list, i_list],
            names=['Weather Description', 'Illumination Description']
        )
        grp = (
            grp.set_index(['Weather Description', 'Illumination Description'])
            .reindex(full, fill_value=0)
            .reset_index()
        )
        grp['avg_val'] = np.where(
            grp['acc_cnt'] > 0,
            grp['total'] / grp['acc_cnt'],
            0
        )

        color_scale = alt.Scale(
            scheme='reds',
            domain=[0, grp['avg_val'].max()]
        )

        return (
            alt.Chart(grp)
            .mark_rect()
            .encode(
                x='Illumination Description:N',
                y='Weather Description:N',
                color=alt.Color(
                    'avg_val:Q',
                    scale=color_scale,
                    title=f'Avg {metric} / Accident'
                ),
                tooltip=[
                    'Weather Description',
                    'Illumination Description',
                    alt.Tooltip('avg_val:Q', format=fmt)
                ]
            )
            .properties(title=title, width=800, height=420)
        )

    df_weather_heat = df[df['Weather Description'].isin(weather_sel_heat)]
    heatmap = build_heat(df_weather_heat, metric_choice, weather_sel_heat, illum_sel)
    st.altair_chart(heatmap, use_container_width=True)

########################################################################################
# Time/Day Heatmap and Map
########################################################################################
st.markdown(
    """
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>🕛</div>
        <div style='font-size: 24px;'> <b>
            How do time and day affect accident frequency?
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander("Click to explore temporal patterns & collision types", expanded=False):

    st.markdown(
        "Click a cell in the grid to filter the bar chart below by day and hour. "
        "**Shift-click** adds cells. **Double-click** anywhere in the grid to reset."
    )

    # 1 ── cached prep (day/hour labels & collision counts) -------------------------------
    @st.cache_data
    def prep_tables(_df):
        day_map  = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',
                    4:'Friday',5:'Saturday',6:'Sunday'}
        hour_map = {h: pd.to_datetime(f"{h}:00", format="%H:%M").strftime("%-I %p")
            for h in range(24)}

        _df = _df.copy()
        _df['day_name']   = _df['day_of_week'].map(day_map)
        _df['hour_label'] = _df['hour'].map(hour_map)
        _df['day_sort']   = _df['day_of_week']
        _df['hour_sort']  = _df['hour']

        freq_tbl = (
            _df.groupby(['day_name','hour_label','day_sort','hour_sort'])
                .size().reset_index(name='accident_count')
        )
        return _df, freq_tbl

    df_labeled, freq_grid = prep_tables(df)

    # 2 ── shared selection on the grid (double-click clears) ------------------------------
    sel_time = alt.selection_point(
        name   = "timeSel",
        fields = ['day_name', 'hour_label'],
        toggle = 'event',
        clear  = 'dblclick',   # built-in reset
        empty  = 'all'
    )

    # 3 ── top chart: day × hour grid ------------------------------------------------------
    grid_chart = (
        alt.Chart(freq_grid)
        .mark_rect()
        .encode(
            x=alt.X('hour_label:N',
                    sort=freq_grid.sort_values('hour_sort')['hour_label'].unique(),
                    title='Hour of Day'),
            y=alt.Y('day_name:N',
                    sort=freq_grid.sort_values('day_sort')['day_name'].unique(),
                    title='Day of Week'),
            color=alt.Color('accident_count:Q',
                            scale=alt.Scale(scheme='reds'),
                            title='Accident Frequency'),
            tooltip=['day_name','hour_label','accident_count']
        )
        .add_params(sel_time)
        .properties(height=300)
    )

    # 4 ── bottom chart: frequency per collision type, filtered by the same selection ------
    # bottom chart: frequency per collision type, filtered by the same selection
    collision_freq = (
    alt.Chart(df_labeled)                       # df_labeled already has day/hour labels
      .transform_filter(sel_time)               # ← same selection as the grid
      .transform_aggregate(
          accident_count='count()',             # create the field
          groupby=['Collision Type Description']
      )
      .mark_bar()
      .encode(
          x=alt.X('accident_count:Q',           # quantitative type specified
                  title='Accident Frequency'),
          y=alt.Y('Collision Type Description:N',
                  sort='-x',
                  title='Collision Type Description'),
          color=alt.Color('accident_count:Q',
                          scale=alt.Scale(scheme='reds'),
                          title='Accident Frequency'),
          tooltip=[
              'Collision Type Description:N',
              alt.Tooltip('accident_count:Q', title='Accident Count')
          ]
      )
      .properties(
          height=350,
          title='Accident Frequency by Collision Type'
      )
    )


    # 5 ── vertically concatenate and show --------------------------------------------------
    stacked = (
        alt.vconcat(grid_chart, collision_freq)
           .resolve_scale(color='independent')
           .configure_axisX(labelAngle=0)
    )

    st.altair_chart(stacked, use_container_width=True)
