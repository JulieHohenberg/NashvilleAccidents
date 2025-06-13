from vega_datasets import data
import os, streamlit as st, pandas as pd, numpy as np, altair as alt, pydeck as pdk

#-------------------------------------------------------------------------------------------------#
# Load CSV (cached) ------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------#
DATA_PATH = "sampled_nashville_accidents (1).csv"

@st.cache_data
def load_sample_csv(path_or_buf, sample_size=10000, seed=42) -> pd.DataFrame:
    df_full = pd.read_csv(path_or_buf)
    return df_full.sample(n=min(sample_size, len(df_full)), random_state=seed)

if os.path.exists(DATA_PATH):
    df = load_sample_csv(DATA_PATH)
else:
    st.warning("`nashville_accidents.csv` not found – upload it below.")
    up = st.file_uploader("📂 Upload CSV", type="csv")
    if up is None:
        st.stop()
    df = load_sample_csv(up)

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
        <h3 style="font-family: 'Segoe UI', sans-serif; font-weight: normal; color: #444; line-height: 1.5; max-width: 800px; margin: 0 auto;">
            Uncovering the Patterns Behind the Wheel:<br>
            How Weather, Light, Time, and Place Interact to Shape Crash Risk in Nashville
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
### Setting the Stage

Every day, drivers in Nashville face a range of environmental challenges—from foggy mornings to poorly lit roads at night. But which conditions are most dangerous? When are accidents most likely to occur?

This dashboard walks you through an exploratory journey of traffic accident patterns, helping you uncover how **weather**, **lighting**, and **timing** affect accident **frequency** and **severity**.

Let's start by examining where these accidents occur.
""")
####################### MAP ##################################################3
# Filter to valid lat/long range for Nashville
df = df[(df['Lat'] >= 36.0) & (df['Lat'] <= 36.4) &
        (df['Long'] >= -87.0) & (df['Long'] <= -86.5)]

st.markdown("""
### 📍 Where Are Crashes Happening Most?

Before diving into what causes crashes, let’s look at **where** they occur. The map below shows accident hot spots across the Nashville area. Darker colors mean more accidents.

Use your mouse to **zoom** and **pan** around the map to explore different neighborhoods and clusters.
""")

# Compute center of the map
midpoint = (np.average(df["Lat"]), np.average(df["Long"]))

# PyDeck heatmap layer with tooltips
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df,
    get_position='[Long, Lat]',
    get_weight=1,
    radius=200,
    intensity=0.4,
    threshold=0.05,
    pickable=True,
    colorRange=[
        [255,255,204],
        [255,237,160],
        [254,217,118],
        [254,178, 76],
        [253,141, 60],
        [240, 59, 32],
    ],
)

# Define map view with zoom/pan enabled
view_state = pdk.ViewState(
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=11,
    pitch=40,
    bearing=0,
)

# Deck object
deck = pdk.Deck(
    layers=[heatmap_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/satellite-v9",
    tooltip={
        "html": """
            <b>Address:</b> {Location} <br/>
            <b>Injuries:</b> {Number of Injuries} <br/>
            <b>Fatalities:</b> {Number of Fatalities} <br/>
            <b>Lat:</b> {Lat} &nbsp; <b>Long:</b> {Long}
        """,
        "style": {"font-size": "12px"},
    },
)

# Display in Streamlit
st.pydeck_chart(deck, use_container_width=True)

#############################################################################################################################





#-------------------------------------------------------------------------------------------------#
# Weather filter  (applies to *both* charts) 
#-------------------------------------------------------------------------------------------------#
st.markdown(
    """
    ---
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>⛅</div>
        <div style='font-size: 24px;'> <b>
           Weather and Crash Severity 
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### When the Skies Turn Dangerous

Nashville drivers don’t just battle traffic — they battle the elements. But which conditions actually lead to more crashes?

Let’s begin with a big-picture look: how common are accidents during different weather types? The chart below lets you explore this, and you can click any weather type to dig deeper into how dangerous it really is.
""")



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
    top_weather = df['Weather Description'].value_counts().nlargest(8).index.tolist()
    weather_sel_bar = st.multiselect(
        "Weather Condition(s)",
        options=list(top_weather),
        default=list(top_weather),
        key="weather_sel_bar"
    )

    df_weather_bar = df[df['Weather Description'].isin(weather_sel_bar)]

    # Preprocess for total + severity stats
    sev_df = (
        df_weather_bar.groupby('Weather Description')
        .agg(
            total_acc=('Weather Description', 'count'),
            inj=('has_injury', 'sum'),
            fat=('has_fatality', 'sum')
        )
        .reset_index()
    )
    sev_df['% with Injury'] = sev_df['inj'] / sev_df['total_acc'] * 100
    sev_df['% with Fatality'] = sev_df['fat'] / sev_df['total_acc'] * 100

    # Melt for bar chart
    sev_melt = sev_df.melt(
        id_vars='Weather Description',
        value_vars=['% with Injury', '% with Fatality'],
        var_name='Severity Type',
        value_name='Percentage'
    )

    # Selection interaction
    weather_select = alt.selection_point(fields=['Weather Description'], toggle=True, clear='click')

    # Scatterplot – Total accidents
    scatter = (
        alt.Chart(sev_df)
        .mark_circle(size=200)
        .encode(
            x=alt.X('Weather Description:N', sort='-y'),
            y=alt.Y('total_acc:Q', title='Total Accidents'),
            color=alt.Color('total_acc:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Weather Description', 'total_acc']
        )
        .add_params(weather_select)
        .properties(
            width=800,
            height=220,
            title='Total Accidents by Weather Type'
        )
    )

    # Bar chart – Severity proportions, filtered by selection
    bar = (
        alt.Chart(sev_melt)
        .transform_filter(weather_select)
        .mark_bar()
        .encode(
            x=alt.X('Weather Description:N', sort='-y',
                    axis=alt.Axis(labelAngle=-35, labelOverlap=False)),
            y=alt.Y('Percentage:Q'),
            color=alt.Color('Severity Type:N',
                            scale=alt.Scale(
                                domain=['% with Injury', '% with Fatality'],
                                range=['orange', 'crimson'])),
            tooltip=['Weather Description', 'Severity Type',
                     alt.Tooltip('Percentage:Q', format='.1f')]
        )
        .properties(
            width=800,
            height=400,
            title='Proportion of Accidents with Injuries or Fatalities'
        )
    )

    # Combine and show
    st.altair_chart(
        alt.vconcat(scatter, bar).resolve_scale(color='independent'),
        use_container_width=True
    )


#-------------------------------------------------------------------------------------------------#
# Illumination filter (below bar chart, affects heat-map only) 
#-------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------#
# Lighting-and-Weather section ─ big emoji header (kept outside the dropdown)
#-------------------------------------------------------------------------------------------------#
st.markdown(
    """
    ---
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>𓍙</div>
        <div style='font-size: 24px;'><b>
            Lighting Conditions Compound Risk
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### The Perfect Storm?

Rain might be dangerous. Darkness might be dangerous. But what happens when they combine?

This next section explores how **lighting conditions** and **weather** interact to influence crash outcomes. Use the dropdowns to filter for different combinations, and ask: does visibility amplify danger?
""")


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
        "Toggle Between Injuries and Fatalities",
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
    ---
    <div style='display: flex; align-items: center;'>
        <div style='font-size: 80px; margin-right: 20px;'>🕛</div>
        <div style='font-size: 24px;'> <b>
            Timing is Everything
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### Not All Hours Are Equal

Morning commutes, late-night drives, weekend errands — when are accidents most likely to strike?

This section dives into the **time of day** and **day of week** patterns in Nashville’s crashes. Start by clicking any square in the heatmap to see which times are most accident-prone.
""")

with st.expander("Click to explore temporal patterns & collision types", expanded=False):
    st.markdown("The bar chart below updates based on your selection. It reveals which **collision types** are most frequent at the times you choose. You might uncover patterns — like rear-end collisions spiking during morning commutes, or more severe crashes on weekend nights.")
    st.markdown(
        "Click a cell in the grid to filter the bar chart below by day and hour. "
        "**Clicking another cell** adds cells. **Triple-click** anywhere in the grid to reset."
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