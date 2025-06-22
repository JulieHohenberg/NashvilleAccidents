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
            How Weather, Light, Time, and Place Interact to Shape Crash Risk in Nashville, Tennessee
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
           Weather Patterns and Risky Hours 
        </b></div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
### When Dangerous Conditions and Timing Collide

Not all weather is equally dangerous — and neither is every hour of the day. When do accidents spike, and under what skies?

The scatterplot below helps you uncover patterns in injury rates by hour and weather type. Each bubble shows how frequent and severe accidents are.
""")



#top_weather = df['Weather Description'].value_counts().nlargest(8).index
#weather_sel = st.multiselect("Weather Condition(s)",
                             #list(top_weather),
                             #default=list(top_weather))

#df_weather = df[df['Weather Description'].isin(weather_sel)]

#-------------------------------------------------------------------------------------------------#
# Bar chart  (filtered **only** by Weather)
#-------------------------------------------------------------------------------------------------#
#======================== 1️⃣ WEATHER-ONLY ANALYSIS (SCATTER & BAR CHARTS) ================================#
with st.expander("Click a bubble to reveal how that hour compares across weather types in overall crash volume.", expanded=False):

    # ------ Weather filter ----------------------------------------------------
    top_weather = df['Weather Description'].value_counts().nlargest(8).index.tolist()
    
    # “Select all” reset button (must appear *before* the multiselect)
    if st.button("Select all weather conditions", key="select_all_weather_btn"):
        st.session_state["weather_sel"] = top_weather
    
    weather_sel = st.multiselect(
        "Weather Condition(s)",
        options=top_weather,
        default=top_weather,
        key="weather_sel"
    )

    # ------ Prepare data ------------------------------------------------------
    df_weather = df[df['Weather Description'].isin(weather_sel)].copy()
    df_weather['hour']       = df_weather['Date and Time'].dt.hour
    df_weather['has_injury'] = df_weather['Number of Injuries'] > 0

    scatter_data = (
        df_weather.groupby(['hour', 'Weather Description'])
        .agg(
            accident_count=('Weather Description', 'count'),
            injury_count=('has_injury', 'sum')
        )
        .reset_index()
    )
    scatter_data['% Injury'] = scatter_data['injury_count'] / scatter_data['accident_count'] * 100

    # ------ Selections --------------------------------------------------------
    hour_select = alt.selection_point(fields=['hour'])         # click-to-filter
    zoom_select = alt.selection_interval(bind='scales')        # drag-zoom / pan

    # ------ TOP CHART: scatter with numeric 0-23 hour labels ------------------
    scatter_chart = (
        alt.Chart(scatter_data)
        .mark_circle()
        .encode(
            x=alt.X(
                'hour:Q',
                title='Hour of Day',
                axis=alt.Axis(tickMinStep=1)   # numeric labels (0–23)
            ),
            y=alt.Y('% Injury:Q', title='% of Accidents with Injury'),
            size=alt.Size('accident_count:Q', scale=alt.Scale(range=[10, 600]), legend=None),
            color=alt.Color('Weather Description:N', legend=alt.Legend(title='Weather')),
            tooltip=[
                alt.Tooltip('hour:Q', title='Hour'),
                'Weather Description',
                alt.Tooltip('accident_count:Q', title='Accident Count'),
                alt.Tooltip('% Injury:Q', format='.1f')
            ]
        )
        .add_params(hour_select, zoom_select)
        .properties(
            width=800,
            height=320,
            title='Injury Rate by Hour and Weather Type'
        )
    )

    # ------ BOTTOM CHART: bar chart filtered by selected hour -----------------
    bar_data = (
        df_weather.groupby(['hour', 'Weather Description'])
        .size()
        .reset_index(name='accident_count')
    )

    bar_chart = (
        alt.Chart(bar_data)
        .transform_filter(hour_select)
        .mark_bar()
        .encode(
            x=alt.X('Weather Description:N', sort='-y'),
            y=alt.Y('accident_count:Q', title='Accident Count'),
            color=alt.Color('Weather Description:N', legend=None),
            tooltip=['Weather Description', 'accident_count']
        )
        .properties(
            width=800,
            height=300,
            title='Accident Volume by Weather (Selected Hour)'
        )
    )
    # ------ Display combined charts -----------------------------------------
    st.caption("🖱️ *Tip: drag to pan or zoom the scatter plot; click a bubble to filter the bar chart.*")
    st.altair_chart(alt.vconcat(scatter_chart, bar_chart), use_container_width=True)

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


#======================== 1️⃣ WEATHER-ONLY ANALYSIS (SCATTER & BAR CHARTS) ================================#
with st.expander("Click a bubble to reveal how that hour compares across weather types in overall crash volume.", expanded=False):

    # ---------- Weather filter ------------------------------------------------
    top_weather = df['Weather Description'].value_counts().nlargest(8).index.tolist()

    if "weather_sel" not in st.session_state:
        st.session_state["weather_sel"] = top_weather

    if st.button("Select all weather conditions", key="select_all_weather_btn"):
        st.session_state["weather_sel"] = top_weather

    weather_sel = st.multiselect(
        "Weather Condition(s)",
        options=top_weather,
        key="weather_sel"
    )

    # ---------- Prepare data --------------------------------------------------
    df_weather = df[df['Weather Description'].isin(weather_sel)].copy()
    df_weather['hour']       = df_weather['Date and Time'].dt.hour
    df_weather['has_injury'] = df_weather['Number of Injuries'] > 0

    scatter_data = (
        df_weather.groupby(['hour', 'Weather Description'])
        .agg(
            accident_count=('Weather Description', 'count'),
            injury_count=('has_injury', 'sum')
        )
        .reset_index()
    )
    scatter_data['% Injury'] = scatter_data['injury_count'] / scatter_data['accident_count'] * 100

    # ---------- Selections ----------------------------------------------------
    hour_select = alt.selection_point(fields=['hour'])         # click-to-filter
    zoom_select = alt.selection_interval(bind='scales')        # drag-zoom / pan (shared)

    # ---------- TOP CHART: scatter -------------------------------------------
    scatter_chart = (
        alt.Chart(scatter_data)
        .mark_circle()
        .encode(
            x=alt.X('hour:Q', title='Hour of Day', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('% Injury:Q', title='% of Accidents with Injury'),
            size=alt.Size('accident_count:Q', scale=alt.Scale(range=[10, 600]), legend=None),
            color=alt.Color('Weather Description:N', legend=alt.Legend(title='Weather')),
            tooltip=[
                alt.Tooltip('hour:Q', title='Hour'),
                'Weather Description',
                alt.Tooltip('accident_count:Q', title='Accident Count'),
                alt.Tooltip('% Injury:Q', format='.1f')
            ]
        )
        .add_params(hour_select, zoom_select)
        .properties(width=800, height=320,
                    title='Injury Rate by Hour and Weather Type')
    )

    # ---------- BOTTOM CHART: bar (now zoom/pan enabled on y-axis) -----------
    bar_data = (
        df_weather.groupby(['hour', 'Weather Description'])
        .size()
        .reset_index(name='accident_count')
    )

    bar_chart = (
        alt.Chart(bar_data)
        .transform_filter(hour_select)
        .mark_bar()
        .encode(
            x=alt.X('Weather Description:N', sort='-y'),
            y=alt.Y('accident_count:Q', title='Accident Count'),
            color=alt.Color('Weather Description:N', legend=None),
            tooltip=['Weather Description', 'accident_count']
        )
        .add_params(zoom_select)   # ⬅️ same zoom/pan interaction
        .properties(width=800, height=300,
                    title='Accident Volume by Weather (Selected Hour)')
    )
    # Apply filters and draw heat-map
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

# ─── 📊 Key Insights & Calls to Action (Vertical Selector) ─────────────────────

st.markdown("## 🔍 Key Insights & Recommended Actions")

# 1️⃣  DATA --------------------------------------------------------------------
insights = {
    "⚠️ Injuries ↑ — Fog at dawn & night sleet/hail": {
        "metric": "Highest <strong>injuries / crash</strong>",
        "why": "Limited dawn visibility & slick surfaces sharply raise injury odds.",
        "actions": [
            "Deploy <strong>weather-triggered variable-speed signs</strong> on fog/sleet corridors",
            "Schedule <strong>pre-dawn sanding/salting</strong> for bridges & overpasses",
        ],
        "owners": "<strong>Nashville DPW & Office of Emergency Mgmt</strong>",
    },
    "🌑 Fatalities ↑ — Clear but dark, unlit roads": {
        "metric": "Highest <strong>fatalities / crash</strong>",
        "why": "Drivers over-estimate visibility; reaction time plummets in darkness.",
        "actions": [
            "<strong>Fast-track LED street-light builds</strong> on high-speed rural arterials",
            "Refresh <strong>reflective lane & shoulder markings</strong> and add rumble strips",
        ],
        "owners": "<strong>Metro Works & THSO</strong>",
    },
    "🕒 Crash Frequency ↑ — 5 AM wknd · 11 PM Fri · Wed 8-10 PM": {
        "metric": "Highest <strong>crash count</strong> windows",
        "why": "Fatigue and impaired driving cluster during these time slots.",
        "actions": [
            "Schedule <strong>DUI checkpoints & visibility patrols</strong> during peaks",
            "Pilot <strong>late-night transit/ride-hail vouchers</strong> to cut drunk-driving exposure",
        ],
        "owners": "<strong>Metro Police & Transit Authority</strong>",
    },
    "🚗 Crash Mode ↑ — Front-to-Rear collisions": {
        "metric": "Most <strong>frequent collision type</strong>",
        "why": "Tailgating and abrupt braking trigger chain-reaction crashes.",
        "actions": [
            "Launch <strong>“Leave 3 Seconds” anti-tailgating media blitz</strong>",
            "<strong>Re-time signals</strong> on high-volume corridors to smooth stop-and-go waves",
        ],
        "owners": "<strong>DPW Traffic Engineering</strong>",
    },
}

# 2️⃣  STYLE TWEAKS ------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Radio labels a bit bigger & bolder */
        div[data-testid="stRadio"] label {font-weight:600;font-size:0.95rem;}
        /* Info card look */
        div.infobox {
            border: 1px solid #e6e6e6;
            border-radius: 0.5rem;
            padding: 1rem 1.25rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        /* Bullet spacing */
        ul {margin-top:0.25rem;}
        ul li {margin-bottom:0.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# 3️⃣  SELECTOR & DISPLAY ------------------------------------------------------
choice = st.radio(
    "Select an insight to view details:",
    list(insights.keys()),
    index=0,
)

item = insights[choice]

st.markdown(f"""
<div class="infobox">
<b>Risk metric:</b> {item['metric']}<br><br>

<b>Why it matters</b><br>
{item['why']}<br><br>

<b>Recommended actions ({item['owners']})</b>
<ul>
    {''.join(f'<li>{act}</li>' for act in item['actions'])}
</ul>
</div>
""", unsafe_allow_html=True)

