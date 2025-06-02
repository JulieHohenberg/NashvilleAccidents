import streamlit as st
import altair as alt
from vega_datasets import data

cars = data.cars()

scatter = alt.Chart(cars).mark_circle(size=60).encode(
    x="Horsepower",
    y="Miles_per_Gallon",
    color="Origin",
    tooltip=["Horsepower", "Miles_per_Gallon", "Origin"]
).interactive()

st.title("Altair Scatterplot in Streamlit")
st.altair_chart(scatter, use_container_width=True, theme=None)
