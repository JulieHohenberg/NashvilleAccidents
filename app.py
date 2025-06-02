from vega_datasets import data
import os
import streamlit as st
import pandas as pd
import altair as alt

# Load data
df = pd.read_csv("data/nashville_accidents.csv")

# Preview
st.write(df.head())

# Preview the DataFrame
print(df.head())